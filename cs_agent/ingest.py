"""Build the Redis knowledge-base index from kb/documents at startup.

Runs before the agent is served (main.py imports it), so the agent card only
becomes available once the index is ready. Embeddings load from the pre-baked
cache (kb/embeddings.json) when present, else fall back to live embedding;
without model credentials the index is BM25-only."""

import base64
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

from rag_tools import DOC_PREFIX, EMBEDDING_DIM, KB_INDEX, REDIS_URL, _embed

MODULE_DIR = Path(__file__).resolve().parent


def _first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


KB_DOCUMENTS_DIR = Path(
    os.environ.get(
        "KB_DOCUMENTS_DIR",
        str(
            _first_existing_path(
                MODULE_DIR / "kb" / "documents",
                MODULE_DIR.parent / "kb" / "documents",
                Path("/app/kb/documents"),
            )
        ),
    )
)
# Pre-baked {doc_id: base64(float32)} cache (see precompute_embeddings.py).
KB_EMBEDDINGS_PATH = Path(
    os.environ.get(
        "KB_EMBEDDINGS_PATH",
        str(KB_DOCUMENTS_DIR.parent / "embeddings.json"),
    )
)

EMBED_BATCH_SIZE = 25
CHUNK_LONG_DOC_CHARS = int(os.environ.get("KB_CHUNK_LONG_DOC_CHARS", "1800"))

TOOL_NAME_RE = re.compile(r"\b[A-Za-z_]\w*_\d{4}\b")
UNDERSCORE_TOOL_RE = re.compile(r"\b[a-z][a-z0-9_]*_[a-z0-9_]+\b")
CALLABLE_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
ACCOUNT_CLASS_RE = re.compile(
    r"\b[A-Z][A-Za-z+]*(?:[- ][A-Z][A-Za-z+]*(?:\s*\([^)]+\))?)* "
    r"(?:Account|Card|Saver|Vault)(?:\s*\([^)]+\))?\b"
)
H2_RE = re.compile(r"(?m)^##\s+(.+?)\s*$")
KEY_LINE_RE = re.compile(
    r"fee|waiv|eligib|requirement|minimum|maximum|limit|apy|annual|monthly|"
    r"deposit|referral|refer|bonus|open|account_class|tool|signature",
    re.IGNORECASE,
)
GENERIC_ACCOUNT_CLASSES = {
    "Account",
    "Bank Account",
    "Business Account",
    "Business Checking Account",
    "Business Credit Card",
    "Business Rewards Card",
    "Business Savings Account",
    "Checking Account",
    "Credit Card",
    "Debit Card",
    "Eligible Business Credit Card",
    "Eligible Business Rewards Card",
    "Linked Checking Account",
    "No Checking Account",
    "Replacement Card",
    "Replacement Credit Card",
    "Rewards Card",
    "Same Card",
    "Savings Account",
    "The Account",
    "Virtual Card",
    "Your Account",
    "Your Business Account",
    "Your Card",
    "Your Credit Card",
    "Your Debit Card",
}
TOOL_TOKEN_EXCLUDES = {
    "account_class",
    "account_id",
    "account_type",
    "agent_tool_name",
    "amount",
    "arguments_json",
    "card_name",
    "destination_account_id",
    "discoverable_tool_name",
    "source_account_id",
    "time_verified",
    "tool_name",
    "user_id",
}


def load_embedding_cache() -> dict[str, bytes]:
    """Load pre-baked embedding bytes by doc id (empty dict if no cache)."""
    if not KB_EMBEDDINGS_PATH.exists():
        return {}
    with open(KB_EMBEDDINGS_PATH) as fp:
        raw = json.load(fp)
    return {doc_id: base64.b64decode(b64) for doc_id, b64 in raw.items()}


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:80] or "section"


def extract_tool_names(text: str) -> list[str]:
    """Extract internal numeric tools plus documented callable signatures."""
    names = set(TOOL_NAME_RE.findall(text))
    for line in text.splitlines():
        lowered = line.lower()
        if "tool" not in lowered and "signature" not in lowered and "`" not in line:
            continue
        names.update(
            name for name in UNDERSCORE_TOOL_RE.findall(line)
            if name not in TOOL_TOKEN_EXCLUDES
        )
        for name in CALLABLE_RE.findall(line):
            if "_" in name and name not in {"e_g", "example_tool"}:
                names.add(name)
    return sorted(names)


def extract_account_classes(text: str) -> list[str]:
    """Extract product/account class names used as high-value RAG keywords."""
    names = set()
    for match in ACCOUNT_CLASS_RE.findall(text):
        name = re.sub(r"\s+", " ", match).strip(" .,:;`'\"")
        if name in GENERIC_ACCOUNT_CLASSES:
            continue
        if name.startswith(("Your ", "The ", "This ", "New ")):
            continue
        names.add(name)
    return sorted(names)


def _enrich_document(doc: dict) -> dict:
    text = f"{doc['title']}\n{doc['content']}"
    account_classes = extract_account_classes(text)
    exact_tools = extract_tool_names(text)
    preset_keywords = doc.get("keywords", [])
    preset_tools = [keyword for keyword in preset_keywords if "_" in keyword]
    # The field intentionally carries both exact tool strings and account-class
    # keywords, matching the RAG rebuild spec's "tool_names / keywords" target.
    tool_names = sorted(set(exact_tools + account_classes + preset_tools))
    keywords = sorted(set(account_classes + exact_tools + preset_keywords))
    return {
        **doc,
        "tool_names": ", ".join(tool_names),
        "keywords": ", ".join(keywords),
        "source_doc_id": doc.get("source_doc_id", doc["id"]),
    }


def _load_raw_documents() -> list[dict]:
    """Load raw KB documents ({id, title, content}) from disk."""
    docs = []
    for path in sorted(KB_DOCUMENTS_DIR.glob("*.json")):
        with open(path) as fp:
            docs.append(json.load(fp))
    return docs


def _chunk_document(doc: dict) -> list[dict]:
    content = doc["content"]
    headings = list(H2_RE.finditer(content))
    if len(content) < CHUNK_LONG_DOC_CHARS or len(headings) <= 1:
        return [doc]

    chunks = []
    preamble = content[: headings[0].start()].strip()
    for index, heading in enumerate(headings, start=1):
        next_start = headings[index].start() if index < len(headings) else len(content)
        section = content[heading.start() : next_start].strip()
        if index == 1 and preamble:
            section = f"{preamble}\n\n{section}"
        heading_text = heading.group(1).strip()
        chunks.append(
            {
                "id": f"{doc['id']}__h{index:02d}_{_slug(heading_text)}",
                "title": f"{doc['title']} - {heading_text}",
                "content": section,
                "source_doc_id": doc["id"],
            }
        )
    return chunks


def _category_for_doc(doc_id: str) -> str:
    if doc_id.startswith("doc_checking_accounts_"):
        return "personal checking"
    if doc_id.startswith("doc_savings_accounts_"):
        return "personal savings"
    if doc_id.startswith("doc_business_checking_accounts_"):
        return "business checking"
    if doc_id.startswith("doc_business_savings_accounts_"):
        return "business savings"
    if doc_id.startswith("doc_credit_cards_"):
        return "personal credit card"
    if doc_id.startswith("doc_business_credit_cards_"):
        return "business credit card"
    return "general"


def _account_type_for_category(category: str) -> str | None:
    if "checking" in category:
        return "checking"
    if "savings" in category:
        return "savings"
    return None


def _digest_lines_for(account_class: str, docs: list[dict], max_lines: int = 48) -> list[str]:
    lines = []
    seen = set()
    for doc in docs:
        for raw_line in doc["content"].splitlines():
            line = raw_line.strip()
            if not line or line in seen:
                continue
            if account_class in line or KEY_LINE_RE.search(line):
                seen.add(line)
                lines.append(line)
                if len(lines) >= max_lines:
                    return lines
    return lines


def _build_account_digest_documents(raw_documents: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for doc in raw_documents:
        text = f"{doc['title']}\n{doc['content']}"
        for account_class in extract_account_classes(text):
            grouped[account_class].append(doc)

    digests = []
    for account_class, docs in sorted(grouped.items()):
        categories = Counter(_category_for_doc(doc["id"]) for doc in docs)
        category = categories.most_common(1)[0][0]
        account_type = _account_type_for_category(category)
        tools = set()
        has_referral_terms = False
        for doc in docs:
            text = f"{doc['title']}\n{doc['content']}"
            tools.update(extract_tool_names(text))
            has_referral_terms = has_referral_terms or bool(
                re.search(r"\b(refer|refers|referred|referral|referrals)\b", text, re.IGNORECASE)
            )
        if account_type is not None:
            tools.add("open_bank_account_4821")
            if account_type == "savings":
                tools.add("transfer_funds_between_bank_accounts_7291")
        if has_referral_terms and ("checking" in category or "credit card" in category):
            tools.add("submit_referral")

        source_ids = [doc["id"] for doc in docs[:12]]
        content_lines = [
            f"## Account-class digest: {account_class}",
            f"- Category: {category}",
            f"- Source docs: {', '.join(source_ids)}",
        ]
        if account_type is not None:
            content_lines.append(
                "- Opening write tool: "
                f"open_bank_account_4821(user_id, account_type='{account_type}', "
                f"account_class='{account_class}')"
            )
            if account_type == "savings":
                content_lines.append(
                    "- Optional funding write tool after opening: "
                    "transfer_funds_between_bank_accounts_7291"
                    "(source_account_id, destination_account_id, amount)"
                )
        if "submit_referral" in tools:
            content_lines.append(
                "- Referral completion: after CS confirms eligibility and terms, "
                "the personal agent should call its live user-side "
                "submit_referral tool with the exact schema/args from the user "
                "tool list."
            )
        if tools:
            content_lines.append(f"- Tool names: {', '.join(sorted(tools))}")
        content_lines.append("")
        content_lines.append("## Fees, eligibility, limits, and referral facts")
        content_lines.extend(_digest_lines_for(account_class, docs))
        digests.append(
            {
                "id": f"doc_digest_account_class_{_slug(account_class)}",
                "title": f"Digest: {account_class}",
                "content": "\n".join(content_lines),
                "source_doc_id": ",".join(source_ids),
                "keywords": [account_class, category, *sorted(tools)],
            }
        )
    return digests


def load_documents() -> list[dict]:
    """Load raw docs, heading chunks, and synthetic account-class digests."""
    raw_documents = _load_raw_documents()
    documents: list[dict] = []
    for doc in raw_documents:
        documents.extend(_chunk_document(doc))
    documents.extend(_build_account_digest_documents(raw_documents))
    return [_enrich_document(doc) for doc in documents]


def build_index() -> None:
    """(Re)create the KB index and load every document, embedding if possible."""
    client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No KB documents found in {KB_DOCUMENTS_DIR}")

    try:
        client.ft(KB_INDEX).dropindex(delete_documents=True)
    except redis.ResponseError:
        pass

    client.ft(KB_INDEX).create_index(
        fields=[
            TextField("title", weight=2.0),
            TextField("content"),
            TextField("tool_names", weight=4.0),
            TextField("keywords", weight=3.0),
            TextField("source_doc_id"),
            VectorField(
                "embedding",
                "HNSW",
                {"TYPE": "FLOAT32", "DIM": EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
            ),
        ],
        definition=IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH),
    )

    # Pre-baked cache first; live-embed only the misses (BM25-only if neither).
    cache = load_embedding_cache()
    embedding_bytes: list[bytes | None] = [cache.get(d["id"]) for d in documents]
    misses = [i for i, b in enumerate(embedding_bytes) if b is None]
    if cache:
        print(
            f"[ingest] embedding cache hit for {len(documents) - len(misses)}/"
            f"{len(documents)} documents",
            file=sys.stderr,
        )
    if misses:
        if os.environ.get("GOOGLE_API_KEY"):
            try:
                for start in range(0, len(misses), EMBED_BATCH_SIZE):
                    idx = misses[start : start + EMBED_BATCH_SIZE]
                    vectors = _embed([f"{documents[i]['title']}\n{documents[i]['content']}" for i in idx])
                    for i, vector in zip(idx, vectors):
                        embedding_bytes[i] = struct.pack(f"{EMBEDDING_DIM}f", *vector)
                print(f"[ingest] live-embedded {len(misses)} uncached documents", file=sys.stderr)
            except Exception as e:
                print(
                    f"[ingest] embeddings unavailable ({e}); {len(misses)} doc(s) "
                    "will be BM25-only. To refresh vectors, set GOOGLE_API_KEY "
                    "and run cs_agent/precompute_embeddings.py for "
                    "gemini-embedding-001 / 768-dim.",
                    file=sys.stderr,
                )
        else:
            print(
                f"[ingest] GOOGLE_API_KEY not set; {len(misses)} doc(s) will be "
                "BM25-only. To refresh vectors for new chunks, set GOOGLE_API_KEY "
                "and run cs_agent/precompute_embeddings.py for "
                "gemini-embedding-001 / 768-dim.",
                file=sys.stderr,
            )

    pipe = client.pipeline(transaction=False)
    for doc, emb in zip(documents, embedding_bytes):
        mapping = {
            "title": doc["title"],
            "content": doc["content"],
            "tool_names": doc.get("tool_names", ""),
            "keywords": doc.get("keywords", ""),
            "source_doc_id": doc.get("source_doc_id", doc["id"]),
        }
        if emb is not None:
            mapping["embedding"] = emb
        pipe.hset(f"{DOC_PREFIX}{doc['id']}", mapping=mapping)
    pipe.execute()
    print(f"[ingest] indexed {len(documents)} documents into {KB_INDEX}", file=sys.stderr)


if __name__ == "__main__":
    build_index()
