"""Knowledge-base search tools backed by Redis (RediSearch).

kb_search performs hybrid retrieval: BM25 plus optional vector search, fused
with reciprocal rank fusion (RRF). Vector search is available only when the
index contains prebaked gemini-embedding-001 vectors and GOOGLE_API_KEY is set
for query embedding.

Replies are parsed via execute_command so both the classic array reply and
the Redis 8 map-style reply work regardless of redis-py version."""

import concurrent.futures
import os
import re
import struct
import sys
from collections import defaultdict

import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
KB_INDEX = "kb_idx"
DOC_PREFIX = "doc:"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
DEFAULT_TOP_K = 10
RRF_K = 60
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "for",
    "how",
    "i",
    "in",
    "is",
    "me",
    "my",
    "of",
    "on",
    "the",
    "to",
}

_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
_genai_client = None


def _get_genai_client():
    """Reused genai client (one connection pool, not a new one per search)."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is required for live query/chunk embedding. "
            "Precompute kb/embeddings.json with gemini-embedding-001 when "
            "adding chunks, then deploy that cache for startup indexing."
        )
    global _genai_client
    if _genai_client is None:
        from google import genai

        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with gemini-embedding-001 via google-genai."""
    from google.genai import types

    # Reduced-dim output is unnormalized; the index uses COSINE, so that's fine.
    result = _get_genai_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    return [e.values for e in result.embeddings]


# A throttled/slow embedding key must never stall search. Cap the live query
# embed with a hard wall-clock timeout; on timeout the caller (vector search)
# raises, is caught, and falls back to BM25 — hybrid is restored automatically
# once the key responds within the cap again.
_EMBED_TIMEOUT_S = float(os.environ.get("KB_EMBED_TIMEOUT_S", "6"))
_EMBED_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _embed_bounded(texts: list[str]) -> list[list[float]]:
    """Embed `texts`, but give up after _EMBED_TIMEOUT_S seconds (raises)."""
    return _EMBED_EXECUTOR.submit(_embed, texts).result(timeout=_EMBED_TIMEOUT_S)


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _parse_search_reply(reply) -> list[dict]:
    """Normalize an FT.SEARCH reply (array or map shape) to result dicts."""
    if isinstance(reply, dict):
        results = reply.get(b"results", reply.get("results")) or []
        out = []
        for row in results:
            attrs = row.get(b"extra_attributes", row.get("extra_attributes")) or {}
            doc = {"doc_id": _decode(row.get(b"id", row.get("id", "")))}
            doc.update({_decode(k): _decode(v) for k, v in attrs.items()})
            out.append(doc)
        return out
    out = []
    for i in range(1, len(reply) - 1, 2):
        doc = {"doc_id": _decode(reply[i])}
        fields = reply[i + 1]
        for j in range(0, len(fields) - 1, 2):
            doc[_decode(fields[j])] = _decode(fields[j + 1])
        out.append(doc)
    return out


def _strip_score(docs: list[dict]) -> list[dict]:
    for doc in docs:
        doc.pop("score", None)
    return docs


def _kb_search_bm25(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Full-text (BM25) search over the Rho-Bank knowledge base."""
    terms = [
        term
        for term in re.findall(r"\w+", query.lower())
        if len(term) > 1 and term not in QUERY_STOPWORDS
    ]
    if not terms:
        return []
    # OR-join: RediSearch defaults to AND, which zeroes out long queries.
    or_query = "|".join(dict.fromkeys(terms))
    reply = _client.execute_command(
        "FT.SEARCH", KB_INDEX, or_query,
        "LIMIT", "0", str(top_k),
        "RETURN", "5", "title", "content", "tool_names", "keywords", "source_doc_id",
    )
    return _parse_search_reply(reply)


def _kb_search_vector(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Semantic (vector) search over the Rho-Bank knowledge base."""
    try:
        vector = struct.pack(f"{EMBEDDING_DIM}f", *_embed_bounded([query])[0])
        reply = _client.execute_command(
            "FT.SEARCH", KB_INDEX, f"*=>[KNN {top_k} @embedding $vec AS score]",
            "PARAMS", "2", "vec", vector,
            "SORTBY", "score",
            "LIMIT", "0", str(top_k),
            "RETURN", "6", "title", "content", "tool_names", "keywords", "source_doc_id", "score",
            "DIALECT", "2",
        )
        return _strip_score(_parse_search_reply(reply))
    except Exception:
        return []


def _rrf_fuse(result_sets: list[list[dict]], top_k: int) -> list[dict]:
    scores: dict[str, float] = defaultdict(float)
    docs_by_id: dict[str, dict] = {}
    for docs in result_sets:
        for rank, doc in enumerate(docs, start=1):
            doc_id = doc.get("doc_id")
            if not doc_id:
                continue
            scores[doc_id] += 1.0 / (RRF_K + rank)
            docs_by_id.setdefault(doc_id, doc)
    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [docs_by_id[doc_id] for doc_id in ranked_ids]


def kb_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Hybrid RAG search over the Rho-Bank knowledge base.

    Args:
        query: Keywords, account class, tool name, or a natural-language
            question. Use several focused searches for complex tasks.
        top_k: Number of documents to return.

    Returns:
        Merged top-k documents with doc_id, title, content, tool_names,
        keywords, and source_doc_id. tool_names contains discovered internal
        tool strings and documented tool signatures when present.
    """
    top_k = max(1, min(int(top_k or DEFAULT_TOP_K), 25))
    print(f"[kb_search] {query!r} top_k={top_k}", file=sys.stderr)  # prove RAG is used
    bm25_docs = _kb_search_bm25(query, top_k=top_k * 2)
    vector_docs = _kb_search_vector(query, top_k=top_k * 2)
    return _rrf_fuse([bm25_docs, vector_docs], top_k=top_k)
