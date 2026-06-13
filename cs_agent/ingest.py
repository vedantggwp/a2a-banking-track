"""Build the Redis knowledge-base index from kb/documents at startup.

Runs before the agent is served (main.py imports it), so the agent card only
becomes available once the index is ready. Embeddings load from the pre-baked
cache (kb/embeddings.json) when present, else fall back to live embedding;
without model credentials the index is BM25-only."""

import base64
import json
import os
import struct
import sys
from pathlib import Path

import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

from rag_tools import DOC_PREFIX, EMBEDDING_DIM, KB_INDEX, REDIS_URL, _embed

KB_DOCUMENTS_DIR = Path(os.environ.get("KB_DOCUMENTS_DIR", "/app/kb/documents"))
# Pre-baked {doc_id: base64(float32)} cache (see precompute_embeddings.py).
KB_EMBEDDINGS_PATH = Path(os.environ.get("KB_EMBEDDINGS_PATH", "/app/kb/embeddings.json"))

EMBED_BATCH_SIZE = 25


def load_embedding_cache() -> dict[str, bytes]:
    """Load pre-baked embedding bytes by doc id (empty dict if no cache)."""
    if not KB_EMBEDDINGS_PATH.exists():
        return {}
    with open(KB_EMBEDDINGS_PATH) as fp:
        raw = json.load(fp)
    return {doc_id: base64.b64decode(b64) for doc_id, b64 in raw.items()}


def load_documents() -> list[dict]:
    """Load all KB documents ({id, title, content})."""
    docs = []
    for path in sorted(KB_DOCUMENTS_DIR.glob("*.json")):
        with open(path) as fp:
            docs.append(json.load(fp))
    return docs


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
                "will be BM25-only (kb_search_bm25 still works)",
                file=sys.stderr,
            )

    pipe = client.pipeline(transaction=False)
    for doc, emb in zip(documents, embedding_bytes):
        mapping = {"title": doc["title"], "content": doc["content"]}
        if emb is not None:
            mapping["embedding"] = emb
        pipe.hset(f"{DOC_PREFIX}{doc['id']}", mapping=mapping)
    pipe.execute()
    print(f"[ingest] indexed {len(documents)} documents into {KB_INDEX}", file=sys.stderr)


if __name__ == "__main__":
    build_index()
