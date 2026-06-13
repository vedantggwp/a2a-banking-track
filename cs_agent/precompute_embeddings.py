"""Precompute the pre-baked KB embedding cache (kb/embeddings.json).

Run this once whenever you change kb/documents so the baked vectors stay in
lockstep with the docs. It embeds every document with the same model/dim the
agent uses, so ingest.py can load them at startup instead of calling the
embedding API (instant startup, no per-restart embedding cost). Needs Google
model credentials (the same GOOGLE_* env the cs-agent uses).

    python precompute_embeddings.py
"""

import base64
import json
import os
import struct
import sys
from pathlib import Path

from ingest import load_documents
from rag_tools import EMBEDDING_DIM, _embed

KB_EMBEDDINGS_PATH = Path(os.environ.get("KB_EMBEDDINGS_PATH", "/app/kb/embeddings.json"))
EMBED_BATCH_SIZE = 25


def main() -> None:
    """Embed every KB document and write the {doc_id: base64(float32)} cache."""
    documents = load_documents()
    if not documents:
        raise RuntimeError("No KB documents found")
    cache: dict[str, str] = {}
    for start in range(0, len(documents), EMBED_BATCH_SIZE):
        batch = documents[start : start + EMBED_BATCH_SIZE]
        vectors = _embed([f"{d['title']}\n{d['content']}" for d in batch])
        for doc, vector in zip(batch, vectors):
            cache[doc["id"]] = base64.b64encode(struct.pack(f"{EMBEDDING_DIM}f", *vector)).decode()
        print(f"[precompute] embedded {min(start + EMBED_BATCH_SIZE, len(documents))}/{len(documents)}", file=sys.stderr)
    KB_EMBEDDINGS_PATH.write_text(json.dumps(cache))
    print(f"[precompute] wrote {len(cache)} embeddings to {KB_EMBEDDINGS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
