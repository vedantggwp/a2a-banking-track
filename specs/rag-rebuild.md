# RAG rebuild — the #1 lever (A2A track)

**Reward truth (confirmed from harness + task JSON):** 70/79 tasks are **DB-basis, BINARY** — reward 1.0 *only* if the exact gated **write** tool-call fires with correct args. Reads don't score. The chain that scores: **verify identity → discover the correct internal tool via RAG → unlock → call with exact args.** So retrieval must surface the EXACT internal tool name + its arg schema, reliably.

Frozen: model `gemini-3.5-flash`, embeddings `gemini-embedding-001` (768-dim); compose shape; **contextId discipline** (keep `to_a2a(root_agent)` or replicate `ctx.session.id == contextId`); env tools fetched live. `a2a-hack smoke` + `test_m1_contextid` are the conformance gate — do not break them.

## Changes (priority order)
1. **Hybrid `kb_search`** (`cs_agent/rag_tools.py`): run BM25 + vector, **RRF-fuse**, return merged top-k. Replaces the BM25/vector either-or. Keep the two underlying fns as internals.
2. **Index a `tool_names` / keywords field at ingest** (`cs_agent/ingest.py`): regex-extract internal tool names (`\w+_\d{4}`) + account-class names from content; store as a field and `RETURN` it — so the exact tool string + args reach the model. This is what makes the gated write fire.
3. **Chunk long docs by `##` heading** (`ingest.py`) + **raise default top_k to 8–10** (`rag_tools.py`). Fixes comparison/fee tasks needing many docs (task_001 lists ~25 required docs).
4. **Per-account-class digest docs at ingest** (`ingest.py`): one synthetic doc per account class with its fee/eligibility/tool table, so a single search yields the full picture.
5. **CS prompt** (`cs_agent/agent.py` RAG_GUIDANCE): mandate **exhaustive multi-search before recommending**; unlock the tool to read its real param schema before calling; never recommend before KB-confirming. Defeats the built-in trap tasks.
6. **Personal prompt** (`personal_agent/agent.py`): **drive to completion** — after CS confirms, FIRE the user-side write tool with exact args, don't just narrate. Many DB tasks need a user-scope write.
7. **Pre-bake the Redis index into the cs-agent image** (`cs_agent/Dockerfile` + `ingest.py`): instant startup, avoid the 10-min task timeout (= reward 0).

## Offline test (no Vertex key needed)
- Redis-stack on `localhost:6379` (RediSearch). Ingest BM25 (+ prebaked `kb/embeddings.json` if present).
- Assert `kb_search` returns the right doc **and** `tool_names` for: "overdraft fee Blue Account", "open a fee-free account", "refer a friend". Vector re-embedding for new chunks waits for the key.
