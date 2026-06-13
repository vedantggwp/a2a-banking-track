# Rho-Bank A2A — Two Interoperable Customer-Service Agents

[![A2A Protocol](https://img.shields.io/badge/A2A-Protocol-blue)](https://a2a-protocol.org) [![Discord](https://img.shields.io/discord/1391916121589944320?color=7289da&label=Discord&logo=discord&logoColor=white)](https://discord.gg/674NGXpAjU)

Two interoperable [A2A](https://a2a-protocol.org) agents for the **Rho-Bank
customer-service track**, scored on Sierra's [tau2-bench](https://github.com/sierra-research/tau2-bench):

- **personal-agent** (`:9001`) — the user's personal assistant. The simulated
  user talks to it over A2A. It holds the *user-side* environment tools (apply
  for a card, deposit, refer a friend, …) and contacts the bank on the user's
  behalf.
- **cs-agent** (`:9002`) — the bank's customer-service agent. It holds the
  *bank-side* environment tools and answers every policy question by searching
  a **~698-document policy knowledge base** with **Redis-backed RAG**.

Both agents run on **`gemini-3.5-flash`** (KB vector search uses
`gemini-embedding-001`). Scoring is **binary per task on the final database
state**: the run earns reward 1 only when the exact write tool-calls land the
DB in the golden state — reads don't score. The two agents are graded in three
pairings, weighted **50 / 25 / 25** (your pair together, then each agent paired
with a held-out opponent), so clean A2A interop is worth as much as task skill.

---

## How it works

```text
        simulated user ──A2A──▶ PERSONAL agent (:9001)
                                  │  user-side env tools (apply_for_card, deposit, refer…)
                                  │
                                  └─ ask_customer_service ──A2A──▶ CS agent (:9002)
                                  (same contextId threaded)         │  bank-side env tools
                                                                    │
                                                                    └─ kb_search ──▶ Redis
                                                                       (hybrid BM25 + vector
                                                                        RAG over ~698 docs)
        final DB state  ──compared to golden──▶  reward ∈ {0, 1}
```

One personal-agent turn, internal session vs. what the caller sees over A2A:

```text
A2A in (message/send, contextId=ctx42):
  "Please refer my friend Dana for a Blue Account."

Inside the agent's session (invisible to the caller):
  tool call    ask_customer_service("How do I submit a referral?")
  tool result  "Verify the customer; then submit it from the customer's side
                with their submit_referral tool."
  tool call    submit_referral({"account_type": "Blue Account", ...})
               → POST {ENV_API_URL}/sessions/ctx42/tools/submit_referral
  tool result  {"content": "Referral submitted.", "error": false}
  final text   "Done — I've submitted the referral for Dana."

A2A out (what the caller receives):
  Task → status.state "completed", artifact text:
    "Done — I've submitted the referral for Dana."
```

Only that final text crosses the wire — tool calls and intermediate updates
stay inside the agent.

---

## Engineering approach

The winning chain for a task is: **retrieve the right policy doc → verify
identity → unlock the discoverable bank tool → call it with byte-exact args.**
The two failure modes that cost reward are (a) answering from the model's memory
instead of the KB, and (b) last-mile argument errors on the final write. Every
design choice below targets one of those.

### Retrieval — hybrid BM25 + vector, RRF-fused
`cs_agent/rag_tools.py` runs **BM25 and `gemini-embedding-001` vector search in
parallel** and fuses them with **Reciprocal Rank Fusion** (`RRF_K = 60`). Lexical
recall catches exact tool names and account classes; vector recall catches
paraphrased policy intent; RRF gives a single ranking that beats either alone.

### Forced KB grounding (hard precondition, not advice)
`kb_search` is a **gate, not a suggestion**. The CS agent's policy prompt
(`cs_agent/agent.py`) makes it invalid to recommend a product, state any
rate/fee/eligibility rule, or make any DB write without having called
`kb_search` that turn and grounded every value in a retrieved doc — and it must
**never state a banking fact from memory**. This kills hallucinated policies and
surfaces the precise rules (net-correction aggregation, Reg-E fields, "best free
tier") that the final write depends on.

### Integer canonicalization of tool-call args
The grader hashes the exact JSON of each write, so `1500.0 ≠ 1500`. Both agents'
toolsets (`personal_agent/env_toolset.py`, `cs_agent/env_toolset.py`) coerce
whole-number floats to ints (`int(x) if x.is_integer() else x`) **before the call
leaves the agent**, so writes hash-match the golden DB state instead of
near-missing on a serialization artifact.

### Pre-baked embedding cache — instant startup
`cs_agent/ingest.py` loads a prebaked vector cache (`kb/embeddings.json`,
1162 vectors) at startup instead of live-embedding the corpus. Ingest is a
cache hit — no API calls, no quota burn, no startup embedding wait — which keeps
the marker's cold start comfortably inside the 10-minute task budget.

### Graceful BM25 fallback when embeddings are unavailable
Query embedding runs under a hard wall-clock cap (`_embed_bounded`,
`KB_EMBED_TIMEOUT_S=6`). If the embedding endpoint is missing or throttled, the
vector leg raises, is caught, and search **degrades to BM25** rather than hanging
the task — so retrieval stays alive (and within turn timeouts) even on a
rate-limited key, including on the held-out marked runs.

### Smarter ingest
`ingest.py` extracts internal tool names (`\w+_\d{4}`) and account classes into
a dedicated searchable field, chunks long docs by heading, and builds
per-account-class **digest docs** — so the doc naming the right tool ranks for
the query that needs it.

### Strict A2A contextId discipline
The incoming A2A `contextId` is the session key. It is threaded onto **every**
environment-tool call and onto **every** personal→CS A2A message, and each
conversation is fully isolated by its contextId. This is what earns the 50/25/25
interop credit — the two agents compose cleanly with held-out opponents, and the
harness's `smoke` conformance check fails loudly on any contextId mistake.

---

## Conformance

- **Model:** both agents run on `gemini-3.5-flash`; KB vector search uses
  `gemini-embedding-001`. No other chat model is used.
- **Compose shape (frozen):** services `personal-agent` (`:9001`),
  `cs-agent` (`:9002`), and `redis`, honoring the injected `ENV_API_URL`,
  `ENV_API_TOKEN`, `CS_AGENT_URL`, and Google model credentials.
- **contextId threading:** reused on every env-tool call and every personal→CS
  message; conversations are stateless across contextIds (simulations run
  concurrently and must stay isolated).
- **Tools fetched live** from the env API each turn — never hardcoded.
- **Plain text only** crosses the wire; only the final message text reaches the
  other agent.

---

## Run it

Requires Docker (with `host.docker.internal` reachable — on Linux the compose
file maps it via `host-gateway`) and a Vertex AI API key.

```bash
# 1. Configure model credentials
cp .env.example .env
#    set GOOGLE_API_KEY to your Vertex AI key in .env

# 2. Build and start both agents + Redis
docker compose up --build
```

This starts `personal-agent` on `:9001`, `cs-agent` on `:9002`, and `redis`.
The CS agent ingests the KB into a Redis full-text + vector index at startup;
with the prebaked `kb/embeddings.json` cache this is a cache hit and starts
fast.

Run simulations with the harness CLI from
[a2anet/a2a-hackathon](https://github.com/a2anet/a2a-hackathon) (clone it next
to this repo, and export the same `GOOGLE_API_KEY` — the simulated user runs on
it too):

```bash
# Smoke-test one task (prints both conversation legs, every env tool call,
# and the reward; fails loudly on contextId mistakes)
uv run a2a-hack smoke \
    --personal-url http://localhost:9001 --cs-url http://localhost:9002

# Run the train split and browse results
uv run a2a-hack run --personal-url http://localhost:9001 \
    --cs-url http://localhost:9002 --tasks train --save-to results/dev --auto-resume
uv run tau2 view results/dev
```

Iterate against the `train` split; run `--tasks test` sparingly, as a
generalisation check. Keep `--concurrency` low: on a single Vertex key, raising
it causes 429s that push turns past the 5-min limit and score the task 0.

> **Secrets:** the Vertex API key lives only in `.env` (gitignored) and is
> submitted separately via the hackathon form — never commit it. Note that
> harness result files (`results/`) can serialize the key in plaintext; keep
> `results/` out of version control.

---

## Layout

```
personal_agent/  agent.py (prompt), env_toolset.py (+ int canonicalization),
                 cs_client_tool.py, main.py
cs_agent/        agent.py (policy prompt, forced KB grounding),
                 rag_tools.py (hybrid BM25 + vector + RRF, bounded embed),
                 env_toolset.py (+ int canonicalization), ingest.py, main.py
kb/              public knowledge base: documents/ (698 docs) + policy.md,
                 embeddings.json (prebaked vector cache)
specs/           rag-rebuild.md (retrieval design spec)
```

---

## Submission

Submit your public repo URL and your Vertex API key at
[hackathon.a2anet.com](https://hackathon.a2anet.com). Each submission pins the
current commit; the latest is what gets marked. Every submission gets a feedback
run (three tasks × three pairings, with full transcripts), and final marking
uses the same three pairings weighted 50 / 25 / 25.

## 🤖 Join A2A Net

[A2A Net](https://a2anet.com) is an open-source community for the [A2A protocol](https://a2a-protocol.org/latest/) and a platform to build AI agents for [Slack](https://slack.com/intl/en-gb/), [Microsoft 365 Copilot](https://m365.cloud.microsoft/), [Microsoft Teams](https://www.microsoft.com/microsoft-teams/), and [Gemini Enterprise](https://cloud.google.com/gemini-enterprise).

[Join the Discord](https://discord.gg/674NGXpAjU) to share your project, ask questions, and stay up to date.
