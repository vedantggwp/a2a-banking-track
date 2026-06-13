# A2A Hackathon Template

[![A2A Protocol](https://img.shields.io/badge/A2A-Protocol-blue)](https://a2a-protocol.org) [![Discord](https://img.shields.io/discord/1391916121589944320?color=7289da&label=Discord&logo=discord&logoColor=white)](https://discord.gg/674NGXpAjU)

A working two-agent starter for the A2A Hackathon track. You build:

- **personal-agent** (`:9001`) — the user's personal assistant. Our simulated
  user talks to it over A2A. It holds the *user-side* environment tools and
  can contact the bank's customer service on the user's behalf.
- **cs-agent** (`:9002`) — the bank's customer service agent. It holds the
  *bank-side* environment tools and answers policy questions by searching the
  public knowledge base (Redis RAG).

Fork it, make it smarter, submit your repo at
[hackathon.a2anet.com](https://hackathon.a2anet.com).

## Rules

- Both agents must run on **`gemini-3.5-flash`** (the template default).
  Experiment with whatever you like, but marked runs must use
  gemini-3.5-flash — we manually verify the winning teams' code and
  transcripts, and any other chat model disqualifies the submission.
  (KB vector search uses `gemini-embedding-001`.)
- Auth is **Vertex AI via an API key** created in your GCP project (Vertex AI
  Agent Platform → API keys), so usage runs on your Google Cloud credits.
  Set it in `.env`; you'll also submit it with your repo so marking runs can
  use it.
- Your submission is a **public GitHub repo** that keeps to "What you can't
  change" below. Everything else is fair game.

## What you can change

Prompts, tool descriptions, the RAG pipeline (`ingest.py`, document layout,
index schema, your own Redis image/modules, or an index pre-baked into the
image so startup is instant), conversation flow between your two agents,
extra internal tools — the knowledge base in `kb/` is public, and
preprocessing it cleverly is encouraged. You can also go further:

### Use another agent framework entirely

Anything A2A-compatible works, in any language — keep the compose shape
below and the inside of each container is yours. The wire contract a
replacement must satisfy:

- Serve A2A over **JSON-RPC**: an agent card at the service root,
  `message/send` with text parts. Our clients call you with streaming off
  and read your reply text from a returned `Message`, or from a returned
  `Task`'s artifacts plus its final `status.message` (all text parts,
  joined). Intermediate status updates are never read.
- Every incoming message carries a **`contextId`** — your session key.
  Reuse it on every env tool call (it's the session id in the URL path)
  and, personal agent only, as the `contextId` on every A2A message to
  `CS_AGENT_URL`.
- Fetch your tools from the env API (see "Environment tools"); don't
  hardcode them.

`a2a-hack smoke` (see "Dev loop") is the conformance check: it fails loudly
on contextId mistakes.

### Write a custom A2A Agent Executor

The template serves both agents with ADK's built-in A2A executor
(`to_a2a(root_agent)`). You can replace it to control exactly what goes on
the wire. Here's one full personal-agent turn — the internal session vs.
what the caller sees over A2A:

```text
A2A in (message/send, contextId=ctx42):
  "Please refer my friend Dana for a Blue Account."

Inside the agent's session (invisible to the caller):
  tool call    ask_customer_service("How do I submit a referral?")
  tool result  "Verify the customer first; then it's submitted from the
                customer's side with their submit_referral tool."
  tool call    submit_referral({"account_type": "Blue Account", ...})
               → POST {ENV_API_URL}/sessions/ctx42/tools/submit_referral
  tool result  {"content": "Referral submitted.", "error": false}
  final text   "Done — I've submitted the referral for Dana."

A2A out (what the caller receives):
  Task
    status.state: "completed"
    artifacts: [{ parts: [{ text: "Done — I've submitted the referral
                            for Dana." }] }]
```

ADK's executor returns the final model message as a Task **artifact**; tool
calls and intermediate updates never leave the agent. Our clients read
artifact text plus the final `status.message` text — either carries your
reply, and responding with a plain `Message` works too. Whatever executor
you use, only that final text reaches the other side.

## What you can't change

1. **The compose shape.** A `docker-compose.yml` with services
   `personal-agent` (`:9001`), `cs-agent` (`:9002`), and `redis`. Both
   agents honor the injected env vars `ENV_API_URL`, `ENV_API_TOKEN`, and
   the Google model credentials; the personal agent honors `CS_AGENT_URL`
   and always contacts customer service through it.
2. **contextId discipline.** Reuse the incoming A2A `contextId` on every
   env tool call and, for the personal agent, on every message to the CS
   agent. (ADK keys its session on the contextId, so the template gets this
   for free — `ctx.session.id` *is* the contextId.)
3. **Statelessness across contextIds.** Simulations run concurrently; each
   conversation must be fully isolated by its contextId.
4. **The model** (see Rules), and the environment itself — the tasks, env
   tools, and simulated user belong to the harness.
5. **The harness timeouts.** Each agent turn must finish within 5 minutes and
   each whole task within 10 minutes, or the run is scored 0. These limits are
   fixed by the harness. You may change the local run `--concurrency`, but
   raising it on a single Vertex key causes 429 rate-limit errors that slow
   your agents into these timeouts, so it isn't recommended.

## Environment tools

Each agent gets its tools from the harness env API (not hardcoded):

- `GET {ENV_API_URL}/sessions/{contextId}/tools` — OpenAI-style schemas for
  your scope (`Authorization: Bearer {ENV_API_TOKEN}`). The personal agent
  sees the user's tools; the CS agent sees the bank's tools.
- `POST {ENV_API_URL}/sessions/{contextId}/tools/{name}` with
  `{"arguments": {...}}` — execute a tool; returns
  `{"content": ..., "error": ...}`.

This template's `EnvApiToolset` fetches these live each turn, so tools that
get granted mid-conversation appear automatically.

## Dev loop

You run simulations locally with the harness CLI from
[a2anet/a2a-hackathon](https://github.com/a2anet/a2a-hackathon) — clone it
next to this repo.

```bash
# 1. Configure (model credentials etc.)
cp .env.example .env

# 2. Run your agents
docker compose up --build

# 3. In the harness repo, smoke-test one task (export the same
#    GOOGLE_API_KEY first — our simulated user runs on it too)
uv run a2a-hack smoke \
    --personal-url http://localhost:9001 --cs-url http://localhost:9002

# 4. Run the train split and browse results
uv run a2a-hack run --personal-url http://localhost:9001 \
    --cs-url http://localhost:9002 --tasks train --save-to results/dev --auto-resume
uv run tau2 view results/dev

# 5. Iterate on prompts/RAG/flow, then submit (see "Submission")
```

`smoke` prints both conversation legs (user↔personal and personal↔CS), every
environment tool call, and the reward — if your contextId handling is wrong
it will tell you loudly.

Iterate against the `train` split. Run `--tasks test` sparingly, as a
generalisation check: if you tune to it, you won't know whether your
improvements actually generalise.

## What's where

```
personal_agent/  agent.py (prompt), env_toolset.py, cs_client_tool.py, main.py
cs_agent/        agent.py (policy prompt), env_toolset.py, rag_tools.py, ingest.py, main.py
kb/              the public knowledge base: documents/ (698 docs) + policy.md
```

The CS agent's `ingest.py` builds a Redis full-text + vector index at startup
(vector search needs model credentials; BM25 works without).

## Submission

Submit your public repo URL and your Vertex API key at
[hackathon.a2anet.com](https://hackathon.a2anet.com). Submit as often as you
like — each submission pins the current commit, and your latest one is what
gets marked.

Every submission gets a feedback run: scores for **three tasks** (the
`feedback` split), each in three pairings — your pair together, your
personal agent with our held-out CS agent, and our held-out personal agent
with your CS agent — with full transcripts on the site.

Final marking uses the same three pairings, weighted 50% (your pair) /
25% / 25% (each held-out pairing). Keep to the contract above and both of
your agents will compose cleanly with ours.

## 🤖 Join A2A Net

[A2A Net](https://a2anet.com) is an open-source community for the [A2A protocol](https://a2a-protocol.org/latest/) and platform to build AI agents for [Slack](https://slack.com/intl/en-gb/), [Microsoft 365 Copilot](https://m365.cloud.microsoft/), [Microsoft Teams](https://www.microsoft.com/microsoft-teams/), and [Gemini Enterprise](https://cloud.google.com/gemini-enterprise).

[Join the Discord](https://discord.gg/674NGXpAjU) to share your project, ask questions, stay up-to-date with the latest news, be the first to hear about open-source releases, tutorials, and more!
