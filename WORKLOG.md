# WORKLOG — RAG Enhancement Pilot

Newest entries at the bottom. Decisions, what was tried, what worked/failed, costs.

---

## 2026-06-13 — Phase 1: re-fetch + read `enterprise-copilot`

- Re-fetched `https://github.com/yx3728/enterprise-copilot.git` into `_ref_enterprise-copilot/`
  (gitignored). Single commit `9ba7c9d`. Read backend fully.
- Full read-through notes in `docs/enterprise-copilot-notes.md`.
- **Reusable core**: chunking strategies (pure fns), the RAG pipeline shape
  (embed-query → top-k cosine → numbered context → "answer from context only, cite [n]" →
  generate), and the eval *intent* (per-case metrics + aggregate run, persisted).
- **Dropped for pilot** (per spec scope guardrails): frontend, FastAPI server, Postgres,
  docker, Azure, and the Qdrant *server* (replaced by in-process numpy cosine index — same
  semantics, no daemon, reproducible). The weak heuristic metrics become secondary; rigorous
  **recall@k** becomes the headline retrieval metric.

### Key decisions

- **No API key needed.** Followed the pattern in `~/cog_moral_tests/run_experiment.py`: call
  Claude headlessly via the `claude` CLI (`claude -p "<prompt>" --model <id> [--effort <e>]
  --output-format json` with clean flags: empty system prompt, no tools, no MCP, no settings,
  no session persistence). Runs on the user's OAuth/Max account. Verified working — returns a
  JSON object with `result`, `usage`, and `total_cost_usd`. This is how we get base-model
  answers, RAG answers, and judge verdicts.
- **Provider = Anthropic Claude** for all generation + judging (the original used OpenAI; no
  OpenAI key here, and we're Claude-native). Per the claude-api skill, default model is
  `claude-opus-4-8`; we'll use a cheaper tier where it doesn't hurt (e.g. answering model) and
  reserve stronger models for the judge as needed.
- **Embeddings = local** (`sentence-transformers`, CPU). Free, offline, fully reproducible —
  lets recall@k run on the entire eval set at zero marginal cost. Swaps out OpenAI embeddings
  while keeping the pipeline shape identical.
- **Judge budget clarified by user: ≤500 LLM-judge calls TOTAL for the whole pilot** (not
  per-trial). So: objective recall@k carries the retrieval axis (free, full set); the judge is
  spent mainly on the crown-jewel answer-quality axis. User also said: ship a product with real
  numbers, be pragmatic — not a scientific publication.

### Cost so far
- 1 test `claude` call (haiku), `total_cost_usd` ≈ $0.001. (Billed to Max account; cost field is
  a useful proxy we'll sum across the pilot.)

---

## 2026-06-13 — Phase 2: substrate selection result

Ran `src/selection.py`: base model = haiku, judge = opus, 14 sampled answered-Q&A per
candidate, scored base correctness vs accepted answer (docs-answerable subset only).
**70 judge calls used (430 remaining), cost $2.14.**

| candidate | base mean score (0–100) | docs-answerable n |
|---|---|---|
| marimo    | 26.2 | 8/14  |
| litestar  | 28.0 | 10/14 |
| dagster   | 30.8 | 12/14 |
| duckdb    | 48.1 | 10/14 |
| prefect   | 53.7 | 9/14  |

The base model is clearly weak (~26–31) on marimo / litestar / dagster — a near-tie
(n≈10 → wide CIs that overlap), and clearly stronger on duckdb / prefect (48 / 54).

**Chosen substrate: `dagster-io/dagster`.** Rationale: among the statistically-tied "worst"
group, dagster best satisfies the spec's tie-break criteria — most answered Q&A (851 total;
120 fetched), highest docs-answerable rate (12/14), clean markdown docs (~672 files under
docs/docs/) — so it can hit the 80–150 question target with a credible (tighter-CI) headline.
marimo is the literal lowest base score but has only 87 discussions total → too few to reach
the eval-set target. dagster is niche-but-credible (well-known data orchestrator), fast-moving
(version-specific config/API the base model demonstrably doesn't know: base scored 30.8/100).
