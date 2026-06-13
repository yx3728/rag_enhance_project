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

---

## 2026-06-13 — Phase 3: eval-set validity fix (important)

Built dagster docs corpus (673 files → 4708 chunks) and a first eval set via free
answer→corpus embedding gold (τ=0.62, kept 110/120). recall@k: vector @5=0.54, bm25 @5=0.30,
hybrid @5=0.48.

**But a diagnostic on a gold-hit/RAG-loss case exposed two validity problems:**
1. **Noisy gold** — answer→corpus embedding picks a chunk lexically similar to the (often
   terse) accepted answer, not the chunk that truly answers the question. Example #20327:
   gold landed on a tutorial chunk about `Definitions` loading, not cross-code-location asset
   deps; meanwhile the RAG retriever found the *correct* `defining-assets-with-asset-dependencies`
   page.
2. **Weak references** — GitHub accepted answers are often conversational ("What made you say
   that?"), link-only, or version-obsolete (point to old docs paths). Judging answer correctness
   against them penalizes genuinely good answers.

**Fix:** an LLM-curation pass (judge model) over the candidate questions. For each, it sees the
question + accepted answer + a *fair* candidate pool (top chunks from answer-emb ∪ question-vector
∪ question-bm25, so no retriever is privileged) and returns {keep, gold_indices, reason}.
keep=true only for self-contained, docs-answerable, still-valid questions whose accepted answer
is substantive AND supported by ≥1 pool chunk. **Reference stays = the real accepted answer**
(authentic, avoids docs-circular bias toward RAG); gold = labeler-verified chunks. This yields
trustworthy recall@k and a fair crown-jewel judge. ~110 curation judge calls budgeted.
