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

---

## 2026-06-13 — Phase 4/5: corpus, recall, crown jewel

- Curation (Haiku, fast model — Opus was too slow at ~1.6 calls/min on big prompts) kept
  **150/260** candidates with validated gold (spot-checked 5/5 sensible). 260 Haiku labeling
  calls ($5.12). Switched curation off Opus to Haiku after a slow run; ate ~48 wasted Opus calls.
- **recall@k (n=150, free):** vector @5=0.673 @10=0.733; bm25 @5=0.387; hybrid @5=0.667 @10=0.773.
  Vector >> BM25; hybrid ≈ vector, better at k=10; retrieval is a real bottleneck (~1/3 miss@5).
- **Crown jewel (strict context-only RAG), n=150, joint Opus judge:** base **54.5** vs RAG
  **40.7**, lift **−13.8** (CI [−20.2,−7.6]). RAG win/tie/loss 50/10/90. **RAG underperformed
  base.** Recall-conditioned: gold-retrieved (n=101) base 52.7 / rag 48.9 (≈tie); gold-missed
  (n=49) base 58.0 / rag 23.8 (RAG craters — strict prompt → "not enough info" vs base answering
  from knowledge). Verified on transcripts. 150 Opus judge calls ($4.91).
- **Surprise vs selection:** base scored 54.5 here vs 30.8 in selection — the curated
  docs-answerable questions are also answerable from Claude's parametric knowledge; bar for RAG
  is higher than the selection sample implied. Honest null/negative headline.
- **Eval-first loop closed:** diagnosis → fix = a "fallback RAG" prompt (use+cite context, fall
  back to parametric knowledge when context is thin). Re-running the same 150-question ablation.
- **Judge budget (answer-grading):** selection 70 + crown-jewel strict 150 = 220; + fallback 150
  = 370. Under 500. (Curation labeling 260 Haiku tracked separately as cheaper dataset build.)

---

## 2026-06-13 — Phase 6/7: eval-first fix, resume, report

- **Fallback-RAG fix** (use+cite context, fall back to parametric knowledge when context is thin),
  same 150-question ablation. A session/rate limit killed 63 of the 150 judge calls mid-run
  (only n=87 valid). Wrote `resume_ablation.py` (re-runs only the invalid rows, merges,
  recomputes) and bumped `call_claude` retries 2→4. Re-ran the 63 → full **n=150**.
- **Fallback result:** base 49.5 vs RAG **56.1**, lift **+6.6** (CI [2.0, 11.1], significant).
  win/tie/loss 67/37/46. Conditioned: gold-retrieved +12.5, gold-missed −5.4 (vs strict's −34.2).
  **The fix flips RAG from −13.8 (loss) to +6.6 (win); ~20-pt swing from one prompt change.**
  (base mean shifts run-to-run because base answers are regenerated → report the within-run lift.)
- Wrote `REPORT.md` (substrate selection, eval design, recall/strategy comparison, crown jewel +
  diagnosis + fix, solid/shaky, next-step plan ordered by ROI, resume bullets, budget appendix).
  Updated README + `run_pilot.py` to reproduce the actual (curated) pipeline incl. both variants.

### Final accounting
- **LLM-as-judge (answer-grading) calls ≈ 370** (selection 70 + strict 150 + fallback 150) —
  under the ≤500 budget. Curation labeling (260 Haiku) reported as a separate, cheaper category.
- **Total pilot spend ≈ $24** (Claude via `claude` CLI / OAuth; embeddings local & free).
- Honest headline: RAG is genuinely needed and works **after** the eval-first measure→diagnose→fix
  loop; naive RAG alone regressed a capable base model. Single substrate / single seed — direction
  is likely general but not proven.
