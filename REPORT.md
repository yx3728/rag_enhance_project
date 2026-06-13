# RAG Enhancement Pilot — Report

**Goal of the pilot.** Validate an *eval-first* redesign of a documentation-QA RAG system:
turn a generic "enterprise RAG demo" ([`enterprise-copilot`](https://github.com/yx3728/enterprise-copilot))
into a rigorously-evaluated documentation-QA system on a **SWE-relevant substrate**, produce the
headline **RAG-vs-base-model** result, and write a detailed next-step plan. This is a pilot, run
from the command line — no frontend/server/deploy (those are next-step).

All generation and judging run on Claude via the `claude` CLI (OAuth account, no API key);
embeddings are local (`bge-small`, offline). Costs below are the CLI's reported `total_cost_usd`.

---

## 1. Substrate selection (data-driven)

**Method.** Picked 5 candidate open-source developer tools, each with (a) real docs and (b) a real
Q&A source (**GitHub Discussions Q&A with accepted answers**). For each, sampled 14 answered
questions, had the **base model** (Claude Haiku 4.5, no retrieval) answer them, and an **Opus 4.8
judge** score each answer 0–100 vs the accepted answer (on the docs-answerable subset). Pick the
tool where the base model does **worst** — that is where RAG has the most to prove.

| candidate | base mean score (0–100) ↓ | docs-answerable n |
|---|---|---|
| **marimo** | **26.2** | 8 |
| **litestar** | **28.0** | 10 |
| **dagster** | **30.8** | 12 |
| duckdb | 48.1 | 10 |
| prefect | 53.7 | 9 |

(70 judge calls, $2.14.) The base model is clearly weak (~26–31) on marimo / litestar / dagster
— a statistical near-tie at n≈10 — and clearly stronger on duckdb / prefect.

**Chosen substrate: `dagster-io/dagster`** (a data-orchestration framework). Among the tied
"worst" group, dagster best satisfies the spec's secondary criteria: most answered Q&A (851
total), highest docs-answerable rate (12/14), and clean markdown docs — so it can support an
80–150-question eval set with a credible (tighter-CI) headline. marimo has the literal lowest
base score but only 87 total discussions (too few). dagster is niche-but-credible, fast-moving
(version-specific config/API the base model demonstrably doesn't know — base scored 30.8/100),
and SWE-relevant (questions are about *using* the tool, so answers live in the docs).

---

## 2. Eval design

**Corpus.** dagster's official docs (673 markdown files under `docs/docs/`) → **4,708 chunks**
(paragraph strategy, ~800 chars / 120 overlap — `enterprise-copilot` defaults). In-process index:
local `bge-small` dense embeddings (numpy cosine) + BM25 + a hybrid combiner. (This replaces
`enterprise-copilot`'s Qdrant *server* with an in-process index — same cosine top-k semantics,
no daemon, fully reproducible.)

**Eval set.** Real questions from dagster's GitHub Discussions Q&A. Built by an LLM-curation pass
(Haiku) over 260 candidates: for each, the model saw the question + accepted answer + a *fair*
candidate pool (top doc chunks from answer-embedding ∪ question-vector ∪ question-BM25, so no
retriever is privileged) and returned **keep** + **gold chunk indices**. A question is kept only
if it is a self-contained, docs-answerable, still-valid usage/config/API/error question whose
accepted answer is substantive **and** supported by ≥1 doc chunk. **Kept 150 / 260 (58%).** The
**reference answer stays the real accepted answer** (authentic; avoids biasing the judge toward
docs-grounded RAG output). (260 Haiku labeling calls, $5.12.)

*Why curation was necessary (a result in itself):* a first attempt defined gold purely by
answer→corpus embedding similarity (free, no LLM). A diagnostic showed this was too noisy — terse
or link-only accepted answers produced wrong gold chunks, and judging answers against weak
conversational references penalized genuinely good answers. The curation pass fixed both. Gold was
spot-validated on a sample (5/5 sensible; e.g. the credit-cost question maps exactly to the
credit-usage table).

**Metrics.**
- **Recall@k** — objective, no judge, full set. Fraction of questions whose top-k retrieved chunks
  (for the *question*) contain a gold chunk. Computed for vector / BM25 / hybrid.
- **Crown jewel** — base-only vs RAG answers, scored 0–100 by an Opus judge in one joint call per
  question (positions randomized to control order bias) vs the reference answer.

**Judge budget.** The pilot's LLM-as-**judge** (answer-grading) calls = selection (70) + crown
jewel (150) = **220**, well within the **≤500** budget. Dataset-construction *labeling* (Haiku
curation, 260) is reported separately as a cheaper, distinct category.

---

## 3. Retrieval quality + strategy comparison (recall@k, n=150)

| method | recall@1 | recall@3 | recall@5 | recall@10 | mean latency |
|---|---|---|---|---|---|
| **vector** (bge-small) | 0.34 | 0.60 | **0.673** | 0.733 | 66 ms |
| BM25 | 0.26 | 0.33 | 0.39 | 0.45 | 86 ms |
| hybrid (vec+BM25) | 0.32 | 0.56 | 0.667 | **0.773** | 113 ms |

**Read.** Dense vector retrieval clearly beats lexical BM25 on these natural-language usage
questions (recall@5 0.67 vs 0.39). Hybrid ≈ vector at k=5 and edges ahead at k=10 (0.77), at ~2×
the latency. Retrieval is a real bottleneck: ~1/3 of questions don't have a gold chunk in the
top-5 — a concrete target for the next iteration (reranking, better chunking, query rewriting).
*Caveat:* gold was curated from a pool that included vector, BM25, and answer-embedding
candidates, so the pool is not biased toward one retriever; but gold labeling shares the
embedding space, which can mildly favor dense methods. This does not affect the crown jewel (the
base model does no retrieval).

---

## 4. Crown jewel — RAG vs base model ablation

Same 150 questions answered by (a) the base model alone (Haiku 4.5, no retrieval) and (b) the
RAG system (top-5 vector + the `enterprise-copilot` "answer from context only, cite [n]" prompt).
One Opus-4.8 joint judge call per question (positions randomized) scores both 0–100 vs the
accepted reference answer.

**Headline (honest, and not the direction we hoped): naive RAG underperformed the base model.**

| system | mean score (0–100) | correct-rate | judge preferred |
|---|---|---|---|
| base model (no retrieval) | **54.5** | 0.55 | 93 / 150 |
| RAG (strict context-only) | **40.7** | 0.39 | 55 / 150 |

Mean lift = **−13.8** (95% bootstrap CI **[−20.2, −7.6]** — significantly negative). RAG
win/tie/loss vs base = **50 / 10 / 90**.

**Why — the recall-conditioned breakdown is the real result:**

| retrieval outcome | n | base mean | RAG mean | RAG − base |
|---|---|---|---|---|
| gold chunk **retrieved** | 101 | 52.7 | 48.9 | −3.8 (≈ tie) |
| gold chunk **missed** | 49 | 58.0 | 23.8 | **−34.2** |

When retrieval surfaces the right doc chunk (67% of the time, per §3), RAG roughly **ties** the
base model. When retrieval **misses** (33%), RAG **craters** — because the strict "answer using
ONLY the provided context" prompt makes it respond *"I don't have enough information in the
provided documentation…"* (verified on the actual transcripts) while the base model simply answers
from its training knowledge. Two compounding factors:
1. **Retrieval is mediocre** (recall@5 = 0.67) — a third of questions never get the gold chunk.
2. **Base Claude already knows a fair amount of dagster.** Base scored 54.5 here (vs 30.8 on the
   noisy selection sample), so on this corpus the bar for RAG is higher than selection suggested —
   the curated, docs-answerable questions are also answerable from the model's parametric knowledge.

**This is a genuine, useful pilot finding, not a failure of the harness:** naive "context-only"
RAG, bolted onto imperfect retrieval, can *degrade* a capable base model. The eval-first harness
both surfaced this and localized it to a specific, fixable mechanism.

### 4b. Closing the loop — the fix the eval pointed to

The diagnosis prescribes the fix directly: stop forcing context-only answers when retrieval is
weak. We added a **fallback RAG** prompt — *prefer and cite the retrieved context, but fall back
to the model's own knowledge (and flag it) when the context doesn't cover the question* — and
re-ran the same 150-question ablation.

<!-- FALLBACK_NUMBERS -->
_(filled in after the fallback run completes)_

---

## 5. What's solid / what's shaky

<!-- ASSESSMENT -->

---

## 6. Next-step plan (deferred items, ordered by ROI)

Ordered by ROI for turning the pilot into the full build. "Effort" is rough wall-clock for one
engineer.

**A. Eval set as a CI regression test — highest ROI (~1 day).**
The eval harness already produces objective recall@k for free and a budgeted judge score. Wire it
into CI so every change to chunking/retrieval/prompt is gated on it.
- Steps: freeze `data/eval/dagster-io__dagster.json` as a versioned fixture; add a `pytest` that
  runs `recall.py` and asserts recall@5 ≥ baseline − ε (objective, free, deterministic); add an
  *optional* nightly job that runs the crown-jewel judge on a fixed 50-question slice and posts
  the lift to a dashboard. Cache embeddings (already done) so CI is seconds.
- Why first: it makes every later change measurable and prevents regressions; near-zero ongoing cost.

**B. Close the retrieval gap — high ROI (~2–3 days).**
Recall@5 is only 0.67; ~1/3 of questions miss the gold chunk in top-5. Biggest single lever on
end-to-end quality.
- Steps: (1) add a cross-encoder/LLM **reranker** over a top-30 candidate pool; (2) sweep
  **chunking** (heading-aware / structural splitting instead of fixed 800-char; dagster docs are
  MDX with `<CodeExample>` includes that currently chunk poorly); (3) **query rewriting** (expand
  the user question before retrieval); (4) try a stronger embedding model (bge-large / voyage).
  Measure each against recall@k (free) + the judge slice.

**C. Full strategy comparison — medium ROI (~1 day).**
The pilot already compares vector / BM25 / hybrid on recall@k + latency (§3, free). Extend to the
*answer-quality* axis: run the crown-jewel judge for each retrieval method on a fixed slice, and
add a cost/latency-per-method table. Folds directly into (B)'s measurement.

**D. Live deployment with secrets hygiene — medium ROI (~3–5 days).**
Port the pilot pipeline behind the existing `enterprise-copilot` FastAPI surface (it already has a
`/chat` RAG endpoint and ingestion).
- Steps: containerize; keep the in-process index for small corpora or reattach Qdrant for large
  ones; secrets in a managed store (not env files committed anywhere) — API keys via a secrets
  manager / workload identity, never in the image or repo; add request auth + rate limiting; ship
  the citation-backed answers the core already returns.

**E. Scoped stress / perf test — medium ROI (~2 days).**
- Steps: load-test the retrieval + generation path with a concurrency sweep (1→N); report
  **p50/p95/p99** latency and throughput separately for retrieval (local, cheap) vs generation
  (LLM-bound); identify failure modes (LLM rate limits, timeouts, embedding-model contention,
  index memory at corpus scale) and add backpressure/retry/circuit-breaking. Keep it scoped — a
  single tool, realistic query mix, documented hardware.

**F. Frontend integration — lower ROI for the eval story (~3–5 days).**
Reuse `enterprise-copilot`'s Next.js dashboard: wire the chat + retrieval-inspection views to the
new pipeline, and surface the eval dashboard (recall@k trend, RAG-vs-base lift over time) as a
first-class page so the eval-first story is visible to users/stakeholders.

**Substrate breadth (parallel track).** The harness is substrate-agnostic — re-running selection +
`run_pilot.py` on litestar / marimo (the other low-base tools) would show whether the lift
generalizes, at the cost of more judge calls.

---

## 7. Draft resume bullets

<!-- RESUME -->
