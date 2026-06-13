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

**Result: the fix flips the sign — RAG now beats the base model.**

| RAG variant | base mean | RAG mean | lift (RAG − base) | 95% CI | win/tie/loss |
|---|---|---|---|---|---|
| strict (context-only) | 54.5 | 40.7 | **−13.8** | [−20.2, −7.6] | 50 / 10 / 90 |
| **fallback (use+cite context, else parametric)** | 49.5 | 56.1 | **+6.6** | **[2.0, 11.1]** | 67 / 37 / 46 |

(n=150 each; base is re-generated per run, so its mean shifts a few points with LLM/judge noise
— the joint judge compares base vs RAG *within* each run, so the **lift** is the reliable
quantity.) Correct-rate rose from 0.48 (base) to 0.61 (fallback RAG); the judge preferred the RAG
answer on 83/150 vs 65 for base.

Recall-conditioned, the fix does exactly what the diagnosis predicted:

| retrieval outcome | n | strict RAG − base | fallback RAG − base |
|---|---|---|---|
| gold retrieved | 101 | −3.8 | **+12.5** |
| gold missed | 49 | −34.2 | **−5.4** |

- When the gold chunk **is** retrieved, the context now clearly helps (+12.5 vs base).
- When retrieval **misses**, the fallback no longer craters (−5.4 vs −34.2) — instead of refusing,
  the model answers from its own knowledge and flags it.

**The headline, honestly stated:** on a corpus where a capable base model already does fairly
well, *naive* context-only RAG **hurt** (−13.8); the **eval-first loop diagnosed the cause
(retrieval misses × a brittle prompt) and produced a fix that turned a −13.8 regression into a
statistically-significant +6.6 win** — a ~20-point swing from one prompt change, measured on the
same 150 real questions. RAG is genuinely needed *and* works **once you measure and fix the
failure mode** — which is the whole point of the eval-first redesign.

---

## 5. What's solid / what's shaky

**Solid.**
- **Recall@k is fully objective and runs free on the whole set** (150 questions), with a clean
  vector / BM25 / hybrid comparison and latency. No judge, no sampling.
- **The crown-jewel lift is measured on all 150 questions with a randomized-position joint judge
  and a bootstrap CI.** Both the strict (−13.8, CI excludes 0) and fallback (+6.6, CI excludes 0)
  results are statistically separable from zero.
- **The recall-conditioned breakdown is the same in both runs** and mechanistically consistent
  (verified on transcripts): the failure is retrieval-miss × brittle prompt, and the fix targets
  exactly that. The improvement is not a fluke of one metric.
- **Reproducible & cheap.** Local embeddings + cached index; ~370 judge-grading calls and ~$24
  total for the entire pilot, tracked per call.

**Shaky / caveats (don't overclaim).**
- **Single substrate, single seed.** All numbers are dagster only, one answer model (Haiku 4.5),
  one judge (Opus 4.8). The direction is likely general but is not shown to be.
- **Reference answers are real GitHub accepted answers** — authentic, but sometimes terse,
  link-only, or tied to an older dagster version. The curation filter dropped the worst, but some
  version drift between 2024-era Q&A and current docs remains and adds noise (it would tend to
  *understate* RAG, since RAG retrieves current docs).
- **Gold labels are LLM-curated (Haiku), not human-adjudicated.** Spot-checked 5/5 sensible and
  drawn from a retriever-fair candidate pool, but not gold-standard; recall numbers should be read
  as "good" not "exact". Gold also shares the embedding space, mildly favoring dense retrieval in
  the §3 strategy comparison (irrelevant to the crown jewel — base does no retrieval).
- **Base score moved between selection (30.8) and the crown jewel (54.5).** This reflects sample +
  protocol differences (14 raw questions / single-answer judge vs 150 curated / joint judge), and
  is why we report the within-run *lift* rather than cross-run absolute scores.
- **LLM-judge bias.** A single judge model; we mitigated position bias (randomized A/B) and
  self-preference (judge ≠ answerer), and cross-checked against the objective recall signal, but
  did not run multi-judge agreement. Judge scores are directional, not absolute truth.

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

Grounded only in what this pilot actually produced (no deploy, no scale claims).

**SWE / full-stack framing**
- Built a command-line RAG documentation-QA pipeline over a 4,700-chunk dagster docs corpus,
  reusing an existing FastAPI/Next.js RAG codebase's core and replacing its Qdrant service with a
  zero-dependency in-process vector + BM25 + hybrid index for reproducible, offline retrieval.
- Designed and built an automated evaluation harness (objective recall@k + an LLM-judge answer
  grader with a per-call cost/budget tracker) that runs end-to-end from one command and persists
  reproducible JSON results — the basis for a CI regression gate on retrieval quality.
- Engineered the full data pipeline: pulled and cached real Q&A from GitHub Discussions and docs
  from the source repo, with rate-limit-aware retries, threaded concurrency, and crash-safe
  checkpoint/resume for long judged runs.

**AI engineer framing**
- Ran a rigorous RAG-vs-base-model ablation on 150 real, docs-answerable developer questions:
  found that *naive* "answer-from-context-only" RAG **underperformed** a strong base model by 13.8
  points (95% CI [−20.2, −7.6]) — a non-obvious, honestly-reported result.
- Diagnosed the failure with a retrieval-conditioned analysis (RAG cratered −34 pts only when
  retrieval missed the gold chunk, ~⅓ of the time) and shipped a prompt fix (cite context, fall
  back to parametric knowledge) that flipped the result to a statistically-significant **+6.6**
  win (95% CI [2.0, 11.1]) — a ~20-point swing, re-measured on the same eval set.
- Built the substrate selection, eval-set curation (LLM-verified gold with a retriever-fair
  candidate pool), and recall@k (vector 0.67 vs BM25 0.39 @5) under a strict ≤500 LLM-judge-call
  budget (~370 used) at ~$24 total, keeping the methodology defensible (randomized-position judge,
  bootstrap CIs, disclosed caveats).

---

## Appendix — budget & cost

| phase | model(s) | calls | kind | cost |
|---|---|---|---|---|
| substrate selection | Haiku gen + Opus judge | 70 + 70 | judge-grading (70) | $2.14 |
| eval-set curation | Haiku | 260 | dataset labeling | $5.12 |
| crown jewel — strict | Haiku gen (300) + Opus judge (150) | 150 | judge-grading | $7.24 |
| crown jewel — fallback (+resume) | Haiku gen + Opus judge | 150 | judge-grading | ~$7.9 |
| smoke tests / one abandoned slow curation | mixed | ~60 | — | ~$2 |

**LLM-as-judge (answer-grading) calls ≈ 370** (selection 70 + strict 150 + fallback 150) — within
the ≤500 budget. Dataset-construction *labeling* (Haiku curation, 260) is reported separately as a
distinct, cheaper category. **Total pilot spend ≈ $24** (Claude via the `claude` CLI on an OAuth
account; embeddings local/free). Raw per-call usage is in each `results/*.json`.
