# Multi-Repo RAG Eval — scale-up (resume artifact)

**What this is.** The diagnosed-and-fixed dagster RAG pipeline (see `DIAGNOSIS.md`) scaled to **3
repos** spanning the base model's familiarity × doc-completeness space. The bar: *show the
deployable (fallback) RAG beats the base LLM where the base is weak, measure corpus-gap as the
covariate that explains the lift, and tell the story with confounds flagged.*

- **Subject / base model:** Claude **Haiku 4.5** (reliable knowledge cutoff **Feb 2025**, training
  Jul 2025) — fixed across all repos, unchanged from the pilot.
- **Judge:** Claude **Opus 4.8**, `effort=medium`, everywhere (coverage + ablation).
- **Headline retrieval metric:** lenient **claim-coverage@5** (gold-independent). `recall@k` is
  diagnostic only — on dagster it inflated 3.7× off content-empty gold stubs (see `DIAGNOSIS.md`).
- **Every judge call is traced** (full prompt + raw response) under `traces/<repo>/`; each headline
  number maps to a raw file in `results/manifest.json`.

## 1. Substrate selection (base-probe, Opus-medium judge)

Method: assemble candidates, probe base familiarity on a real-question sample (`multirepo_probe.py`,
n=14, docs-answerable subset), pick 3 spanning the familiarity axis while keeping docs rich. Probe
table (`results/multirepo/probe_forum.json`):

| candidate | base familiarity (Haiku, 0–100) | chosen as |
|---|---|---|
| duckdb | 43.5 | cell 1 — moderate familiarity, rich docs |
| marimo | 28.3 | (alternate for cell 2) |
| litestar | 26.7 | cell 2 — low familiarity, rich docs |

Cell 3 (**zero familiarity / post-cutoff**) = **pydantic-ai**: v1.0 released **2025-09-04**, after
Haiku's Jul-2025 training cutoff; GitHub Discussions are disabled (no forum Q&A) → **synthetic
doc-grounded eval mode** (flagged distinctly). Cutoffs verified online (Anthropic docs).

## 2. Per-repo pipeline (reused, not rebuilt)

Each repo: corpus build with the diagnosis fixes (`build_corpus.py` — heading-aware/code-safe
chunking; API **docstring** extraction via `apidocs.py`, since Sphinx/mkdocstrings rST/md are
autodoc stubs) → eval-set build → claim-coverage analyze → recall (diagnostic) → ablation
(fallback = deployable, strict = diagnostic), joint randomized-position Opus-medium judge, bootstrap CI.

| repo | corpus chunks | API-docstring chunks | eval n | eval mode |
|---|---|---|---|---|
| duckdb (duckdb-web) | 4,204 | 0 (self-contained markdown) | 136 | forum |
| litestar | 1,426 | 525 | 51 | forum (its full available answered Q&A) |
| pydantic-ai | 2,600 | 790 | 110 | synthetic (doc-grounded) |

> **Session-limit incident (logged):** the first dual-curation ran during a Max-account session
> limit — traces showed 41% (duckdb) / 98% (litestar) of calls errored. Detected via per-call
> `is_error` in traces, archived the degraded runs, and re-ran after recovery (0% error). This is
> exactly why the trace-everything rule matters.

## 3. Cross-repo result (`results/multirepo/cross_repo.json`)

| repo | cell | mode | n | base familiarity¹ | claim-cov@5 | **corpus-gap** | **fallback lift** | 95% CI | W/T/L |
|---|---|---|---|---|---|---|---|---|---|
| duckdb | moderate | forum | 136 | 43.5 | 0.507 | 0.228 | **+11.1** | [6.1, 15.9] | 64/42/30 |
| litestar | low | forum | 51 | 26.7 | 0.608 | 0.137 | **+14.5** | [4.1, 25.0] | 27/12/12 |
| pydantic-ai | zero/post-cutoff | synthetic | 110 | 24.9² | 0.891 | 0.045 | **+65.4** | [60.1, 70.3] | 106/3/1 |

¹ forum repos use the selection probe; ² pydantic-ai familiarity = base mean on its (synthetic)
eval — a *different* measurement (see confounds). The diagnostic **strict**-RAG lifts were
−1.0 (duckdb), +1.8 (litestar), +64.8 (pydantic-ai).

**The bar is cleared in all three repos:** the deployable fallback RAG beats the base model with a
95% CI that excludes 0, everywhere.

## 4. The trend (stated as a hypothesis, not a proof)

**Hypothesis:** the RAG-over-base lift grows as base familiarity falls *and* corpus-gap falls
(answer present in docs but unknown to the base). The 3 points are consistent with it:

```
base familiarity ↓     43.5  →  26.7  →  24.9
corpus-gap        ↓    0.228  → 0.137  → 0.045
fallback lift     ↑   +11.1  → +14.5  → +65.4
```

- duckdb → litestar (both **forum**, the cleaner comparison): lower familiarity + lower corpus-gap
  → larger lift (+11.1 → +14.5). Modest and CI-overlapping, but directionally on-hypothesis.
- pydantic-ai is the dramatic point (+65.4): a post-cutoff tool the base genuinely doesn't know
  (base 24.9), with answers fully in the docs (gap 0.045) → RAG injects the missing knowledge and
  wins 106/110. **But this point is not directly comparable to the forum repos** (see confounds).

**We do NOT claim causal isolation of familiarity vs doc-richness** — 3 entangled points can't
separate them. The claim is: *where the base is weak and the docs contain the answer, fallback RAG
delivers a large, CI-clean lift; the lift co-varies with both factors in the expected direction.*

## 5. Confounds (explicit, with direction of effect)

- **Familiarity × doc-richness × eval-mode are entangled.** This is not a factorial design (that's
  the paper we deliberately didn't do). The cleanest like-for-like is duckdb-vs-litestar (both forum).
- **Synthetic mode inflates pydantic-ai.** Synthetic questions are *doc-shaped* (written from a doc
  chunk), so retrieval is far easier (cov@5 0.89, corpus-gap 0.045 by construction) than real user
  questions. Direction: **overstates** pydantic-ai's lift vs a hypothetical forum-Q version. The
  +65.4 conflates true post-cutoff ignorance (real) with synthetic-retrieval ease (artifact). It is
  a clean *knowledge-injection* demo, not a forum-comparable lift.
- **Base-familiarity is measured two ways.** Forum repos use the probe (43.5 / 26.7); pydantic-ai
  uses its synthetic-eval base mean (24.9). The probe and the ablation-base also differ within a
  repo (e.g. litestar probe 26.7 vs ablation-base 48.7) because curated docs-answerable questions
  are easier for the base than the raw probe sample. Treat the familiarity axis as ordinal, not exact.
- **corpus-gap is partly a selection effect.** Forum questions skew toward what docs *can't*
  answer; corpus-gap-driven misses are **intrinsic** (answer absent from docs), not retriever
  failures — better embeddings/rerankers can't fix them.
- **Single seed; single judge model (Opus-medium); base model fixed (Haiku 4.5); litestar n=51**
  (its full available Q&A → wider CI). No reranker (it hurt on dagster; not re-justified here).

## 6. What's solid / what's shaky

**Solid.**
- Fallback RAG > base in all 3 repos, CI excludes 0 — the core claim, on 297 questions total.
- The forum-vs-forum comparison (duckdb vs litestar) supports the trend without the synthetic confound.
- corpus-gap is *measured* per repo and is the covariate that orders the forum lifts.
- Fully audited: 0 trace errors on the trusted runs; every number → raw file in `manifest.json`.

**Shaky.**
- 3 entangled points can't prove the familiarity-vs-doc-richness decomposition — it's a hypothesis.
- pydantic-ai's magnitude is synthetic-mode-inflated; use it as a knowledge-injection illustration.
- litestar's small n widens its CI; its lift overlaps duckdb's.
- Familiarity axis is ordinal (two measurement methods).

## 7. Draft resume / interview bullets (grounded only in what was produced)

- Built a documentation-QA RAG eval that shows a small base LLM + retrieval (fallback RAG) **beats
  the base model on all 3 evaluated repos** (lift +11 / +15 / +65, each 95% CI excluding 0; 297
  questions), with every judge call traced and reconstructable.
- Showed the RAG lift **tracks a measured corpus-gap covariate** (and base familiarity): on the
  two like-for-like forum repos, lower familiarity + lower corpus-gap → larger lift; quantified the
  **intrinsic corpus ceiling** (answers absent from docs) that no retriever can fix.
- Caught and fixed a **metric-integrity bug** — exact recall@k was inflated 3.7× by content-empty
  gold stubs — and replaced it with a gold-independent **claim-coverage** judge; carried the lesson
  across repos (recall is diagnostic-only).
- Demonstrated **knowledge injection on a post-cutoff tool** (Pydantic AI v1, released after the base
  model's training cutoff): base near-floor (25/100), RAG 90/100 — explicitly flagged as a
  synthetic, doc-grounded eval (distinct from the forum-question repos).

## 8. Reproduce / audit

`results/manifest.json` maps each number to its raw files. Per repo: `data/eval/<repo>.json`,
`results/analyze_<repo>.json`, `results/ablation_<repo>{,_fallback}.json`, `traces/<repo>/*.jsonl`
(full judge prompts + responses). Pipeline: `build_corpus.py` → `curate_eval.py`/`synth_eval.py` →
`run_metrics.py` (`analyze.py` + `recall.py` + `ablation.py`) → `synthesize.py`.
