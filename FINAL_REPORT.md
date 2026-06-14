# An Eval-First Reconstruction of a Documentation-QA RAG System

### From a generic enterprise demo to a measured, debugged, multi-repo retrieval evaluation

**Author's note / framing.** This is the consolidated final report for a four-phase project that
takes an existing portfolio RAG demo (`enterprise-copilot`) and rebuilds it around *evaluation* as
the primary artifact. It is written to publication structure (abstract → methods → results →
threats to validity → conclusions) but is honest about its scope: a single-engineer, single-seed,
single-judge-model study run from the command line on a personal compute budget. Where a result is
an artifact, a confound, or a null, it is reported as such. Every headline number is reconstructable
from raw judge transcripts (`traces/`) via `results/manifest.json`.

- **Subject (answering) model throughout:** Claude **Haiku 4.5** (reliable knowledge cutoff Feb 2025).
- **Judge model throughout:** Claude **Opus 4.8** (`effort=medium`), distinct from the subject to
  reduce self-preference bias.
- **Access:** all generation and judging run through the `claude` CLI on a personal OAuth account
  (no API key); embeddings are local (`bge-small-en-v1.5`), offline, free.
- **Scale:** 38 commits; ~2,800 LOC across 24 Python modules; 4 substrates; ~447 evaluation
  questions; ~$64 of traced LLM spend (plus earlier untraced pilot/diagnosis work).

---

## Abstract

We reconstruct a documentation question-answering RAG system as an *eval-first* artifact and use it
to study when retrieval-augmented generation actually beats a capable base LLM. Starting from a
generic enterprise RAG demo, we (1) build a rigorous, mostly-objective evaluation harness on a
developer-tool substrate; (2) discover and repair a **metric-integrity failure** in which exact
chunk-recall@5 reads 0.673 while honest answer-coverage is ~0.18, because the auto-labeled "gold"
chunks were content-empty include-directive **stubs**; (3) forensically attribute residual
retrieval failures to four sources and show the dominant ones (corpus gaps, measurement artifacts)
are *not* fixable by better embeddings or rerankers; and (4) scale to three repositories spanning
the base model's familiarity, demonstrating that a deployable "fallback" RAG prompt beats the base
model in all three (lift +11.1, +14.5, +65.4; each 95% CI excludes 0) with the lift co-varying with
a directly-measured **corpus-gap** covariate. The central methodological contribution is that
*recall@k is an unreliable headline for auto-labeled RAG evals*; a gold-independent claim-coverage
metric and explicit corpus-gap measurement are required to tell the truth.

---

## 1. Starting point: the original `enterprise-copilot` repository

The project began from [`enterprise-copilot`](https://github.com/yx3728/enterprise-copilot), a
full-stack "AI Enterprise Copilot" demo: a Next.js dashboard + FastAPI backend + PostgreSQL +
**Qdrant** vector DB + OpenAI, containerized for Docker / Azure Container Apps. Its RAG core did
document ingestion (PDF/Markdown, three chunking strategies), OpenAI embeddings, Qdrant vector
search, and citation-backed answering. Its "evaluation" was a set of **heuristics**: keyword
recall, keyword-overlap "retrieval relevance," and a token-overlap "hallucination" proxy, aggregated
into an "answer-quality proxy."

**Assessment carried into this project.** The RAG *pipeline shape* was sound and worth reusing:
chunk → embed → top-k cosine → numbered context → "answer from context only, cite [n]" → generate.
The *evaluation* was the weak point — keyword overlap is neither a retrieval metric nor an answer
metric. The eval-first reconstruction keeps the pipeline shape and replaces everything around the
measurement.

**What was reused vs. replaced/added** (see `docs/enterprise-copilot-notes.md`):

| Component | Disposition |
|---|---|
| Chunking strategies (fixed/paragraph/sentence) | **Reused** (ported, then upgraded — §3) |
| Pipeline shape + "context-only, cite [n]" prompt philosophy | **Reused** |
| Qdrant *server* | **Replaced** with an in-process NumPy cosine index + BM25 + hybrid (no daemon, reproducible) |
| OpenAI embeddings/generation | **Replaced** with local `bge-small` embeddings + Claude via CLI |
| Heuristic keyword/hallucination metrics | **Replaced** with recall@k, LLM-judge answer scoring, and a gold-independent claim-coverage metric |
| Frontend / server / DB / Docker / Azure | **Dropped** (out of scope; this is a CLI eval artifact) |

---

## 2. Phase I — Eval-first pilot (substrate selection + the crown-jewel ablation)

**Substrate selection by data, not guess.** We required a *code/usage* substrate where ground truth
lives in docs, and selected among five developer tools by **base-model performance** (Haiku, no
retrieval, judged vs. accepted GitHub-Discussions answers): lower base score = more for RAG to
prove (`results/selection.json`):

| candidate | base score (0–100) |
|---|---|
| marimo | 26.2 |
| litestar | 28.0 |
| **dagster** | **30.8** ← chosen |
| duckdb | 48.1 |
| prefect | 53.7 |

Dagster was chosen from the statistically-tied "worst" group on the secondary criteria the task set
(most answered Q&A, highest docs-answerable rate, cleanest Markdown docs) so a credible 80–150
question eval set was reachable.

**Eval set.** 150 real, docs-answerable dagster Discussions questions (accepted answer = reference),
curated by an LLM pass that verified each had a supporting gold chunk; corpus = dagster docs (4,708
chunks). **Metrics:** objective recall@k (free, full set) + a budgeted LLM-judge answer-quality
ablation.

**The crown jewel (RAG vs. base, n=150, joint randomized-position Opus judge, bootstrap CI):**

| RAG variant | base | RAG | lift | 95% CI | W/T/L |
|---|---|---|---|---|---|
| strict ("context only") | 54.5 | 40.7 | **−13.8** | [−20.2, −7.6] | 50/10/90 |
| fallback (context, else parametric) | 49.5 | 56.1 | **+6.6** | [2.0, 11.1] | 67/37/46 |

The pilot's honest, non-obvious headline: **naive context-only RAG *underperformed* the base
model.** Recall-conditioning localized it — RAG cratered only when retrieval missed the gold chunk
(the strict prompt answers "not enough information" while the base answers from training). A
one-line **fallback prompt** (prefer+cite context, otherwise answer from the model's own knowledge
and flag it) flipped the lift to +6.6. This is where Phase I stopped, and where the numbers looked
"weak" — motivating Phase II.

---

## 3. Phase II — Diagnose-then-fix (the metric-integrity result)

A dedicated diagnostic pass tested the hypothesis that the weak numbers were an
*implementation/measurement* problem, not a task ceiling.

**3.1 The recall illusion.** We defined a gold-independent **answer-coverage@5** metric (Opus judge:
"do the top-5 retrieved chunks contain the information to produce the reference answer?"). On the
frozen baseline:

> **exact recall@5 = 0.673  vs.  answer-coverage@5 = 0.18.**

The gap is *inverted* from the usual direction. Root cause: **24% of all chunks were unexpanded
`<CodeExample path=.../>` include-directive stubs**, and many were labeled "gold." Retrieval found
the stub (recall looks high) but the stub contains no answer (coverage is the truth). **recall@5 was
an artifact; honest retrieval was ~0.18.** This is the project's core methodological finding:
*never headline exact recall@k when gold is auto-labeled.*

**3.2 Mechanical corpus repair (one pass).** Expanded every `<CodeExample>` by inlining the
referenced source snippet (1,152 stubs → real code); added the API reference; replaced blind
800-char chunking with heading-aware, code-fence-safe chunking + size caps. Result: strict
coverage barely moved (0.18 → 0.20) — proving the *content in chunks* was not the coverage
bottleneck (it had been poisoning the *gold/recall*, not the retrieval).

**3.3 Forensic triage of the residual** (`results/residual_triage.json`). Every residual failure
classified into four sources:

| source | share of residual | retrieval-fixable? |
|---|---|---|
| (a) corpus gap (answer absent from docs — community/debug knowledge) | **49%** | No |
| (b) metric artifact (answer *was* in top-5; strict judge false-negative) | **30%** | No |
| (c) chunking (present but fragmented) | 12% | Partly |
| (d) embedding rank (good chunk exists, ranked > k) | 9% | Yes |

The 30% (b) showed the strict coverage judge has a large verbosity-driven false-negative rate
against human reference answers; a **lenient claim-coverage** metric (partial/reworded support
counts) put honest coverage@5 at **0.527** (docs-only), not 0.18. Only **~17%** of questions are
retrieval-architecture-fixable; **~39%** are genuine corpus gaps. *The weak numbers were a corpus +
measurement story, not an embedding ceiling.* (External literature agreed; see
`docs/diagnosis-research.md`.)

**3.4 Levers, measured honestly** (`results/improvement_sweep.json`, lenient claim-coverage@5):

| config | cov@5 | corpus-gap |
|---|---|---|
| docs only | 0.527 | 0.247 |
| **+ all-package API docstrings ("wide")** | **0.567** | **0.213** |
| wide + off-the-shelf cross-encoder reranker | 0.48 ↓ | 0.213 |

The real API reference lives in **source docstrings** (Sphinx/mkdocstrings rST/md are autodoc
stubs); extracting all packages' docstrings lifted coverage and cut the gap. An off-the-shelf
reranker **hurt** (demoted good dense hits on long, messy queries) — a reported negative lever.

**3.5 Crown jewel on the fixed corpus.** Re-running the same ablation on the widened corpus nearly
doubled the deployable lift and erased the strict regression:

| corpus | strict lift | fallback lift | 95% CI |
|---|---|---|---|
| original | −13.8 | +6.6 | [2.0, 11.1] |
| **widened (fixed)** | **−3.4 (≈tie)** | **+12.2** | **[6.9, 17.6]** |

Recall-conditioned, the fix did exactly what the diagnosis predicted: fallback RAG now beats base
even when the exact gold chunk isn't in top-5 (+5.1) because the richer corpus supplies supporting
context anyway. **Hypothesis confirmed for the fixable portion; the residual ~21% corpus gap is a
real, non-retrieval-fixable ceiling.**

---

## 4. Phase III — Multi-repo scale-up (when does RAG beat the base?)

We scaled the fixed pipeline to three repositories spanning the base model's familiarity, verifying
cutoffs online: Haiku 4.5 reliable cutoff **Feb 2025** (training Jul 2025), so "post-cutoff" tools
must postdate mid-2025. We chose **Pydantic AI** (v1.0 released 2025-09-04, after Haiku's training
cutoff; Discussions disabled → a flagged **synthetic, doc-grounded** eval mode).

**Headline result** (`results/multirepo/cross_repo.json`; subject Haiku 4.5, judge Opus-medium;
deployable = fallback RAG; bootstrap CI):

| repo | familiarity cell | mode | n | base fam. | claim-cov@5 | **corpus-gap** | **fallback lift** | 95% CI |
|---|---|---|---|---|---|---|---|---|
| duckdb | moderate | forum | 136 | 43.5 | 0.51 | 0.228 | **+11.1** | [6.1, 15.9] |
| litestar | low | forum | 51 | 26.7 | 0.61 | 0.137 | **+14.5** | [4.1, 25.0] |
| pydantic-ai | zero / post-cutoff | synthetic | 110 | 24.9 | 0.89 | 0.045 | **+65.4** | [60.1, 70.3] |

**The deployable RAG beats the base model in all three repos (every 95% CI excludes 0; 297
questions).** And the lift co-varies with the measured covariates in the hypothesized direction:

```
base familiarity ↓   43.5 → 26.7 → 24.9
corpus-gap        ↓  0.228 → 0.137 → 0.045
fallback lift     ↑  +11.1 → +14.5 → +65.4
```

The post-cutoff point is dramatic (base 24.9 → RAG 90.3, wins 106/110): a tool the base genuinely
does not know, whose answers are fully in the docs, so RAG injects the missing knowledge. It is
presented as a **knowledge-injection demonstration**, explicitly flagged because synthetic
doc-shaped questions make retrieval artificially easy.

---

## 5. Engineering delivered (beyond the original repo)

A reusable, audited CLI evaluation toolkit (~2,800 LOC, 24 modules):

- **LLM access & audit:** `llm.py` — Claude via the `claude` CLI (no API key), per-call cost/token
  tracking, model-aware effort policy, and a thread-safe **trace writer** persisting the full prompt
  + raw response of every judge/gen call (39 MB of transcripts, committed).
- **Corpus tooling:** `corpus.py` / `build_corpus.py` (heading-aware, code-fence-safe chunking;
  MDX cleanup; include-directive expansion), `apidocs.py` (AST extraction of public API docstrings —
  the real API reference behind autodoc stubs), `fetch_docs.py` / `fetch_discussions.py` (rate-limit-
  aware GitHub fetchers).
- **Retrieval:** `index.py` — in-process dense (cached embeddings) + BM25 + hybrid, replacing the
  Qdrant service.
- **Eval-set construction:** `curate_eval.py` (LLM-verified, content-bearing gold from a
  retriever-fair candidate pool), `synth_eval.py` (doc-grounded synthetic questions for post-cutoff
  repos), `remap_gold.py`.
- **Metrics:** `recall.py` (recall@k), `coverage.py` + `analyze.py` (gold-independent
  claim-coverage + measured corpus-gap + optional reranker), `ablation.py` (joint
  randomized-position RAG-vs-base judge with bootstrap CI and recall-conditioning),
  `triage.py` (forensic a/b/c/d attribution).
- **Orchestration & robustness:** `run_metrics.py`, `synthesize.py`, `resume_ablation.py`
  (re-runs only the calls a session limit killed and merges them — used after a real incident),
  `manifest.json` (number → raw-file provenance).

**Problems solved along the way:** GitHub secondary-rate-limit backoff; Opus judge too slow/
expensive at scale → fast model for labeling + `effort=medium` for judging; a mid-run **session
limit** that silently errored 41–98% of curation calls, *caught via the per-call error traces*,
archived, and cleanly re-run; oversized/zero-content chunks; gold remapping across re-chunked
corpora.

---

## 6. Threats to validity (explicit)

- **Single seed, single judge model** (Opus-medium), **single subject model** (Haiku 4.5). Judge
  scores are directional, not absolute; position bias mitigated by randomization, self-preference by
  judge≠subject, and we cross-checked judge verdicts against objective recall.
- **Entangled axes.** Familiarity × doc-richness × eval-mode are not a factorial design — the trend
  in §4 is a **hypothesis**, not a causal decomposition. The clean like-for-like is duckdb vs.
  litestar (both forum).
- **Synthetic mode inflates pydantic-ai.** Doc-shaped questions ⇒ easy retrieval (cov@5 0.89,
  gap 0.045 by construction). The +65.4 conflates real post-cutoff ignorance with synthetic-
  retrieval ease; it is not forum-comparable.
- **Selection effect in corpus-gap.** Forum questions skew toward what docs *can't* answer;
  corpus-gap misses are intrinsic (answer absent), not retriever failures.
- **Eval-set sizes** vary (51–150); litestar's n=51 is its full available Q&A and widens its CI.
- **Base familiarity is ordinal** (measured by a probe for forum repos, by synthetic-eval base for
  pydantic-ai; probe and ablation-base also differ within a repo).
- **Reference answers are human/community text**; the strict coverage judge under-scored them ~30%
  — the reason we headline lenient claim-coverage.

---

## 7. Conclusions

1. **Measurement is the deliverable.** The single most important finding is negative and
   methodological: auto-labeled exact recall@k can be off by 3.7× (0.67 vs. 0.18) due to
   content-empty gold; a gold-independent claim-coverage metric plus measured corpus-gap are
   required. This generalizes beyond these repos.
2. **Naive RAG can hurt; the prompt and the corpus matter more than the retriever.** Context-only
   RAG underperformed a capable base model (−13.8) until (a) a fallback prompt and (b) corpus
   repair (include-expansion + real API docstrings) turned it into a significant win (+12.2).
3. **RAG beats the base where the base is weak and the docs hold the answer** — shown across three
   repos (+11.1 / +14.5 / +65.4, all CIs exclude 0), with the lift tracking a measured corpus-gap
   covariate. Where it wouldn't help — community-knowledge corpus gaps — we measured and named the
   ceiling rather than hiding it.
4. **Honesty held throughout.** Negatives (naive RAG loss, reranker regression, the recall
   illusion, the session-limit incident) are reported with their direction of effect and their
   receipts.

**Reproduce / audit.** `results/manifest.json` maps every headline number to raw files; full judge
transcripts are in `traces/<repo>/*.jsonl`. Pipeline per repo:
`build_corpus.py → curate_eval.py | synth_eval.py → run_metrics.py (analyze + recall + ablation) →
synthesize.py`. Companion documents: `REPORT.md` (Phase I), `DIAGNOSIS.md` (Phase II),
`REPORT_MULTIREPO.md` (Phase III), `WORKLOG.md` (decisions + per-phase cost).

*Total traced LLM spend ≈ $64 (multi-repo + diagnosis traces); subject = Claude Haiku 4.5,
judge = Claude Opus 4.8 at medium effort, embeddings local and free.*
