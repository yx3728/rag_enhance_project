# RAG Enhancement Pilot — eval-first documentation QA

> **📄 Full project write-up: [`FINAL_REPORT.md`](FINAL_REPORT.md)** — publication-style report spanning the original repo, the eval-first pilot, the diagnose-and-fix metric work, and the multi-repo scale-up.

An **eval-first** redesign of a documentation-QA RAG system, built on a **developer-tool /
library substrate** (questions about *using* the software, so ground truth lives in the docs).
It reuses the RAG core of [`enterprise-copilot`](https://github.com/yx3728/enterprise-copilot)
(chunking, retrieve → numbered context → "answer from context only, cite [n]" → generate) and
swaps the corpus to a real dev tool, then wraps it in a rigorous, mostly-objective eval harness.

The headline deliverable is a **RAG-vs-base-model ablation**: the same real questions answered
(a) by the base model alone and (b) by the RAG system, with the lift measured. See `REPORT.md`.

> ## 🌐 Multi-repo scale-up (latest) — see `REPORT_MULTIREPO.md`
>
> The fixed pipeline was scaled to **3 repos** spanning the base model's (Haiku 4.5, cutoff Feb
> 2025) familiarity × doc-completeness. The deployable **fallback RAG beats the base model on all
> three** (95% CI excludes 0), and the lift tracks a **measured corpus-gap** covariate:
>
> | repo | base familiarity | corpus-gap | claim-cov@5 | **fallback lift [95% CI]** | mode |
> |---|---|---|---|---|---|
> | duckdb | 43.5 (moderate) | 0.228 | 0.51 | **+11.1 [6.1,15.9]** | forum |
> | litestar | 26.7 (low) | 0.137 | 0.61 | **+14.5 [4.1,25.0]** | forum |
> | pydantic-ai | ~25 (post-cutoff) | 0.045 | 0.89 | **+65.4 [60.1,70.3]** | synthetic |
>
> pydantic-ai (v1 released after the base's training cutoff) is a knowledge-injection demo, flagged
> as synthetic (doc-shaped Qs ⇒ easy retrieval). Confounds (entangled axes, single seed, synthetic
> mode) are laid out in `REPORT_MULTIREPO.md`. Every judge call is traced (`traces/`, `manifest.json`).

---

**Original pilot substrate (chosen by data): `dagster-io/dagster`** — picked because the base model
did worst on it among the candidates (it's fast-moving, version-specific, SWE-relevant).

**Headline result (after the diagnose-then-fix pass — see `DIAGNOSIS.md`):** on 150 real
docs-answerable questions, the deployable **fallback RAG beats the base model by +12.2** (95% CI
[6.9, 17.6]) on the diagnosed+widened corpus. Honest retrieval quality is **claim-coverage@5 ≈
0.57** (the originally-reported recall@5 = 0.67 was an artifact of content-empty `<CodeExample>`
gold stubs). The dominant remaining limit is a **~21% corpus gap** (community/debugging knowledge
not in the docs), not the retriever.

<details><summary>Before diagnosis (original pilot numbers)</summary>

Naive context-only RAG *underperformed* base (lift −13.8); a fallback prompt got it to +6.6;
recall@5 reported as 0.67. The diagnosis showed recall was inflated by stub gold and the corpus
was missing the real API reference; fixing those raised the fallback lift to +12.2.
</details>

## What's here (and what's deliberately not)

This is a **pilot**: a command-line RAG pipeline + eval harness. There is **no** frontend,
server, database, deployment, or dashboard — those are next-step items in `REPORT.md`. The
focus is the pipeline + eval + the crown-jewel result.

```
src/
  llm.py              Claude client via the `claude` CLI (OAuth account, no API key) + UsageTracker
  fetch_discussions.py  pull answered GitHub Discussions Q&A for a repo (cached)
  fetch_docs.py       pull a tool's docs (markdown/rst) from its repo (the corpus)
  chunking.py         chunking strategies (ported from enterprise-copilot)
  index.py            in-process retrieval: numpy cosine + BM25 + hybrid (replaces Qdrant server)
  rag.py              base vs RAG generation (reuses enterprise-copilot prompt philosophy)
  build_eval.py       construct the eval set (real Qs + gold doc chunks, docs-answerable filter)
  recall.py           recall@k (objective, no judge) for vector/bm25/hybrid
  ablation.py         crown-jewel RAG-vs-base ablation (judge-scored) + bootstrap CI
  selection.py        substrate selection: base-model performance across candidate tools
  run_pilot.py        end-to-end runner for one substrate
  # --- diagnose-then-fix pass (see DIAGNOSIS.md) ---
  corpus.py           rebuild corpus: expand <CodeExample> stubs, MDX cleanup, heading-aware chunks (+wide: API docstrings)
  apidocs.py          extract public API docstrings from all dagster packages (the real API reference)
  coverage.py         answer-coverage@k (gold-independent Opus judge)
  analyze.py          lenient claim-coverage analyzer + corpus-gap + optional cross-encoder reranker
  triage.py           forensic residual triage into (a)corpus-gap/(b)mislabel/(c)chunking/(d)embedding
  remap_gold.py       remap eval gold onto a re-chunked corpus (lexical overlap)
docs/                 design notes (enterprise-copilot read-through, eval design, diagnosis-research)
data/                 eval sets + (gitignored) fetched corpora, indices, caches
results/              persisted JSON results (selection, recall, ablation)
```

## Requirements

- Python 3.12, a virtualenv with `sentence-transformers`, `rank_bm25`, `scikit-learn`, `numpy`.
- The `claude` CLI logged in (OAuth/Max account) — the pilot calls Claude through it, so **no
  API key is needed**. (Embeddings run locally and offline; only generation + the judge use Claude.)
- `gh` CLI authenticated (for fetching docs + Discussions from GitHub).

```bash
python -m venv .venv && .venv/bin/pip install sentence-transformers rank_bm25 scikit-learn numpy requests beautifulsoup4
```

## How to run

```bash
# 0. (one time) fetch answered Discussions Q&A for the candidate repos
.venv/bin/python src/fetch_discussions.py dagster-io dagster 320
#    (repeat for PrefectHQ/prefect, litestar-org/litestar, marimo-team/marimo, duckdb/duckdb)

# 1. substrate selection — pick the tool where the base model does worst
.venv/bin/python src/selection.py                         # -> results/selection.json

# 2. end-to-end pilot for the chosen substrate:
#    fetch docs -> curate eval set -> recall@k -> crown-jewel ablation (strict + fallback)
.venv/bin/python src/run_pilot.py dagster-io__dagster
```

Individual steps are also runnable directly: `fetch_docs.py`, `curate_eval.py`, `recall.py`,
`ablation.py [--rag-variant strict|fallback]`, and `resume_ablation.py` (re-runs any judge calls
dropped by a rate/session limit and merges them). Results land in `results/` as JSON; the eval set
in `data/eval/<tool>.json`.

## Eval design (summary)

- **Eval set**: real questions from the tool's GitHub Discussions Q&A (accepted answers as
  reference), filtered to docs-answerable.
- **Recall@k** (objective, no judge, full set): does retrieval surface the gold doc chunk?
  Gold = the chunk(s) the accepted answer points to (answer→corpus embedding match). Computed
  for vector / BM25 / hybrid.
- **Crown jewel**: base-only vs RAG answers, scored 0–100 by a stronger judge model vs the
  reference answer; report lift, win/tie/loss, recall-conditioned breakdown, bootstrap CI.
- **Judge budget**: ≤500 judge calls for the whole pilot; tracked and reported.

Full methodology, caveats, and the next-step plan are in `REPORT.md` and `docs/eval-design.md`.
