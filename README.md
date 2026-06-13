# RAG Enhancement Pilot — eval-first documentation QA

An **eval-first** redesign of a documentation-QA RAG system, built on a **developer-tool /
library substrate** (questions about *using* the software, so ground truth lives in the docs).
It reuses the RAG core of [`enterprise-copilot`](https://github.com/yx3728/enterprise-copilot)
(chunking, retrieve → numbered context → "answer from context only, cite [n]" → generate) and
swaps the corpus to a real dev tool, then wraps it in a rigorous, mostly-objective eval harness.

The headline deliverable is a **RAG-vs-base-model ablation**: the same real questions answered
(a) by the base model alone and (b) by the RAG system, with the lift measured. See `REPORT.md`.

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
docs/                 design notes (enterprise-copilot read-through, eval design)
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
.venv/bin/python src/fetch_discussions.py <owner> <name> 120

# 1. substrate selection — pick the tool where the base model does worst
.venv/bin/python src/selection.py

# 2. end-to-end pilot for the chosen substrate (fetch docs -> eval set -> recall -> ablation)
.venv/bin/python src/run_pilot.py <owner__name>
```

Results land in `results/` as JSON; the eval set in `data/eval/<tool>.json`.

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
