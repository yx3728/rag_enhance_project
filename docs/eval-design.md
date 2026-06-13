# Eval design (pilot)

Pragmatic, ship-real-numbers methodology. Two axes; objective metric carries the load,
LLM judge is reserved for the crown jewel (≤500 judge calls TOTAL for the pilot).

## Corpus
The chosen tool's official docs (markdown from its repo), chunked (paragraph strategy,
~800 chars, 120 overlap — enterprise-copilot defaults). In-process index (numpy cosine +
BM25 + hybrid), local `bge-small` embeddings.

## Eval set
- Real questions from the tool's GitHub Discussions **Q&A with accepted answers**.
- **Reference answer** = the accepted answer (used by the crown-jewel judge).
- **Gold chunk(s)** for recall@k = doc chunk(s) the accepted answer points to. Located for
  FREE (no judge) via two signals:
  1. **Doc links in the answer** (high precision): if the accepted answer links to the tool's
     docs, the linked page's chunks are gold.
  2. **Answer→corpus embedding match** (fallback): chunks with cosine ≥ τ to the accepted
     answer text. answer→corpus is a *different* query than the question→corpus retrieval we
     evaluate, so it is not fully circular.
- **Docs-answerable filter** (free): keep a question only if ≥1 gold chunk is found above
  threshold. Roadmap / "it's a bug" / opinion answers don't match any doc chunk → dropped.
- Target ~100–150 kept questions.
- Caveat (disclosed): gold labeling shares the embedding space with the dense retriever, so it
  mildly favors the vector retriever in the *strategy comparison*. It does NOT affect the crown
  jewel (base has no retrieval). We validate a sample of gold labels by hand and report it.

## Metrics
1. **Recall@k (objective, no judge, full set).** For retriever R and k∈{1,3,5,10}:
   fraction of questions where retrieved top-k ∩ gold ≠ ∅. Computed for vector / BM25 / hybrid
   → this also gives the optional strategy comparison for free.
2. **Crown-jewel ablation (LLM judge).** Same questions, two answers:
   - base: model answers with NO retrieval.
   - RAG: model answers from top-k retrieved context.
   Judge (stronger, different model) scores each answer 0–100 for correctness vs the reference
   answer. Report base vs RAG mean + win/tie/loss + a recall-conditioned breakdown
   (RAG quality when gold was vs wasn't retrieved). Report n and a bootstrap CI on the lift.

## Budget
Judge calls = selection (~70) + crown jewel (2 × N). Recall@k and the strategy comparison are
free. Sample gold-label validation: a few judge calls. Kept well under 500.
