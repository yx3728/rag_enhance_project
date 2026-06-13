# RAG Diagnosis: Research on Best Practices

**Context.** Documentation-QA RAG over the Dagster docs. Forensic triage of retrieval
failures puts the residual at roughly: **~49% corpus-gap** (answer is community/debugging
knowledge not present in the docs), **~30% measurement artifact** (strict LLM
"answer-coverage" judge false-negatives — answer WAS in top-5, but judged against verbose
human reference answers), **~12% chunking**, **~9% embedding-rank** (good chunk exists but
ranks below k=5). Only **~17% is retrieval-architecture-fixable**.

## Summary (read this first)

- The external literature **agrees with our triage shape**: practitioners increasingly
  argue the corpus/data, not the retriever, is the dominant bottleneck, and that strict
  LLM judges over verbose references systematically under-score (length/verbosity bias is
  real and measurable, ~+17% pull toward longer text).
- Our two largest slices (corpus-gap 49% + measurement 30% = ~79%) are **not** fixed by
  better embeddings or rerankers. Rerankers and stronger embedding models only address the
  ~9% embedding-rank slice, and only when the right chunk is already in the candidate pool
  (the "recall ceiling").
- The single highest-leverage *retrieval-architecture* move for the 12% chunking + 9% rank
  slices is **contextual retrieval + hybrid (dense+BM25) + a reranker** — Anthropic reports
  this stack cuts top-k retrieval-failure rate by up to ~67% on its corpora. But that only
  reclaims the ~21% architecture-adjacent slice, capped at our ~17% truly fixable.
- The highest-leverage move *overall* is fixing **measurement** (re-score with a
  fact/claim-coverage judge instead of full-reference matching) and **corpus** (close
  documented gaps or set honest "not-in-corpus" expectations). These attack ~79% of the
  residual.
- Recommended priority for our case: **(1) Fix the judge → (2) Quantify/triage the corpus
  gap → (3) Add reranker + hybrid + contextual chunk prefixes for the ~17% fixable.**

---

## Q1. Honest RAG retrieval measurement (maps to our 30% measurement artifact)

**The core failure mode we hit is well documented.** LLM-as-judge scoring against a full,
verbose, human-written reference answer conflates two different things: "is the needed
information present in the retrieved context?" vs. "does the retrieved text match this
particular wording/level of detail?" When references are community-style and verbose, the
judge penalizes correct-but-terse coverage. Studies quantify a strong, systematic
**length/verbosity bias** in LLM judges — they prefer longer responses by ~+17.3% (vs.
~+12.9% for humans), i.e. they over-reward verbosity independent of correctness
(arXiv 2410.02736, "Justice or Prejudice?"; arXiv 2510.12462). This is exactly the
direction that produces false-negatives when the *reference* is verbose and the
*retrieved/answer* is concise.

**Best practices to measure retrieval honestly:**

1. **Decompose the reference into atomic claims/facts, then check coverage per claim — not
   whole-answer match.** This is precisely how Ragas computes **Context Recall**: break the
   reference into individual claims and test whether each claim is attributable to the
   retrieved context; recall = fraction of reference claims supported. **Context Precision**
   then measures whether relevant chunks are ranked above irrelevant ones. Claim-level
   attribution is far more robust to verbosity/paraphrase than reference-string matching
   (Ragas docs). MLflow's **RetrievalSufficiency** judge implements the same idea: it scores
   whether the context contains enough info to satisfy the request, graded against
   `expected_facts` rather than the full `expected_response`.

2. **Prefer "context sufficiency / answer-coverage" framed as: does the union of retrieved
   chunks contain the supporting facts?** This is reference-free or fact-list-based and
   sidesteps the verbose-reference trap. (Patronus, Anyscale, LangSmith RAG eval guides all
   converge on coverage + precision rather than reference-string similarity.)

3. **Top-k pooling + judge to build/repair gold labels.** When exact-chunk `recall@k` is
   measured against a single pre-labeled "gold chunk," it is misleading because *multiple
   chunks can carry the same fact* and labelers miss valid alternates. The fix used in IR is
   **pooling**: pool the top-k from several retrievers/configs, have the judge mark every
   pooled chunk that supports the answer, and treat *any* supporting chunk as a hit. This
   directly removes the "good chunk present but not the labeled one" false-negative —
   relevant to our 30% slice.

4. **Mitigate judge bias explicitly:** give the judge the *fact list*, not the verbose
   reference; instruct it to ignore length/style and judge only factual support; consider
   truncating/normalizing length before judging; use reference-guided grading (gold facts
   ground the judge in truth, reducing both hallucination and verbosity bias). Validate the
   judge against a small human-labeled set and report judge–human agreement.

5. **When exact-chunk recall@k is misleading:** (a) when facts are redundantly distributed
   across chunks; (b) when chunk boundaries differ between index variants so the "gold
   chunk id" is config-specific; (c) when the answer needs *synthesis across several
   chunks* — single-gold-chunk recall undercounts. In all three, prefer **claim-coverage
   over the retrieved set** to per-chunk id matching.

> **For us:** ~30% of the residual is almost certainly recoverable by re-scoring with a
> claim/fact-coverage judge (Ragas Context Recall style or MLflow RetrievalSufficiency)
> plus top-k pooling, instead of matching against verbose community reference answers.
> This is the cheapest, highest-confidence win.

## Q2. Cross-encoder / LLM rerankers (maps to our 9% embedding-rank slice)

**What lift to expect.** Rerankers reorder an existing candidate pool with full
query–document cross-attention, recovering precision that single-vector bi-encoders lose.
Reported lifts on documentation/technical retrieval: Anthropic's stack shows **reranking on
top of contextual embeddings + BM25 takes the top-20 retrieval-failure reduction from ~49%
to ~67%** — i.e. reranking contributed a meaningful additional chunk of the gain. Vendor and
community benchmarks put `bge-reranker-v2-m3` at roughly Cohere-Rerank-3.5 quality at zero
API cost; it is the strongest open-weight option for English+multilingual.

**Latency cost.** Cross-encoders run a full transformer pass *per query–document pair*, so
cost scales linearly with pool size. Practical figures from benchmarks: `ms-marco-MiniLM`
~1.1s and `bge-reranker-v2-m3` ~12s to score 1000 candidates on comparable hardware
(~10x slower); on GPU, `bge-reranker-v2-m3` is ~50–100ms for small pools. The standard
deployment is **retrieve top ~50–100 with the bi-encoder, rerank down to top 5**, keeping
added latency in the tens-of-ms-to-low-hundreds range. Pinecone notes bi-encoder vector
search returns in <100ms while naive full-corpus cross-encoding is infeasible (their example:
>50 hours over 40M records) — hence the two-stage pattern.

**When rerankers DON'T help — the recall ceiling.** A reranker only reorders the candidate
pool; it **cannot recover a relevant chunk that the first-stage retriever never surfaced**.
If the answer isn't in the corpus at all (our 49%), or the right chunk isn't in the
candidate pool, the reranker is "an expensive sorter of partial misses" (Pinecone; multiple
practitioner writeups). So a reranker addresses our **9% embedding-rank** slice (right chunk
in pool, ranked >k) but does **nothing** for the 49% corpus-gap or the 30% measurement
slices.

> **For us:** A reranker is worth adding — it directly targets the 9% rank slice and some of
> the 12% chunking slice — but budget its impact at single-digit percentage points of the
> residual, not a transformation. Increase first-stage `k` to ~50–100 before reranking so
> the chunk is actually in the pool.

## Q3. Embedding model choice for code/technical docs

**MTEB landscape (general retrieval):** Cohere embed-v4 ~65, OpenAI `text-embedding-3-large`
~64.6, BGE-M3 ~63. For **code/developer** content specifically, **code-tuned models are the
real differentiator**: `voyage-code-3` and Gemini-embedding-class models score notably
higher on MTEB-Code (~84) than general-purpose embedders. `text-embedding-3-large` has not
been updated since Jan 2024 and is now matched/beaten by newer hosted and open models.

**Small vs large open model.** The jump from `bge-small` → `bge-large` gives a real but
**modest** retrieval-quality bump on general benchmarks (a few MTEB points), at higher
latency/memory. The larger, more reliable gains come from: (a) switching to an
**instruction-tuned** model (E5/GTE/BGE-M3 with query instructions) and (b) for code-heavy
docs, a **code-specialized** model (`voyage-code-3`). For mostly-prose docs with code
snippets, a strong general open model (BGE-M3 / GTE-large / E5-large) is usually sufficient.

> **For us:** Embedding choice is a second-order lever given the corpus-gap dominance. If we
> change anything, instruction-tuned BGE-M3 (also enables native hybrid dense+sparse) is the
> pragmatic pick; `voyage-code-3` only if Dagster answers hinge on code-symbol matching.
> The small→large upgrade alone is rarely worth it relative to fixing measurement/corpus.

## Q4. Chunking for MDX/markdown technical docs (maps to our 12% chunking slice)

- **Heading-aware / structural splitting.** Use Markdown/MDX-structure-aware splitters
  (LangChain Markdown splitter, LlamaIndex `MarkdownNodeParser`) so chunks align to
  `#`/`##` sections and **code blocks and tables are never split mid-fence**. Naive
  fixed-size splitting that bisects a code example is a known failure source for dev docs.
- **Parent-document / small-to-big retrieval.** Embed small precise child chunks
  (~100–300 tokens) for matching, but **return the larger parent section** (e.g. the whole
  heading section, ~500–2000 tokens) to the LLM. LangChain `ParentDocumentRetriever` /
  LlamaIndex `HierarchicalNodeParser`; practitioners report ~15–30% accuracy gains on
  context-heavy queries. This is well suited to docs where a code example needs its
  surrounding prose.
- **Code-aware chunking.** Keep a code block with its explanatory paragraph and the heading
  it lives under; prefer recursive splitting that respects fences over token-count splitting.
- **Contextual chunk prefixes (Anthropic Contextual Retrieval).** Prepend a 50–100 token
  LLM-generated description of each chunk's role in its parent doc before embedding.
  Anthropic reports **contextual embeddings alone cut top-20 retrieval failures ~35%**, and
  **+contextual BM25 ~49%**. This is highly applicable to fragmented MDX where a chunk loses
  its "which page/feature" context.

> **For us:** For the 12% chunking slice, the best-bang-for-buck combo is **heading/MDX-aware
> structural splitting + parent-document retrieval + contextual prefixes**, keeping code
> blocks intact. This overlaps with the reranker recommendation as the "architecture fix"
> bundle for our ~17% fixable.

## Q5. Corpus-coverage problem vs. retrieval-architecture problem — fix priority

There is clear, converging guidance that this is **three distinct questions** and must be
diagnosed separately (Pinecone "Beyond Retrieval"; Deepchecks; TDS "10 Common RAG Mistakes"):

1. **Does the corpus contain the answer?** (content/coverage question)
2. **Does the retriever surface the right passage?** (search/architecture question)
3. **Does the generator stay faithful to it?** (generation question)

**How teams distinguish coverage vs. architecture failures:**
- **Oracle/upper-bound test.** If you place the known answer passage in context and the
  system answers correctly, the failure is *retrieval*, not generation. If no passage in the
  *entire corpus* supports the answer (manual or judge-assisted search), it's a **corpus
  gap** — unfixable by any retriever.
- **Context Recall as the splitter.** Low Context Recall = retrieval/coverage gap; high
  Context Recall but wrong answer = generation problem (Ragas, Qdrant eval guide). Crucially,
  Context Recall is computed against the *reference*, so when the reference fact isn't in the
  corpus at all, recall is structurally 0 — that flags **corpus gap**, distinct from
  ranking failure.
- **Recall@large-k probe.** If the gold chunk appears at k=100 but not k=5, it's a
  **ranking** problem (reranker/embedding territory). If it never appears even at large k,
  it's **corpus or chunking**.

**Fix priority — corpus usually beats retriever.** A growing body of practitioner and
research opinion argues the **data/corpus is the bottleneck more often than the retriever**
(e.g. "Less LLM, More Documents" arXiv 2510.02657; TDS "RAG Is Not Machine Learning"). The
recommended order is: **fix measurement → fix corpus coverage → then retrieval architecture
(chunking, hybrid, rerank) → last, embedding model swaps.** Over-investing in rerankers/
embeddings when the answer isn't in the corpus is the classic wasted effort.

> **For us:** Our triage already matches this guidance. 49% corpus-gap means the dominant
> "fix" is **corpus actions, not retriever actions**: either ingest the missing
> community/debugging knowledge (e.g. GitHub issues, Slack/forum answers, changelogs) or
> explicitly scope these as out-of-corpus and report them honestly rather than as retrieval
> failures.

---

## Concrete recommendations for OUR situation

Priority order (impact-weighted by our residual slices):

1. **Re-measure with a claim/fact-coverage judge (attacks ~30%).** Replace
   verbose-reference matching with Ragas-style **Context Recall** (claim decomposition +
   attribution) or MLflow **RetrievalSufficiency** (grade against an expected-facts list).
   Add **top-k pooling** so any supporting chunk counts as a hit. Validate judge vs. a small
   human set. This is the cheapest, highest-confidence recovery and likely reclassifies most
   of the 30% as passing.
2. **Quantify and triage the corpus gap (attacks ~49%).** For the gap questions, confirm
   via large-k + corpus-wide search that the fact truly isn't present. Then either (a)
   **expand the corpus** with the community/debugging sources where answers actually live,
   or (b) **scope them out** and stop counting them as retrieval failures. No retriever
   change helps here.
3. **Ship the architecture bundle for the ~17% fixable (attacks ~12% chunking + ~9% rank).**
   - Hybrid retrieval: **dense + BM25** (BGE-M3 supports both natively).
   - **MDX/heading-aware structural chunking** with code blocks kept intact +
     **parent-document retrieval** (small child for match, parent section returned).
   - **Contextual chunk prefixes** (Anthropic-style) before embedding.
   - **Add a reranker** (`bge-reranker-v2-m3`, open-weight; or Cohere Rerank if hosted is
     preferred) over a **top-50–100** first-stage pool, reranked to top-5.
4. **Embedding model: low priority.** Only revisit after 1–3. If changed, prefer
   instruction-tuned **BGE-M3**; consider `voyage-code-3` only if answers hinge on code
   symbols. The small→large open swap alone is rarely worth the cost here.

**Expected ceiling:** even a perfect retrieval architecture can only address the ~17%
fixable plus whatever the reranker reclaims from the rank slice. The big numbers move only
by fixing the **judge** and the **corpus**.

---

## Citations

**Measurement / LLM-judge / context metrics**
- Ragas — Context Recall (claim decomposition + attribution): https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/
- Ragas — Context Precision: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/
- MLflow — RetrievalSufficiency LLM judge (grade against expected_facts): https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/context-sufficiency
- Patronus — RAG Evaluation Metrics: https://www.patronus.ai/llm-testing/rag-evaluation-metrics
- Anyscale — RAG evaluation: https://docs.anyscale.com/rag/evaluation
- LangChain/LangSmith — Evaluate a RAG application: https://docs.langchain.com/langsmith/evaluate-rag-tutorial
- Qdrant — Best Practices in RAG Evaluation: https://qdrant.tech/blog/rag-evaluation-guide/
- "Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge" (length/verbosity bias): https://arxiv.org/pdf/2410.02736
- "Evaluating and Mitigating LLM-as-a-judge Bias" (+17.3% length bias): https://arxiv.org/pdf/2510.12462
- Evidently AI — LLM-as-a-judge guide: https://www.evidentlyai.com/llm-guide/llm-as-a-judge

**Rerankers**
- Pinecone — Rerankers and Two-Stage Retrieval (recall ceiling, latency): https://www.pinecone.io/learn/series/rag/rerankers/
- Reranker benchmark comparison (bge-v2-m3, Cohere, ms-marco; latency figures): https://aimultiple.com/rerankers
- BGE Reranker v2 M3 vs Cohere Rerank 3.5: https://agentset.ai/rerankers/compare/baaibge-reranker-v2-m3-vs-cohere-rerank-35
- "Recall Ceiling Problem" explainer: https://medium.com/@gowthami.mv105/how-rag-actually-works-from-encoders-to-the-recall-ceiling-problem-7eabd37d1c47

**Embedding models**
- Embedding Models 2026 benchmark/comparison (MTEB, voyage-code, OpenAI, BGE): https://app.ailog.fr/en/blog/news/embedding-models-2026
- Choosing embedding models (MTEB scores): https://app.ailog.fr/en/blog/guides/choosing-embedding-models
- E5 / "Improving Text Embeddings with Large Language Models" (E5-mistral): https://arxiv.org/pdf/2401.00368

**Chunking**
- Anthropic — Contextual Retrieval (35% / 49% / 67% failure-rate reductions): https://www.anthropic.com/news/contextual-retrieval
- Anthropic Cookbook — Contextual embeddings guide: https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide
- LangChain ParentDocumentRetriever / parent-child chunking: https://medium.com/@seahorse.technologies.sl/parent-child-chunking-in-langchain-for-advanced-rag-e7c37171995a
- Chunking techniques with LangChain & LlamaIndex (Markdown-aware): https://www.lancedb.com/blog/chunking-techniques-with-langchain-and-llamaindex
- "Reconstructing Context: Evaluating Advanced Chunking Strategies for RAG": https://arxiv.org/pdf/2504.19754

**Corpus vs. retrieval diagnosis / fix priority**
- Pinecone — The Applicability Problem in RAG (corpus vs retrieval vs generation): https://www.pinecone.io/learn/series/beyond-retrieval/rag-applicability-problem/
- Deepchecks — Retrieval Quality vs Answer Quality: https://deepchecks.com/retrieval-vs-answer-quality-rag-evaluation/
- "Less LLM, More Documents: Searching for Improved RAG" (data as bottleneck): https://arxiv.org/pdf/2510.02657
- TDS — 10 Common RAG Mistakes in Production: https://towardsdatascience.com/10-common-rag-mistakes-we-keep-seeing-in-production/
- TDS — RAG Is Not Machine Learning: https://towardsdatascience.com/rag-is-not-machine-learning-and-the-ml-toolkit-solves-the-wrong-problem/
