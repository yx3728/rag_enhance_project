# `enterprise-copilot` — read-through notes (Phase 1)

Source: https://github.com/yx3728/enterprise-copilot.git (re-fetched 2026-06-13, single commit
`9ba7c9d Initial public release`). Re-fetched into `_ref_enterprise-copilot/` (gitignored).

## What it is
A full-stack enterprise knowledge assistant: Next.js frontend + FastAPI backend + PostgreSQL +
Qdrant + OpenAI. RAG over uploaded PDFs/Markdown. Containerized (docker-compose) with Azure
Container Apps Bicep. MIT licensed.

## Architecture (backend `app/`)
- `config.py` — pydantic-settings. OpenAI (`text-embedding-3-small`, `gpt-4o-mini`), Postgres,
  Qdrant, chunk size 800 / overlap 120.
- `services/chunking.py` — **pure, reusable.** `chunk_text(text, strategy)` for
  fixed / paragraph / sentence. `_merge_parts` greedily packs parts up to `chunk_size`,
  `_fixed_chunks` does sliding window with overlap. No external deps. char-based sizing.
- `services/embeddings.py` — async OpenAI embeddings, batched. Thin wrapper.
- `services/vector_store.py` — Qdrant: ensure_collection, upsert, search (cosine, top_k,
  optional doc filter), delete_by_document. VECTOR_SIZE=1536 (tied to OpenAI embedder).
- `services/rag.py` — **the RAG core.** `PROMPTS` dict (default/concise/detailed system prompts,
  all "answer using ONLY the provided context, cite [n]"). `retrieve()` (embed query -> search),
  `build_context()` (numbered context blocks + Citation objects), `generate_answer()`
  (OpenAI chat, temp 0.2), `chat()` (DB-coupled orchestration: sessions/messages/usage).
- `services/ingestion.py` — orchestrates extract -> chunk -> embed -> upsert; DB-coupled.
- `services/parsers.py` — PDF (pypdf) + md/txt extraction.
- `services/evaluation.py` — **heuristic eval (the part we redesign).** `keyword_recall`
  (fraction of expected keywords appearing in answer string), `retrieval_relevance` (mean
  keyword overlap in retrieved chunks), `hallucination_heuristic` (token-overlap proxy).
  `run_evaluation()` aggregates avg/p95 latency + the three heuristics + an "answer_quality_proxy".
- `models/entities.py` — SQLAlchemy: Document, DocumentChunk, ChatSession, ChatMessage,
  MessageFeedback, EvaluationRun, UsageEvent.
- `api/` — FastAPI routers: chat, documents, evaluation, health.

## What we REUSE (continuity with Joey's project)
- **Chunking strategies** (`chunking.py`) — port near-verbatim; pure functions, no deps.
- **RAG shape**: embed-query -> vector-search top_k -> build numbered context -> "answer from
  context only, cite [n]" system prompt -> generate. We keep this exact pipeline shape and the
  prompt philosophy from `rag.py::PROMPTS`.
- **Vector search semantics**: cosine similarity, top_k, payload carries chunk_id/doc/index/content.
- **Eval *intent***: per-case metrics + aggregate, persisted as a run. We keep the structure but
  replace the heuristics with a rigorous, objective **recall@k** (the spec's headline metric) and
  a budgeted LLM judge for answer quality.

## What we DROP for the pilot (explicitly out of scope per spec §4/§5)
- Next.js frontend, FastAPI server, Postgres, docker-compose, Azure Bicep — none of it. The pilot
  is a CLI pipeline + eval harness.
- Qdrant **server** dependency: replaced with an in-process numpy cosine index (small corpus,
  reproducible, no daemon). Same search semantics, just no network service. Keeps "design decisions
  visible" per spec §4.
- The heuristic `keyword_recall` / `hallucination_heuristic` as *primary* metrics — they're weak
  (string matching). We keep keyword-overlap only as a cheap secondary signal, not the headline.

## Provider change (decision recorded in WORKLOG)
Original is OpenAI (embeddings + gpt-4o-mini). No OpenAI key is available in this environment.
For the pilot we use:
- **Local embeddings** (sentence-transformers, CPU) for retrieval — free, offline, fully
  reproducible recall@k at zero marginal cost. This is the bulk of the eval.
- **Anthropic Claude** for generation (base-model & RAG answers) and the LLM judge.
This keeps the *pipeline shape* identical to enterprise-copilot while making the eval cheap and
reproducible and the generation use the latest capable model.
