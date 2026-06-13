"""RAG pipeline — generation. Reuses enterprise-copilot's prompt philosophy
(answer using ONLY the provided context, cite [n]) and pipeline shape
(retrieve -> numbered context -> generate), swapped onto the local index + Claude CLI.

Two generation modes for the crown-jewel ablation:
  - base_answer():  no retrieval. The model answers from its own knowledge.
  - rag_answer():   retrieve top-k, build numbered context, answer from context only.
"""
from __future__ import annotations

from dataclasses import dataclass

import config as C
from index import Index
from llm import call_claude, LLMResult

# Reused from enterprise-copilot/backend/app/services/rag.py PROMPTS["default"], adapted to
# a generic developer-tool documentation assistant.
RAG_SYSTEM = (
    "You are a documentation assistant for a software developer tool. Answer the user's "
    "question using ONLY the provided documentation context. If the context is insufficient, "
    "say you do not have enough information. Cite sources inline as [1], [2] matching the "
    "context blocks. Be concise and concrete (commands, config, code)."
)

BASE_SYSTEM = (
    "You are a helpful assistant answering a developer's question about a software tool. "
    "Answer concretely and concisely (commands, config, code where relevant)."
)

# Fix motivated by the crown-jewel diagnosis: the strict "context only" prompt craters when
# retrieval misses (recall@5≈0.67). This variant prefers the retrieved context but falls back to
# the model's own knowledge instead of refusing, and flags when it does so.
RAG_FALLBACK_SYSTEM = (
    "You are a documentation assistant for a software developer tool. Prefer the provided "
    "documentation context and cite it inline as [1], [2]. If the context does not fully cover "
    "the question, use your own knowledge of the tool to give the best concrete answer you can, "
    "and note which parts are not grounded in the provided docs. Be concise and concrete "
    "(commands, config, code)."
)

SYSTEMS = {"strict": RAG_SYSTEM, "fallback": RAG_FALLBACK_SYSTEM}


@dataclass
class RagOutput:
    answer: str
    retrieved_ids: list[str]
    context: str
    llm: LLMResult


def build_context(idx: Index, retrieved: list[tuple[str, float]]) -> str:
    blocks = []
    for i, (cid, _score) in enumerate(retrieved, start=1):
        ch = idx.chunk_by_id(cid)
        loc = f"{ch.doc_path}" + (f" § {ch.heading}" if ch.heading else "")
        blocks.append(f"[{i}] ({loc})\n{ch.content}")
    return "\n\n".join(blocks)


def base_answer(question: str, *, model: str = C.ANSWER_MODEL) -> LLMResult:
    return call_claude(question, model=model, system=BASE_SYSTEM)


_reranker = None


def _rerank(question: str, idx: Index, k: int, pool: int = 30):
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    cand = idx.vector(question, pool)
    cids = [c for c, _ in cand]
    scores = _reranker.predict([[question, idx.chunk_by_id(c).content[:1200]] for c in cids])
    order = sorted(range(len(cids)), key=lambda i: -scores[i])[:k]
    return [(cids[i], float(scores[i])) for i in order]


def rag_answer(question: str, idx: Index, *, k: int = C.TOP_K, method: str = "vector",
               model: str = C.ANSWER_MODEL, variant: str = "strict", rerank: bool = False) -> RagOutput:
    retrieved = _rerank(question, idx, k) if rerank else idx.retrieve(question, k, method=method)
    context = build_context(idx, retrieved)
    prompt = f"Documentation context:\n{context}\n\nQuestion: {question}"
    llm = call_claude(prompt, model=model, system=SYSTEMS[variant])
    return RagOutput(answer=llm.text, retrieved_ids=[c for c, _ in retrieved],
                     context=context, llm=llm)
