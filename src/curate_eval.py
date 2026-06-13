"""Curated eval-set builder (replaces the free answer->embedding gold with LLM-verified gold
and a validity filter). See WORKLOG 2026-06-13 Phase 3.

For each candidate discussion:
  - build a FAIR candidate pool of doc chunks = (answer-embedding top-A) ∪ (question-vector top-Q)
    ∪ (question-bm25 top-B), deduped. No retriever is privileged in the pool.
  - one judge call returns {keep, gold_indices, reason}:
      keep=true iff self-contained docs-answerable usage/config/API/error question whose
      accepted answer is substantive (not conversational / roadmap / "fixed next release" /
      version-obsolete) AND supported by >=1 pool chunk.
  - reference answer stays = the real accepted answer (authentic; no docs-circular bias).

Output: data/eval/<tool>.json  (questions with verified gold_chunk_ids).
Concurrent (threaded) over questions; checkpointed.

Usage: python src/curate_eval.py <owner__name> [limit]
"""
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config as C
from evalkit import load_index
from index import embed_texts
from llm import call_claude, UsageTracker

WORKERS = 8
POOL_ANS, POOL_VEC, POOL_BM = 5, 5, 5   # candidates from each source
MIN_Q_LEN = 40
MIN_ANS_LEN = 100

CURATE_PROMPT = """You are curating an evaluation set for a documentation-QA system for the
developer tool "{tool}". Decide whether to KEEP this question, and if so, identify which of the
provided documentation chunks actually contain the answer.

QUESTION:
{question}

ACCEPTED ANSWER (from the project's community/maintainers):
{answer}

CANDIDATE DOCUMENTATION CHUNKS (numbered):
{pool}

Reply with ONLY a JSON object (no markdown fence):
{{
  "keep": <true|false>,
  "gold_indices": [<indices of chunks that actually contain/support the answer, [] if none>],
  "reason": "<one short sentence>"
}}

KEEP=true ONLY IF ALL hold:
- It is a concrete, self-contained usage / config / API / error question about {tool}
  (not opinion, roadmap, "is this a bug", a feature request, or a vague discussion).
- The accepted answer is a substantive, still-valid answer (NOT just a clarifying question,
  a bare link, "fixed in a future release", or clearly about an obsolete version).
- At least one candidate chunk genuinely contains or directly supports the answer
  (those are the gold_indices). If no chunk does, set keep=false and gold_indices=[].
"""


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def shorten(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + " …"


def run(tool: str, limit: int | None = None):
    idx = load_index(tool)
    discussions = json.load(open(C.RAW / "discussions" / f"{tool}.json"))
    # basic pre-filter
    cands = [d for d in discussions
             if len(f"{d['title']} {d['question']}") >= MIN_Q_LEN and len(d["answer"]) >= MIN_ANS_LEN]
    if limit:
        cands = cands[:limit]
    # precompute answer embeddings -> answer-emb pool
    ans_emb = embed_texts([d["answer"] for d in cands])
    sims = ans_emb @ idx.emb.T

    usage = UsageTracker()
    lock = threading.Lock()
    tool_name = tool.split("__")[1]
    kept = []
    done = [0]

    def pool_for(i, d):
        a_top = list(np.argsort(-sims[i])[:POOL_ANS])
        v_top = [idx.ids.index(c) for c, _ in idx.vector(f"{d['title']}\n{d['question']}", POOL_VEC)]
        b_top = [idx.ids.index(c) for c, _ in idx.bm25(f"{d['title']}\n{d['question']}", POOL_BM)]
        seen, order = set(), []
        for j in a_top + v_top + b_top:
            if j not in seen:
                seen.add(j); order.append(j)
        return order

    def work(args):
        i, d = args
        pool_idx = pool_for(i, d)
        pool_chunks = [idx.chunks[j] for j in pool_idx]
        pool_str = "\n\n".join(
            f"[{k}] ({c.doc_path}{' § '+c.heading if c.heading else ''})\n{shorten(c.content,500)}"
            for k, c in enumerate(pool_chunks))
        prompt = CURATE_PROMPT.format(
            tool=tool_name,
            question=shorten(f"{d['title']}\n\n{d['question']}", 2500),
            answer=shorten(d["answer"], 2000),
            pool=pool_str)
        r = call_claude(prompt, model=C.CURATE_MODEL)
        with lock:
            usage.record(r, is_judge=True)
            done[0] += 1
        v = parse_json(r.text)
        keep = bool(v.get("keep"))
        gold_local = [g for g in (v.get("gold_indices") or []) if isinstance(g, int) and 0 <= g < len(pool_chunks)]
        row = None
        if keep and gold_local:
            row = {
                "id": f"{tool}#{d['number']}", "number": d["number"], "title": d["title"],
                "question": f"{d['title']}\n\n{d['question']}".strip(),
                "reference_answer": d["answer"], "url": d["url"],
                "gold_chunk_ids": [pool_chunks[g].chunk_id for g in gold_local],
                "curate_reason": v.get("reason", ""),
            }
        with lock:
            print(f"  [{done[0]}/{len(cands)}] #{d['number']} keep={keep} "
                  f"gold={len(gold_local)} judge_used={usage.judge_calls}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for row in ex.map(work, list(enumerate(cands))):
            if row:
                kept.append(row)

    out = {"tool": tool, "method": "llm-curated gold + validity filter",
           "n_candidates": len(cands), "n_eval": len(kept),
           "n_corpus_chunks": len(idx.chunks), "questions": kept,
           "curation_usage": usage.summary()}
    (C.EVAL / f"{tool}.json").write_text(json.dumps(out, indent=2))
    print(f"\ncurated eval set: kept {len(kept)}/{len(cands)} -> data/eval/{tool}.json")
    print(f"curation usage: {usage.summary()}")


if __name__ == "__main__":
    tool = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(tool, limit)
