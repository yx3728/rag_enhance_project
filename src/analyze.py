"""Unified claim-coverage analyzer (supersedes the strict coverage judge per Phase-3 research:
LLM judges over verbose human references systematically under-score).

For each question: pool = vector top-20 ∪ BM25 top-20 (each candidate tagged with its vector
rank). One Opus call returns which candidates contain information that supports the reference
answer (lenient: partial/differently-phrased support counts). Derive:
  - coverage@k (vector): a supporting candidate has vector_rank < k    [k = 1,3,5,10]
  - in_corpus: ANY candidate supports (vector- or BM25-reachable in top-20) -> complement ≈ corpus gap
  - category per question: a_corpus_gap / b_in_top5 / c_or_d_rank (present but ranked >= 5)

Run on multiple corpora to compare. Output results/analyze_<tool>.json.

Usage: python src/analyze.py <tool> [--pool 20]
"""
import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import config as C
from evalkit import load_index, load_eval
from llm import call_claude, UsageTracker, enable_trace, disable_trace

WORKERS = 5
INF = 10**9
_reranker = None


def get_reranker(name="BAAI/bge-reranker-base"):
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(name)
    return _reranker

PROMPT = """You are measuring retrieval quality for the developer tool "{tool}".

QUESTION:
{question}

REFERENCE ANSWER (the facts a correct answer must convey):
{reference}

CANDIDATE CHUNKS retrieved from the docs (numbered):
{pool}

Identify which candidates contain information that SUPPORTS the reference answer. Count a
candidate as supporting if it contains facts/steps/API details needed for the answer, EVEN IF
phrased differently, more concisely, or only partially — do not require verbatim or complete
overlap. Then judge whether the supporting candidates TOGETHER would let someone produce the
reference answer.
Reply with ONLY a JSON object (no markdown fence):
{{"supporting": [<candidate indices, [] if none>], "sufficient": <true|false>, "reason": "<short>"}}
"""


def parse_json(t):
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def run(tool, pool=20, rerank=False):
    idx = load_index(tool)
    ev = load_eval(tool)
    qs = ev["questions"]
    tool_name = tool.split("__")[1].replace("_mech", "").replace("_wide", "")
    usage = UsageTracker(); lock = threading.Lock(); done = [0]
    rr = get_reranker() if rerank else None
    tagp = "_rerank" if rerank else ""
    enable_trace(C.ROOT / "traces" / tool / f"analyze{tagp}.jsonl")

    def work(q):
        vec = idx.vector(q["question"], max(pool, 30) if rerank else pool)   # ordered by dense score
        if rr is not None:
            cids = [c for c, _ in vec]
            scores = rr.predict([[q["question"], idx.chunk_by_id(c).content[:1200]] for c in cids])
            order_r = sorted(range(len(cids)), key=lambda i: -scores[i])
            vec = [(cids[i], float(scores[i])) for i in order_r][:pool]       # reranked order
        bm = idx.bm25(q["question"], pool)
        vrank = {cid: r for r, (cid, _) in enumerate(vec)}
        order = []
        seen = set()
        for cid, _ in vec + bm:
            if cid not in seen:
                seen.add(cid); order.append(cid)
        cand = order[:2 * pool]
        pool_str = "\n\n".join(
            f"[{i}] ({idx.chunk_by_id(c).doc_path})\n{idx.chunk_by_id(c).content[:550]}"
            for i, c in enumerate(cand))
        r = call_claude(PROMPT.format(tool=tool_name, question=q["question"][:2500],
                        reference=q["reference_answer"][:2500], pool=pool_str[:15000]),
                        model=C.JUDGE_MODEL,
                        trace_meta={"repo": tool, "phase": "analyze"+tagp, "kind": "claim_coverage", "q": q["id"]})
        v = parse_json(r.text)
        supp = [s for s in (v.get("supporting") or []) if isinstance(s, int) and 0 <= s < len(cand)]
        supp_vranks = [vrank.get(cand[s], INF) for s in supp]
        best = min(supp_vranks) if supp_vranks else INF
        row = {"id": q["id"], "in_corpus": bool(supp), "best_vrank": (None if best == INF else best),
               "sufficient": bool(v.get("sufficient")), "supporting_n": len(supp)}
        with lock:
            usage.record(r, is_judge=True); done[0] += 1
            print(f"  [{done[0]}/{len(qs)}] {q['id']} in_corpus={row['in_corpus']} best_vrank={row['best_vrank']}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(work, qs))

    n = len(rows)
    def cov(k):
        return round(sum(1 for r in rows if r["best_vrank"] is not None and r["best_vrank"] < k) / n, 3)
    in_corpus = sum(1 for r in rows if r["in_corpus"])
    out = {"tool": tool, "n": n, "rerank": rerank,
           "coverage_at": {str(k): cov(k) for k in (1, 3, 5, 10)},
           "in_corpus_rate": round(in_corpus / n, 3),
           "corpus_gap_rate": round(1 - in_corpus / n, 3),
           "present_but_ranked_below5": round(sum(1 for r in rows if r["in_corpus"] and (r["best_vrank"] is None or r["best_vrank"] >= 5)) / n, 3),
           "usage": usage.summary(), "rows": rows}
    tag = "_rerank" if rerank else ""
    (C.RESULTS / f"analyze_{tool}{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\n=== ANALYZE {tool} (n={n}) ===")
    print(f"  coverage@: {out['coverage_at']}")
    print(f"  in_corpus(reachable@20): {out['in_corpus_rate']}  | corpus_gap: {out['corpus_gap_rate']}")
    print(f"  present-but-ranked>=5 (reranker-addressable): {out['present_but_ranked_below5']}")
    print(f"  usage: {usage.summary()}")
    disable_trace()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tool")
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--rerank", action="store_true")
    a = ap.parse_args()
    run(a.tool, a.pool, a.rerank)
