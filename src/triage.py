"""Phase-2 forensic triage on RESIDUAL retrieval failures.

Residual = questions whose top-5 vector retrieval failed to surface the answer content
(answer-coverage@5 == False on the post-mechanical corpus — the honest "retrieval missed",
robust to gold mislabel).

For each residual question, retrieve a broad pool (union of vector + BM25 top-20 over the whole
corpus) and ask one Opus judge call to locate the answer within the pool, then classify into:
  (a) corpus gap      — answer content is in NONE of the pooled candidates (not in corpus)
  (b) gold mislabel   — a top-5 candidate DOES contain it (coverage@5 was a measurement artifact)
  (c) chunking        — content present but no single chunk suffices (fragmented/split/stub)
  (d) embedding       — a single good chunk exists but ranked below k=5

Outputs results/residual_triage.json + data/diag/residual_failures.json.

Usage: python src/triage.py <tool> --coverage <coverage_json> [--k 5] [--pool 20]
"""
import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import config as C
from evalkit import load_index, load_eval
from llm import call_claude, UsageTracker

WORKERS = 5

PROMPT = """You are doing failure analysis on a retrieval system for the tool "{tool}".

QUESTION:
{question}

REFERENCE ANSWER (what a correct answer must contain):
{reference}

CANDIDATE CHUNKS retrieved from the whole corpus (numbered; rank 0 = top vector hit):
{pool}

Find where (if anywhere) the information needed for the reference answer lives in these candidates.
Reply with ONLY a JSON object (no markdown fence):
{{
  "supporting_ranks": [<indices of candidates that contain answer info, [] if none>],
  "single_chunk_sufficient": <true if ONE candidate alone is enough to answer>,
  "reason": "<one short sentence>"
}}
"""


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def classify(supporting, single_ok, k):
    if not supporting:
        return "a_corpus_gap"
    if min(supporting) < k:
        return "b_gold_mislabel"      # answer was within top-k; coverage/gold said miss
    if single_ok:
        return "d_embedding"          # one good chunk exists but ranked >= k
    return "c_chunking"               # present but spread across chunks / no single sufficient one


def run(tool, coverage_json, k=5, pool=20):
    cov = json.load(open(coverage_json))
    residual_ids = [r["id"] for r in cov["rows"] if r.get("covered") is False]
    idx = load_index(tool)
    ev = {q["id"]: q for q in load_eval(tool)["questions"]}
    tool_name = tool.split("__")[1].replace("_mech", "")
    usage = UsageTracker(); lock = threading.Lock(); done = [0]

    # persist the raw residual set
    (C.DATA / "diag").mkdir(exist_ok=True)
    raw = []
    for qid in residual_ids:
        q = ev[qid]
        retr = idx.retrieve(q["question"], 10, method="vector")
        raw.append({"id": qid, "title": q["title"], "url": q["url"],
                    "reference_answer": q["reference_answer"],
                    "gold_chunk_ids": q["gold_chunk_ids"],
                    "top10": [{"chunk_id": c, "score": round(s, 3)} for c, s in retr]})
    (C.DATA / "diag" / "residual_failures.json").write_text(json.dumps(raw, indent=2))

    def pooled(q):
        v = idx.vector(q["question"], pool)
        b = idx.bm25(q["question"], pool)
        seen, order = set(), []
        for cid, _ in v + b:           # vector first → low ranks reflect vector order
            if cid not in seen:
                seen.add(cid); order.append(cid)
        return order[:pool]

    def work(qid):
        q = ev[qid]
        cids = pooled(q)
        pool_str = "\n\n".join(
            f"[{i}] ({idx.chunk_by_id(c).doc_path})\n{idx.chunk_by_id(c).content[:600]}"
            for i, c in enumerate(cids))
        r = call_claude(PROMPT.format(tool=tool_name, question=q["question"][:2500],
                        reference=q["reference_answer"][:2500], pool=pool_str[:14000]),
                        model=C.JUDGE_MODEL)
        v = parse_json(r.text)
        supp = [s for s in (v.get("supporting_ranks") or []) if isinstance(s, int) and 0 <= s < len(cids)]
        cat = classify(supp, bool(v.get("single_chunk_sufficient")), k)
        row = {"id": qid, "title": q["title"], "category": cat,
               "supporting_ranks": supp, "single_chunk_sufficient": bool(v.get("single_chunk_sufficient")),
               "reason": v.get("reason", "")}
        with lock:
            usage.record(r, is_judge=True); done[0] += 1
            print(f"  [{done[0]}/{len(residual_ids)}] {qid} -> {cat} (supp={supp[:3]})", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(work, residual_ids))

    from collections import Counter
    dist = Counter(r["category"] for r in rows)
    out = {"tool": tool, "residual_n": len(residual_ids), "k": k, "pool": pool,
           "distribution": dict(dist),
           "distribution_pct": {kk: round(100 * vv / len(rows), 1) for kk, vv in dist.items()} if rows else {},
           "usage": usage.summary(), "rows": rows}
    (C.RESULTS / "residual_triage.json").write_text(json.dumps(out, indent=2))
    print(f"\n=== RESIDUAL TRIAGE (n={len(residual_ids)}) ===")
    for cat, c in dist.most_common():
        print(f"  {cat:20} {c:4}  ({round(100*c/len(rows),1)}%)")
    print(f"usage: {usage.summary()}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tool")
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pool", type=int, default=20)
    a = ap.parse_args()
    run(a.tool, a.coverage, a.k, a.pool)
