"""answer-coverage@k (Phase 0 metric).

For each question, an Opus 4.8 judge sees the question + accepted reference answer + the top-k
retrieved chunks and returns yes/no: "do these chunks contain the information needed to produce
the reference answer?" This does NOT depend on the exact labeled gold chunk, so it is robust to
gold-mislabel (failure category (b)). Report alongside exact-chunk recall@k.

Concurrent + checkpointed (a session/rate limit can kill a batch; resumes from the checkpoint).

Usage: python src/coverage.py <owner__name> [--method vector|bm25|hybrid] [--k 5] [--tag baseline]
"""
import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import config as C
from evalkit import load_index, load_eval
from rag import build_context
from llm import call_claude, UsageTracker

WORKERS = 5

COVERAGE_PROMPT = """You are auditing a retrieval system for the developer tool "{tool}".

QUESTION:
{question}

ACCEPTED REFERENCE ANSWER (what a correct answer should contain):
{reference}

RETRIEVED DOCUMENTATION CHUNKS (top-{k}):
{chunks}

Question: do the retrieved chunks TOGETHER contain the information needed to produce the reference
answer? Be strict — partial/tangential overlap that wouldn't let someone actually answer is "no".
Reply with ONLY a JSON object (no markdown fence):
{{"covered": <true|false>, "reason": "<one short sentence>"}}
"""


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def run(tool: str, method: str = "vector", k: int = 5, tag: str = "", existing: dict | None = None):
    idx = load_index(tool)
    ev = load_eval(tool)
    qs = ev["questions"]
    tool_name = tool.split("__")[1]
    usage = UsageTracker()
    lock = threading.Lock()
    done = [0]
    prior = existing or {}

    def work(q):
        if q["id"] in prior and isinstance(prior[q["id"]].get("covered"), bool):
            return prior[q["id"]]
        retrieved = idx.retrieve(q["question"], k, method=method)
        chunks = build_context(idx, retrieved)
        r = call_claude(COVERAGE_PROMPT.format(
            tool=tool_name, question=q["question"][:3000], reference=q["reference_answer"][:3000],
            k=k, chunks=chunks[:9000]), model=C.JUDGE_MODEL)
        v = parse_json(r.text)
        row = {"id": q["id"], "covered": v.get("covered"), "reason": v.get("reason", ""),
               "gold_hit": bool(set(q["gold_chunk_ids"]) & set(c for c, _ in retrieved))}
        with lock:
            usage.record(r, is_judge=True)
            done[0] += 1
            print(f"  [{done[0]}/{len(qs)}] {q['id']} covered={row['covered']} gold_hit={row['gold_hit']}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(work, qs))

    valid = [r for r in rows if isinstance(r.get("covered"), bool)]
    n = len(valid)
    cov = sum(1 for r in valid if r["covered"]) / n if n else None
    gold_recall = sum(1 for r in valid if r["gold_hit"]) / n if n else None
    out = {"tool": tool, "method": method, "k": k, "tag": tag, "n": n, "n_invalid": len(rows) - n,
           "answer_coverage_at_k": round(cov, 3) if cov is not None else None,
           "exact_recall_at_k": round(gold_recall, 3) if gold_recall is not None else None,
           "usage": usage.summary(), "rows": rows}
    suffix = f"_{tag}" if tag else ""
    (C.RESULTS / f"coverage_{tool}_{method}_k{k}{suffix}.json").write_text(json.dumps(out, indent=2))
    print(f"\nanswer-coverage@{k} ({method}, n={n}): {out['answer_coverage_at_k']} "
          f"| exact-recall@{k}: {out['exact_recall_at_k']} | invalid={out['n_invalid']}")
    print(f"usage: {usage.summary()}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tool")
    ap.add_argument("--method", default="vector")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume-from", default="", help="existing coverage json to resume invalid rows from")
    a = ap.parse_args()
    existing = None
    if a.resume_from:
        existing = {r["id"]: r for r in json.load(open(a.resume_from)).get("rows", [])}
    run(a.tool, a.method, a.k, a.tag, existing)
