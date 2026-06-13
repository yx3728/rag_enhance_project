"""Re-run only the questions whose scores came back invalid (e.g. killed by a session/rate
limit) and merge them back into an existing ablation result. Recomputes the summary.

Usage: python src/resume_ablation.py <owner__name> [--rag-variant strict|fallback]
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import config as C
from evalkit import load_index, load_eval
from llm import UsageTracker
from ablation import score_question, compute_summary, _print_summary

WORKERS = 4


def run(tool: str, variant: str = "strict"):
    suffix = "" if variant == "strict" else f"_{variant}"
    path = C.RESULTS / f"ablation_{tool}{suffix}.json"
    data = json.load(open(path))
    rows = data["rows"]
    by_id = {r["id"]: r for r in rows}

    bad_ids = [r["id"] for r in rows
               if not isinstance(r.get("base_score"), (int, float))
               or not isinstance(r.get("rag_score"), (int, float))]
    print(f"{path.name}: {len(bad_ids)} invalid rows to re-run")
    if not bad_ids:
        return

    idx = load_index(tool)
    ev = load_eval(tool)
    qmap = {q["id"]: q for q in ev["questions"]}
    tool_name = tool.split("__")[1]
    usage = UsageTracker()
    lock = threading.Lock()
    done = [0]

    def work(qid):
        row, calls = score_question(tool_name, qmap[qid], idx, variant)
        with lock:
            for c, is_j in calls:
                usage.record(c, is_judge=is_j)
            done[0] += 1
            print(f"  [{done[0]}/{len(bad_ids)}] {qid} base={row['base_score']} "
                  f"rag={row['rag_score']}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for row in ex.map(work, bad_ids):
            by_id[row["id"]] = row

    merged = [by_id[r["id"]] for r in rows]
    # carry forward original usage + add the re-run usage (approximate cumulative spend)
    prev = data["summary"].get("usage", {})
    u = usage.summary()
    u["note"] = f"re-run of {len(bad_ids)} invalid rows; prior run judge_calls={prev.get('judge_calls')}"
    summary = compute_summary(tool, variant, merged, u)
    path.write_text(json.dumps({"summary": summary, "rows": merged}, indent=2))
    _print_summary(tool, summary)
    print(f"  re-run usage: {usage.summary()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tool")
    ap.add_argument("--rag-variant", default="strict", choices=["strict", "fallback"])
    a = ap.parse_args()
    run(a.tool, a.rag_variant)
