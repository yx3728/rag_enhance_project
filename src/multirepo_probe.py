"""Base-familiarity probe (Opus-medium judge) for the multi-repo scale-up.

For each forum repo: sample N real answered-Q&A, have the BASE model (Haiku 4.5, no retrieval)
answer, and an Opus-medium judge score correctness vs the accepted answer (docs-answerable subset).
Lower base score = lower familiarity. Full judge traces persisted.

Usage: python src/multirepo_probe.py <owner__name> [owner__name ...] [--n 14]
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import config as C
from llm import call_claude, UsageTracker, enable_trace, disable_trace
from selection import JUDGE_PROMPT, question_text, shorten, parse_json

WORKERS = 6


def probe(repo: str, n: int, usage: UsageTracker):
    items = json.load(open(C.RAW / "discussions" / f"{repo}.json"))
    step = max(1, len(items) // n)
    sample = items[::step][:n]
    tool = repo.split("__")[1]
    lock = threading.Lock()
    rows = []

    def work(it):
        q = question_text(it)
        base = call_claude(shorten(q, 4000), model=C.ANSWER_MODEL,
                           trace_meta={"repo": repo, "phase": "probe", "kind": "base_answer", "q": it["number"]})
        if base.is_error:
            return None
        jr = call_claude(JUDGE_PROMPT.format(tool=tool, question=shorten(q, 2500),
                         reference=shorten(it["answer"], 2500), candidate=shorten(base.text, 2500)),
                         model=C.JUDGE_MODEL,
                         trace_meta={"repo": repo, "phase": "probe", "kind": "judge", "q": it["number"]})
        v = parse_json(jr.text) or {}
        with lock:
            usage.record(base, is_judge=False); usage.record(jr, is_judge=True)
        return {"q": it["number"], "docs_answerable": bool(v.get("docs_answerable")),
                "base_score": v.get("score")}

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = [r for r in ex.map(work, sample) if r]
    ans = [r for r in rows if r["docs_answerable"] and isinstance(r["base_score"], (int, float))]
    scores = [r["base_score"] for r in ans]
    return {"repo": repo, "sampled": len(rows), "docs_answerable": len(ans),
            "base_mean_score": round(sum(scores) / len(scores), 1) if scores else None,
            "base_scores": scores, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--n", type=int, default=14)
    a = ap.parse_args()
    enable_trace(C.ROOT / "traces" / "_probe" / "probe.jsonl")
    usage = UsageTracker()
    out = {"judge_model": C.JUDGE_MODEL, "judge_effort": "medium", "answer_model": C.ANSWER_MODEL,
           "n": a.n, "repos": {}}
    for repo in a.repos:
        r = probe(repo, a.n, usage)
        out["repos"][repo] = r
        print(f"{repo:26} base_mean={r['base_mean_score']} (docs_answerable {r['docs_answerable']}/{r['sampled']})")
    out["usage"] = usage.summary()
    (C.RESULTS / "multirepo" / "probe_forum.json").write_text(json.dumps(out, indent=2))
    disable_trace()
    print("usage:", usage.summary())


if __name__ == "__main__":
    main()
