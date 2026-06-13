"""Crown jewel: RAG-vs-base ablation (spec §1).

Same questions answered (a) by the base model alone and (b) by the RAG system. A stronger,
different judge model scores each answer 0-100 for correctness vs the accepted reference
answer. Reports base vs RAG mean, win/tie/loss, a recall-conditioned breakdown, and a
bootstrap CI on the lift. Persists raw per-question results.

Judge calls = 2 * N (base + RAG). Stays within the pilot's 500-judge budget.

Usage: python src/ablation.py <owner__name> [limit]
"""
import json
import random
import re
import sys

import config as C
from evalkit import load_index, load_eval
from llm import call_claude, UsageTracker
from rag import base_answer, rag_answer

JUDGE_PROMPT = """You are grading an answer to a developer's question about the tool "{tool}".

QUESTION:
{question}

REFERENCE ANSWER (accepted answer from the project's maintainers/community — treat as ground truth):
{reference}

ANSWER TO GRADE:
{candidate}

Score how correct and complete the ANSWER is relative to the REFERENCE. Reply with ONLY a JSON
object (no markdown fence):
{{"score": <integer 0-100>, "correct": <true if it would actually solve the user's problem>, "reason": "<one short sentence>"}}
"""


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def judge(tool, question, reference, candidate, usage) -> dict:
    r = call_claude(JUDGE_PROMPT.format(
        tool=tool, question=question[:3000], reference=reference[:3000], candidate=candidate[:3000]),
        model=C.JUDGE_MODEL)
    usage.record(r, is_judge=True)
    return parse_json(r.text)


def bootstrap_ci(diffs, iters=2000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iters):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def run(tool: str, limit: int | None = None):
    idx = load_index(tool)
    ev = load_eval(tool)
    qs = ev["questions"]
    if limit:
        qs = qs[:limit]
    usage = UsageTracker()
    tool_name = tool.split("__")[1]

    rows = []
    for i, q in enumerate(qs):
        gold = set(q["gold_chunk_ids"])
        # base (no retrieval)
        b = base_answer(q["question"])
        usage.record(b, is_judge=False)
        # rag (vector top-k)
        rg = rag_answer(q["question"], idx, k=C.TOP_K, method="vector")
        usage.record(rg.llm, is_judge=False)
        gold_hit = bool(gold & set(rg.retrieved_ids))

        bj = judge(tool_name, q["question"], q["reference_answer"], b.text, usage)
        rj = judge(tool_name, q["question"], q["reference_answer"], rg.answer, usage)
        rows.append({
            "id": q["id"], "url": q["url"], "title": q["title"],
            "gold_hit": gold_hit,
            "base_score": bj.get("score"), "base_correct": bj.get("correct"),
            "rag_score": rj.get("score"), "rag_correct": rj.get("correct"),
        })
        print(f"  [{i+1}/{len(qs)}] base={rows[-1]['base_score']} rag={rows[-1]['rag_score']} "
              f"gold_hit={gold_hit}  judge_left={usage.judge_remaining()}", flush=True)

    valid = [r for r in rows if isinstance(r["base_score"], (int, float))
             and isinstance(r["rag_score"], (int, float))]
    n = len(valid)
    base_scores = [r["base_score"] for r in valid]
    rag_scores = [r["rag_score"] for r in valid]
    diffs = [r["rag_score"] - r["base_score"] for r in valid]
    wins = sum(1 for d in diffs if d > 5)
    losses = sum(1 for d in diffs if d < -5)
    ties = n - wins - losses
    lo, hi = bootstrap_ci(diffs) if n else (None, None)

    cond = {}
    for label, subset in [("gold_retrieved", [r for r in valid if r["gold_hit"]]),
                          ("gold_missed", [r for r in valid if not r["gold_hit"]])]:
        if subset:
            cond[label] = {
                "n": len(subset),
                "base_mean": round(sum(r["base_score"] for r in subset) / len(subset), 1),
                "rag_mean": round(sum(r["rag_score"] for r in subset) / len(subset), 1),
            }

    summary = {
        "tool": tool, "answer_model": C.ANSWER_MODEL, "judge_model": C.JUDGE_MODEL,
        "n": n, "top_k": C.TOP_K,
        "base_mean_score": round(sum(base_scores) / n, 1) if n else None,
        "rag_mean_score": round(sum(rag_scores) / n, 1) if n else None,
        "mean_lift": round(sum(diffs) / n, 1) if n else None,
        "lift_ci95": [round(lo, 1), round(hi, 1)] if n else None,
        "rag_wins": wins, "ties": ties, "rag_losses": losses,
        "base_correct_rate": round(sum(1 for r in valid if r["base_correct"]) / n, 3) if n else None,
        "rag_correct_rate": round(sum(1 for r in valid if r["rag_correct"]) / n, 3) if n else None,
        "conditioned_on_retrieval": cond,
        "usage": usage.summary(),
    }
    (C.RESULTS / f"ablation_{tool}.json").write_text(json.dumps(
        {"summary": summary, "rows": rows}, indent=2))

    print(f"\n==== CROWN JEWEL: RAG vs base  ({tool}, n={n}) ====")
    print(f"  base mean score : {summary['base_mean_score']}")
    print(f"  RAG  mean score : {summary['rag_mean_score']}")
    print(f"  mean lift       : {summary['mean_lift']}  (95% CI {summary['lift_ci95']})")
    print(f"  RAG win/tie/loss: {wins}/{ties}/{losses}")
    print(f"  correct rate    : base {summary['base_correct_rate']} -> RAG {summary['rag_correct_rate']}")
    print(f"  conditioned     : {cond}")
    print(f"  usage           : {usage.summary()}")
    return summary


if __name__ == "__main__":
    tool = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(tool, limit)
