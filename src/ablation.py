"""Crown jewel: RAG-vs-base ablation (spec §1).

Same questions answered (a) by the base model alone and (b) by the RAG system. A stronger,
different judge model scores BOTH answers in ONE joint call (positions randomized to control
order bias) for correctness vs the accepted reference answer — 1 judge call per question.

Reports base vs RAG mean, win/tie/loss, judge-preferred rate, a recall-conditioned breakdown,
and a bootstrap CI on the lift. Persists raw per-question results + a checkpoint JSONL.

Usage: python src/ablation.py <owner__name> [--limit N]
"""
import argparse
import hashlib
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import config as C
from evalkit import load_index, load_eval
from llm import call_claude, UsageTracker
from rag import base_answer, rag_answer

WORKERS = 5

JUDGE_PROMPT = """You are grading two answers to a developer's question about the tool "{tool}".

QUESTION:
{question}

REFERENCE ANSWER (accepted answer from the project's maintainers/community — treat as ground truth):
{reference}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Independently score how correct and complete EACH answer is relative to the REFERENCE
(0 = wrong/non-answer, 100 = fully correct and complete). Judge on substance, not style or length.
Reply with ONLY a JSON object (no markdown fence):
{{"a_score": <0-100>, "b_score": <0-100>, "a_correct": <bool>, "b_correct": <bool>,
  "preferred": "A"|"B"|"tie", "reason": "<one short sentence>"}}
"""


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _seeded_swap(qid: str) -> bool:
    """Deterministic per-question position assignment (base=A or base=B)."""
    return int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 2 == 0


def score_question(tool_name, q, idx, variant, rerank=False):
    """Run base + RAG + joint judge for one question. Returns (row, [(LLMResult, is_judge), ...])."""
    gold = set(q["gold_chunk_ids"])
    b = base_answer(q["question"])                                   # base: no retrieval
    rg = rag_answer(q["question"], idx, k=C.TOP_K, method="vector", variant=variant, rerank=rerank)  # RAG: vector top-k
    gold_hit = bool(gold & set(rg.retrieved_ids))
    base_is_a = _seeded_swap(q["id"])
    a_text, b_text = (b.text, rg.answer) if base_is_a else (rg.answer, b.text)
    jr = call_claude(JUDGE_PROMPT.format(
        tool=tool_name, question=q["question"][:3000], reference=q["reference_answer"][:3000],
        answer_a=a_text[:3000], answer_b=b_text[:3000]), model=C.JUDGE_MODEL)
    v = parse_json(jr.text)
    if base_is_a:
        base_score, rag_score = v.get("a_score"), v.get("b_score")
        base_correct, rag_correct = v.get("a_correct"), v.get("b_correct")
        pref = {"A": "base", "B": "rag"}.get(v.get("preferred"), v.get("preferred"))
    else:
        base_score, rag_score = v.get("b_score"), v.get("a_score")
        base_correct, rag_correct = v.get("b_correct"), v.get("a_correct")
        pref = {"B": "base", "A": "rag"}.get(v.get("preferred"), v.get("preferred"))
    row = {
        "id": q["id"], "url": q["url"], "title": q["title"], "gold_hit": gold_hit,
        "base_score": base_score, "rag_score": rag_score,
        "base_correct": base_correct, "rag_correct": rag_correct,
        "preferred": pref, "reason": v.get("reason", ""),
    }
    return row, [(b, False), (rg.llm, False), (jr, True)]


def bootstrap_ci(diffs, iters=2000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def run(tool: str, limit: int | None = None, judge_cap: int | None = None, variant: str = "strict", rerank: bool = False):
    idx = load_index(tool)
    ev = load_eval(tool)
    qs = ev["questions"]
    suffix = "" if variant == "strict" else f"_{variant}"
    if judge_cap is not None and len(qs) > judge_cap:
        print(f"NOTE: capping eval from {len(qs)} to {judge_cap} questions to stay within judge budget.")
        qs = qs[:judge_cap]
    if limit:
        qs = qs[:limit]
    usage = UsageTracker()
    tool_name = tool.split("__")[1]
    lock = threading.Lock()
    ckpt = (C.RESULTS / f"ablation_{tool}{suffix}.checkpoint.jsonl").open("w")
    done = [0]

    def work(q):
        row, calls = score_question(tool_name, q, idx, variant, rerank)
        with lock:
            for c, is_j in calls:
                usage.record(c, is_judge=is_j)
            done[0] += 1
            ckpt.write(json.dumps(row) + "\n"); ckpt.flush()
            print(f"  [{done[0]}/{len(qs)}] base={row['base_score']} rag={row['rag_score']} "
                  f"pref={row['preferred']} gold_hit={row['gold_hit']} judge_used={usage.judge_calls}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(work, qs))
    ckpt.close()

    summary = compute_summary(tool, variant, rows, usage.summary())
    (C.RESULTS / f"ablation_{tool}{suffix}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2))
    _print_summary(tool, summary)
    return summary


def compute_summary(tool: str, variant: str, rows: list[dict], usage_summary: dict) -> dict:
    valid = [r for r in rows if isinstance(r["base_score"], (int, float))
             and isinstance(r["rag_score"], (int, float))]
    n = len(valid)
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

    return {
        "tool": tool, "rag_variant": variant, "answer_model": C.ANSWER_MODEL, "judge_model": C.JUDGE_MODEL,
        "n": n, "n_invalid": len(rows) - n, "top_k": C.TOP_K,
        "scoring": "joint single-call, positions randomized",
        "base_mean_score": round(sum(r["base_score"] for r in valid) / n, 1) if n else None,
        "rag_mean_score": round(sum(r["rag_score"] for r in valid) / n, 1) if n else None,
        "mean_lift": round(sum(diffs) / n, 1) if n else None,
        "lift_ci95": [round(lo, 1), round(hi, 1)] if n else None,
        "rag_wins": wins, "ties": ties, "rag_losses": losses,
        "rag_preferred": sum(1 for r in valid if r["preferred"] == "rag"),
        "base_preferred": sum(1 for r in valid if r["preferred"] == "base"),
        "tie_preferred": sum(1 for r in valid if r["preferred"] == "tie"),
        "base_correct_rate": round(sum(1 for r in valid if r["base_correct"]) / n, 3) if n else None,
        "rag_correct_rate": round(sum(1 for r in valid if r["rag_correct"]) / n, 3) if n else None,
        "conditioned_on_retrieval": cond,
        "usage": usage_summary,
    }


def _print_summary(tool, s):
    print(f"\n==== CROWN JEWEL: RAG vs base  ({tool}, n={s['n']}, variant={s['rag_variant']}) ====")
    print(f"  base mean : {s['base_mean_score']}    RAG mean : {s['rag_mean_score']}")
    print(f"  mean lift : {s['mean_lift']}  (95% CI {s['lift_ci95']})")
    print(f"  RAG win/tie/loss (>5pt): {s['rag_wins']}/{s['ties']}/{s['rag_losses']}")
    print(f"  judge preferred: rag={s['rag_preferred']} base={s['base_preferred']} tie={s['tie_preferred']}")
    print(f"  correct rate: base {s['base_correct_rate']} -> RAG {s['rag_correct_rate']}")
    print(f"  conditioned on retrieval: {s['conditioned_on_retrieval']}")
    print(f"  usage: {s['usage']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tool")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge-cap", type=int, default=None)
    ap.add_argument("--rag-variant", default="strict", choices=["strict","fallback"])
    ap.add_argument("--rerank", action="store_true")
    a = ap.parse_args()
    run(a.tool, a.limit, a.judge_cap, a.rag_variant, a.rerank)
