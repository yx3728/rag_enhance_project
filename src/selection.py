"""Substrate selection (spec §2): for each candidate dev tool, run the BASE model (no
retrieval) on a sample of real docs-answerable questions and measure how well it does.
Pick the tool where the base model does WORST — that's where RAG has the most to prove.

One judge call per sampled question does double duty: (a) classify docs-answerable, and
(b) score the base answer's correctness vs the accepted reference answer. We average base
correctness over the docs-answerable subset.

Outputs results/selection.json and prints a ranked table.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from llm import call_claude, UsageTracker

CANDIDATES = [
    "dagster-io__dagster", "PrefectHQ__prefect", "litestar-org__litestar",
    "marimo-team__marimo", "duckdb__duckdb",
]
SAMPLE_PER_CANDIDATE = 14   # deterministic slice; some get filtered as not docs-answerable

BASE_SYSTEM = ""  # bare model

JUDGE_PROMPT = """You are evaluating a developer-tools Q&A system.

TOOL: {tool}

USER QUESTION:
{question}

REFERENCE ANSWER (accepted answer from the project's maintainers/community):
{reference}

CANDIDATE ANSWER (produced by a language model with NO access to the docs):
{candidate}

Do two things and reply with ONLY a JSON object (no prose, no markdown fence):
1. "docs_answerable": true if this is a concrete usage/config/API/error question whose
   correct answer lives in the tool's documentation or source code. Set false if the
   reference answer is essentially "it's a bug / fixed in a future release", a roadmap or
   opinion, a request for more info, or otherwise not derivable from docs/source.
2. "score": integer 0-100 for how correct and complete the CANDIDATE answer is relative to
   the REFERENCE answer (100 = fully correct and complete; 0 = wrong or non-answer).
3. "reason": one short sentence.

JSON:"""


def shorten(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + " …[truncated]"


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def question_text(item: dict) -> str:
    return f"{item['title']}\n\n{item['question']}".strip()


def run():
    usage = UsageTracker()
    per_candidate = {}
    details = {}

    for cand in CANDIDATES:
        items = json.load(open(C.RAW / "discussions" / f"{cand}.json"))
        # deterministic sample: evenly spaced across the fetched (recency-ordered) set
        step = max(1, len(items) // SAMPLE_PER_CANDIDATE)
        sample = items[::step][:SAMPLE_PER_CANDIDATE]
        rows = []
        for it in sample:
            q = question_text(it)
            base = call_claude(shorten(q, 4000), model=C.ANSWER_MODEL, system=BASE_SYSTEM)
            usage.record(base, is_judge=False)
            if base.is_error:
                continue
            jprompt = JUDGE_PROMPT.format(
                tool=cand.split("__")[1],
                question=shorten(q, 2500),
                reference=shorten(it["answer"], 2500),
                candidate=shorten(base.text, 2500),
            )
            jr = call_claude(jprompt, model=C.JUDGE_MODEL)
            usage.record(jr, is_judge=True)
            verdict = parse_json(jr.text) or {}
            rows.append({
                "number": it["number"], "title": it["title"], "url": it["url"],
                "docs_answerable": bool(verdict.get("docs_answerable", False)),
                "base_score": verdict.get("score"),
                "reason": verdict.get("reason", ""),
            })
            print(f"  [{cand}] #{it['number']} docs_answerable={rows[-1]['docs_answerable']} "
                  f"base_score={rows[-1]['base_score']}")

        answerable = [r for r in rows if r["docs_answerable"] and isinstance(r["base_score"], (int, float))]
        scores = [r["base_score"] for r in answerable]
        per_candidate[cand] = {
            "sampled": len(rows),
            "docs_answerable": len(answerable),
            "base_mean_score": round(sum(scores) / len(scores), 1) if scores else None,
            "base_scores": scores,
        }
        details[cand] = rows
        print(f"=== {cand}: docs_answerable={len(answerable)}/{len(rows)} "
              f"base_mean_score={per_candidate[cand]['base_mean_score']}")

    out = {
        "answer_model": C.ANSWER_MODEL,
        "judge_model": C.JUDGE_MODEL,
        "sample_per_candidate": SAMPLE_PER_CANDIDATE,
        "per_candidate": per_candidate,
        "usage": usage.summary(),
        "details": details,
    }
    (C.RESULTS / "selection.json").write_text(json.dumps(out, indent=2))

    print("\n==== SUBSTRATE SELECTION (lower base score = stronger RAG case) ====")
    ranked = sorted(per_candidate.items(),
                    key=lambda kv: (kv[1]["base_mean_score"] is None, kv[1]["base_mean_score"] or 999))
    for cand, m in ranked:
        print(f"  {cand:24} base_mean={str(m['base_mean_score']):6} "
              f"(n_docs_answerable={m['docs_answerable']})")
    print(f"\nusage: {usage.summary()}")


if __name__ == "__main__":
    run()
