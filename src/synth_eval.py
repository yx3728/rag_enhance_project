"""Synthetic, doc-grounded eval set for a post-cutoff repo with no forum Q&A (spec §2 caveat).

For each sampled substantive doc chunk, an Opus-medium call writes a realistic, self-contained
developer question that the chunk answers + a concise reference answer derived ONLY from the chunk.
gold = that chunk. The base model later sees only the question (chunk held out), isolating
knowledge-injection. Corpus-gap ≈ 0 by construction.

FLAG (recorded in outputs): synthetic questions are doc-shaped, so retrieval is easier than real
user questions — this eval mode is distinct from the forum-Q repos and is labelled "synthetic".

Usage: python src/synth_eval.py <repo_key> [--n 110]
"""
import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import config as C
from evalkit import load_index
from llm import call_claude, UsageTracker, enable_trace, disable_trace

WORKERS = 6

GEN_PROMPT = """You are creating one evaluation item for a documentation-QA system for the tool "{tool}".

DOCUMENTATION CHUNK (the only source of truth for this item):
{chunk}

Write:
1. a realistic, SELF-CONTAINED developer question that THIS chunk answers — phrased as a user would
   ask it (do NOT say "this doc"/"the chunk"/"above"; name the tool/API concretely). It must be
   answerable from the chunk.
2. a concise REFERENCE answer derived ONLY from this chunk (1-4 sentences, concrete: API/config/code).

If the chunk is not substantive enough to support a concrete usage/API question (e.g. pure
navigation, a title, a changelog line), set "usable": false.
Reply with ONLY a JSON object (no markdown fence):
{{"usable": <true|false>, "question": "...", "reference": "..."}}
"""


def parse_json(t):
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def run(repo_key: str, n: int = 110):
    idx = load_index(repo_key)
    tool = repo_key.split("__")[1]
    # candidate chunks: substantive prose, reasonable size, prefer those with a heading
    cands = [c for c in idx.chunks if 300 <= len(c.content) <= 2000
             and c.content.count(" ") > 40 and c.content.count("\n") < 60]
    # deterministic, even spread across the corpus; oversample then filter to n usable
    step = max(1, len(cands) // (n * 2))
    sample = cands[::step][: n * 2]
    usage = UsageTracker(); enable_trace(C.ROOT / "traces" / repo_key / "synth.jsonl")
    lock = threading.Lock(); kept = []

    def work(ch):
        if len(kept) >= n:
            return None
        r = call_claude(GEN_PROMPT.format(tool=tool, chunk=ch.content[:2200]),
                        model=C.JUDGE_MODEL,
                        trace_meta={"repo": repo_key, "phase": "synth", "kind": "gen_q", "chunk": ch.chunk_id})
        v = parse_json(r.text)
        with lock:
            usage.record(r, is_judge=True)
        if not v.get("usable") or not v.get("question") or not v.get("reference"):
            return None
        return {"id": f"{repo_key}#{ch.chunk_id}", "title": (ch.heading or ch.doc_path)[:80],
                "question": v["question"].strip(), "reference_answer": v["reference"].strip(),
                "url": ch.doc_path, "gold_chunk_ids": [ch.chunk_id], "eval_mode": "synthetic"}

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for row in ex.map(work, sample):
            if row and len(kept) < n:
                kept.append(row)

    out = {"tool": repo_key, "eval_mode": "synthetic", "n_eval": len(kept),
           "n_corpus_chunks": len(idx.chunks),
           "note": "doc-grounded synthetic Qs (gold = source chunk; corpus-gap ≈0 by construction). "
                   "Confound: doc-shaped questions make retrieval easier than real user questions.",
           "questions": kept, "gen_usage": usage.summary()}
    (C.EVAL / f"{repo_key}.json").write_text(json.dumps(out, indent=2))
    disable_trace()
    print(f"synthetic eval: {len(kept)} questions -> data/eval/{repo_key}.json | usage {usage.summary()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_key")
    ap.add_argument("--n", type=int, default=110)
    a = ap.parse_args()
    run(a.repo_key, a.n)
