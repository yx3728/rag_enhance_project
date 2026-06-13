"""Remap eval-set gold_chunk_ids from the baseline chunks to a re-chunked corpus.

Re-chunking changes chunk boundaries/ids, so the old gold_chunk_ids no longer exist. For each
old gold chunk we find the new chunk(s) FROM THE SAME DOC with the highest word-overlap to the
old gold's content (lexical, not embedding — avoids biasing the dense retriever we evaluate).
Questions/references are unchanged. Honest caveat: exact-recall on the new corpus is therefore
measured against *remapped* gold; answer-coverage@k (gold-independent) is the primary comparable.

Usage: python src/remap_gold.py <src_tool> <dst_tool>   # e.g. dagster-io__dagster dagster-io__dagster_mech
"""
import json
import re
import sys

import config as C

WORD = re.compile(r"[a-z0-9_]+")


def words(s: str) -> set:
    return set(WORD.findall(s.lower()))


def run(src: str, dst: str):
    old_chunks = {c["chunk_id"]: c for c in json.load(open(C.INDEX / src / "chunks.json"))}
    new_chunks = json.load(open(C.INDEX / dst / "chunks.json"))
    by_doc = {}
    for c in new_chunks:
        by_doc.setdefault(c["doc_path"], []).append(c)
    # precompute word sets for new chunks
    new_words = {c["chunk_id"]: words(c["content"]) for c in new_chunks}

    ev = json.load(open(C.EVAL / f"{src}.json"))
    out_qs = []
    n_lost = 0
    for q in ev["questions"]:
        new_gold = []
        for gid in q["gold_chunk_ids"]:
            oc = old_chunks.get(gid)
            if not oc:
                continue
            cands = by_doc.get(oc["doc_path"], [])
            if not cands:
                continue
            ow = words(oc["content"]) or {oc["doc_path"]}
            best = max(cands, key=lambda c: len(ow & new_words[c["chunk_id"]]) / (len(ow) or 1))
            new_gold.append(best["chunk_id"])
        new_gold = list(dict.fromkeys(new_gold))  # dedup, keep order
        if not new_gold:
            n_lost += 1
            continue
        nq = dict(q)
        nq["gold_chunk_ids"] = new_gold
        nq["gold_chunk_ids_orig"] = q["gold_chunk_ids"]
        out_qs.append(nq)

    out = dict(ev)
    out["tool"] = dst
    out["gold_remapped_from"] = src
    out["n_eval"] = len(out_qs)
    out["questions"] = out_qs
    (C.EVAL / f"{dst}.json").write_text(json.dumps(out, indent=2))
    print(f"remapped gold: {len(out_qs)} questions ({n_lost} dropped: gold doc absent in new corpus) -> data/eval/{dst}.json")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
