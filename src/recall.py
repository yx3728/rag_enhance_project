"""Recall@k on the full eval set — objective, no LLM judge (spec §3).

recall@k = fraction of questions where the top-k retrieved chunks (for the QUESTION) contain
at least one gold chunk. Computed for vector / BM25 / hybrid -> doubles as the optional
strategy comparison. Also reports mean retrieval latency per method.

Usage: python src/recall.py <owner__name>
"""
import json
import sys
import time

import config as C
from evalkit import load_index, load_eval

KS = [1, 3, 5, 10]
METHODS = ["vector", "bm25", "hybrid"]


def run(tool: str):
    idx = load_index(tool)
    ev = load_eval(tool)
    qs = ev["questions"]
    maxk = max(KS)

    results = {}
    for method in METHODS:
        hits = {k: 0 for k in KS}
        latencies = []
        for q in qs:
            gold = set(q["gold_chunk_ids"])
            t0 = time.perf_counter()
            retrieved = [cid for cid, _ in idx.retrieve(q["question"], maxk, method=method)]
            latencies.append((time.perf_counter() - t0) * 1000)
            for k in KS:
                if gold & set(retrieved[:k]):
                    hits[k] += 1
        n = len(qs)
        results[method] = {
            "recall_at": {str(k): round(hits[k] / n, 3) for k in KS},
            "mean_latency_ms": round(sum(latencies) / len(latencies), 1),
            "p95_latency_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1),
        }

    out = {"tool": tool, "n": len(qs), "embed_model": C.EMBED_MODEL,
           "ks": KS, "results": results}
    (C.RESULTS / f"recall_{tool}.json").write_text(json.dumps(out, indent=2))

    print(f"\n==== recall@k  ({tool}, n={len(qs)}) ====")
    header = "method   " + "  ".join(f"@{k:<5}" for k in KS) + "  lat_ms"
    print(header)
    for m in METHODS:
        r = results[m]["recall_at"]
        row = f"{m:8} " + "  ".join(f"{r[str(k)]:<6}" for k in KS) + f"  {results[m]['mean_latency_ms']}"
        print(row)
    return out


if __name__ == "__main__":
    run(sys.argv[1])
