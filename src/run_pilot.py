"""End-to-end pilot runner for one substrate.

Steps: fetch docs -> build eval set -> recall@k (free) -> crown-jewel ablation (judge).
Discussions are assumed already fetched (src/fetch_discussions.py) and cached.

Usage:
  python src/run_pilot.py <owner__name> [--tau 0.62] [--limit N] [--skip-ablation]
"""
import argparse
import sys

import config as C
import fetch_docs
import build_eval
import recall
import ablation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool")
    ap.add_argument("--tau", type=float, default=build_eval.TAU)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    sub = C.SUBSTRATES[args.tool]
    print(f"\n### 1. fetch docs for {args.tool}")
    fetch_docs.fetch(sub["owner"], sub["name"], sub["ref"], sub["prefixes"], exts=tuple(sub["exts"]))

    print(f"\n### 2. build eval set (tau={args.tau})")
    build_eval.build(args.tool, args.tau)

    print("\n### 3. recall@k (objective, full set)")
    recall.run(args.tool)

    if not args.skip_ablation:
        print("\n### 4. crown-jewel RAG-vs-base ablation")
        ablation.run(args.tool, args.limit)


if __name__ == "__main__":
    main()
