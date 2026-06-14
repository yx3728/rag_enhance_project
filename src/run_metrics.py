"""Run the metric chain for one repo whose eval set already exists:
  claim-coverage analyze (Opus-medium, traced) -> recall@k (free) -> ablation strict + fallback.
All judge calls traced under traces/<repo>/. Sequential within one process.

Usage: python src/run_metrics.py <repo_key>
"""
import sys
import analyze
import recall
import ablation


def main():
    repo = sys.argv[1]
    print(f"\n##### {repo}: claim-coverage analyze #####")
    analyze.run(repo)                       # -> results/analyze_<repo>.json (cov@k, corpus-gap)
    print(f"\n##### {repo}: recall@k (diagnostic) #####")
    recall.run(repo)                        # -> results/recall_<repo>.json
    print(f"\n##### {repo}: ablation — strict (diagnostic) #####")
    ablation.run(repo, variant="strict")    # -> results/ablation_<repo>.json
    print(f"\n##### {repo}: ablation — fallback (headline) #####")
    ablation.run(repo, variant="fallback")  # -> results/ablation_<repo>_fallback.json


if __name__ == "__main__":
    main()
