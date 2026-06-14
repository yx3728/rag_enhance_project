"""Cross-repo synthesis + manifest for the multi-repo scale-up.

Reads the persisted per-repo results and emits:
  - results/multirepo/cross_repo.json (the one table: familiarity, corpus-gap, coverage, lift, W/T/L)
  - results/manifest.json (each headline number -> the raw file(s) that produced it)

No API calls. Run after all per-repo metrics exist.
"""
import json
from pathlib import Path

import config as C

REPOS = [
    ("duckdb__duckdb", "moderate", "forum"),
    ("litestar-org__litestar", "low", "forum"),
    ("pydantic__pydantic-ai", "zero/post-cutoff", "synthetic"),
]


def load(p):
    p = C.RESULTS / p
    return json.load(open(p)) if p.exists() else None


def main():
    sel = load("multirepo/multirepo_selection.json")
    probe = (sel or {}).get("base_probe", {}) if sel else {}
    rows = []
    manifest = {}
    for key, cell, mode in REPOS:
        analyze = load(f"analyze_{key}.json")
        recall = load(f"recall_{key}.json")
        abl_f = load(f"ablation_{key}_fallback.json")
        abl_s = load(f"ablation_{key}.json")
        ev = json.load(open(C.EVAL / f"{key}.json"))
        base_fam = None
        if mode == "forum" and key in probe:
            base_fam = probe[key].get("base_mean_score")
        elif abl_f:
            base_fam = abl_f["summary"]["base_mean_score"]  # base score on this repo's eval
        row = {
            "repo": key, "cell": cell, "eval_mode": mode, "n_eval": ev["n_eval"],
            "base_familiarity_probe": base_fam,
            "claim_coverage_at5": (analyze or {}).get("coverage_at", {}).get("5"),
            "corpus_gap": (analyze or {}).get("corpus_gap_rate"),
            "recall_at5_vector": ((recall or {}).get("results", {}).get("vector", {}) or {}).get("recall_at", {}).get("5"),
            "fallback_lift": (abl_f or {}).get("summary", {}).get("mean_lift"),
            "fallback_lift_ci95": (abl_f or {}).get("summary", {}).get("lift_ci95"),
            "fallback_base_mean": (abl_f or {}).get("summary", {}).get("base_mean_score"),
            "fallback_rag_mean": (abl_f or {}).get("summary", {}).get("rag_mean_score"),
            "fallback_wtl": (abl_f or {}).get("summary") and
                f"{abl_f['summary']['rag_wins']}/{abl_f['summary']['ties']}/{abl_f['summary']['rag_losses']}",
            "strict_lift": (abl_s or {}).get("summary", {}).get("mean_lift"),
        }
        rows.append(row)
        manifest[key] = {
            "claim_coverage_at5 / corpus_gap": f"results/analyze_{key}.json",
            "recall@k": f"results/recall_{key}.json",
            "fallback ablation (headline lift)": f"results/ablation_{key}_fallback.json",
            "strict ablation (diagnostic)": f"results/ablation_{key}.json",
            "eval set": f"data/eval/{key}.json",
            "judge traces": f"traces/{key}/",
        }

    cross = {"subject_model": "claude-haiku-4-5", "judge_model": "claude-opus-4-8 (effort=medium)",
             "rows": rows}
    (C.RESULTS / "multirepo" / "cross_repo.json").write_text(json.dumps(cross, indent=2))
    manifest["_global"] = {"selection/base-probe": "results/multirepo/multirepo_selection.json + results/multirepo/probe_forum.json",
                           "cross-repo table": "results/multirepo/cross_repo.json",
                           "dagster diagnosis (prior)": "DIAGNOSIS.md + results/*dagster*"}
    (C.RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # print table
    print(f"\n{'repo':24} {'cell':16} {'mode':9} {'n':4} {'baseFam':8} {'cov@5':6} {'gap':6} {'lift':7} {'ci95':16} {'W/T/L'}")
    for r in rows:
        print(f"{r['repo']:24} {r['cell']:16} {r['eval_mode']:9} {str(r['n_eval']):4} "
              f"{str(r['base_familiarity_probe']):8} {str(r['claim_coverage_at5']):6} "
              f"{str(r['corpus_gap']):6} {str(r['fallback_lift']):7} {str(r['fallback_lift_ci95']):16} {r['fallback_wtl']}")


if __name__ == "__main__":
    main()
