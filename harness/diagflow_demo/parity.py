"""Reproduce DiagFlowBench's published numbers with the WP3 metrics, offline (no API calls).

Reads the released per-model results and judge cache, scores them with our implementation in the
repository's own mode (``expected="gold"``) and compares with ``analysis_report.json``. A match shows our
scorers are the same function as the paper's, so numbers from new runs are comparable to the published ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_graphs
from .judge import CacheJudge
from .scoring import aggregate_metrics, score_conversation

# our metric id -> (location in the original report, key)
REPORT_KEYS = {
    "step_accuracy": [("clean", "sanity_checks", "position_tracking_exact"), ("clean", "metrics", "step_accuracy")],
    "branch_following": [("clean", "metrics", "branch_following"), ("clean", "diagnostics", "branch_following")],
    "termination_recognition": [("clean", "metrics", "termination_recognition")],
    "correct_abstention_rate": [("mixed", "correct_abstention_rate")],
    "forced_mapping_rate": [("mixed", "forced_mapping_rate")],
    "fabrication_rate": [("mixed", "fabrication_rate")],
    "error_recovery_rate": [("mixed", "recovery_rate")],
    "over_abstention_rate": [("mixed", "inappropriate_abstention_rate")],
}


# Not reproducible from the released code: the released report predates the current scorer (on Llama 3.3
# 70B the code gives 0.8% termination recognition, the report 83.4%). Listed, not counted as a failure.
KNOWN_UNREPRODUCIBLE = {"termination_recognition"}


def _lookup(report: dict, path: tuple[str, ...]):
    cur = report
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def reference_value(report: dict, metric_id: str):
    for path in REPORT_KEYS[metric_id]:
        v = _lookup(report, path)
        if v is not None:
            return v
    return None


def rescore_model(results: list[dict], graphs: dict, judge: CacheJudge) -> dict:
    scores = [score_conversation(r, graphs[r["graph_id"]], judge, expected="gold") for r in results if r.get("graph_id") in graphs]
    return aggregate_metrics(scores)


def run(repo: str | Path, tol: float = 1e-9, only: list[str] | None = None) -> tuple[list[dict], bool]:
    repo = Path(repo)
    graphs = load_graphs(repo)
    judge = CacheJudge.from_file(repo / "evaluation_results" / "judge_cache.json")
    report = json.loads((repo / "evaluation_results" / "analysis_report.json").read_text(encoding="utf-8"))
    rows, ok = [], True
    for path in sorted((repo / "evaluation_results" / "per_model").glob("results_*.json")):
        name = " ".join(path.stem.split("_")[1:])
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        if name not in report:
            continue
        ours = rescore_model(json.loads(path.read_text(encoding="utf-8")), graphs, judge)
        for mid in REPORT_KEYS:
            ref = reference_value(report[name], mid)
            got = (ours.get(mid) or {}).get("rate")
            if ref is None and got is None:
                continue
            match = ref is not None and got is not None and abs(ref - got) <= tol
            known = mid in KNOWN_UNREPRODUCIBLE and not match
            ok &= match or known
            rows.append({"model": name, "metric": mid, "published": ref, "ours": got, "match": match, "known": known})
    return rows, ok


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagflow", required=True, help="path of the cloned DiagFlowBench repository")
    ap.add_argument("--models", help="comma-separated substrings of model names to check")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args(argv)
    rows, ok = run(args.diagflow, args.tol, [m.strip() for m in args.models.split(",")] if args.models else None)
    for r in rows:
        f = lambda v: "n/a" if v is None else f"{v:.6f}"  # noqa: E731
        tag = "OK  " if r["match"] else ("NOTE" if r["known"] else "DIFF")
        print(f"{tag} {r['model']:22s} {r['metric']:26s} published={f(r['published'])} ours={f(r['ours'])}")
    n_known = sum(r["known"] for r in rows)
    print(f"\n{sum(r['match'] for r in rows)}/{len(rows)} values match" + (f", {n_known} known difference(s) marked NOTE" if n_known else "")
          + ("" if ok else "  (unexpected differences above)"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
