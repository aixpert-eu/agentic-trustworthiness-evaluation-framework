"""
Offline evaluation of captured MaintIE runs with the WP3 metrics.

Per run (single-trajectory metrics through the standard harness):
    task_success, subgoal_progress, log_completeness, step_efficiency
Per configuration (model x pattern), across work orders and runs:
    pass^k (outcome reliability), cross_run_consistency (repeatability),
    schema_violation_rate, latency, cost_per_query
plus entity / relation precision, recall and F1 (task quality, micro-averaged).

``optimal_steps`` for step_efficiency is a demo choice, not a MaintIE fact:
ReAct = one lookup per gold entity (a lookup is a tool step), Basic RAG = 1.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contracts.evaluator import EvaluationResult
from harness.adapters import from_wp4_pattern_result
from harness.evaluate_trajectory import evaluate_trajectory
from metrics.aggregate.cross_run_consistency import CrossRunConsistencyEvaluator
from metrics.aggregate.pass_hat_k import PassHatKAgentEvaluator
from metrics.common.schema_violation_rate import SchemaViolationRateEvaluator
from metrics.efficiency.cost_per_query import CostPerQueryEvaluator
from metrics.efficiency.latency import LatencyEvaluator

from .data import Item, load_split, select_eval_items
from .scoring import build_task, parse_prediction, prf, score_prediction

DEFAULT_LATENCY_BUDGET_S = 30.0
DEFAULT_COST_BUDGET_USD = 0.01


def optimal_steps(pattern: str, item: Item) -> int:
    return max(1, len(item.entities)) if pattern == "react" else 1


def res_dict(r: EvaluationResult | Exception | None) -> dict | None:
    if r is None:
        return None
    if isinstance(r, Exception):
        return {"error": f"{type(r).__name__}: {r}"}
    return {"score": r.score, "is_success": r.is_success, "band_level": r.metadata.get("band_level"),
            "band_label": r.metadata.get("band_label"), "explanation": r.explanation}


def load_runs(runs_dir: str | Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(runs_dir).rglob("*__r*.json"))]


def token_usage(messages: list[dict]) -> tuple[int, int]:
    tin = tout = 0
    for m in messages:
        u = m.get("usage_metadata") or {}
        tin += int(u.get("input_tokens", 0))
        tout += int(u.get("output_tokens", 0))
    return tin, tout


def evaluate_run(rec: dict, item: Item) -> dict:
    """Score one captured run. Runs that ended in an error count as failures."""
    pattern = rec["pattern"]
    task = build_task(item, optimal_steps=optimal_steps(pattern, item))
    result = {"messages": rec["messages"], "structured_response": rec["structured_response"]}
    traj = from_wp4_pattern_result(result, input_query=rec["input"])
    quality = score_prediction(item, traj.output)
    pred = parse_prediction(traj.output)
    out = {
        "run": rec["run"], "error": rec.get("error"), "seconds": rec["wall_seconds"],
        "quality": quality, "predicted": pred, "structured": quality["valid"],
        "tokens_in": 0, "tokens_out": 0,
    }
    if rec.get("error") or not rec["messages"]:
        out.update(task_success=0.0, subgoal_progress=0.0, log_completeness=None, step_efficiency=None, metrics={})
        return out
    report = evaluate_trajectory(traj, task)
    out["metrics"] = {k: res_dict(v) for k, v in report.items()}
    g = lambda k: (report[k].score if k in report and not isinstance(report[k], Exception) else None)  # noqa: E731
    out.update(task_success=g("task_success") or 0.0, subgoal_progress=g("subgoal_progress") or 0.0,
               log_completeness=g("log_completeness"), step_efficiency=g("step_efficiency"))
    out["tokens_in"], out["tokens_out"] = token_usage(rec["messages"])
    out["n_tool_calls"] = sum(len(m.get("tool_calls") or []) for m in rec["messages"])
    return out


def _mean(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _safe(fn) -> dict | None:
    try:
        return res_dict(fn())
    except Exception as e:  # keep the dashboard populated when one aggregate cannot run
        return {"error": f"{type(e).__name__}: {e}"}


def aggregate_config(model: str, pattern: str, per_item: dict[str, list[dict]], items: dict[str, Item],
                     price: tuple[float, float], latency_budget: float, cost_budget: float) -> dict:
    runs = [r for rs in per_item.values() for r in rs]
    micro = {}
    for kind in ("entity", "relation"):
        micro[kind] = prf(sum(r["quality"][kind]["tp"] for r in runs), sum(r["quality"][kind]["n_pred"] for r in runs),
                          sum(r["quality"][kind]["n_gold"] for r in runs))
    problems = [(len(rs), int(sum(r["task_success"] >= 1.0 for r in rs))) for rs in per_item.values()]
    f1_by_item = [[r["quality"]["entity"]["f1"] for r in rs] for rs in per_item.values()]
    n_runs_min = min(len(rs) for rs in per_item.values())
    mean_s = _mean([r["seconds"] for r in runs])
    cost = _mean([r["tokens_in"] / 1000 * price[0] + r["tokens_out"] / 1000 * price[1] for r in runs])
    return {
        "model": model, "pattern": pattern, "n_items": len(per_item), "n_runs": len(runs),
        "n_errors": sum(1 for r in runs if r["error"]),
        "task_success_rate": _mean([r["task_success"] for r in runs]),
        "subgoal_progress": _mean([r["subgoal_progress"] for r in runs]),
        "log_completeness": _mean([r["log_completeness"] for r in runs]),
        "step_efficiency": _mean([r["step_efficiency"] for r in runs]),
        "entity": micro["entity"], "relation": micro["relation"],
        "pass_hat_k": _safe(lambda: PassHatKAgentEvaluator().evaluate_problems(problems)),
        "cross_run_consistency": _safe(lambda: CrossRunConsistencyEvaluator().evaluate_runs(f1_by_item)) if n_runs_min >= 2 else None,
        "schema_conformance": _safe(lambda: SchemaViolationRateEvaluator().evaluate_batch(
            [r["predicted"] if r["structured"] else None for r in runs])),
        "latency": _safe(lambda: LatencyEvaluator().evaluate_measurement(mean_s, latency_budget)) if mean_s is not None else None,
        "latency_mean_s": mean_s,
        "cost": _safe(lambda: CostPerQueryEvaluator().evaluate_measurement(cost, cost_budget)),
        "cost_mean_usd": cost,
        "tokens_in_mean": _mean([r["tokens_in"] for r in runs]), "tokens_out_mean": _mean([r["tokens_out"] for r in runs]),
        "tool_calls_mean": _mean([r.get("n_tool_calls") for r in runs]),
        "items": [
            {"item_id": iid, "text": items[iid].text,
             "gold": {"entities": [{"text": e.text, "type": e.coarse} for e in items[iid].entities],
                      "relations": [{"head": r.head, "tail": r.tail, "type": r.type} for r in items[iid].relations]},
             "runs": [{k: r[k] for k in ("run", "error", "task_success", "subgoal_progress", "predicted", "seconds")}
                      | {"entity_f1": r["quality"]["entity"]["f1"], "relation_f1": r["quality"]["relation"]["f1"]} for r in rs]}
            for iid, rs in per_item.items()
        ],
    }


def filter_records(records: list[dict], *, item_ids: set[str] | None = None, max_runs: int | None = None,
                   models: list[str] | None = None, patterns: list[str] | None = None,
                   common: bool = False) -> list[dict]:
    """Restrict captured runs so configurations are compared on the same footing.

    ``item_ids``: keep only these work orders. ``max_runs``: keep runs 0..max_runs-1.
    ``models``: case-insensitive substrings of the model id. ``patterns``: exact pattern names.
    ``common``: keep only (work order, run) pairs that every remaining configuration has, so a
    model captured on fewer items or runs cannot look better or worse because of that.
    """
    keep = []
    needles = [m.lower() for m in models or []]
    for r in records:
        if item_ids is not None and r["item_id"] not in item_ids:
            continue
        if max_runs is not None and r["run"] >= max_runs:
            continue
        if needles and not any(n in r["model"].lower() for n in needles):
            continue
        if patterns and r["pattern"] not in patterns:
            continue
        keep.append(r)
    if common and keep:
        per_cfg: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
        for r in keep:
            per_cfg[(r["model"], r["pattern"])].add((r["item_id"], r["run"]))
        shared = set.intersection(*per_cfg.values())
        keep = [r for r in keep if (r["item_id"], r["run"]) in shared]
    return keep


def coverage_warning(configs: list[dict]) -> str | None:
    """A message when configurations differ in number of work orders or runs (not comparable as is)."""
    shapes = {(c["n_items"], c["n_runs"]) for c in configs}
    if len(shapes) <= 1:
        return None
    detail = "; ".join(f"{c['model']} | {c['pattern']}: {c['n_items']} items, {c['n_runs']} runs" for c in configs)
    return f"configurations differ in coverage ({detail}). Use --n/--runs/--common to compare like with like."


def evaluate_all(records: list[dict], items: dict[str, Item], prices: dict[str, tuple[float, float]] | None = None,
                 latency_budget: float = DEFAULT_LATENCY_BUDGET_S, cost_budget: float = DEFAULT_COST_BUDGET_USD,
                 subset_note: str | None = None) -> dict:
    prices = prices or {}
    grouped: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    provenance: dict[str, Any] = {}
    for rec in records:
        if rec["item_id"] not in items:
            continue
        grouped[(rec["model"], rec["pattern"])][rec["item_id"]].append(evaluate_run(rec, items[rec["item_id"]]))
        provenance[rec["model"]] = rec.get("provenance", {})
    configs = []
    for (model, pattern), per_item in sorted(grouped.items()):
        for rs in per_item.values():
            rs.sort(key=lambda r: r["run"])
        configs.append(aggregate_config(model, pattern, per_item, items, tuple(prices.get(model, (0.0, 0.0))),
                                        latency_budget, cost_budget))
    return {
        "meta": {"generated_at": datetime.now(timezone.utc).isoformat(), "dataset": "MaintIE (nlp-tlp/maintie, MIT), gold split",
                 "latency_budget_s": latency_budget, "cost_budget_usd": cost_budget, "prices_usd_per_1k": prices,
                 "provenance": provenance,
                 "notes": ["Entity match = (normalised text, top-level class); relation match = (head text, tail text, type).",
                           "task_success = every gold entity found with the right class; subgoal_progress = fraction of gold entities and relations found.",
                           "step_efficiency optimal steps: ReAct = one lookup per gold entity, Basic RAG = 1."]
                 + ([subset_note] if subset_note else [])},
        "configs": configs,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maintie", required=True)
    ap.add_argument("--runs", default="runs", help="directory written by capture")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--n", type=int, default=None, help="score only the first N work orders (same selection as capture --n)")
    ap.add_argument("--seed", type=int, default=0, help="seed used at capture time (only matters with --n)")
    ap.add_argument("--max-runs", type=int, default=None, help="score only runs 0..K-1 of each work order")
    ap.add_argument("--models", help="comma-separated, case-insensitive substrings of model ids to include")
    ap.add_argument("--patterns", help="comma-separated pattern names to include (react,basic_rag)")
    ap.add_argument("--common", action="store_true",
                    help="keep only (work order, run) pairs that every configuration has, for like-for-like comparison")
    ap.add_argument("--prices", help='JSON file {"model": [usd_per_1k_input, usd_per_1k_output]}')
    ap.add_argument("--latency-budget", type=float, default=DEFAULT_LATENCY_BUDGET_S)
    ap.add_argument("--cost-budget", type=float, default=DEFAULT_COST_BUDGET_USD)
    args = ap.parse_args(argv)

    gold = {i.item_id: i for i in load_split(Path(args.maintie) / "data" / "gold_release.json", "g")}
    prices = json.loads(Path(args.prices).read_text()) if args.prices else {}
    item_ids = {i.item_id for i in select_eval_items(list(gold.values()), n=args.n, seed=args.seed)} if args.n else None
    split = lambda v: [x.strip() for x in v.split(",") if x.strip()] if v else None  # noqa: E731
    records = filter_records(load_runs(args.runs), item_ids=item_ids, max_runs=args.max_runs,
                             models=split(args.models), patterns=split(args.patterns), common=args.common)
    if not records:
        raise SystemExit("no runs left after filtering; check --runs, --n, --models, --patterns")
    parts = [f"first {args.n} work orders (seed {args.seed})" if args.n else None,
             f"runs 0..{args.max_runs - 1}" if args.max_runs else None,
             f"models matching {args.models}" if args.models else None,
             f"patterns {args.patterns}" if args.patterns else None,
             "only (work order, run) pairs present for every configuration" if args.common else None]
    note = ("Subset: " + "; ".join(p for p in parts if p) + ".") if any(parts) else None
    results = evaluate_all(records, gold, prices, args.latency_budget, args.cost_budget, subset_note=note)
    warn = coverage_warning(results["configs"])
    if warn:
        print("WARNING:", warn)
        results["meta"]["notes"].append("WARNING: " + warn)
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    for c in results["configs"]:
        print(f"{c['model']:40s} {c['pattern']:10s} success={c['task_success_rate']:.2f} "
              f"entF1={c['entity']['f1']:.2f} relF1={c['relation']['f1']:.2f} errors={c['n_errors']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
