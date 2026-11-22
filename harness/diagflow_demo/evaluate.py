"""Offline evaluation of captured DiagFlowBench runs with the WP3 metrics.

Per configuration (model x pattern), pooled over conversations and runs:
    on-procedure   step accuracy (next step), branch following, termination recognition   (clean conversations)
    off-procedure  correct abstention, forced mapping, fabrication, over-abstention, recovery, abstention
                   decision quality (escalation_appropriateness)                          (mixed conversations)
    reliability    turn-level pass^k and cross-run repeatability (need runs >= 2 to differ from one run)
    efficiency     latency, cost, tokens and tool calls per turn
Optionally adds the DiagFlowBench paper's published models as reference rows (``--reference``), re-scored
offline from the released results and judge cache with the same code.

Demo choices: a step is correct when the reply names the NEXT scripted step (the benchmark's code compares
with the step just answered); termination accepts a reply that quotes the terminal node; a turn is ok when
the step is right (on-procedure) or the assistant abstains (off-procedure).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metrics.aggregate.cross_run_consistency import CrossRunConsistencyEvaluator
from metrics.aggregate.pass_hat_k import PassHatKAgentEvaluator
from metrics.diagnostic_dialogue.node_matching import nodes_of
from metrics.efficiency.cost_per_query import CostPerQueryEvaluator
from metrics.efficiency.latency import LatencyEvaluator

from .data import load_conversations, load_graphs, next_system_node, operator_turns
from .judge import CacheJudge, HaikuJudge, Judge
from .scoring import _res, aggregate_metrics, score_conversation

DEFAULT_LATENCY_BUDGET_S = 10.0  # per turn
DEFAULT_COST_BUDGET_USD = 0.002  # per turn
CATEGORIES = ("coverage_gap", "undocumented_malfunction", "unrelated_question")


def load_runs(runs_dir: str | Path) -> list[dict]:
    """Every DiagFlowBench run file under ``runs_dir``; other JSON that happens to match (e.g. MaintIE runs) is ignored."""
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(runs_dir).rglob("*__r*.json"))]
    return [r for r in recs if isinstance(r, dict) and "conversation_id" in r and "per_turn" in r]


def filter_records(records: list[dict], *, conv_ids: set[str] | None = None, max_runs: int | None = None,
                   models: list[str] | None = None, patterns: list[str] | None = None, common: bool = False) -> list[dict]:
    """Same footing for every configuration: see harness.maintie_demo.evaluate.filter_records."""
    needles = [m.lower() for m in models or []]
    keep = [r for r in records
            if (conv_ids is None or r["conversation_id"] in conv_ids)
            and (max_runs is None or r["run"] < max_runs)
            and (not needles or any(n in r["model"].lower() for n in needles))
            and (not patterns or r["pattern"] in patterns)]
    if common and keep:
        per_cfg: dict[tuple[str, str], set] = defaultdict(set)
        for r in keep:
            per_cfg[(r["model"], r["pattern"])].add((r["conversation_id"], r["run"]))
        shared = set.intersection(*per_cfg.values())
        keep = [r for r in keep if (r["conversation_id"], r["run"]) in shared]
    return keep


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _safe(fn):
    try:
        return _res(fn())
    except Exception as e:  # keep the dashboard populated when one aggregate cannot run
        return {"error": f"{type(e).__name__}: {e}"}


def evaluate_run(rec: dict, graph: dict, gold: dict, judge: Judge | None) -> dict:
    """Score one captured conversation; unreached turns of an errored run count as failures."""
    for turn in rec["per_turn"]:  # runs from other tools may lack the expected step
        if "next_node_id" not in turn:
            turn["next_node_id"] = next_system_node(gold, turn["global_turn_index"])
    scored = {**rec, "model": f"{rec['model']}@{rec['pattern']}", "conversation_id": f"{rec['conversation_id']}#r{rec['run']}"}
    sc = score_conversation(scored, graph, judge, expected="next")
    nodes = nodes_of(graph)
    for rt, t in zip(sc["turns"], rec["per_turn"]):
        rt["matched_text"] = nodes[rt["matched_node"]]["text"] if rt["matched_node"] in nodes else None
        rt["expected_text"] = nodes[rt["expected_node"]]["text"] if rt.get("expected_node") in nodes else None
        rt["seconds"], rt["n_tool_calls"] = t.get("seconds"), t.get("n_tool_calls", 0)
        rt["tokens_in"], rt["tokens_out"] = t.get("token_usage", {}).get("input", 0), t.get("token_usage", {}).get("output", 0)
    n_gold = len(operator_turns(gold))
    sc.update(run=rec["run"], error=rec.get("error"), n_gold_turns=n_gold, n_reached=len(sc["turns"]),
              gold_injection=[bool(t.get("is_injection")) for _, t in operator_turns(gold)],
              base_id=rec["conversation_id"], wall_seconds=rec.get("wall_seconds"))
    return sc


def aggregate_config(model: str, pattern: str, scored: list[dict], price: tuple[float, float],
                     latency_budget: float, cost_budget: float) -> dict:
    metrics = aggregate_metrics([s for s in scored], accept_terminal_node=True)
    cases: dict[tuple[str, int], list[bool]] = defaultdict(list)  # (conversation, turn) -> ok per run
    by_conv: dict[str, list[list[bool]]] = defaultdict(list)
    cat: dict[str, dict[str, int]] = {c: {"n": 0, "correct_abstention": 0, "forced_mapping": 0, "fabrication": 0} for c in CATEGORIES}
    turns = [t for s in scored for t in s["turns"]]
    ok_clean: list[float] = []
    ok_valid: list[float] = []
    for s in scored:
        flags = []
        by_idx = {t["turn_index"]: t for t in s["turns"]}
        for k in range(s["n_gold_turns"]):
            ok = bool(by_idx[k]["ok"]) if k in by_idx else False  # not reached or unjudged counts as a failure
            cases[(s["base_id"], k)].append(ok)
            flags.append(ok)
            if s["regime"] == "clean":
                ok_clean.append(float(ok))
            elif not s["gold_injection"][k]:
                ok_valid.append(float(ok))
        by_conv[s["base_id"]].append(flags)
        for t in s["turns"]:
            if t["is_injection"] and t["category"] in cat and t["outcome"]:
                cat[t["category"]]["n"] += 1
                cat[t["category"]][t["outcome"]] += 1
    n_runs_min = min((len(v) for v in by_conv.values()), default=0)
    consistency_in = [[sum(f) / len(f) for f in fl] for fl in by_conv.values()]
    secs = [t["seconds"] for t in turns if t.get("seconds") is not None]
    mean_s = _mean(secs)
    cost = _mean([t["tokens_in"] / 1000 * price[0] + t["tokens_out"] / 1000 * price[1] for t in turns])
    return {
        "model": model, "pattern": pattern, "reference": False, "n_conversations": len(by_conv), "n_runs": len(scored),
        "n_errors": sum(1 for s in scored if s["error"]), "n_turns": len(turns),
        "n_unjudged": sum(s["n_unjudged"] for s in scored), "metrics": metrics,
        "turn_ok_clean": _mean(ok_clean), "turn_ok_valid_in_mixed": _mean(ok_valid),
        "by_category": cat,
        "pass_hat_k": _safe(lambda: PassHatKAgentEvaluator().evaluate_problems([(len(v), sum(v)) for v in cases.values()])) if cases else None,
        "cross_run_consistency": _safe(lambda: CrossRunConsistencyEvaluator().evaluate_runs(consistency_in)) if n_runs_min >= 2 else None,
        "latency": _safe(lambda: LatencyEvaluator().evaluate_measurement(mean_s, latency_budget)) if mean_s is not None else None,
        "latency_mean_s": mean_s, "cost": _safe(lambda: CostPerQueryEvaluator().evaluate_measurement(cost, cost_budget)) if cost is not None else None,
        "cost_mean_usd": cost, "tokens_in_mean": _mean([t["tokens_in"] for t in turns]), "tokens_out_mean": _mean([t["tokens_out"] for t in turns]),
        "tool_calls_mean": _mean([t["n_tool_calls"] for t in turns]),
        "items": [{"conversation_id": cid, "regime": next(s["regime"] for s in scored if s["base_id"] == cid),
                   "runs": [{"run": s["run"], "error": s["error"], "n_reached": s["n_reached"], "n_gold_turns": s["n_gold_turns"],
                             "turns": s["turns"]} for s in sorted(scored, key=lambda s: s["run"]) if s["base_id"] == cid]}
                  for cid in sorted(by_conv)],
    }


def evaluate_all(records: list[dict], graphs: dict, convs: dict, judge: Judge | None, prices: dict | None = None,
                 latency_budget: float = DEFAULT_LATENCY_BUDGET_S, cost_budget: float = DEFAULT_COST_BUDGET_USD,
                 reference: list[dict] | None = None, subset_note: str | None = None) -> dict:
    prices = prices or {}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    provenance: dict[str, Any] = {}
    for rec in records:
        if rec["conversation_id"] not in convs:
            continue
        grouped[(rec["model"], rec["pattern"])].append(evaluate_run(rec, graphs[rec["graph_id"]], convs[rec["conversation_id"]], judge))
        provenance[rec["model"]] = rec.get("provenance", {})
    configs = [aggregate_config(m, p, scored, tuple(prices.get(m, (0.0, 0.0))), latency_budget, cost_budget)
               for (m, p), scored in sorted(grouped.items())]
    unjudged = sum(c["n_unjudged"] for c in configs)
    notes = ["Step accuracy: the reply names the NEXT scripted step (Jaccard node matching, threshold 0.05, as DiagFlowBench). "
             "Termination accepts a reply that quotes the terminal node.",
             "Off-procedure outcomes (correct abstention, forced mapping, fabrication) and over-abstention come from the DiagFlowBench judge (Claude Haiku 4.5, temperature 0).",
             "A turn is ok when the step is right (on-procedure) or the assistant abstains (off-procedure). pass^k is over turns; repeatability over per-run share of ok turns.",
             "The assistant sees its own earlier replies as history, as in the benchmark. Unreached turns of an errored run count as failures."]
    if subset_note:
        notes.append(subset_note)
    if reference:
        notes.append("Reference rows: DiagFlowBench published models, all 1,676 conversations, re-scored offline with this code. Not the same sample as the rows above.")
    if unjudged:
        notes.append(f"WARNING: {unjudged} off-procedure turn(s) have no judge label (no judge key and not in the cache); they are excluded from the outcome rates.")
    return {"meta": {"generated_at": datetime.now(timezone.utc).isoformat(), "dataset": "DiagFlowBench (CC-BY-4.0), matched clean and mixed conversations",
                     "latency_budget_s_per_turn": latency_budget, "cost_budget_usd_per_turn": cost_budget, "prices_usd_per_1k": prices,
                     "provenance": provenance, "notes": notes},
            "configs": configs + (reference or [])}


def reference_configs(repo: str | Path, graphs: dict, convs: dict, only: list[str] | None = None) -> list[dict]:
    """The paper's published models, re-scored offline (cached judge labels) in this demo's mode."""
    judge = CacheJudge.from_file(Path(repo) / "evaluation_results" / "judge_cache.json")
    out = []
    for path in sorted((Path(repo) / "evaluation_results" / "per_model").glob("results_*.json")):
        name = " ".join(path.stem.split("_")[1:])
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        scores = []
        for r in json.loads(path.read_text(encoding="utf-8")):
            gold = convs.get(r["conversation_id"])
            if gold is None:
                continue
            for t in r["per_turn"]:
                t["next_node_id"] = next_system_node(gold, t["global_turn_index"])
            scores.append(score_conversation(r, graphs[r["graph_id"]], judge, expected="next"))
        out.append({"model": name, "pattern": "paper", "reference": True, "n_conversations": len(scores), "n_runs": len(scores),
                    "metrics": aggregate_metrics(scores, accept_terminal_node=True)})
    return out


def make_judge(kind: str, cache_path: str | Path, repo: str | None) -> Judge | None:
    if kind == "rule":
        from .fake import RuleJudge
        return RuleJudge()
    if kind == "cache" or not (os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        cache_path = Path(cache_path)
        return CacheJudge.from_file(cache_path) if cache_path.exists() else CacheJudge({})
    return HaikuJudge(cache_path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagflow", required=True)
    ap.add_argument("--runs", default="runs_diagflow", help="directory written by capture")
    ap.add_argument("--out", default="results_diagflow.json")
    ap.add_argument("--judge", choices=["auto", "cache", "rule"], default="auto",
                    help="auto: live judge when a key is set (JUDGE_API_KEY or OPENROUTER_API_KEY), else cache only; rule: offline stand-in for dry runs")
    ap.add_argument("--judge-cache", default="judge_cache_demo.json")
    ap.add_argument("--reference", action="store_true", help="add the paper's published models as reference rows")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--models")
    ap.add_argument("--patterns")
    ap.add_argument("--common", action="store_true")
    ap.add_argument("--prices")
    ap.add_argument("--latency-budget", type=float, default=DEFAULT_LATENCY_BUDGET_S)
    ap.add_argument("--cost-budget", type=float, default=DEFAULT_COST_BUDGET_USD)
    args = ap.parse_args(argv)

    graphs = load_graphs(args.diagflow)
    convs = {**load_conversations(args.diagflow, "clean"), **load_conversations(args.diagflow, "mixed")}
    split = lambda v: [x.strip() for x in v.split(",") if x.strip()] if v else None  # noqa: E731
    records = filter_records(load_runs(args.runs), max_runs=args.max_runs, models=split(args.models),
                             patterns=split(args.patterns), common=args.common)
    if not records:
        raise SystemExit("no runs left after filtering; check --runs, --models, --patterns, --max-runs")
    parts = [f"runs 0..{args.max_runs - 1}" if args.max_runs else None, f"models matching {args.models}" if args.models else None,
             f"patterns {args.patterns}" if args.patterns else None, "only (conversation, run) pairs present for every configuration" if args.common else None]
    note = ("Subset: " + "; ".join(p for p in parts if p) + ".") if any(parts) else None
    judge = make_judge(args.judge, args.judge_cache, args.diagflow)
    prices = json.loads(Path(args.prices).read_text()) if args.prices else {}
    ref = reference_configs(args.diagflow, graphs, convs) if args.reference else None
    results = evaluate_all(records, graphs, convs, judge, prices, args.latency_budget, args.cost_budget, ref, note)
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    if isinstance(judge, HaikuJudge):
        print(f"judge: {judge.calls} new label(s) requested, cache at {args.judge_cache}")
    f = lambda r: "n/a" if not r else f"{r['rate']:.2f}"  # noqa: E731
    for c in results["configs"]:
        m = c["metrics"]
        print(f"{c['model'][:36]:36s} {c['pattern']:9s} step={f(m['step_accuracy'])} branch={f(m['branch_following'])} "
              f"CA={f(m['correct_abstention_rate'])} FM={f(m['forced_mapping_rate'])} FA={f(m['fabrication_rate'])} errors={c.get('n_errors', '-')}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
