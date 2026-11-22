"""Offline evaluation of captured DiagFlowBench runs with the WP3 metrics.

Per configuration (model x pattern), pooled over conversations and runs:
    on-procedure   step accuracy (next step), branch following, termination recognition   (clean conversations)
    off-procedure  correct abstention, forced mapping, fabrication, over-abstention, recovery, abstention
                   decision quality (escalation_appropriateness)                          (mixed conversations)
    reliability    turn-level pass^k and pass@k, cross-run repeatability (need runs >= 2 to differ from one run)
    auditability   log completeness of tool calls (only for patterns that make tool calls)
    across configs discriminative_power: does the share of ok turns per run separate the configurations
    efficiency     latency, cost, tokens and tool calls per turn
Optionally adds the DiagFlowBench paper's published models as reference rows (``--reference``), re-scored
offline from the released results and judge cache with the same code.

Every configuration is scored with the same k (the fewest runs any conversation has, at most 5, or ``--k``).
95% intervals on the rates are percentile bootstraps over conversations (see harness.stats); the published
reference rows have none.

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

from contracts.evaluator import EvaluationInput
from harness import stats
from harness.runfiles import load_run_files
from metrics.aggregate.cross_run_consistency import CrossRunConsistencyEvaluator
from metrics.aggregate.pass_at_k import PassAtKAgentEvaluator
from metrics.aggregate.pass_hat_k import DEFAULT_K, PassHatKAgentEvaluator
from metrics.diagnostic_dialogue.node_matching import nodes_of
from metrics.efficiency.cost_per_query import CostPerQueryEvaluator
from metrics.efficiency.latency import LatencyEvaluator
from metrics.meta_evaluation.discriminative_power import DiscriminativePowerEvaluator
from metrics.trajectory.log_completeness import LogCompletenessEvaluator

from .data import load_conversations, load_graphs, next_system_node, operator_turns
from .judge import CacheJudge, HaikuJudge, Judge
from .scoring import _res, aggregate_metrics, score_conversation

DEFAULT_LATENCY_BUDGET_S = 10.0  # per turn
DEFAULT_COST_BUDGET_USD = 0.002  # per turn
CATEGORIES = ("coverage_gap", "undocumented_malfunction", "unrelated_question")


def load_runs(runs_dir: str | Path) -> list[dict]:
    """Every DiagFlowBench run file under ``runs_dir``; other JSON that happens to match (e.g. MaintIE runs) is ignored."""
    recs = load_run_files(runs_dir)
    return [r for r in recs if isinstance(r, dict) and "conversation_id" in r and "per_turn" in r]


def filter_records(records: list[dict], *, conv_ids: set[str] | None = None, max_runs: int | None = None,
                   models: list[str] | None = None, patterns: list[str] | None = None, common: bool = False,
                   min_runs: int | None = None) -> list[dict]:
    """Same footing for every configuration: see harness.maintie_demo.evaluate.filter_records.

    ``min_runs``: keep only conversations that have runs 0..min_runs-1 in every remaining configuration."""
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
    if min_runs and keep:
        have: dict[tuple[str, str], dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
        for r in keep:
            have[(r["model"], r["pattern"])][r["conversation_id"]].add(r["run"])
        needed = set(range(min_runs))
        full = set.intersection(*[{c for c, runs in per_conv.items() if needed <= runs} for per_conv in have.values()])
        keep = [r for r in keep if r["conversation_id"] in full]
    return keep


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _safe(fn):
    try:
        return _res(fn())
    except Exception as e:  # keep the dashboard populated when one aggregate cannot run
        return {"error": f"{type(e).__name__}: {e}"}


def tool_call_log(per_turn: list[dict]) -> tuple[int, int]:
    """(complete tool calls, tool calls seen) over a run, checked turn by turn with the log_completeness metric."""
    seen = complete = 0
    for t in per_turn:
        res = LogCompletenessEvaluator().evaluate(EvaluationInput(messages=t.get("messages") or [], input="", output=None))
        n = int(res.metadata.get("tool_calls_seen", 0))
        seen += n
        complete += n - len(res.metadata.get("incomplete_ids", []))
    return complete, seen


def ok_flags(s: dict) -> list[bool]:
    """One flag per gold turn of a scored run; a turn the run never reached or that was not judged counts as a failure."""
    by_idx = {t["turn_index"]: t for t in s["turns"]}
    return [bool(by_idx[k]["ok"]) if k in by_idx else False for k in range(s["n_gold_turns"])]


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
    sc["log_complete"], sc["log_seen"] = tool_call_log(rec["per_turn"])
    return sc


def _headline(res: dict | None) -> float | None:
    return None if not res else res["rate"] if res.get("rate") is not None else res.get("f1")


def _with_ci(res: dict | None, ci: tuple[float, float] | None) -> dict | None:
    if res is not None and "error" not in res and ci is not None:
        res["ci95"] = [ci[0], ci[1]]
    return res


def aggregate_config(model: str, pattern: str, scored: list[dict], price: tuple[float, float],
                     latency_budget: float, cost_budget: float, k: int | None = None, n_boot: int = stats.N_BOOT) -> dict:
    metrics = aggregate_metrics([s for s in scored], accept_terminal_node=True)
    cases: dict[tuple[str, int], list[bool]] = defaultdict(list)  # (conversation, turn) -> ok per run
    by_conv: dict[str, list[list[bool]]] = defaultdict(list)
    cat: dict[str, dict[str, int]] = {c: {"n": 0, "correct_abstention": 0, "forced_mapping": 0, "fabrication": 0} for c in CATEGORIES}
    turns = [t for s in scored for t in s["turns"]]
    ok_clean: list[float] = []
    ok_valid: list[float] = []
    for s in scored:
        flags = ok_flags(s)
        for turn_k, ok in enumerate(flags):  # not reached or unjudged counts as a failure
            cases[(s["base_id"], turn_k)].append(ok)
            if s["regime"] == "clean":
                ok_clean.append(float(ok))
            elif not s["gold_injection"][turn_k]:
                ok_valid.append(float(ok))
        by_conv[s["base_id"]].append(flags)
        for t in s["turns"]:
            if t["is_injection"] and t["category"] in cat and t["outcome"]:
                cat[t["category"]]["n"] += 1
                cat[t["category"]][t["outcome"]] += 1
    n_runs_min = min((len(v) for v in by_conv.values()), default=0)
    k = k if k is not None else min(DEFAULT_K, max(n_runs_min, 1))
    problems = [(len(v), sum(v)) for v in cases.values()]
    by_base: dict[str, list[list[tuple[int, int]]]] = defaultdict(list)  # conversation -> its turn cases (the resampling unit)
    for (base, _turn), v in cases.items():
        by_base[base].append([(len(v), sum(v))])
    problem_clusters = [[p for c in cl for p in c] for cl in by_base.values()]
    pass_hat = _safe(lambda: PassHatKAgentEvaluator().evaluate_problems(problems, k)) if problems else None
    pass_at = _safe(lambda: PassAtKAgentEvaluator().evaluate_problems(problems, k)) if problems else None
    for res, ev in ((pass_hat, PassHatKAgentEvaluator), (pass_at, PassAtKAgentEvaluator)):
        if res is not None and "error" not in res:
            _with_ci(res, stats.cluster_bootstrap(problem_clusters, lambda sample, ev=ev: ev().evaluate_problems([p for c in sample for p in c], k).score, n_boot=n_boot))
    clusters_by_conv: dict[str, list[dict]] = defaultdict(list)
    for s in scored:
        clusters_by_conv[s["base_id"]].append(s)
    ids = list(metrics)
    cis = stats.cluster_bootstrap_many(
        list(clusters_by_conv.values()),
        lambda sample: {mid: _headline(r) for mid, r in aggregate_metrics([s for c in sample for s in c], accept_terminal_node=True).items() if mid in ids},
        n_boot=n_boot)
    for mid, res in metrics.items():
        _with_ci(res, cis.get(mid))
    log_seen = sum(s["log_seen"] for s in scored)
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
        "k": k, "pass_hat_k": pass_hat, "pass_at_k": pass_at,
        "log_completeness": (sum(s["log_complete"] for s in scored) / log_seen) if log_seen else None, "tool_calls_logged": log_seen,
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


def common_k(grouped: dict, k: int | None) -> int:
    """The k every configuration is scored with: ``k`` if given, else the fewest runs of any conversation (at most 5)."""
    fewest = min((len(v) for scored in grouped.values() for v in _runs_by_conversation(scored).values()), default=1)
    if k is None:
        return min(DEFAULT_K, fewest)
    if k > fewest:
        raise ValueError(f"--k {k} needs at least {k} runs of every conversation in every configuration; the fewest is {fewest}. "
                         "Capture more runs, or use --max-runs/--common to compare on the same footing.")
    return k


def _runs_by_conversation(scored: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for s in scored:
        out[s["base_id"]].append(s)
    return out


def discriminative_power(grouped: dict) -> dict | None:
    """Does the share of ok turns per run separate the configurations? Meta-evaluation, whole comparison."""
    by_system = {f"{m} | {p}": [sum(f) / len(f) for s in scored for f in [ok_flags(s)] if f] for (m, p), scored in grouped.items()}
    by_system = {name: xs for name, xs in by_system.items() if xs}
    if len(by_system) < 2:
        return None
    res = _safe(lambda: DiscriminativePowerEvaluator().evaluate_groups(by_system))
    if res is not None and "error" not in res:
        res.update(score_used="share of ok turns per run", n_systems=len(by_system))
    return res


def evaluate_all(records: list[dict], graphs: dict, convs: dict, judge: Judge | None, prices: dict | None = None,
                 latency_budget: float = DEFAULT_LATENCY_BUDGET_S, cost_budget: float = DEFAULT_COST_BUDGET_USD,
                 reference: list[dict] | None = None, subset_note: str | None = None, k: int | None = None,
                 n_boot: int = stats.N_BOOT) -> dict:
    prices = prices or {}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    provenance: dict[str, Any] = {}
    for rec in records:
        if rec["conversation_id"] not in convs:
            continue
        grouped[(rec["model"], rec["pattern"])].append(evaluate_run(rec, graphs[rec["graph_id"]], convs[rec["conversation_id"]], judge))
        provenance[rec["model"]] = rec.get("provenance", {})
    k_used = common_k(grouped, k)
    configs = [aggregate_config(m, p, scored, tuple(prices.get(m, (0.0, 0.0))), latency_budget, cost_budget, k=k_used, n_boot=n_boot)
               for (m, p), scored in sorted(grouped.items())]
    unjudged = sum(c["n_unjudged"] for c in configs)
    notes = ["Step accuracy: the reply names the NEXT scripted step (Jaccard node matching, threshold 0.05, as DiagFlowBench). "
             "Termination accepts a reply that quotes the terminal node.",
             "Off-procedure outcomes (correct abstention, forced mapping, fabrication) and over-abstention come from the DiagFlowBench judge (Claude Haiku 4.5, temperature 0).",
             "A turn is ok when the step is right (on-procedure) or the assistant abstains (off-procedure). pass^k is over turns; repeatability over per-run share of ok turns.",
             "The assistant sees its own earlier replies as history, as in the benchmark. Unreached turns of an errored run count as failures.",
             f"pass^k and pass@k use k={k_used} for every configuration. 95% intervals are percentile bootstraps over conversations; with few conversations they indicate how far a number could move.",
             "Log completeness is shown only for patterns that make tool calls (with none there is nothing to log, so it is not scored as a pass)."]
    if subset_note:
        notes.append(subset_note)
    if reference:
        notes.append("Reference rows: DiagFlowBench published models, all 1,676 conversations, re-scored offline with this code. Not the same sample as the rows above.")
    if unjudged:
        notes.append(f"WARNING: {unjudged} off-procedure turn(s) have no judge label (no judge key and not in the cache); they are excluded from the outcome rates.")
    return {"meta": {"generated_at": datetime.now(timezone.utc).isoformat(), "dataset": "DiagFlowBench (CC-BY-4.0), matched clean and mixed conversations",
                     "latency_budget_s_per_turn": latency_budget, "cost_budget_usd_per_turn": cost_budget, "prices_usd_per_1k": prices,
                     "provenance": provenance, "pass_k": k_used,
                     "bootstrap": {"method": "percentile bootstrap over conversations", "n_boot": n_boot, "level": stats.LEVEL, "seed": stats.SEED},
                     "meta_evaluation": {"discriminative_power": discriminative_power(grouped)}, "notes": notes},
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
    ap.add_argument("--min-runs", type=int, default=None, metavar="M",
                    help="keep only conversations that have runs 0..M-1 in every configuration (use after extending a capture part of the way)")
    ap.add_argument("--prices")
    ap.add_argument("--latency-budget", type=float, default=DEFAULT_LATENCY_BUDGET_S)
    ap.add_argument("--cost-budget", type=float, default=DEFAULT_COST_BUDGET_USD)
    ap.add_argument("--k", type=int, default=None, help="k for pass^k and pass@k, the same for every configuration (default: fewest runs per conversation, at most 5)")
    ap.add_argument("--bootstrap", type=int, default=stats.N_BOOT, metavar="N", help="bootstrap draws for the 95%% intervals (default %(default)s)")
    args = ap.parse_args(argv)

    graphs = load_graphs(args.diagflow)
    convs = {**load_conversations(args.diagflow, "clean"), **load_conversations(args.diagflow, "mixed")}
    split = lambda v: [x.strip() for x in v.split(",") if x.strip()] if v else None  # noqa: E731
    records = filter_records(load_runs(args.runs), max_runs=args.max_runs, models=split(args.models),
                             patterns=split(args.patterns), common=args.common, min_runs=args.min_runs)
    if not records:
        raise SystemExit("no runs left after filtering; check --runs, --models, --patterns, --max-runs, --min-runs")
    parts = [f"runs 0..{args.max_runs - 1}" if args.max_runs else None, f"models matching {args.models}" if args.models else None,
             f"patterns {args.patterns}" if args.patterns else None, "only (conversation, run) pairs present for every configuration" if args.common else None,
             f"only conversations with at least {args.min_runs} runs in every configuration" if args.min_runs else None]
    note = ("Subset: " + "; ".join(p for p in parts if p) + ".") if any(parts) else None
    judge = make_judge(args.judge, args.judge_cache, args.diagflow)
    prices = json.loads(Path(args.prices).read_text()) if args.prices else {}
    ref = reference_configs(args.diagflow, graphs, convs) if args.reference else None
    try:
        results = evaluate_all(records, graphs, convs, judge, prices, args.latency_budget, args.cost_budget, ref, note, k=args.k, n_boot=args.bootstrap)
    except ValueError as e:
        raise SystemExit(str(e))
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
