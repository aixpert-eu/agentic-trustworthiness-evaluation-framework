"""Capture multi-turn DiagFlowBench runs, one JSON file per (model, pattern, conversation, run).

A conversation is replayed turn by turn: the scripted operator message is sent, the agent replies, and its
own reply (not the scripted system turn) becomes part of the history for the next turn, as in the
benchmark. Files use the benchmark's per-turn result schema, plus the expected next step, timing and token
counts, so ``scoring.py`` handles our runs and the published results alike.

Same safeguards as the MaintIE capture: existing runs are skipped (resume), authentication errors abort
without writing, ``--timeout`` caps one agent call, ``--min-free-requests`` stops cleanly when the free
daily budget runs low. A conversation that hits an error is saved with the turns completed so far.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from harness.maintie_demo.capture import (
    CaptureAborted,
    _call_with_retries,
    free_requests_remaining,
    serialize_messages,
    slug,
)

from harness.runfiles import read_run_file, write_run_file

from .data import load_conversations, load_graphs, matched_pairs, next_system_node, operator_turns, select_pairs

AgentFn = Callable[[list[tuple[str, str]]], dict]
AgentFactory = Callable[[dict, dict], AgentFn]  # (graph, conversation) -> agent_fn


def run_path(out_dir: str | Path, model: str, pattern: str, conv_id: str, run: int) -> Path:
    return Path(out_dir) / slug(model) / pattern / f"{conv_id}__r{run}.json"


def _usage(result: dict) -> dict[str, int]:
    if result.get("usage"):
        return {"input": int(result["usage"].get("input", 0)), "output": int(result["usage"].get("output", 0))}
    tin = tout = 0
    for m in result.get("messages", []):
        u = (m.get("usage_metadata") if isinstance(m, dict) else getattr(m, "usage_metadata", None)) or {}
        tin += int(u.get("input_tokens", 0))
        tout += int(u.get("output_tokens", 0))
    return {"input": tin, "output": tout}


def replay_conversation(conv: dict, agent_fn: AgentFn, *, retries: int, backoff: float, timeout: float | None,
                        sleep: Callable[[float], None]) -> tuple[list[dict], str | None]:
    """Run one conversation. Returns (per_turn records, error or None)."""
    history: list[tuple[str, str]] = []
    per_turn: list[dict] = []
    for k, (gi, turn) in enumerate(operator_turns(conv)):
        history.append(("user", turn["utterance"]))
        snapshot = list(history)
        t0 = time.perf_counter()
        result, err, attempts = _call_with_retries(lambda _t, s=snapshot: agent_fn(s), "", retries, backoff, sleep, timeout)
        seconds = time.perf_counter() - t0
        if err:
            return per_turn, f"turn {k}: {err}"
        reply = result["reply"]
        history.append(("assistant", reply))
        msgs = serialize_messages(result.get("messages", []))
        per_turn.append({
            "turn_index": k, "global_turn_index": gi, "operator_utterance": turn["utterance"], "model_response": reply,
            "ground_truth": turn.get("ground_truth", {}), "edge_label_to_next": turn.get("edge_label_to_next"),
            "is_injection": turn.get("is_injection", False), "injection_category": turn.get("injection_category"),
            "post_injection": turn.get("post_injection", False), "next_node_id": next_system_node(conv, gi),
            "token_usage": _usage(result), "seconds": seconds, "attempts": attempts + 1,
            "n_tool_calls": sum(len(m.get("tool_calls") or []) for m in msgs), "messages": msgs,
        })
    return per_turn, None


def capture(conv_ids: list[str], convs: dict[str, dict], graphs: dict[str, dict], agent_factory: AgentFactory, *,
            model: str, pattern: str, runs: int, out_dir: str | Path, retries: int = 3, backoff: float = 15.0,
            timeout: float | None = None, provenance: dict | None = None, retry_errors: bool = False,
            min_free_requests: int | None = None, remaining_fn: Callable[[], int | None] | None = None,
            sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print) -> dict:
    counts: dict = {"done": 0, "skipped": 0, "errors": 0}
    for conv_id in conv_ids:
        conv = convs[conv_id]
        for run in range(runs):
            path = run_path(out_dir, model, pattern, conv_id, run)
            prev = read_run_file(path)  # None when missing or cut short by a killed job: run it (again)
            if prev is not None and not (prev.get("error") and retry_errors):
                counts["skipped"] += 1
                continue
            if min_free_requests is not None and remaining_fn is not None:
                left = remaining_fn()
                if left is not None and left < min_free_requests:
                    counts["stopped"] = f"only {left} free requests left today (guard: {min_free_requests})"
                    log(f"[{model} | {pattern}] stopping: {counts['stopped']}")
                    return counts
            graph = graphs[conv["graph_id"]]
            t0 = time.perf_counter()
            per_turn, err = replay_conversation(conv, agent_factory(graph, conv), retries=retries, backoff=backoff,
                                                timeout=timeout, sleep=sleep)
            md = conv.get("metadata", {})
            record = {
                "conversation_id": conv_id, "graph_id": conv["graph_id"], "model": model, "pattern": pattern, "run": run,
                "regime": conv.get("regime", "clean"), "error": err, "wall_seconds": time.perf_counter() - t0,
                "num_turns": len(per_turn), "n_operator_turns": len(operator_turns(conv)), "per_turn": per_turn,
                "metadata": {"depth": md.get("depth"), "num_decisions": md.get("num_decisions"),
                             "terminator_type": md.get("terminator_type"), "true_terminator": md.get("true_terminator", True),
                             "variant": conv.get("variant"), "robustness_axis": conv.get("robustness_axis")},
                "provenance": provenance or {}, "captured_at": datetime.now(timezone.utc).isoformat(),
            }
            write_run_file(path, record)
            counts["errors" if err else "done"] += 1
            log(f"[{model} | {pattern}] {conv_id} run {run}: {'ERROR ' + err if err else f'ok, {len(per_turn)} turns'} "
                f"({record['wall_seconds']:.0f}s)")
    return counts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagflow", required=True, help="path of the cloned DiagFlowBench repository")
    ap.add_argument("--pattern", choices=["baseline", "react", "basic_rag"], required=True,
                    help="baseline = the benchmark's own setup (whole graph in the prompt, no WP4)")
    ap.add_argument("--model", required=True, help="OpenRouter model id")
    ap.add_argument("--pairs", type=int, default=5, help="number of matched clean/mixed pairs")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--min-turns", type=int, default=4, help="shortest clean conversation (operator turns) to sample")
    ap.add_argument("--max-turns", type=int, default=14, help="longest clean conversation to sample (caps requests per run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="runs_diagflow")
    ap.add_argument("--wp4-commit", default="unknown")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--timeout", type=float, default=None, help="seconds allowed per agent call")
    ap.add_argument("--min-free-requests", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="built-in fake agent, no API calls")
    args = ap.parse_args(argv)

    graphs = load_graphs(args.diagflow)
    clean, mixed = load_conversations(args.diagflow, "clean"), load_conversations(args.diagflow, "mixed")
    convs = {**clean, **mixed}
    eligible = [p for p in matched_pairs(clean, mixed) if args.min_turns <= len(operator_turns(clean[p[0]])) <= args.max_turns]
    pairs = select_pairs(eligible, args.pairs, args.seed)
    conv_ids = [cid for pair in pairs for cid in pair]

    if args.dry_run:
        from .fake import make_fake_factory
        factory = make_fake_factory(args.pattern, seed=args.seed)
    elif args.pattern == "baseline":
        from .agents import make_baseline_agent
        factory = lambda graph, conv: make_baseline_agent(args.model, graph, args.temperature)  # noqa: E731
    else:
        try:
            from .diagflow_agents import build_agent  # imports WP4's inobo package
        except ImportError as e:
            raise SystemExit(f"diagflow_agents.py could not be imported ({e}). ReAct and Basic RAG need WP4's inobo package: "
                             "uv sync --extra inobo. --dry-run and --pattern baseline need neither. See harness/diagflow_demo/README.md.")
        factory = lambda graph, conv: build_agent(args.pattern, args.model, graph, temperature=args.temperature)  # noqa: E731

    guard = args.min_free_requests if args.model.endswith(":free") and not args.dry_run else None
    prov = {"wp4_commit": args.wp4_commit, "temperature": args.temperature, "seed": args.seed, "pairs": args.pairs,
            "min_turns": args.min_turns, "max_turns": args.max_turns, "dry_run": args.dry_run,
            "history": "model's own earlier replies"}
    try:
        counts = capture(conv_ids, convs, graphs, factory, model=args.model, pattern=args.pattern, runs=args.runs,
                         out_dir=args.out, retries=args.retries, timeout=args.timeout, provenance=prov,
                         retry_errors=args.retry_errors, min_free_requests=guard,
                         remaining_fn=free_requests_remaining if guard is not None else None)
    except CaptureAborted as e:
        raise SystemExit(f"ABORTED (authentication or credit error, nothing recorded for this run): {e}\n"
                         "Check OPENROUTER_API_KEY in this shell / job, then re-run; finished runs are kept.")
    print(counts)
    if "stopped" in counts:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
