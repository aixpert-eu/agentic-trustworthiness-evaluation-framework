"""
Capture raw agent runs to disk, one JSON file per (model, pattern, item, run).

Capture is kept separate from evaluation on purpose: live runs cost money,
time and rate-limit budget, so they are written once and every metric is then
computed offline from the saved files (re-scoring never calls a model).
Existing successful run files are skipped, so an interrupted capture resumes.

The agents themselves come from ``wp4_agents.py`` (local file that imports
WP4's private package; gitignored, see README). Use ``--dry-run`` to exercise
the whole pipeline with a built-in fake agent and no API calls.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .data import Item, load_split, select_eval_items

AgentFn = Callable[[str], dict]


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _g(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def serialize_messages(messages: list[Any]) -> list[dict]:
    """LangChain message objects or dicts -> plain JSON-able dicts."""
    out = []
    for m in messages:
        d = {"type": _g(m, "type", "unknown"), "content": _g(m, "content", "")}
        if not isinstance(d["content"], str):
            d["content"] = json.dumps(d["content"], default=str)
        for key in ("tool_calls", "tool_call_id", "name", "usage_metadata"):
            val = _g(m, key)
            if val:
                d[key] = json.loads(json.dumps(val, default=str))
        out.append(d)
    return out


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    return json.loads(json.dumps(obj, default=str))


def run_path(out_dir: Path, model: str, pattern: str, item_id: str, run: int) -> Path:
    return Path(out_dir) / slug(model) / pattern / f"{item_id}__r{run}.json"


def _call_with_retries(fn: AgentFn, text: str, retries: int, backoff: float, sleep: Callable) -> tuple[dict | None, str | None, int]:
    err = None
    for attempt in range(retries + 1):
        try:
            return fn(text), None, attempt
        except Exception as e:  # provider/rate-limit/validation errors are data, not crashes
            err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                sleep(backoff * (attempt + 1))
    return None, err, retries


def capture(
    items: list[Item],
    agent_fn: AgentFn,
    *,
    model: str,
    pattern: str,
    runs: int,
    out_dir: str | Path,
    retries: int = 3,
    backoff: float = 15.0,
    provenance: dict | None = None,
    retry_errors: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> dict:
    counts = {"done": 0, "skipped": 0, "errors": 0}
    for item in items:
        for run in range(runs):
            path = run_path(Path(out_dir), model, pattern, item.item_id, run)
            if path.exists():
                prev = json.loads(path.read_text(encoding="utf-8"))
                if not (prev.get("error") and retry_errors):
                    counts["skipped"] += 1
                    continue
            t0 = time.perf_counter()
            result, err, attempts = _call_with_retries(agent_fn, item.text, retries, backoff, sleep)
            elapsed = time.perf_counter() - t0
            record = {
                "item_id": item.item_id, "run": run, "model": model, "pattern": pattern,
                "input": item.text, "wall_seconds": elapsed, "attempts": attempts + 1,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "provenance": provenance or {}, "error": err,
                "messages": serialize_messages(result["messages"]) if result else [],
                "structured_response": to_jsonable(result.get("structured_response")) if result else None,
                "context": to_jsonable(result.get("context")) if result else None,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1), encoding="utf-8")
            counts["errors" if err else "done"] += 1
            log(f"[{model} | {pattern}] {item.item_id} run {run}: {'ERROR ' + err if err else 'ok'} ({elapsed:.1f}s)")
    return counts


def build_eval_items(gold_path: str, n: int, seed: int) -> list[Item]:
    return select_eval_items(load_split(gold_path, "g"), n=n, seed=seed)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maintie", required=True, help="path to the cloned nlp-tlp/maintie repo")
    ap.add_argument("--pattern", choices=["react", "basic_rag"], required=True)
    ap.add_argument("--model", required=True, help="OpenRouter model id, e.g. google/gemma-4-31b-it:free")
    ap.add_argument("--n", type=int, default=50, help="number of gold work orders")
    ap.add_argument("--runs", type=int, default=3, help="independent runs per work order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--wp4-commit", default="unknown", help="WP4 commit hash, recorded in every run file")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-errors", action="store_true", help="re-run files that previously ended in an error")
    ap.add_argument("--dry-run", action="store_true", help="use the built-in fake agent (no API calls)")
    args = ap.parse_args(argv)

    root = Path(args.maintie) / "data"
    items = build_eval_items(str(root / "gold_release.json"), args.n, args.seed)
    silver = load_split(root / "silver_release.json", "s")

    if args.dry_run:
        from .fake import make_fake_agent
        agent_fn = make_fake_agent(args.pattern, items, seed=args.seed)
    else:
        try:
            from .wp4_agents import build_agent  # local, gitignored: imports WP4's private package
        except ImportError as e:
            raise SystemExit(f"wp4_agents.py not found or failing to import ({e}). See harness/maintie_demo/README.md.")
        agent_fn = build_agent(args.pattern, args.model, silver=silver, exclude_texts={i.text for i in items},
                               temperature=args.temperature)

    prov = {"wp4_commit": args.wp4_commit, "temperature": args.temperature, "seed": args.seed,
            "n_items": len(items), "dry_run": args.dry_run}
    counts = capture(items, agent_fn, model=args.model, pattern=args.pattern, runs=args.runs, out_dir=args.out,
                     retries=args.retries, provenance=prov, retry_errors=args.retry_errors)
    print(counts)


if __name__ == "__main__":
    main()
