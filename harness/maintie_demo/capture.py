"""
Capture raw agent runs to disk, one JSON file per (model, pattern, item, run).

Capture is kept separate from evaluation on purpose: live runs cost money,
time and rate-limit budget, so they are written once and every metric is then
computed offline from the saved files (re-scoring never calls a model).
Existing successful run files are skipped, so an interrupted capture resumes.

The agents themselves come from ``wp4_agents.py``, which imports WP4's
``inobo`` package (``uv sync --extra inobo``; see README). Use ``--dry-run`` to
exercise the whole pipeline with a built-in fake agent and no API calls, and
without ``inobo``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import threading
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from harness.runfiles import read_run_file, write_run_file

from .data import Item, load_split, select_eval_items

AgentFn = Callable[[str], dict]


class CaptureAborted(RuntimeError):
    """Raised on errors that no retry can fix (bad or missing API key, no credits).

    Nothing is written for the run that triggered it, so a later capture resumes
    exactly there instead of finding a pile of error files.
    """


class RunTimeout(BaseException):
    """One agent attempt exceeded ``--timeout``. A BaseException so that library code
    with a blanket ``except Exception`` cannot swallow the alarm."""


_FATAL = re.compile(r"unauthorized|user not found|invalid api key|no auth credentials|authenticationerror|"
                    r"payment required|insufficient credits|\b40[12]\b", re.IGNORECASE)


def is_fatal_error(message: str) -> bool:
    """True for authentication / credit errors, which will fail identically for every run."""
    return bool(_FATAL.search(message))


@contextmanager
def deadline(seconds: float | None):
    """Interrupt the enclosed call after ``seconds`` (POSIX main thread only; otherwise a no-op)."""
    usable = (seconds and seconds > 0 and hasattr(signal, "setitimer")
              and threading.current_thread() is threading.main_thread())
    if not usable:
        yield
        return

    def _raise(signum, frame):
        raise RunTimeout(f"run exceeded {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def free_requests_remaining(api_key: str | None = None, fetch: Callable[[str, dict], dict] | None = None) -> int | None:
    """Remaining daily requests for free models on this OpenRouter key, or None if unknown."""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return None

    def _http(url: str, headers: dict) -> dict:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))

    try:
        body = (fetch or _http)("https://openrouter.ai/api/v1/key", {"Authorization": f"Bearer {api_key}"})
        info = (body.get("data", body) or {}).get("free_model_daily_requests") or {}
        remaining = info.get("remaining")
        return int(remaining) if remaining is not None else None
    except Exception:  # an unreachable status endpoint must not block a run
        return None


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


def _call_with_retries(fn: AgentFn, text: str, retries: int, backoff: float, sleep: Callable,
                       timeout: float | None = None) -> tuple[dict | None, str | None, int]:
    err = None
    for attempt in range(retries + 1):
        try:
            with deadline(timeout):
                return fn(text), None, attempt
        except RunTimeout as e:  # a run that is too slow is data; retrying would only double the wait
            return None, f"RunTimeout: {e}", attempt
        except Exception as e:  # provider/rate-limit/validation errors are data, not crashes
            err = f"{type(e).__name__}: {e}"
            if is_fatal_error(err):
                raise CaptureAborted(err) from e
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
    timeout: float | None = None,
    min_free_requests: int | None = None,
    remaining_fn: Callable[[], int | None] | None = None,
    check_every: int = 10,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> dict:
    """Capture every missing (item, run). Returns counts; adds ``stopped`` when the request guard ended it early.

    Raises ``CaptureAborted`` on authentication or credit errors. With ``min_free_requests`` and a
    ``remaining_fn``, stops cleanly (nothing half-written) when the remaining daily request budget,
    re-checked every ``check_every`` live runs, falls below that number.
    """
    counts: dict = {"done": 0, "skipped": 0, "errors": 0}
    live = 0
    for item in items:
        for run in range(runs):
            path = run_path(Path(out_dir), model, pattern, item.item_id, run)
            prev = read_run_file(path)  # None when missing or cut short by a killed job: run it (again)
            if prev is not None and not (prev.get("error") and retry_errors):
                counts["skipped"] += 1
                continue
            if min_free_requests is not None and remaining_fn is not None and live % check_every == 0:
                left = remaining_fn()
                if left is not None and left < min_free_requests:
                    counts["stopped"] = f"only {left} free requests left today (guard: {min_free_requests})"
                    log(f"[{model} | {pattern}] stopping: {counts['stopped']}")
                    return counts
            live += 1
            t0 = time.perf_counter()
            result, err, attempts = _call_with_retries(agent_fn, item.text, retries, backoff, sleep, timeout)
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
            write_run_file(path, record)
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
    ap.add_argument("--timeout", type=float, default=None,
                    help="seconds allowed per attempt; a slower run is recorded as an error (RunTimeout) and not retried")
    ap.add_argument("--min-free-requests", type=int, default=None,
                    help="stop cleanly when fewer than N free-model requests remain today (OpenRouter /api/v1/key; ':free' models only)")
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
            from .wp4_agents import build_agent  # imports WP4's inobo package
        except ImportError as e:
            raise SystemExit(f"wp4_agents.py could not be imported ({e}). ReAct and Basic RAG need WP4's inobo package: "
                             "uv sync --extra inobo. --dry-run needs neither. See harness/maintie_demo/README.md.")
        agent_fn = build_agent(args.pattern, args.model, silver=silver, exclude_texts={i.text for i in items},
                               temperature=args.temperature)

    prov = {"wp4_commit": args.wp4_commit, "temperature": args.temperature, "seed": args.seed,
            "n_items": len(items), "dry_run": args.dry_run}
    guard = args.min_free_requests if args.model.endswith(":free") and not args.dry_run else None
    try:
        counts = capture(items, agent_fn, model=args.model, pattern=args.pattern, runs=args.runs, out_dir=args.out,
                         retries=args.retries, provenance=prov, retry_errors=args.retry_errors, timeout=args.timeout,
                         min_free_requests=guard, remaining_fn=free_requests_remaining if guard is not None else None)
    except CaptureAborted as e:
        raise SystemExit(f"ABORTED (authentication or credit error, nothing recorded for this run): {e}\n"
                         "Check OPENROUTER_API_KEY in this shell / job, then re-run; finished runs are kept.")
    print(counts)
    if "stopped" in counts:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
