"""The public, WP4-independent agent: the benchmark's own baseline (whole graph in the system prompt).

WP4 patterns (ReAct, Basic RAG) are built in the local, gitignored ``diagflow_agents.py``, which imports
WP4's private package. Every agent has the same shape:

    agent_fn(history) -> {"reply": str, "messages": [...], "usage": {"input": int, "output": int}}

where ``history`` is ``[("user", text), ("assistant", text), ..., ("user", text)]`` ending with the
operator's latest message (earlier assistant entries are the agent's own previous replies).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable

from . import judge_prompts

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _post(url: str, body: dict, key: str, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from OpenRouter: {e.read().decode(errors='replace')[:200]}") from e


def make_baseline_agent(model: str, graph_data: dict[str, Any], temperature: float = 0.0, max_tokens: int = 512,
                        api_key: str | None = None, post: Callable[[str, dict, str], dict] = _post):
    """The benchmark's setup: system prompt with the full graph JSON, no tools, no retrieval."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set.")
    system = judge_prompts.SYSTEM.format(graph_json=json.dumps(graph_data["graph"], indent=2))

    def agent_fn(history: list[tuple[str, str]]) -> dict:
        msgs = [{"role": "system", "content": system}] + [{"role": r, "content": c} for r, c in history]
        body = post(OPENROUTER_URL, {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": msgs}, key)
        choice = body["choices"][0]["message"]
        reply = choice.get("content") or ""
        u = body.get("usage") or {}
        usage = {"input": int(u.get("prompt_tokens", 0)), "output": int(u.get("completion_tokens", 0))}
        return {"reply": reply, "usage": usage,
                "messages": [{"type": "human", "content": history[-1][1]},
                             {"type": "ai", "content": reply, "usage_metadata": {"input_tokens": usage["input"], "output_tokens": usage["output"]}}]}

    return agent_fn
