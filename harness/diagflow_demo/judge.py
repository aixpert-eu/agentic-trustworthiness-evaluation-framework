"""Off-procedure outcome labels (LLM judge) with a persistent cache.

Same judge as DiagFlowBench: Claude Haiku 4.5 via OpenRouter at temperature 0, same prompts
(``judge_prompts``), same cache key format, so a cache file produced by the original repository can
be read here and ours can be merged back. Without a key, ``CacheJudge`` serves labels from a cache
only (this is how the published results are re-scored offline).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from . import judge_prompts

JUDGE_MODEL = "anthropic/claude-haiku-4.5"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OUTCOMES = {"CORRECT_ABSTENTION": "correct_abstention", "FORCED_MAPPING": "forced_mapping", "FABRICATION": "fabrication"}


class Missing(LookupError):
    """A label is not in the cache and no live judge is available."""


class Judge(Protocol):
    def injection(self, *, model: str, conv_id: str, turn_index: int, response: str, utterance: str,
                  graph_nodes: dict[str, Any]) -> str: ...

    def post_injection(self, *, model: str, conv_id: str, turn_index: int, response: str, utterance: str,
                       expected_node_text: str) -> bool | None: ...


def cache_key(task: str, model: str, conv_id: str, turn_index: int) -> str:
    return f"{task}|{model}|{conv_id}|{turn_index}"


def graph_summary(graph_nodes: dict[str, Any]) -> str:
    """Every node is listed: the judge prompt calls them the ONLY valid steps."""
    return "\n".join(f"  {nid} ({n.get('type', 'process')}): {n.get('text', '').replace(chr(10), ' ').strip()}"
                     for nid, n in graph_nodes.items())


def parse_outcome(raw: str) -> str:
    """Judge text to a label. Unrecognised output falls back to fabrication, as in the original.

    One deliberate difference: an empty reply is unrecognised too. The original's substring test treats
    the empty string as a match for the first label, which would count it as a correct abstention.
    """
    if not raw.split():
        return "fabrication"
    first = raw.upper().split()[0]
    for label, outcome in OUTCOMES.items():
        if label in first or first in label:
            return outcome
    return "fabrication"


class CacheJudge:
    """Serves labels from a cache dict only. Raises ``Missing`` for an uncached injection turn;
    returns None for an uncached post-injection turn (the original only counts cached ones)."""

    def __init__(self, cache: dict[str, Any]):
        self.cache = cache

    @classmethod
    def from_file(cls, path: str | Path) -> "CacheJudge":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def injection(self, *, model, conv_id, turn_index, response, utterance, graph_nodes) -> str:
        key = cache_key("inj", model, conv_id, turn_index)
        if key not in self.cache:
            raise Missing(key)
        return self.cache[key]["outcome"]

    def post_injection(self, *, model, conv_id, turn_index, response, utterance, expected_node_text) -> bool | None:
        hit = self.cache.get(cache_key("post", model, conv_id, turn_index))
        return None if hit is None else bool(hit["inappropriate_abstention"])


class HaikuJudge(CacheJudge):
    """Cache first, then the live judge. New labels are written to ``cache_path`` as they are made."""

    def __init__(self, cache_path: str | Path, api_key: str | None = None, call=None, retries: int = 6):
        self.path = Path(cache_path)
        super().__init__(json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {})
        self.api_key = api_key or os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        self._call = call or self._openrouter
        self.retries = retries
        self.calls = 0
        self._wait: float | None = None

    def _openrouter(self, prompt: str, max_tokens: int) -> str:
        if not self.api_key:
            raise RuntimeError("No judge key: set JUDGE_API_KEY or OPENROUTER_API_KEY.")
        body = json.dumps({"model": JUDGE_MODEL, "max_tokens": max_tokens, "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(OPENROUTER_URL, data=body, headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(min(60.0, 2.0 * 2 ** (attempt - 1)) if self._wait is None else self._wait)
            self._wait = None
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:  # honour the provider's Retry-After on rate limits
                last = e
                try:
                    self._wait = min(120.0, float(e.headers.get("Retry-After", "")))
                except (TypeError, ValueError, AttributeError):
                    self._wait = None
            except Exception as e:  # network or provider error: retry, then give up
                last = e
        raise RuntimeError(f"judge call failed after {self.retries + 1} attempts: {last}")

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def injection(self, *, model, conv_id, turn_index, response, utterance, graph_nodes) -> str:
        key = cache_key("inj", model, conv_id, turn_index)
        if key not in self.cache:
            prompt = judge_prompts.JUDGE_INJECTION.format(
                graph_summary=graph_summary(graph_nodes), utterance=utterance[:500], response=response[:600])
            raw = self._call(prompt, 20)
            self.calls += 1
            outcome = parse_outcome(raw)
            self.cache[key] = {"outcome": outcome, "fabrication": outcome == "fabrication",
                               "forced_mapping": outcome == "forced_mapping",
                               "correct_abstention": outcome == "correct_abstention", "judge_raw": raw}
            self._save()
        return self.cache[key]["outcome"]

    def post_injection(self, *, model, conv_id, turn_index, response, utterance, expected_node_text) -> bool | None:
        key = cache_key("post", model, conv_id, turn_index)
        if key not in self.cache:
            prompt = judge_prompts.JUDGE_POST_INJECTION.format(
                expected_node_text=expected_node_text[:300], utterance=utterance[:400], response=response[:500])
            raw = self._call(prompt, 5)
            self.calls += 1
            self.cache[key] = {"inappropriate_abstention": raw.upper().startswith("YES"), "judge_raw": raw}
            self._save()
        return bool(self.cache[key]["inappropriate_abstention"])
