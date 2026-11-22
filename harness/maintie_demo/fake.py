"""Deterministic fake agents for tests and ``--dry-run``. They mimic the
result shapes confirmed for WP4 main: ReactAgent -> messages +
structured_response; BasicRAG -> messages + context (JSON answer as text)."""

from __future__ import annotations

import json
import random

from .data import Item
from .scoring import norm


def _gold_json(item: Item, rng: random.Random, drop: float, wrong_class: float) -> dict:
    ents = []
    for e in item.entities:
        if rng.random() < drop:
            continue
        t = e.coarse
        if rng.random() < wrong_class:
            t = "State" if t != "State" else "Process"
        ents.append({"text": e.text, "type": t})
    kept = {norm(e["text"]) for e in ents}
    rels = [{"head": r.head, "tail": r.tail, "type": r.type} for r in item.relations
            if norm(r.head) in kept and norm(r.tail) in kept and rng.random() > drop]
    return {"entities": ents, "relations": rels}


def make_fake_agent(pattern: str, items: list[Item], seed: int = 0, drop: float = 0.25, wrong_class: float = 0.1):
    by_text = {i.text: i for i in items}
    rng = random.Random(seed)

    def agent(text: str) -> dict:
        item = by_text[text]
        out = _gold_json(item, rng, drop, wrong_class)
        usage = {"input_tokens": 300, "output_tokens": 120, "total_tokens": 420}
        if pattern == "react":
            msgs = [{"type": "human", "content": text}]
            for i, e in enumerate(item.entities):
                msgs.append({"type": "ai", "content": "", "usage_metadata": usage,
                             "tool_calls": [{"name": "ontology_lookup", "args": {"term": e.text}, "id": f"c{i}"}]})
                msgs.append({"type": "tool", "content": e.coarse, "tool_call_id": f"c{i}", "name": "ontology_lookup"})
            msgs.append({"type": "ai", "content": "", "usage_metadata": usage,
                         "tool_calls": [{"name": "finish_task", "args": out, "id": "cf"}]})
            msgs.append({"type": "tool", "content": "Task completed.", "tool_call_id": "cf", "name": "finish_task"})
            return {"messages": msgs, "structured_response": out}
        msgs = [
            {"type": "human", "content": text},
            {"type": "ai", "content": "", "usage_metadata": usage,
             "tool_calls": [{"name": "retrieve_examples", "args": {"query": text}, "id": "q0"}]},
            {"type": "tool", "content": "Text: example", "tool_call_id": "q0", "name": "retrieve_examples"},
            {"type": "ai", "content": json.dumps(out), "usage_metadata": usage},
        ]
        return {"messages": msgs, "context": ["Text: example"]}

    return agent
