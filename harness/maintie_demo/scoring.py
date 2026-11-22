"""
Scoring of a MaintIE extraction against gold, and the TaskDefinition
builder that connects it to the WP3 metrics.

Matching is on normalised text, not character offsets: an entity matches
when (normalised span text, top-level class) equals a gold entity; a
relation matches when (head text, tail text, relation type) does. Counts are
multiset-aware so a text with two identical gold entities needs two
predictions.

Mapping onto existing WP3 metrics:
    task_success      oracle_fn: every gold entity found with the right class
    subgoal_progress  one sub-goal per gold entity and per gold relation
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from contracts.task_definition import TaskDefinition

from .data import COARSE_CLASSES, RELATION_TYPES, Item


def norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _json_from_text(text: str) -> Any | None:
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE.findall(text)]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except (ValueError, TypeError):
            continue
    return None


def parse_prediction(output: Any) -> dict | None:
    """Return {"entities": [...], "relations": [...]} (validated shape and
    labels) or None when the output is missing or violates the schema."""
    if output is None:
        return None
    if hasattr(output, "model_dump"):
        output = output.model_dump()
    if isinstance(output, str):
        output = _json_from_text(output)
    if not isinstance(output, dict):
        return None
    ents, rels = output.get("entities"), output.get("relations", [])
    if not isinstance(ents, list) or not isinstance(rels, list):
        return None
    try:
        e_out = [{"text": str(e["text"]), "type": e["type"]} for e in ents]
        r_out = [{"head": str(r["head"]), "tail": str(r["tail"]), "type": r["type"]} for r in rels]
    except (KeyError, TypeError):
        return None
    if any(e["type"] not in COARSE_CLASSES for e in e_out) or any(r["type"] not in RELATION_TYPES for r in r_out):
        return None
    return {"entities": e_out, "relations": r_out}


def _entity_keys(pred: dict | None) -> Counter:
    return Counter((norm(e["text"]), e["type"]) for e in (pred or {}).get("entities", []))


def _relation_keys(pred: dict | None) -> Counter:
    return Counter((norm(r["head"]), norm(r["tail"]), r["type"]) for r in (pred or {}).get("relations", []))


def gold_entity_keys(item: Item) -> Counter:
    return Counter((norm(e.text), e.coarse) for e in item.entities)


def gold_relation_keys(item: Item) -> Counter:
    return Counter((norm(r.head), norm(r.tail), r.type) for r in item.relations)


def prf(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "n_pred": n_pred, "n_gold": n_gold, "precision": p, "recall": r, "f1": f}


def score_prediction(item: Item, output: Any) -> dict:
    """Task-level counts and PRF for one run. Invalid output scores zero but
    keeps the gold counts so micro averages stay correct."""
    pred = parse_prediction(output)
    out = {"valid": pred is not None}
    for name, p_keys, g_keys in (
        ("entity", _entity_keys(pred), gold_entity_keys(item)),
        ("relation", _relation_keys(pred), gold_relation_keys(item)),
    ):
        tp = sum((p_keys & g_keys).values())
        out[name] = prf(tp, sum(p_keys.values()), sum(g_keys.values()))
    return out


def _has_entity(key: tuple[str, str]):
    def check(output: Any) -> bool:
        return _entity_keys(parse_prediction(output))[key] >= 1
    return check


def _has_relation(key: tuple[str, str, str]):
    def check(output: Any) -> bool:
        return _relation_keys(parse_prediction(output))[key] >= 1
    return check


def build_task(item: Item, optimal_steps: int | None = None) -> TaskDefinition:
    """TaskDefinition for one MaintIE item (gold labels for the WP3 metrics)."""
    g_ent = gold_entity_keys(item)

    def oracle(output: Any) -> bool:
        pred = _entity_keys(parse_prediction(output))
        return all(pred[k] >= n for k, n in g_ent.items())

    checks = [_has_entity(k) for k in g_ent] + [_has_relation(k) for k in gold_relation_keys(item)]
    return TaskDefinition(
        task_id=item.item_id,
        input_query=item.text,
        oracle_fn=oracle,
        subgoal_checks=checks,
        optimal_steps=optimal_steps,
        metadata={"n_gold_entities": sum(g_ent.values()), "n_gold_relations": len(item.relations)},
    )
