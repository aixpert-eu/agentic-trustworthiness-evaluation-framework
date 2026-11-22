"""
Scoring of a MaintIE extraction against gold, and the TaskDefinition
builder that connects it to the WP3 metrics.

Matching is on normalised text, not character offsets: an entity matches
when (normalised span text, top-level class) equals a gold entity; a
relation matches when (head text, tail text, relation type) does. Counts are
multiset-aware so a text with two identical gold entities needs two
predictions.

Mapping onto existing WP3 metrics (scoring is lenient: out-of-scheme items
are dropped and counted, see parse_prediction):
    task_success      oracle_fn: every gold entity found with the right class
    subgoal_progress  one sub-goal per gold entity and per gold relation
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from contracts.task_definition import TaskDefinition

from .data import COARSE_CLASSES, RELATION_TYPES, Item, canonical_relation


def norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _json_from_text(text: str) -> Any | None:
    """Extract the answer object from model text. Small reasoning models often
    think out loud before (and sometimes draft JSON inside) the answer, so
    take the LAST JSON object that has an "entities" key; fall back to the
    whole text parsed as JSON."""
    try:
        return json.loads(text.strip())
    except (ValueError, TypeError):
        pass
    decoder, found = json.JSONDecoder(), []
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and "entities" in obj:
            found.append(obj)
    return found[-1] if found else None


def parse_prediction(output: Any) -> dict | None:
    """Return {"entities", "relations", "n_invalid"} or None when no answer
    object can be recovered at all.

    Parsing is lenient on purpose: an entity or relation with a label outside
    the scheme (or a malformed item) is dropped and counted in ``n_invalid``,
    instead of zeroing the whole run. Strict schema conformance is reported
    separately (n_invalid == 0). The silver-style short relation names
    (hasPatient, hasAgent) are accepted as aliases of the gold names."""
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
    e_out, r_out, bad = [], [], 0
    for e in ents:
        if isinstance(e, dict) and "text" in e and e.get("type") in COARSE_CLASSES:
            e_out.append({"text": str(e["text"]), "type": e["type"]})
        else:
            bad += 1
    for r in rels:
        rtype = canonical_relation(r.get("type")) if isinstance(r, dict) else None
        if isinstance(r, dict) and "head" in r and "tail" in r and rtype in RELATION_TYPES:
            r_out.append({"head": str(r["head"]), "tail": str(r["tail"]), "type": rtype})
        else:
            bad += 1
    return {"entities": e_out, "relations": r_out, "n_invalid": bad}


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
    out = {"valid": pred is not None and pred["n_invalid"] == 0}
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
