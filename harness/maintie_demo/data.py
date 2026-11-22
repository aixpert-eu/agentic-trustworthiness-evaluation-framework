"""
MaintIE data loading for the WP3 live demo.

MaintIE (nlp-tlp/maintie, MIT) is a set of short maintenance work-order
texts annotated with entities (a class hierarchy, e.g.
``PhysicalObject/SensingObject/TemperatureSensingObject``) and relations
between them. The demo scores the agents on the 5 top-level classes and
the 6 relation types.

Format notes verified against the released files: ``entities[].end`` is
EXCLUSIVE, ``tokens`` begin with a ``<id>`` placeholder that entities index
into, and relation ``head``/``tail`` are indices into ``entities``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

COARSE_CLASSES = ("PhysicalObject", "State", "Process", "Activity", "Property")
RELATION_TYPES = (
    "hasPart",
    "contains",
    "hasProperty",
    "isA",
    "hasParticipant/hasPatient",
    "hasParticipant/hasAgent",
)
ID_TOKEN = "<id>"


@dataclass(frozen=True)
class Entity:
    text: str
    fine_type: str

    @property
    def coarse(self) -> str:
        return self.fine_type.split("/")[0]


@dataclass(frozen=True)
class Relation:
    head: str  # entity text
    tail: str
    type: str


@dataclass(frozen=True)
class Item:
    item_id: str
    text: str
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]


def clean_text(tokens: list[str]) -> str:
    return " ".join(t for t in tokens if t != ID_TOKEN)


def parse_record(record: dict, item_id: str) -> Item:
    tokens = record["tokens"]
    ents = tuple(Entity(" ".join(tokens[e["start"]:e["end"]]), e["type"]) for e in record["entities"])
    rels = []
    for r in record.get("relations", []):
        rels.append(Relation(ents[r["head"]].text, ents[r["tail"]].text, r["type"]))
    return Item(item_id=item_id, text=clean_text(tokens), entities=ents, relations=tuple(rels))


def load_split(path: str | Path, prefix: str) -> list[Item]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [parse_record(r, f"{prefix}{i:05d}") for i, r in enumerate(records)]


def select_eval_items(gold: list[Item], n: int, seed: int = 0, min_tokens: int = 3, max_tokens: int = 25) -> list[Item]:
    """Deterministic evaluation sample: gold items with at least one entity
    and a moderate length, shuffled with ``seed`` and truncated to ``n``."""
    pool = [
        i for i in gold
        if i.entities and min_tokens <= len(i.text.split()) <= max_tokens
        and all(e.coarse in COARSE_CLASSES for e in i.entities)
        and all(r.type in RELATION_TYPES for r in i.relations)
    ]
    pool.sort(key=lambda i: i.item_id)
    random.Random(seed).shuffle(pool)
    return pool[:n]
