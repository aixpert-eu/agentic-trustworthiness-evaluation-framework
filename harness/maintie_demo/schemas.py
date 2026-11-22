"""Output schema handed to the agents. Class and relation names are
constrained with Literal so the agent cannot invent labels (a first live
connectivity run showed an unconstrained model inventing relation types)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

EntityClass = Literal["PhysicalObject", "State", "Process", "Activity", "Property"]
RelationType = Literal[
    "hasPart", "contains", "hasProperty", "isA", "hasParticipant/hasPatient", "hasParticipant/hasAgent"
]


class Entity(BaseModel):
    text: str
    type: EntityClass


class Relation(BaseModel):
    head: str
    tail: str
    type: RelationType


class Extraction(BaseModel):
    entities: list[Entity]
    relations: list[Relation]
