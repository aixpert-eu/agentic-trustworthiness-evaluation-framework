"""Loading DiagFlowBench graphs and conversations, and picking matched clean/mixed pairs."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

DATASET_DIR = "DiagFlowBench_Dataset"


def dataset_root(repo: str | Path) -> Path:
    root = Path(repo) / DATASET_DIR
    if not root.is_dir():
        raise SystemExit(f"{root} not found. Pass the path of the cloned DiagFlowBench repository.")
    return root


def load_graphs(repo: str | Path) -> dict[str, dict[str, Any]]:
    graphs = {}
    for p in sorted((dataset_root(repo) / "graphs" / "json").glob("GRAPH*.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        graphs[g["id"]] = g
    return graphs


def load_conversations(repo: str | Path, regime: str) -> dict[str, dict[str, Any]]:
    """regime is 'clean' or 'mixed'."""
    out = {}
    for p in sorted((dataset_root(repo) / "conversations" / regime).glob("*.json")):
        c = json.loads(p.read_text(encoding="utf-8"))
        out[c["conversation_id"]] = c
    return out


def matched_pairs(clean: dict[str, Any], mixed: dict[str, Any]) -> list[tuple[str, str]]:
    """(clean_id, mixed_id) for every clean conversation that has its mixed counterpart."""
    return [(cid, cid + "_mixed") for cid in sorted(clean) if cid + "_mixed" in mixed]


def select_pairs(pairs: list[tuple[str, str]], n: int, seed: int = 0) -> list[tuple[str, str]]:
    """Deterministic sample of matched pairs; the first n for a seed are a prefix of the first m > n."""
    pool = sorted(pairs)
    random.Random(seed).shuffle(pool)
    return pool[:n]


def operator_turns(conv: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    """(global turn index, turn) for every operator turn, in order."""
    return [(i, t) for i, t in enumerate(conv["turns"]) if t.get("speaker") == "operator"]


def next_system_node(conv: dict[str, Any], global_idx: int) -> str | None:
    """Node of the next scripted system turn: the step a correct reply to the operator turn names."""
    for t in conv["turns"][global_idx + 1:]:
        if t.get("speaker") == "system":
            return t.get("node_id")
    return None
