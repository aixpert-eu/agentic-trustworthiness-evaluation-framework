"""Resolving a free-text assistant reply to a node of a procedure graph.

A reply is matched to the graph node whose text has the highest word-level
Jaccard similarity; below the threshold it resolves to no node. This is the
matching used by DiagFlowBench (threshold 0.05 calibrated there), kept
identical so numbers are comparable with the published ones.

A graph is the DiagFlowBench shape::

    {"graph": {"nodes": {"box_1": {"type": "process", "text": "..."}, ...},
               "edges": [{"source": "box_1", "target": "box_2", "label": None}, ...]}}
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

DEFAULT_THRESHOLD = 0.05


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


@lru_cache(maxsize=None)
def _words(text: str) -> frozenset[str]:
    return frozenset(normalize(text).split())


def jaccard(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def nodes_of(graph_data: dict[str, Any]) -> dict[str, Any]:
    return graph_data["graph"]["nodes"]


def adjacency(graph_data: dict[str, Any]) -> dict[str, list[tuple[str, str | None]]]:
    adj: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for e in graph_data["graph"]["edges"]:
        adj[e["source"]].append((e["target"], e.get("label")))
    return adj


def best_node(reply: str, nodes: dict[str, Any]) -> tuple[str | None, float]:
    """Highest-similarity node and its score, with no threshold applied."""
    best, best_score = None, 0.0
    for nid, node in nodes.items():
        s = jaccard(reply, node.get("text", ""))
        if s > best_score:
            best, best_score = nid, s
    return best, round(best_score, 4)


def match_node(reply: str, nodes: dict[str, Any], threshold: float = DEFAULT_THRESHOLD) -> tuple[str | None, float]:
    """Node the reply names, or (None, 0.0) when nothing reaches the threshold."""
    nid, score = best_node(reply, nodes)
    return (None, 0.0) if score < threshold else (nid, score)


def forward_reachable(node_id: str, adj: dict[str, list[tuple[str, str | None]]], max_hops: int = 2) -> set[str]:
    reachable: set[str] = set()
    frontier = {node_id}
    for _ in range(max_hops):
        nxt = set()
        for n in frontier:
            for target, _label in adj.get(n, []):
                if target not in reachable:
                    nxt.add(target)
                    reachable.add(target)
        frontier = nxt
    return reachable
