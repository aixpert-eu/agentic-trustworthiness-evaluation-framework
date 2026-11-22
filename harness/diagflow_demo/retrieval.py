"""Knowledge access over one procedure graph (pure python, no extra dependencies).

``GraphIndex`` serves the two WP4 patterns:
    * Basic RAG: ``retrieve(query)`` returns the best-matching steps with the steps that follow them.
    * ReAct tools: ``get_node``, ``next_steps`` and ``find_steps``.
Step text only; edge labels (Yes/No) are shown with each successor so a decision can be followed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class GraphIndex:
    def __init__(self, graph_data: dict[str, Any], k1: float = 1.5, b: float = 0.75):
        g = graph_data["graph"]
        self.nodes: dict[str, dict[str, Any]] = g["nodes"]
        self.succ: dict[str, list[tuple[str, str | None]]] = {}
        for e in g["edges"]:
            self.succ.setdefault(e["source"], []).append((e["target"], e.get("label")))
        self.ids = list(self.nodes)
        self.docs = [Counter(_tok(self.nodes[i].get("text", ""))) for i in self.ids]
        self.lens = [sum(d.values()) or 1 for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        n = len(self.docs)
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 4) -> list[str]:
        scores: Counter = Counter()
        for t in set(_tok(query)):
            if t not in self.idf:
                continue
            for i, d in enumerate(self.docs):
                tf = d.get(t, 0)
                if tf:
                    scores[i] += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return [self.ids[i] for i, _ in scores.most_common(k)]

    def next_steps(self, node_id: str) -> str:
        succ = self.succ.get(node_id)
        if node_id not in self.nodes:
            return "unknown step"
        if not succ:
            return "This is a terminal step: the procedure ends here."
        return "\n".join(f"- {('if ' + lbl) if lbl else 'then'}: {self.nodes[t]['text']}" for t, lbl in succ)

    def describe(self, node_id: str) -> str:
        n = self.nodes.get(node_id)
        if n is None:
            return "unknown step"
        return f"[{node_id}] ({n.get('type', 'process')}) {n.get('text', '')}\nFollowed by:\n{self.next_steps(node_id)}"

    def find_steps(self, query: str, k: int = 4) -> str:
        hits = self.search(query, k)
        return "\n".join(f"[{i}] ({self.nodes[i].get('type', 'process')}) {self.nodes[i]['text']}" for i in hits) or "no matching step"

    def retrieve(self, query: str, k: int = 4) -> str:
        hits = self.search(query, k)
        return "\n\n".join(self.describe(i) for i in hits) or "No matching procedure step found."
