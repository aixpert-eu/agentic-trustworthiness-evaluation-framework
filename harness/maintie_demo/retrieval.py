"""
Knowledge sources for the demo, built from the MaintIE silver set only
(never from the evaluation items, so nothing leaks into the retriever):

    * BM25Index: retrieves similar annotated work orders (Basic RAG).
    * OntologyLookup: majority top-level class of a term (ReAct tool).

Pure python, no extra dependencies.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .data import Item
from .scoring import norm

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def format_example(item: Item) -> str:
    ents = "; ".join(f"{e.text} [{e.coarse}]" for e in item.entities)
    rels = "; ".join(f"{r.head} -{r.type}-> {r.tail}" for r in item.relations) or "none"
    return f"Text: {item.text}\nEntities: {ents}\nRelations: {rels}"


class BM25Index:
    def __init__(self, items: list[Item], exclude_texts: set[str] | None = None, k1: float = 1.5, b: float = 0.75):
        excluded = {norm(t) for t in (exclude_texts or set())}
        self.items = [i for i in items if norm(i.text) not in excluded]
        self.k1, self.b = k1, b
        self.docs = [Counter(_tok(i.text)) for i in self.items]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[int]] = defaultdict(list)
        for idx, d in enumerate(self.docs):
            for t in d:
                self.postings[t].append(idx)

    def search(self, query: str, k: int = 3) -> list[Item]:
        scores: Counter = Counter()
        for t in set(_tok(query)):
            for idx in self.postings.get(t, ()):
                tf = self.docs[idx][t]
                denom = tf + self.k1 * (1 - self.b + self.b * self.lens[idx] / self.avg)
                scores[idx] += self.idf[t] * tf * (self.k1 + 1) / denom
        return [self.items[i] for i, _ in scores.most_common(k)]

    def retrieve(self, query: str, k: int = 3) -> str:
        hits = self.search(query, k)
        if not hits:
            return "No similar annotated work orders found."
        return "\n\n".join(format_example(h) for h in hits)


class OntologyLookup:
    """term -> top-level class distribution observed in the silver set."""

    def __init__(self, items: list[Item], exclude_texts: set[str] | None = None):
        excluded = {norm(t) for t in (exclude_texts or set())}
        self.by_term: dict[str, Counter] = defaultdict(Counter)
        for it in items:
            if norm(it.text) in excluded:
                continue
            for e in it.entities:
                self.by_term[norm(e.text)][e.coarse] += 1

    def lookup(self, term: str) -> str:
        counts = self.by_term.get(norm(term))
        if not counts:
            return "unknown"
        total = sum(counts.values())
        return ", ".join(f"{c} ({n / total:.0%})" for c, n in counts.most_common(3))
