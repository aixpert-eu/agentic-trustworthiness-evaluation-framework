"""Small statistics helpers for the demo evaluations.

Repeated runs of the same work order or conversation are not independent samples, so the unit that is
resampled is the item, never the run. ``cluster_bootstrap`` draws items with replacement, recomputes the
statistic on each draw and returns the percentile interval. With few items (the demos use 10 to 50) read the
interval as an indication of how much the number could move, not as an exact coverage guarantee.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

N_BOOT = 1000
LEVEL = 0.95
SEED = 0


def cluster_bootstrap_many(clusters: Sequence[T], stat: Callable[[list[T]], dict[str, float | None]], *, n_boot: int = N_BOOT,
                           level: float = LEVEL, seed: int = SEED) -> dict[str, tuple[float, float] | None]:
    """Percentile bootstrap intervals for several statistics from the same resamples of ``clusters``.

    ``stat`` returns ``{name: value or None}`` for one resample. A name whose value is undefined on most
    resamples (or when there are fewer than two clusters) maps to None. The seed is fixed, so the same data
    always gives the same intervals.
    """
    n = len(clusters)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {}
    if n >= 2 and n_boot >= 2:
        for _ in range(n_boot):
            for name, v in stat([clusters[rng.randrange(n)] for _ in range(n)]).items():
                if v is not None:
                    draws.setdefault(name, []).append(v)
    tail = (1.0 - level) / 2.0
    out: dict[str, tuple[float, float] | None] = {}
    for name, xs in draws.items():
        if len(xs) < max(2, n_boot // 2):
            out[name] = None
            continue
        xs.sort()
        out[name] = (xs[int(tail * len(xs))], xs[min(len(xs) - 1, int((1.0 - tail) * len(xs)))])
    return out


def cluster_bootstrap(clusters: Sequence[T], stat: Callable[[list[T]], float | None], *, n_boot: int = N_BOOT,
                      level: float = LEVEL, seed: int = SEED) -> tuple[float, float] | None:
    """Percentile bootstrap interval of one statistic over ``clusters`` (one cluster = one item with all its runs).

    Returns None when there are fewer than two clusters, or when ``stat`` is undefined on most draws.
    """
    return cluster_bootstrap_many(clusters, lambda sample: {"x": stat(sample)}, n_boot=n_boot, level=level, seed=seed).get("x")


def pooled_mean(items: Sequence[Sequence[float | None]]) -> float | None:
    """Mean of every value across items (None values skipped); None when there is nothing to average."""
    xs = [x for item in items for x in item if x is not None]
    return sum(xs) / len(xs) if xs else None


def mean_ci(values_by_item: Sequence[Sequence[float | None]], **kw) -> list[float] | None:
    """Interval of the pooled mean, resampling items. Returns ``[lo, hi]`` (JSON-friendly) or None."""
    ci = cluster_bootstrap(list(values_by_item), pooled_mean, **kw)
    return None if ci is None else [ci[0], ci[1]]
