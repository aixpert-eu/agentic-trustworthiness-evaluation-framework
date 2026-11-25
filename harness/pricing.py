"""Which price applies to a model, and where it came from.

Cost is only a measurement when the price is known. A model id ending in ``:free`` costs nothing by
definition. Any other model needs an entry in the price table given at evaluation time; without one its
cost is unknown, and an unknown cost must not be scored as a cost of zero (it would come out as
"Excellent" and count as a measured Efficiency metric).
"""

from __future__ import annotations

from typing import Mapping, Sequence

FREE_SUFFIX = ":free"


def resolve_price(model: str, prices: Mapping[str, Sequence[float]] | None) -> tuple[tuple[float, float] | None, str]:
    """Return ``(price, basis)`` for ``model``.

    ``price`` is ``(usd_per_1k_input, usd_per_1k_output)`` or ``None`` when it is unknown. ``basis`` says why:
    ``"table"`` (an entry in ``prices``, which also wins for a ``:free`` id), ``"free"`` (a ``:free`` id with no
    entry, so zero) or ``"none"`` (no entry and not free, so the cost is not measured).
    """
    entry = (prices or {}).get(model)
    if entry is not None:
        price_in, price_out = entry
        return (float(price_in), float(price_out)), "table"
    if model.endswith(FREE_SUFFIX):
        return (0.0, 0.0), "free"
    return None, "none"
