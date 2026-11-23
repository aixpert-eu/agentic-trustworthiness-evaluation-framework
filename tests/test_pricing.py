"""An unknown price is not a price of zero."""

from harness.pricing import resolve_price


def test_table_entry_wins_even_for_a_free_id():
    assert resolve_price("m/a", {"m/a": [0.001, 0.002]}) == ((0.001, 0.002), "table")
    assert resolve_price("m/a:free", {"m/a:free": (0.5, 1.0)}) == ((0.5, 1.0), "table")


def test_free_suffix_costs_nothing_without_an_entry():
    assert resolve_price("m/a:free", {}) == ((0.0, 0.0), "free")
    assert resolve_price("m/a:free", None) == ((0.0, 0.0), "free")


def test_any_other_model_without_an_entry_is_not_priced():
    assert resolve_price("m/paid", {}) == (None, "none")
    assert resolve_price("m/paid", {"m/other": (0.1, 0.2)}) == (None, "none")
    assert resolve_price("m/free-ish", None) == (None, "none")  # the suffix must be exactly ":free"
