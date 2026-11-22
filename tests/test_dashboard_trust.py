"""The trustworthiness-by-dimension section shared by the demo dashboards: taxonomy, selections, embedded data."""

from __future__ import annotations

import json
import re

import pytest

from contracts.registry import REGISTRY
from harness import dashboard_trust as trust
from harness.diagflow_demo import dashboard as diagflow_dashboard
from harness.maintie_demo import dashboard as maintie_dashboard

DASHBOARDS = {"diagflow": diagflow_dashboard, "maintie": maintie_dashboard}
EMPTY = {"meta": {"dataset": "d", "generated_at": "2026-01-01T00:00:00", "notes": []}, "configs": []}


def _data(html: str) -> dict:
    raw = html.split('id="data" type="application/json">')[1].split("</script>")[0]
    return json.loads(raw.replace("<\\/", "</"))


def test_every_registry_dimension_is_a_framework_dimension():
    used = {trust.canonical_dimension(d) for spec in REGISTRY for d in spec.dimensions}
    assert used <= set(trust.DIMENSIONS), f"unknown dimension tags: {sorted(used - set(trust.DIMENSIONS))}"


def test_taxonomy_counts_match_the_registry():
    tax = trust.taxonomy()
    assert [d["name"] for d in tax["dimensions"]] == list(trust.DIMENSIONS)
    assert set(tax["registry"]) == {spec.metric_id for spec in REGISTRY}
    total_tags = sum(len(set(trust.canonical_dimension(d) for d in spec.dimensions)) for spec in REGISTRY)
    assert sum(d["registered"] for d in tax["dimensions"]) == total_tags
    # a taxonomy dimension can exist without a registered metric (the page shows that as a gap)
    assert {d["name"]: d["registered"] for d in tax["dimensions"]}["Accountability"] == 0


def test_the_legacy_transparency_tag_lands_in_the_framework_dimension():
    assert trust.taxonomy()["registry"]["log_completeness"]["dims"][0] == "Transparency & Explainability"


@pytest.mark.parametrize("name", DASHBOARDS)
def test_metrics_shown_by_each_dashboard_sit_in_their_registry_dimension(name):
    """The page lists `{id: '...', dim: '...'}` per metric; the dimension must be the registry's own."""
    pairs = re.findall(r"id: '([a-z_]+)', dim: '([^']+)'", DASHBOARDS[name].TEMPLATE)
    assert pairs, "no trust metrics found in the template"
    registry = trust.taxonomy()["registry"]
    for metric_id, dim in pairs:
        assert metric_id in registry, f"{metric_id} is not a registered metric"
        assert dim == registry[metric_id]["dims"][0], f"{metric_id}: page says {dim}, registry says {registry[metric_id]['dims'][0]}"


def test_load_selection_accepts_the_catalogue_export_a_list_and_plain_text(tmp_path):
    export = tmp_path / "uc3.json"
    export.write_text(json.dumps({"schema": trust.SELECTION_SCHEMA, "label": "UC3", "metrics": ["latency", "pass_hat_k_agent", "latency"]}))
    assert trust.load_selection(export) == {"label": "UC3", "metrics": ["latency", "pass_hat_k_agent"]}

    as_list = tmp_path / "ids.json"
    as_list.write_text('["latency", " cost_per_query "]')
    assert trust.load_selection(as_list) == {"label": "", "metrics": ["latency", "cost_per_query"]}

    as_text = tmp_path / "ids.txt"
    as_text.write_text("latency, cost_per_query\nstep_efficiency;pass_hat_k_agent")
    assert trust.load_selection(as_text)["metrics"] == ["latency", "cost_per_query", "step_efficiency", "pass_hat_k_agent"]

    assert trust.load_selection(None) is None
    assert trust.load_selection({"label": "x", "metrics": ["latency"]}) == {"label": "x", "metrics": ["latency"]}


def test_load_selection_rejects_empty_or_malformed_input(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text('{"metrics": []}')
    with pytest.raises(ValueError):
        trust.load_selection(empty)
    with pytest.raises(ValueError):
        trust.load_selection({"label": "no metrics key"})


@pytest.mark.parametrize("name", DASHBOARDS)
def test_trust_placeholders_are_filled_and_taxonomy_is_embedded(name):
    html = DASHBOARDS[name].build_html(EMPTY)
    for placeholder in ("__TRUST_CSS__", "__TRUST_SECTION__", "__TRUST_JS__"):
        assert placeholder not in html
    assert 'id="tcover"' in html and "function trustInit" in html
    data = _data(html)
    assert [d["name"] for d in data["taxonomy"]["dimensions"]] == list(trust.DIMENSIONS)
    assert "selection" not in data


@pytest.mark.parametrize("name", DASHBOARDS)
def test_trustworthiness_comes_before_the_task_specific_results(name):
    html = DASHBOARDS[name].build_html(EMPTY)
    assert html.index("Part 1 &middot; Trustworthiness metrics") < html.index("Part 2 &middot; Task-specific") < html.index("Part 3 &middot; Details")


@pytest.mark.parametrize("name", DASHBOARDS)
def test_a_selection_is_embedded_as_data_never_as_markup(name):
    hostile = {"label": "</script><b>x</b>", "metrics": ["latency", "</script>"]}
    html = DASHBOARDS[name].build_html(EMPTY, hostile)
    assert html.count('id="data"') == 1
    assert _data(html)["selection"] == hostile


@pytest.mark.parametrize("name", DASHBOARDS)
def test_results_are_not_modified_by_building_the_page(name):
    results = json.loads(json.dumps(EMPTY))
    before = json.dumps(results, sort_keys=True)
    DASHBOARDS[name].build_html(results, {"label": "", "metrics": ["latency"]})
    assert json.dumps(results, sort_keys=True) == before


@pytest.mark.parametrize("name", DASHBOARDS)
def test_command_line_accepts_a_selection_file(name, tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps(EMPTY))
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"label": "demo", "metrics": ["latency", "pass_hat_k_agent"]}))
    out = tmp_path / "out.html"
    DASHBOARDS[name].main(["--results", str(results), "--out", str(out), "--metrics", str(sel)])
    assert _data(out.read_text(encoding="utf-8"))["selection"] == {"label": "demo", "metrics": ["latency", "pass_hat_k_agent"]}
