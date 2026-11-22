"""The single-file catalogue pages are generated from docs/catalogue/src and docs/catalogue/picker."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

from contracts.registry import REGISTRY

CATALOGUE = Path(__file__).resolve().parents[1] / "docs" / "catalogue"


def _builder():
    spec = importlib.util.spec_from_file_location("build_catalogue", CATALOGUE / "build_catalogue.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pages() -> dict[str, str]:
    return _builder().build()


def test_committed_pages_are_up_to_date():
    """Fails when src/ or picker/ changed without rebuilding: run `python docs/catalogue/build_catalogue.py`."""
    assert _builder().main(["--check"]) == 0


@pytest.mark.parametrize("name", ["aixpert_catalogue.html", "aixpert_catalogue_picker.html"])
def test_pages_are_self_contained(pages, name):
    html = pages[name]
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:|\.{0,2}/|[a-z_]+\.(js|css)\b|assets/)""", html), "page refers to a file or URL"
    assert "<link" not in html and "@import" not in html and "url(http" not in html
    assert 'src="data:image/png;base64,' in html
    assert html.count("<script>") == html.count("</script>")


def test_original_page_has_no_picker(pages):
    plain = pages["aixpert_catalogue.html"]
    for marker in ("pickable", "sel-download", "dialog-pick", 'id="tray"'):
        assert marker not in plain


def test_picker_page_adds_selection_controls_only(pages):
    picker = pages["aixpert_catalogue_picker.html"]
    for marker in ("id=\"sel-download\"", "id=\"sel-load\"", "id=\"dialog-pick\"", "id=\"tray\"", "aixpert-metric-selection/1"):
        assert marker in picker
    # every id the original page's script looks up is still there
    for element_id in ("grid", "search", "dimension", "agent", "evidence", "scope", "reset", "count", "detail", "dimensions"):
        assert f'id="{element_id}"' in picker


def _catalogue_ids() -> list[str]:
    text = (CATALOGUE / "src" / "catalogue-data.js").read_text(encoding="utf-8")
    start = text.index("const metrics = ") + len("const metrics = ")
    metrics, _ = json.JSONDecoder().raw_decode(text[start:])
    return [m["id"] for m in metrics]


def test_catalogue_ids_exist_in_the_registry():
    """Metric ids a person picks must be ones the dashboards and registry know (bias_language is implemented but unregistered)."""
    unknown = set(_catalogue_ids()) - {s.metric_id for s in REGISTRY} - {"bias_language"}
    assert not unknown, f"catalogue lists metrics the registry does not have: {sorted(unknown)}"


def test_a_moved_anchor_fails_the_build_instead_of_dropping_a_feature(tmp_path, monkeypatch):
    module = _builder()
    src = tmp_path / "src"
    shutil.copytree(CATALOGUE / "src", src)
    index = (src / "index.html").read_text(encoding="utf-8")
    (src / "index.html").write_text(index.replace('id="reset"', 'id="reset-filters"'), encoding="utf-8")
    monkeypatch.setattr(module, "SRC", src)
    with pytest.raises(ValueError, match="anchor not found"):
        module.build()
