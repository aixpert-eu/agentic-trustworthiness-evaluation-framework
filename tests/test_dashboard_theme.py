"""The two demo dashboards share one theme; these checks keep them self-contained and consistent."""

from __future__ import annotations

import json
import re

import pytest

from harness import dashboard_theme as theme
from harness.diagflow_demo import dashboard as diagflow_dashboard
from harness.maintie_demo import dashboard as maintie_dashboard

DASHBOARDS = {"diagflow": diagflow_dashboard, "maintie": maintie_dashboard}
EMPTY = {"meta": {"dataset": "d", "generated_at": "2026-01-01T00:00:00", "notes": []}, "configs": []}


@pytest.mark.parametrize("name", DASHBOARDS)
def test_all_placeholders_are_filled(name):
    html = DASHBOARDS[name].build_html(EMPTY)
    for placeholder in ("__THEME_CSS__", "__HEADER__", "__FOOTER__", "__DATA__"):
        assert placeholder not in html
    assert html.count('<header class="top">') == 1 and html.count("<footer>") == 1
    assert 'id="sub"' in html and "AIXPERT WP3" in html


@pytest.mark.parametrize("name", DASHBOARDS)
def test_every_colour_token_used_is_defined(name):
    template = DASHBOARDS[name].TEMPLATE
    used = set(re.findall(r"var\((--[a-z0-9-]+)\)", template + theme.CSS))
    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", theme.CSS))
    assert used <= defined, f"undefined tokens: {sorted(used - defined)}"


@pytest.mark.parametrize("name", DASHBOARDS)
def test_dashboard_is_self_contained(name):
    html = DASHBOARDS[name].build_html(EMPTY)
    assert not re.search(r"""(src|href)\s*=\s*["']?https?:""", html)
    assert "@import" not in html and "<link" not in html and "url(http" not in html
    assert "data:image/png;base64," in html


@pytest.mark.parametrize("name", DASHBOARDS)
def test_run_data_is_never_rewritten_by_the_theme(name):
    hostile = {**EMPTY, "meta": {**EMPTY["meta"], "notes": ["__HEADER__ __FOOTER__ __THEME_CSS__ </script>"]}}
    html = DASHBOARDS[name].build_html(hostile)
    data = html.split('id="data" type="application/json">')[1].split("</script>")[0]
    assert "__HEADER__ __FOOTER__ __THEME_CSS__" in data
    assert json.loads(data.replace("<\\/", "</"))["meta"]["notes"][0].endswith("</script>")
    assert html.count('<header class="top">') == 1


def test_missing_logo_degrades_to_wordmark_only(monkeypatch, tmp_path):
    monkeypatch.setattr(theme, "LOGO_PATH", tmp_path / "missing.png")
    header = theme.header_html("one", "two")
    assert "<img" not in header and "AI<span>XPERT</span>" in header


def test_header_text_is_escaped():
    header = theme.header_html("a<b", "c&d")
    assert "a&lt;b" in header and "c&amp;d" in header
