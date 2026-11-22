"""Shared look for the demo dashboards (AIXPERT catalogue style, dark navy with a coral accent).

Both dashboards stay one self-contained HTML file: the CSS is inlined and the small logo is embedded as a
data URI. Dashboard templates only need three placeholders, filled by `apply_theme`:

    __THEME_CSS__   inside <style>
    __HEADER__      first element inside <main> (logo, title, run summary in #sub)
    __FOOTER__      last element inside <main>

Colour tokens the dashboard scripts use (keep in sync with the templates; tests check this):
    --react --rag --base --paper     pattern colours
    --good --mid --bad               band chips and ok/no marks
    --fm --fa --ca                   forced mapping, fabrication, correct abstention
"""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path

LOGO_PATH = Path(__file__).parent / "assets" / "aixpert_logo.png"

CSS = r"""
:root{color-scheme:dark;--ink:#e8edf8;--mute:#acb9d5;--line:#ffffff24;--coral:#f17858;--panel:#8098d00a;--track:#ffffff1f;
--react:#6ea0ff;--rag:#f0a255;--base:#4cc3b3;--paper:#a3abbd;--good:#5fc08a;--mid:#e0b04f;--bad:#f58a8a;--fm:#e8b25a;--fa:#f58a8a;--ca:#5fc08a}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(ellipse at 90% 0,#402d4933,transparent 50%),linear-gradient(160deg,#192c54,#0b1430 65%,#030711);background-attachment:fixed;color:var(--ink);font:15px/1.55 Arial,system-ui,sans-serif;min-height:100vh}
button,input,select{font:inherit}
select:focus-visible,summary:focus-visible,a:focus-visible{outline:3px solid var(--coral);outline-offset:3px}
main{max-width:1240px;margin:0 auto;padding:30px 40px 56px}
.top{display:flex;align-items:center;justify-content:space-between;gap:24px;border-bottom:1px solid var(--line);padding-bottom:23px;margin-bottom:8px}
.brand{display:flex;align-items:center;gap:14px;font-size:25px;font-weight:800;white-space:nowrap}
.brand span{font-weight:400}.brand b{color:var(--coral)}
.logo{width:64px;height:64px;object-fit:contain;border-radius:12px;background:#fff;border:1px solid var(--coral);display:block}
.eyebrow{font:11px/1.6 ui-monospace,monospace;letter-spacing:1.5px;text-transform:uppercase;color:var(--mute);font-weight:400}
.heading{text-align:right}
h1{font:600 clamp(24px,2.6vw,38px)/1.15 Georgia,serif;margin:0 0 9px}
.meta{color:var(--mute);font-size:12px}.meta b{color:var(--coral)}
h2{font:600 22px/1.25 Georgia,serif;margin:36px 0 4px}
p.sub{margin:0 0 12px;color:var(--mute);font-size:13px;max-width: 980px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:16px 20px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mute);font-weight:600}td.n{font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child>td{border-bottom:0}
.chip{display:inline-block;padding:1px 9px;border-radius:99px;font-size:12px;border:1px solid var(--line);background:color-mix(in srgb,currentColor 14%,transparent)}
.b4,.b3{color:var(--good);border-color:var(--good)}.b2{color:var(--mid);border-color:var(--mid)}.b1,.b0{color:var(--bad);border-color:var(--bad)}
.row{display:grid;grid-template-columns:minmax(150px,320px) 1fr 52px;gap:8px;align-items:center;margin:4px 0;font-size:13px}
.track{position:relative;background:var(--track);height:10px;border-radius:5px;overflow:hidden}.fill{height:100%;border-radius:5px}
.div{display:grid;grid-template-columns:minmax(150px,320px) 1fr 1fr 110px;gap:0 8px;align-items:center;margin:4px 0;font-size:13px}
.left,.right{display:flex;height:14px;background:var(--track)}.left{flex-direction:row-reverse;border-radius:7px 0 0 7px}.right{border-radius:0 7px 7px 0}
.seg{height:100%}.lbl{overflow-wrap:anywhere;line-height:1.3}
.ciw{position:absolute;top:50%;height:2px;margin-top:-1px;background:var(--ink);box-shadow:0 0 0 1px #0b1430}
.ciw::before,.ciw::after{content:"";position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink);box-shadow:0 0 0 1px #0b1430}
.ciw::before{left:0}.ciw::after{right:0}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
.grp{margin:16px 0 4px;font-weight:600;font-size:13px}.grp:first-child{margin-top:2px}
.legend{font-size:12px;color:var(--mute);margin:4px 0 8px}.legend span.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 5px 0 14px}.legend span.sw:first-child{margin-left:0}
select{padding:6px 10px;background:#0a1531;color:var(--ink);border:1px solid #ffffff3a;border-radius:8px;margin-bottom:6px;max-width:100%}
.ok{color:var(--good)}.no{color:var(--bad)}.mono{font-family:ui-monospace,monospace;font-size:12px}.mute{color:var(--mute)}
details{border-bottom:1px solid var(--line);padding:8px 0}summary{cursor:pointer}
tr.inj td{background:rgba(240,122,122,.10)}
.notes{font-size:12px;line-height:1.7;color:var(--mute);border-left:2px solid var(--coral);padding:4px 16px;margin:28px 0 0}
footer{margin-top:35px;border-top:1px solid var(--line);padding-top:15px;color:var(--mute);font-size:11px}
@media(max-width:760px){main{padding:20px 16px 40px}.top{flex-direction:column;align-items:flex-start}.heading{text-align:left}.brand{font-size:21px}.logo{width:52px;height:52px}
.row{grid-template-columns:1fr 52px}.row>.lbl{grid-column:1;grid-row:1}.row>.n{grid-column:2;grid-row:1}.row>.track{grid-column:1/-1;grid-row:2}
.div{grid-template-columns:1fr 1fr;gap:4px 8px;margin:10px 0}.div>.lbl,.div>.n{grid-column:1/-1}}
"""


def logo_data_uri() -> str | None:
    """The logo as a data URI, or None when the asset is missing (the header then shows the wordmark only)."""
    try:
        return "data:image/png;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    except OSError:
        return None


def header_html(line1: str, line2: str) -> str:
    uri = logo_data_uri()
    logo = f'<img class="logo" alt="AIXPERT logo" src="{uri}">' if uri else ""
    return (
        '<header class="top"><div class="brand">' + logo
        + '<div>AI<span>XPERT</span><b>&bull;</b><div class="eyebrow">Trustworthiness evaluation</div></div></div>'
        + f'<div class="heading"><h1>{escape(line1)}<br>{escape(line2)}</h1><div class="meta" id="sub"></div></div></header>'
    )


def footer_html(text: str) -> str:
    return f"<footer>{escape(text)}</footer>"


def apply_theme(template: str, *, line1: str, line2: str, footer: str) -> str:
    """Fill the theme placeholders. Call this BEFORE inserting the run data, so data can never be rewritten."""
    return (
        template.replace("__THEME_CSS__", CSS)
        .replace("__HEADER__", header_html(line1, line2))
        .replace("__FOOTER__", footer_html(footer))
    )
