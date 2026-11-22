"""Build the single-file metric catalogue pages from the multi-file sources.

    python docs/catalogue/build_catalogue.py            # write the two pages next to this file
    python docs/catalogue/build_catalogue.py --check    # exit 1 when the committed pages are out of date

src/                  the catalogue page as designed by Chara (ARC): index.html, styles.css, catalogue.js,
                      catalogue-data.js, assets/logo.png. Edit these, then rebuild.
picker/               the metric picker added on top: picker.css, picker.js, selection.html

Writes:
    aixpert_catalogue.html          the original page, with styles, data, script and logo inlined so it opens
                                    from a plain file with nothing else next to it
    aixpert_catalogue_picker.html   the same page plus pick-and-choose metric selection: add metrics from the cards,
                                    see per-dimension coverage, download the choice as JSON, load it again. The
                                    dashboards read that JSON under "Metrics shown" (or with --metrics).

The picker is applied with explicit one-occurrence patches, so an edit to src/ that moves an anchor fails the
build instead of silently dropping a feature.
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
PICKER = HERE / "picker"
PLAIN_NAME = "aixpert_catalogue.html"
PICKER_NAME = "aixpert_catalogue_picker.html"


def _sub(text: str, old: str, new: str) -> str:
    """Replace exactly one occurrence; fail loudly when the source drifted."""
    if text.count(old) != 1:
        raise ValueError(f"anchor not found exactly once ({text.count(old)}): {old[:70]!r}")
    return text.replace(old, new)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _inline(index: str, *, css: str, data: str, app: str, logo: str, before_app: str = "", after_app: str = "") -> str:
    for name, body in (("css", css), ("data", data), ("app", app), ("picker", before_app + after_app)):
        if "</script" in body.lower() or "<!--" in body:
            raise ValueError(f"{name} cannot be inlined safely (contains '</script' or '<!--')")
    page = _sub(index, '<link href="styles.css" rel="stylesheet"/>', f"<style>\n{css}\n</style>")
    page = _sub(page, 'src="assets/logo.png"', f'src="{logo}"')
    page = _sub(page, '<script defer="" src="catalogue-data.js">\n  </script>', f"<script>\n{data}\n</script>")
    scripts = (f"<script>\n{before_app}\n</script>\n  " if before_app else "") + f"<script>\n{app}\n</script>"
    scripts += f"\n  <script>\n{after_app}\n</script>" if after_app else ""
    page = _sub(page, '<script defer="" src="catalogue.js">\n  </script>', scripts)
    return _sub(page, "Keep this folder together.", "This copy is one self-contained file (styles, data, script and logo are inlined).")


def build() -> dict[str, str]:
    """File name -> content for both pages."""
    index, css = _read(SRC / "index.html"), _read(SRC / "styles.css")
    data, app = _read(SRC / "catalogue-data.js"), _read(SRC / "catalogue.js")
    logo = "data:image/png;base64," + base64.b64encode((SRC / "assets" / "logo.png").read_bytes()).decode("ascii")

    plain = _inline(index, css=css, data=data, app=app, logo=logo)

    page = _inline(index, css=css + "\n" + _read(PICKER / "picker.css"), data=data, app=app, logo=logo,
                   before_app=_read(PICKER / "picker.js"), after_app="refresh();")
    page = _sub(page, """   <p class="hint">
    Select a metric to explore its description, required evidence and implementation details.
   </p>""", _read(PICKER / "selection.html").rstrip("\n"))
    page = _sub(page, """     <button class="filter" id="reset">
      Reset filters
     </button>""", """     <button class="filter" id="reset">
      Reset filters
     </button>
     <button class="filter" id="sel-all" type="button">
      Add all shown to my selection
     </button>""")
    page = _sub(page, '<span class="meta" id="position">\n    </span>',
                '<span class="meta" id="position">\n    </span>\n    <button aria-pressed="false" class="filter pick-dialog" id="dialog-pick" type="button">\n     + Add to my selection\n    </button>')
    page = _sub(page, "<!-- METRIC DETAILS: filled when a metric card is selected -->", """<div class="tray" hidden="" id="tray" role="region" aria-label="Selected metrics">
   <span><b id="tray-count"></b></span>
   <a class="filter" href="#selection" style="text-decoration:none">Review selection</a>
   <button class="filter" id="tray-download" type="button">Download JSON</button>
   <button class="filter" id="tray-clear" type="button">Clear</button>
  </div>
  <!-- METRIC DETAILS: filled when a metric card is selected -->""")
    page = _sub(page, "AIXPERT · Evaluation framework catalogue", "AIXPERT · Evaluation framework catalogue · metric picker")
    # hook the picker into the original script (two one-line patches)
    page = _sub(page, "    $('grid').append(b)\n  });", "    $('grid').append(pickable(b, m))\n  });")
    page = _sub(page, "  if (!$('detail').open) $('detail').showModal();", "  syncDialogPick(m);\n  if (!$('detail').open) $('detail').showModal();")
    return {PLAIN_NAME: plain, PICKER_NAME: page}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE, help="directory to write the pages to (default: next to this file)")
    ap.add_argument("--check", action="store_true", help="do not write; fail when the pages in --out are out of date")
    args = ap.parse_args(argv)
    pages = build()
    if args.check:
        stale = [n for n, text in pages.items() if not (args.out / n).is_file() or _read(args.out / n) != text]
        if stale:
            print("out of date: " + ", ".join(stale) + "  (run: python docs/catalogue/build_catalogue.py)", file=sys.stderr)
            return 1
        print("catalogue pages are up to date")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in pages.items():
        (args.out / name).write_text(text, encoding="utf-8")
        print(f"wrote {args.out / name} ({len(text) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
