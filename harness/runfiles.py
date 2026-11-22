"""Reading and writing captured run files so that a job killed at its time limit cannot corrupt a capture.

A capture is resumed by looking at which run files already exist. A file cut short in the middle of a write
(the job was killed at its walltime) must neither be mistaken for a finished run nor crash the resume, so
files are written to a temporary name and renamed, and a file that cannot be parsed counts as not done.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def write_run_file(path: Path, record: dict[str, Any]) -> None:
    """Write ``record`` as JSON so that ``path`` either has the whole record or does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_run_file(path: Path) -> dict[str, Any] | None:
    """The record in ``path``, or None when the file is missing or unreadable (cut short, or not JSON)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_run_files(runs_dir: str | Path, glob: str = "*__r*.json") -> list[dict[str, Any]]:
    """Every readable run file under ``runs_dir``; unreadable ones are reported on stderr and left out."""
    out = []
    for p in sorted(Path(runs_dir).rglob(glob)):
        rec = read_run_file(p)
        if rec is None:
            print(f"WARNING: skipping unreadable run file {p} (a capture killed mid-write?). Capture again to redo it.", file=sys.stderr)
            continue
        out.append(rec)
    return out
