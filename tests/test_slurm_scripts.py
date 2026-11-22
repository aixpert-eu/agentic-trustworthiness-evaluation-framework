"""The extra-run job scripts: shell syntax, the order of rounds, the quota stop and what is evaluated afterwards.

The scripts are run with a stand-in ``python`` that records every call, so nothing is captured and no API is used.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

SLURM = Path(__file__).resolve().parents[1] / "harness" / "slurm"
SCRIPTS = {"diagflow": SLURM / "diagflow_extra_runs.sbatch", "maintie": SLURM / "maintie_extra_runs.sbatch"}

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

SHIM = """#!/bin/bash
echo "$*" >> "$SHIM_LOG"
if [ "$1" = "-" ]; then cat >/dev/null; echo 500; exit 0; fi        # the free-request check
if [ "$1" = "-m" ] && [[ "$2" == *.capture ]]; then
  n=$(grep -c -- "-m harness.*capture" "$SHIM_LOG")
  [ "$n" -eq "${STOP_ON:-0}" ] && exit 3                           # the guard: only RESERVE free requests left
  [ "$n" -eq "${FAIL_ON:-0}" ] && exit 1
fi
exit 0
"""


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_are_valid_shell(name):
    assert subprocess.run(["bash", "-n", str(SCRIPTS[name])], capture_output=True).returncode == 0


def run(name, tmp_path, rounds, **env_extra):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    shim = bindir / "python"
    shim.write_text(SHIM)
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "repo").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    log = tmp_path / "calls.txt"
    log.write_text("")
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "SHIM_LOG": str(log), "ROUNDS": rounds, "REPO": str(tmp_path / "repo"),
           "DIAGFLOW": str(tmp_path / "data"), "MAINTIE": str(tmp_path / "data"), "OPENROUTER_API_KEY": "dummy", "OUT_DIR": str(tmp_path / "out"),
           **{k: str(v) for k, v in env_extra.items()}}
    done = subprocess.run(["bash", str(SCRIPTS[name])], env=env, capture_output=True, text=True)
    modules = [c for c in log.read_text().splitlines() if c.startswith("-m ")]  # the `python -m ...` calls, in order
    return done, [c for c in modules if c.split()[1].endswith(".capture")], [c for c in modules if c.split()[1].endswith(".evaluate")]


def min_runs(evaluate_call):
    parts = evaluate_call.split()
    return int(parts[parts.index("--min-runs") + 1])


def test_diagflow_runs_every_round_for_every_pattern_then_evaluates_the_whole_rectangle(tmp_path):
    done, captures, evaluations = run("diagflow", tmp_path, "10:1 15:1")
    assert done.returncode == 0 and len(captures) == 6
    order = [(c.split("--pairs ")[1].split()[0], c.split("--pattern ")[1].split()[0]) for c in captures]
    assert order == [("10", "baseline"), ("10", "basic_rag"), ("10", "react"), ("15", "baseline"), ("15", "basic_rag"), ("15", "react")]
    assert all("--temperature 0 " in c and "--min-free-requests 200" in c and "--min-turns 4 --max-turns 14" in c for c in captures)
    assert len(evaluations) == 1 and min_runs(evaluations[0]) == 1 and "--common" in evaluations[0]


def test_maintie_runs_every_model_and_pattern_in_a_round_before_the_next_round(tmp_path):
    done, captures, evaluations = run("maintie", tmp_path, "15:3 20:3", MODELS="a/x:free b/y:free")
    assert done.returncode == 0 and len(captures) == 2 * 2 * 2
    assert [c.split("--n ")[1].split()[0] for c in captures] == ["15"] * 4 + ["20"] * 4
    assert all("--runs 3" in c for c in captures)
    assert min_runs(evaluations[0]) == 3


def test_a_quota_stop_ends_the_capture_and_evaluates_only_what_was_completed(tmp_path):
    # guard fires in the second round of the repeat plan (call 6 = ReAct of "5:3"): round "5:2" is complete, "5:3" is not
    done, captures, evaluations = run("diagflow", tmp_path, "5:2 5:3", STOP_ON=6)
    assert done.returncode == 0 and len(captures) == 6 and "daily free-request quota" in done.stdout
    assert min_runs(evaluations[0]) == 2
    # guard fires in the very first round: only the runs every item already has (one) can be claimed
    done, captures, evaluations = run("diagflow", tmp_path, "10:1 15:1", STOP_ON=3)
    assert len(captures) == 3 and min_runs(evaluations[0]) == 1


@pytest.mark.parametrize("name", SCRIPTS)
def test_a_failed_capture_is_not_evaluated_and_fails_the_job(tmp_path, name):
    done, captures, evaluations = run(name, tmp_path, "10:1" if name == "diagflow" else "15:3", FAIL_ON=1)
    assert done.returncode == 1 and len(captures) == 1 and evaluations == []


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_key_is_required_for_a_real_run_but_not_for_a_rehearsal(tmp_path, name):
    done, captures, _ = run(name, tmp_path, "10:1", OPENROUTER_API_KEY="")
    assert done.returncode == 2 and "OPENROUTER_API_KEY is not set" in done.stdout and captures == []
    done, captures, _ = run(name, tmp_path, "10:1" if name == "diagflow" else "15:3", OPENROUTER_API_KEY="", DRY_RUN=1)
    assert done.returncode == 0 and captures and all("--dry-run" in c and "_dryrun" in c for c in captures)
