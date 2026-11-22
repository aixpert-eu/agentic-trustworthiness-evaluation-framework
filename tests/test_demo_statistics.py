"""Statistics added to the demo evaluations: item-level bootstrap intervals, pass@k with one k for every
configuration, log completeness that is not a pass when nothing was logged, and discriminative power."""

from __future__ import annotations

import json

import pytest

from harness import stats
from harness.diagflow_demo.capture import capture as diagflow_capture
from harness.diagflow_demo.dashboard import build_html as diagflow_html
from harness.diagflow_demo.evaluate import evaluate_all as diagflow_evaluate
from harness.diagflow_demo.evaluate import load_runs as diagflow_runs
from harness.diagflow_demo.fake import RuleJudge, make_fake_factory
from harness.maintie_demo.capture import capture as maintie_capture
from harness.maintie_demo.dashboard import build_html as maintie_html
from harness.maintie_demo.data import parse_record
from harness.maintie_demo.evaluate import evaluate_all as maintie_evaluate
from harness.maintie_demo.evaluate import load_runs as maintie_runs
from harness.maintie_demo.fake import make_fake_agent
from tests.test_diagflow_demo import GRAPH, clean_conv, mixed_conv
from tests.test_maintie_demo import REC_A, REC_B


# ---- the bootstrap ----------------------------------------------------------------------------------------

def test_bootstrap_is_reproducible_and_needs_two_items():
    clusters = [[0.0, 0.2], [0.5, 0.6], [0.9, 1.0], [0.4, 0.4]]
    assert stats.mean_ci(clusters) == stats.mean_ci(clusters)
    assert stats.mean_ci(clusters, seed=1) != stats.mean_ci(clusters, seed=2)
    assert stats.mean_ci([[0.5, 0.5, 0.5]]) is None and stats.mean_ci([]) is None


def test_interval_brackets_the_estimate_and_narrows_with_more_items():
    few = [[0.0], [1.0], [0.0], [1.0]]
    many = few * 10
    lo, hi = stats.mean_ci(few)
    lo_m, hi_m = stats.mean_ci(many)
    assert lo <= 0.5 <= hi and lo_m <= 0.5 <= hi_m
    assert (hi_m - lo_m) < (hi - lo)


def test_the_item_is_resampled_not_the_run():
    """Two items with three identical runs each: six numbers, but only two independent items."""
    lo, hi = stats.mean_ci([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert (lo, hi) == (0.0, 1.0)


def test_undefined_statistics_give_no_interval():
    assert stats.cluster_bootstrap([1, 2, 3], lambda s: None) is None
    many = stats.cluster_bootstrap_many([1, 2, 3], lambda s: {"a": 1.0, "b": None})
    assert many["a"] == (1.0, 1.0) and "b" not in many


# ---- MaintIE ------------------------------------------------------------------------------------------------

@pytest.fixture
def items():
    return [parse_record(REC_A, "g00000"), parse_record(REC_B, "g00001")]


def _maintie_world(tmp_path, items, runs=3):
    for pattern in ("react", "basic_rag"):
        maintie_capture(items, make_fake_agent(pattern, items, drop=0.0, wrong_class=0.0), model="m/a", pattern=pattern, runs=runs,
                        out_dir=tmp_path, log=lambda s: None)
    maintie_capture(items, make_fake_agent("react", items, drop=0.7, wrong_class=0.5), model="m/c", pattern="react", runs=runs,
                    out_dir=tmp_path, log=lambda s: None)
    return {i.item_id: i for i in items}


def test_maintie_scores_every_configuration_with_the_same_k_and_reports_intervals(tmp_path, items):
    res = maintie_evaluate(maintie_runs(tmp_path) if _maintie_world(tmp_path, items) else [], {i.item_id: i for i in items}, n_boot=200)
    cfg = {(c["model"], c["pattern"]): c for c in res["configs"]}
    assert res["meta"]["pass_k"] == 3 and all(c["k"] == 3 for c in cfg.values())
    perfect = cfg[("m/a", "react")]
    assert perfect["pass_at_k"]["score"] == 1.0 and perfect["pass_hat_k"]["score"] == 1.0
    for res_ in (perfect["pass_hat_k"], perfect["pass_at_k"]):
        lo, hi = res_["ci95"]
        assert lo <= res_["score"] <= hi
    assert set(perfect["ci95"]) == {"task_success_rate", "subgoal_progress", "entity_f1", "relation_f1"}
    weak = cfg[("m/c", "react")]
    assert weak["ci95"]["entity_f1"][0] <= weak["entity"]["f1"] <= weak["ci95"]["entity_f1"][1]
    assert weak["pass_at_k"]["score"] >= weak["pass_hat_k"]["score"]  # at least one of k tries is never worse than all k
    json.dumps(res)


def test_maintie_k_can_be_chosen_but_not_beyond_the_runs_captured(tmp_path, items):
    world = _maintie_world(tmp_path, items)
    recs = maintie_runs(tmp_path)
    res = maintie_evaluate(recs, world, k=2, n_boot=50)
    assert all(c["k"] == 2 and "pass^2" in c["pass_hat_k"]["explanation"] for c in res["configs"])
    with pytest.raises(ValueError, match="fewest is 3"):
        maintie_evaluate(recs, world, k=5, n_boot=50)


def test_log_completeness_is_not_a_pass_when_there_were_no_tool_calls(tmp_path, items):
    world = _maintie_world(tmp_path, items)
    base = make_fake_agent("basic_rag", items, drop=0.0, wrong_class=0.0)

    def without_tool_calls(text):  # a pattern that answers directly, with nothing to log
        out = base(text)
        out["messages"] = [m for m in out["messages"] if m["type"] == "human" or (m["type"] == "ai" and not m.get("tool_calls"))]
        return out

    maintie_capture(items, without_tool_calls, model="m/n", pattern="basic_rag", runs=3, out_dir=tmp_path, log=lambda s: None)
    res = maintie_evaluate(maintie_runs(tmp_path), world, n_boot=50)
    cfg = {(c["model"], c["pattern"]): c for c in res["configs"]}
    assert cfg[("m/a", "react")]["log_completeness"] == 1.0  # tool calls made and all logged
    assert cfg[("m/n", "basic_rag")]["log_completeness"] is None  # nothing to log, so no score rather than a default pass


def test_maintie_discriminative_power_compares_configurations(tmp_path, items):
    world = _maintie_world(tmp_path, items)
    res = maintie_evaluate(maintie_runs(tmp_path), world, n_boot=50)
    dp = res["meta"]["meta_evaluation"]["discriminative_power"]
    assert dp["n_systems"] == 3 and 0.0 <= dp["score"] <= 1.0 and "eta-squared" in dp["explanation"]
    one = maintie_evaluate([r for r in maintie_runs(tmp_path) if r["model"] == "m/a" and r["pattern"] == "react"], world, n_boot=50)
    assert one["meta"]["meta_evaluation"]["discriminative_power"] is None  # nothing to compare
    html = maintie_html(res)
    assert '"discriminative_power"' in html


# ---- DiagFlowBench -------------------------------------------------------------------------------------------

@pytest.fixture
def dfworld():
    c, m = clean_conv(), mixed_conv()
    return {"graphs": {"G1": GRAPH}, "convs": {c["conversation_id"]: c, m["conversation_id"]: m}}


def _diagflow_world(tmp_path, dfworld, runs=3):
    ids = list(dfworld["convs"])
    for model, pattern, kw in (("m/a", "react", dict(p_correct=1.0, p_force=0.0)), ("m/a", "basic_rag", dict(p_correct=1.0, p_force=0.0)),
                               ("m/c", "react", dict(p_correct=0.0, p_force=1.0))):
        diagflow_capture(ids, dfworld["convs"], dfworld["graphs"], make_fake_factory(pattern, **kw), model=model, pattern=pattern, runs=runs,
                         out_dir=tmp_path, retries=1, backoff=0, sleep=lambda s: None, log=lambda s: None)


def test_diagflow_scores_with_one_k_adds_pass_at_k_and_intervals_on_the_rates(tmp_path, dfworld):
    _diagflow_world(tmp_path, dfworld)
    res = diagflow_evaluate(diagflow_runs(tmp_path), dfworld["graphs"], dfworld["convs"], RuleJudge(), n_boot=200)
    cfg = {(c["model"], c["pattern"]): c for c in res["configs"]}
    assert res["meta"]["pass_k"] == 3 and all(c["k"] == 3 for c in cfg.values())
    good = cfg[("m/a", "react")]
    assert good["pass_at_k"]["score"] == good["pass_hat_k"]["score"]  # a deterministic agent: every try gives the same result
    assert len(good["pass_hat_k"]["ci95"]) == 2 and good["pass_hat_k"]["ci95"][0] <= good["pass_hat_k"]["score"] <= good["pass_hat_k"]["ci95"][1]
    step = good["metrics"]["step_accuracy"]
    assert step["ci95"][0] <= step["rate"] <= step["ci95"][1]
    bad = cfg[("m/c", "react")]
    assert bad["pass_at_k"]["score"] >= bad["pass_hat_k"]["score"]
    fm = bad["metrics"]["forced_mapping_rate"]
    assert fm["rate"] == 1.0 and fm["ci95"][0] <= 1.0 <= fm["ci95"][1]
    json.dumps(res)
    assert "__DATA__" not in diagflow_html(res)


def test_diagflow_k_cannot_exceed_the_runs_captured(tmp_path, dfworld):
    _diagflow_world(tmp_path, dfworld)
    with pytest.raises(ValueError, match="fewest is 3"):
        diagflow_evaluate(diagflow_runs(tmp_path), dfworld["graphs"], dfworld["convs"], RuleJudge(), k=4, n_boot=20)
    res = diagflow_evaluate(diagflow_runs(tmp_path), dfworld["graphs"], dfworld["convs"], RuleJudge(), k=1, n_boot=20)
    assert all(c["k"] == 1 for c in res["configs"])


def test_diagflow_log_completeness_only_where_tool_calls_were_made(tmp_path, dfworld):
    _diagflow_world(tmp_path, dfworld)
    res = diagflow_evaluate(diagflow_runs(tmp_path), dfworld["graphs"], dfworld["convs"], RuleJudge(), n_boot=20)
    cfg = {(c["model"], c["pattern"]): c for c in res["configs"]}
    assert cfg[("m/a", "react")]["log_completeness"] == 1.0 and cfg[("m/a", "react")]["tool_calls_logged"] > 0
    assert cfg[("m/a", "basic_rag")]["log_completeness"] is None and cfg[("m/a", "basic_rag")]["tool_calls_logged"] == 0


def test_diagflow_discriminative_power_separates_a_good_and_a_bad_configuration(tmp_path, dfworld):
    _diagflow_world(tmp_path, dfworld)
    res = diagflow_evaluate(diagflow_runs(tmp_path), dfworld["graphs"], dfworld["convs"], RuleJudge(), n_boot=20)
    dp = res["meta"]["meta_evaluation"]["discriminative_power"]
    assert dp["n_systems"] == 3 and dp["score"] > 0 and "share of ok turns" in dp["score_used"]
