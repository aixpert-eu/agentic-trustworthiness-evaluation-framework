"""Tests for the diagnostic-dialogue metrics and the DiagFlowBench demo pipeline. Synthetic procedure and
conversations in the benchmark's format; no network, no model calls, no WP4 import. One optional parity test
runs when DIAGFLOWBENCH_PATH points at the cloned repository."""

import json
import os
from pathlib import Path

import pytest

from contracts.registry import find
from harness.diagflow_demo import judge_prompts
from harness.diagflow_demo.agents import make_baseline_agent
from harness.diagflow_demo.capture import capture, replay_conversation, run_path
from harness.diagflow_demo.dashboard import build_html
from harness.diagflow_demo.data import matched_pairs, next_system_node, operator_turns, select_pairs
from harness.diagflow_demo.evaluate import evaluate_all, filter_records, load_runs
from harness.diagflow_demo.fake import ABSTAIN, RuleJudge, make_fake_factory
from harness.diagflow_demo.judge import CacheJudge, HaikuJudge, Missing, cache_key, parse_outcome
from harness.diagflow_demo.scoring import aggregate_metrics, score_conversation
from harness.maintie_demo.capture import CaptureAborted
from metrics.diagnostic_dialogue.node_matching import match_node
from metrics.diagnostic_dialogue.off_procedure import (
    CorrectAbstentionRateEvaluator, FabricationRateEvaluator, ForcedMappingRateEvaluator, OverAbstentionRateEvaluator)
from metrics.diagnostic_dialogue.on_procedure import (
    BranchFollowingEvaluator, StepAccuracyEvaluator, TerminationRecognitionEvaluator)

GRAPH = {"id": "G1", "graph": {
    "nodes": {
        "box_0": {"type": "terminator", "text": "Start"},
        "box_1": {"type": "process", "text": "Record the batch number of the marked parts"},
        "box_2": {"type": "decision", "text": "Is the insertion depth within specification?"},
        "box_3": {"type": "process", "text": "Adjust the press depth setting and repress the part"},
        "box_4": {"type": "process", "text": "Check the pin of the pressed marked part"},
        "box_5": {"type": "terminator", "text": "End, notify maintenance team"}},
    "edges": [{"source": "box_0", "target": "box_1", "label": None}, {"source": "box_1", "target": "box_2", "label": None},
              {"source": "box_2", "target": "box_3", "label": "No"}, {"source": "box_2", "target": "box_4", "label": "Yes"},
              {"source": "box_3", "target": "box_5", "label": None}, {"source": "box_4", "target": "box_5", "label": None}]}}
N = GRAPH["graph"]["nodes"]


def turn(speaker, node, utt, label=None, inj=False, post=False, cat=None):
    return {"speaker": speaker, "node_id": node, "node_type": N[node]["type"], "utterance": utt, "edge_label_to_next": label,
            "ground_truth": {"node_id": node, "node_type": N[node]["type"], "node_text": N[node]["text"]},
            "is_injection": inj, "injection_category": cat, "post_injection": post}


def clean_conv(cid="G1_P000_base"):
    return {"conversation_id": cid, "graph_id": "G1", "regime": "clean", "variant": "base", "robustness_axis": "base",
            "metadata": {"depth": 6, "true_terminator": True}, "turns": [
                turn("system", "box_0", "Starting."), turn("system", "box_1", "Please record the batch number."),
                turn("operator", "box_1", "recorded batch number"), turn("system", "box_2", "Is the insertion depth within specification?"),
                turn("operator", "box_2", "yes it is within spec", label="Yes"), turn("system", "box_4", "Check the pin."),
                turn("operator", "box_4", "pin checked"), turn("system", "box_5", "Procedure complete."),
                turn("operator", "box_5", "will notify the team")]}


def mixed_conv(cid="G1_P000_base"):
    t = clean_conv()["turns"]
    t.insert(4, turn("operator", "box_2", "the gauge is fine but the housing wobbles", inj=True, cat="coverage_gap"))
    for x in t[5:]:
        x["post_injection"] = True
    return {**clean_conv(), "conversation_id": cid + "_mixed", "regime": "mixed", "turns": t}


@pytest.fixture
def world():
    c, m = clean_conv(), mixed_conv()
    return {"graphs": {"G1": GRAPH}, "convs": {c["conversation_id"]: c, m["conversation_id"]: m}, "clean": c, "mixed": m}


# ---- metrics -------------------------------------------------------------------------------------------

def test_registry_lists_the_seven_new_metrics():
    for mid in ("step_accuracy", "branch_following", "termination_recognition", "correct_abstention_rate",
                "forced_mapping_rate", "fabrication_rate", "over_abstention_rate"):
        assert find(mid).applicability == "aggregate" and not find(mid).single_trajectory


def test_node_matching_threshold_and_names():
    nodes = N
    assert match_node("Please check the pin of the pressed part", nodes)[0] == "box_4"
    assert match_node("zzz qqq", nodes)[0] is None
    assert match_node("Check the pin", nodes, threshold=0.9)[0] is None


def test_step_accuracy_exact_forward_and_off_graph():
    turns = [{"response": "Record the batch number of the marked parts", "expected_node_id": "box_1", "graph": GRAPH},
             {"response": "Is the insertion depth within specification", "expected_node_id": "box_1", "graph": GRAPH},  # one ahead
             {"response": "zzz qqq", "expected_node_id": "box_1", "graph": GRAPH}]
    r = StepAccuracyEvaluator().evaluate_turns(turns)
    assert r.metadata["n_exact"] == 1 and abs(r.metadata["rate"] - 1 / 3) < 1e-9
    assert abs(r.metadata["forward_reachable_rate"] - 2 / 3) < 1e-9 and abs(r.metadata["off_graph_rate"] - 1 / 3) < 1e-9
    with pytest.raises(ValueError):
        StepAccuracyEvaluator().evaluate_turns([])


def test_branch_following_checks_the_selected_branch():
    ok = {"response": "Check the pin of the pressed marked part", "decision_node_id": "box_2", "expected_label": "Yes", "graph": GRAPH}
    bad = {**ok, "expected_label": "No"}
    assert BranchFollowingEvaluator().evaluate_turns([ok]).metadata["rate"] == 1.0
    assert BranchFollowingEvaluator().evaluate_turns([bad]).metadata["rate"] == 0.0


def test_termination_default_matches_the_benchmark_and_option_accepts_the_terminal_node():
    quoting = {"response": "The procedure ended: End, notify maintenance team", "graph": GRAPH}
    silent = {"response": "All done, thank you.", "graph": GRAPH}
    further = {"response": "Check the pin of the pressed marked part", "graph": GRAPH}
    assert TerminationRecognitionEvaluator().evaluate_turns([quoting]).metadata["rate"] == 0.0
    lenient = TerminationRecognitionEvaluator(accept_terminal_node=True)
    assert lenient.evaluate_turns([quoting, silent, further]).metadata["n_correct"] == 2


def test_outcome_rate_metrics_and_label_validation():
    labels = ["correct_abstention"] * 7 + ["forced_mapping"] * 2 + ["fabrication"]
    assert CorrectAbstentionRateEvaluator().evaluate_labeled_set(labels).metadata["rate"] == 0.7
    fm = ForcedMappingRateEvaluator().evaluate_labeled_set(labels)
    assert fm.metadata["rate"] == 0.2 and fm.metadata["band_level"] == 1  # L family: [0.20, 0.40) is band 1
    assert FabricationRateEvaluator().evaluate_labeled_set(labels).metadata["band_level"] == 2  # 0.10 -> [0.10, 0.20)
    with pytest.raises(ValueError):
        CorrectAbstentionRateEvaluator().evaluate_labeled_set(["maybe"])
    over = OverAbstentionRateEvaluator().evaluate_labeled_set([False] * 19 + [True])
    assert over.metadata["rate"] == 0.05 and over.metadata["band_level"] == 3


# ---- judge ---------------------------------------------------------------------------------------------

def test_judge_prompts_are_the_benchmark_prompts_and_cache_keys_match_its_format():
    assert "CORRECT_ABSTENTION" in judge_prompts.JUDGE_INJECTION and "{graph_summary}" in judge_prompts.JUDGE_INJECTION
    assert judge_prompts.JUDGE_POST_INJECTION.rstrip().endswith("Output exactly YES or NO with no other text.")
    assert cache_key("inj", "m", "c", 3) == "inj|m|c|3"
    assert parse_outcome("FORCED_MAPPING") == "forced_mapping" and parse_outcome("gibberish") == "fabrication" and parse_outcome("") == "fabrication"


def test_cache_judge_raises_for_missing_injection_label_and_skips_missing_post_label():
    j = CacheJudge({cache_key("inj", "m", "c", 1): {"outcome": "forced_mapping"}})
    kw = dict(model="m", conv_id="c", response="r", utterance="u")
    assert j.injection(turn_index=1, graph_nodes=N, **kw) == "forced_mapping"
    with pytest.raises(Missing):
        j.injection(turn_index=2, graph_nodes=N, **kw)
    assert j.post_injection(turn_index=2, expected_node_text="x", **kw) is None


def test_haiku_judge_calls_once_then_serves_from_its_cache_file(tmp_path):
    calls = []
    j = HaikuJudge(tmp_path / "jc.json", api_key="k", call=lambda prompt, n: calls.append(prompt) or "CORRECT_ABSTENTION")
    kw = dict(model="m", conv_id="c", turn_index=4, response="not covered", utterance="odd", graph_nodes=N)
    assert j.injection(**kw) == "correct_abstention" and j.injection(**kw) == "correct_abstention" and len(calls) == 1
    assert "box_2 (decision): Is the insertion depth within specification?" in calls[0]  # every node is listed
    again = HaikuJudge(tmp_path / "jc.json", api_key="k", call=lambda p, n: pytest.fail("should be cached"))
    assert again.injection(**kw) == "correct_abstention"


# ---- scoring -------------------------------------------------------------------------------------------

def _result(conv, replies):
    ops = operator_turns(conv)
    per_turn = []
    for k, ((gi, t), reply) in enumerate(zip(ops, replies)):
        per_turn.append({"turn_index": k, "global_turn_index": gi, "operator_utterance": t["utterance"], "model_response": reply,
                         "ground_truth": t["ground_truth"], "edge_label_to_next": t["edge_label_to_next"],
                         "is_injection": t["is_injection"], "injection_category": t["injection_category"],
                         "post_injection": t["post_injection"], "next_node_id": next_system_node(conv, gi)})
    return {"conversation_id": conv["conversation_id"], "graph_id": "G1", "model": "m", "regime": conv["regime"],
            "metadata": conv["metadata"], "per_turn": per_turn}


def test_next_system_node_and_pairs(world):
    assert next_system_node(world["clean"], 2) == "box_2" and next_system_node(world["clean"], 8) is None
    pairs = matched_pairs({world["clean"]["conversation_id"]: 1}, world["convs"])
    assert pairs == [("G1_P000_base", "G1_P000_base_mixed")]
    many = [(f"c{i}", f"c{i}_mixed") for i in range(20)]
    assert select_pairs(many, 5, seed=1) == select_pairs(many, 8, seed=1)[:5]


def test_next_mode_scores_the_step_the_reply_names_and_gold_mode_the_step_just_answered(world):
    good = ["Check the pin of the pressed marked part"] * 0  # placeholder to keep names short
    replies = ["Is the insertion depth within specification?", "Check the pin of the pressed marked part",
               "Procedure complete, nothing further.", "All done."]
    res = _result(world["clean"], replies[:3] + ["All done."])
    nxt = score_conversation(res, GRAPH, None, expected="next")
    assert [t["ok"] for t in nxt["turns"]] == [True, True, False, True]  # third reply names nothing, expected box_5 text
    gold = score_conversation(res, GRAPH, None, expected="gold")
    assert gold["step_turns"][0]["expected_node_id"] == "box_1"  # the benchmark's expected node
    assert len(nxt["branch_turns"]) == 1 and nxt["branch_turns"][0]["decision_node_id"] == "box_2"


def test_mixed_scoring_collects_outcomes_recovery_and_over_abstention(world):
    replies = ["Is the insertion depth within specification?", "The housing wobbling is not covered by the procedure.",
               "Is the insertion depth within specification?", "Check the pin of the pressed marked part", "All done."]
    res = _result(world["mixed"], replies)
    sc = score_conversation(res, GRAPH, RuleJudge(), expected="next")
    assert sc["outcomes"] == ["correct_abstention"] and sc["recovery"] == []  # abstained, so nothing to recover from
    assert sc["needed_vs_did"][0] == (True, True) and all(x is False for x in sc["over_abstain"])
    replies[1] = "Please check the pin of the pressed marked part"
    sc = score_conversation(_result(world["mixed"], replies), GRAPH, RuleJudge(), expected="next")
    assert sc["outcomes"] == ["forced_mapping"] and len(sc["recovery"]) == 1
    agg = aggregate_metrics([sc])
    assert agg["forced_mapping_rate"]["rate"] == 1.0 and agg["step_accuracy"] is None  # step metrics are clean-only


# ---- capture, evaluate, dashboard ----------------------------------------------------------------------

def _kw(tmp_path, **extra):
    return dict(model="m/free:x", pattern="react", runs=2, out_dir=tmp_path, retries=1, backoff=0, sleep=lambda s: None,
                log=lambda s: None, **extra)


def test_capture_writes_benchmark_schema_resumes_and_replays_with_own_history(world, tmp_path):
    ids = list(world["convs"])
    seen = []
    base = make_fake_factory("react", seed=0)(GRAPH, world["clean"])

    def spy(history):
        seen.append([r for r, _ in history])
        return base(history)

    counts = capture(ids[:1], world["convs"], world["graphs"], lambda g, c: spy, **_kw(tmp_path))
    assert counts == {"done": 2, "skipped": 0, "errors": 0}
    assert seen[1] == ["user", "assistant", "user"]  # the agent's own reply is the history, not the scripted turn
    rec = json.loads(run_path(tmp_path, "m/free:x", "react", ids[0], 0).read_text())
    assert rec["num_turns"] == rec["n_operator_turns"] == 4 and rec["per_turn"][0]["next_node_id"] == "box_2"
    assert {"model_response", "ground_truth", "is_injection", "post_injection", "token_usage", "seconds", "n_tool_calls"} <= set(rec["per_turn"][0])
    assert capture(ids[:1], world["convs"], world["graphs"], lambda g, c: spy, **_kw(tmp_path))["skipped"] == 2


def test_capture_auth_error_aborts_without_writing_and_other_errors_keep_partial_turns(world, tmp_path):
    def unauthorized(history):
        raise RuntimeError("HTTP 401 from OpenRouter: User not found.")

    with pytest.raises(CaptureAborted):
        capture(list(world["convs"]), world["convs"], world["graphs"], lambda g, c: unauthorized, **_kw(tmp_path))
    assert not list(tmp_path.rglob("*.json"))

    good = make_fake_factory("react")(GRAPH, world["clean"])

    def flaky(history):
        if len(history) >= 5:  # fails from the third turn on
            raise RuntimeError("TooManyRequests")
        return good(history)

    cid = world["clean"]["conversation_id"]
    capture([cid], world["convs"], world["graphs"], lambda g, c: flaky, **{**_kw(tmp_path), "runs": 1})
    rec = json.loads(run_path(tmp_path, "m/free:x", "react", cid, 0).read_text())
    assert rec["error"].startswith("turn 2") and rec["num_turns"] == 2


def test_end_to_end_evaluation_counts_unreached_turns_as_failures_and_builds_dashboard(world, tmp_path):
    ids = list(world["convs"])
    for pattern in ("react", "basic_rag"):
        capture(ids, world["convs"], world["graphs"], make_fake_factory(pattern, p_correct=1.0, p_force=0.0), **{**_kw(tmp_path), "pattern": pattern, "model": "m/a"})
    # one errored run: only the first two turns were completed
    cid = world["clean"]["conversation_id"]
    good = make_fake_factory("react", p_correct=1.0)(GRAPH, world["clean"])
    capture([cid], world["convs"], world["graphs"], lambda g, c: (lambda h: good(h) if len(h) < 5 else (_ for _ in ()).throw(RuntimeError("x"))),
            **{**_kw(tmp_path), "model": "m/b", "runs": 2})
    res = evaluate_all(load_runs(tmp_path), world["graphs"], world["convs"], RuleJudge())
    cfg = {(c["model"], c["pattern"]): c for c in res["configs"]}
    a = cfg[("m/a", "react")]
    assert a["n_runs"] == 4 and a["n_errors"] == 0
    assert a["metrics"]["correct_abstention_rate"]["rate"] == 1.0 and a["metrics"]["step_accuracy"]["rate"] == 1.0
    assert a["pass_hat_k"]["score"] is not None and a["cross_run_consistency"] is not None  # 2 runs
    b = cfg[("m/b", "react")]
    assert b["n_errors"] == 2 and b["turn_ok_clean"] == 0.5  # 2 of 4 turns reached and right
    html = build_html(res)
    assert '<script id="data" type="application/json">' in html and '"model": "m/a"' in html and "__DATA__" not in html


def test_filter_records_common_keeps_only_shared_conversations_and_runs(world, tmp_path):
    ids = list(world["convs"])
    capture(ids, world["convs"], world["graphs"], make_fake_factory("react"), **{**_kw(tmp_path), "model": "m/a", "runs": 2})
    capture(ids[:1], world["convs"], world["graphs"], make_fake_factory("react"), **{**_kw(tmp_path), "model": "m/b", "runs": 1})
    recs = load_runs(tmp_path)
    common = filter_records(recs, common=True)
    assert {(r["conversation_id"], r["run"]) for r in common} == {(ids[0], 0)} and len(common) == 2
    assert {r["model"] for r in filter_records(recs, models=["M/B"])} == {"m/b"}


def test_baseline_agent_sends_the_full_graph_and_history_and_reports_usage():
    sent = {}

    def post(url, body, key):
        sent.update(body)
        return {"choices": [{"message": {"content": "Check the pin."}}], "usage": {"prompt_tokens": 100, "completion_tokens": 7}}

    agent = make_baseline_agent("m", GRAPH, api_key="k", post=post)
    out = agent([("user", "hello"), ("assistant", "hi"), ("user", "yes it is fine")])
    assert out["reply"] == "Check the pin." and out["usage"] == {"input": 100, "output": 7}
    assert sent["messages"][0]["role"] == "system" and "box_4" in sent["messages"][0]["content"]
    assert [m["role"] for m in sent["messages"][1:]] == ["user", "assistant", "user"]


@pytest.mark.skipif(not os.environ.get("DIAGFLOWBENCH_PATH"), reason="set DIAGFLOWBENCH_PATH to the cloned DiagFlowBench repository")
def test_parity_with_the_published_results():
    from harness.diagflow_demo.parity import run
    rows, ok = run(Path(os.environ["DIAGFLOWBENCH_PATH"]), only=["Llama 3.3", "Gemini"])
    assert rows and ok
    assert all(r["match"] for r in rows if r["metric"] != "termination_recognition")


def test_load_runs_ignores_foreign_json(tmp_path):
    import json
    from harness.diagflow_demo.evaluate import load_runs
    (tmp_path / "m" / "react").mkdir(parents=True)
    (tmp_path / "m" / "react" / "WO1__r0.json").write_text(json.dumps({"item_id": "WO1", "run": 0}))
    (tmp_path / "m" / "react" / "C1__r0.json").write_text(json.dumps({"conversation_id": "C1", "per_turn": []}))
    assert [r["conversation_id"] for r in load_runs(tmp_path)] == ["C1"]
