"""Tests for the MaintIE live-demo pipeline. Uses a tiny synthetic corpus in
the real MaintIE record format; no network, no model calls, no WP4 import."""

import json

import pytest

from harness.adapters import from_wp4_pattern_result
from harness.maintie_demo.capture import capture, run_path, serialize_messages
from harness.maintie_demo.dashboard import build_html
from harness.maintie_demo.data import parse_record, select_eval_items
from harness.maintie_demo.evaluate import evaluate_all, load_runs
from harness.maintie_demo.fake import make_fake_agent
from harness.maintie_demo.retrieval import BM25Index, OntologyLookup
from harness.maintie_demo.scoring import build_task, parse_prediction, score_prediction

REC_A = {
    "text": "<id> air conditioner thermostat not working",
    "tokens": ["<id>", "air", "conditioner", "thermostat", "not", "working"],
    "entities": [
        {"start": 1, "end": 3, "type": "PhysicalObject/EmittingObject/ElectricCoolingObject"},
        {"start": 3, "end": 4, "type": "PhysicalObject/SensingObject/TemperatureSensingObject"},
        {"start": 4, "end": 6, "type": "State/UndesirableState/FailedState"},
    ],
    "relations": [{"head": 0, "tail": 1, "type": "hasPart"}, {"head": 2, "tail": 1, "type": "hasParticipant/hasPatient"}],
}
REC_B = {
    "text": "<id> replace pump seal",
    "tokens": ["<id>", "replace", "pump", "seal"],
    "entities": [
        {"start": 1, "end": 2, "type": "Activity/MaintenanceActivity/Replace"},
        {"start": 2, "end": 3, "type": "PhysicalObject/Pump"},
        {"start": 3, "end": 4, "type": "PhysicalObject/Seal"},
    ],
    "relations": [{"head": 0, "tail": 2, "type": "hasParticipant/hasPatient"}, {"head": 1, "tail": 2, "type": "hasPart"}],
}


@pytest.fixture
def items():
    return [parse_record(REC_A, "g00000"), parse_record(REC_B, "g00001")]


def test_parse_record_uses_exclusive_spans_and_drops_id_token(items):
    a = items[0]
    assert a.text == "air conditioner thermostat not working"
    assert [e.text for e in a.entities] == ["air conditioner", "thermostat", "not working"]
    assert [e.coarse for e in a.entities] == ["PhysicalObject", "PhysicalObject", "State"]
    assert a.relations[0].head == "air conditioner" and a.relations[0].tail == "thermostat"
    assert a.relations[1].head == "not working"


def test_parse_prediction_accepts_dict_json_and_fenced_json():
    d = {"entities": [{"text": "pump", "type": "PhysicalObject"}], "relations": []}
    want = dict(d, n_invalid=0)
    assert parse_prediction(d) == want
    assert parse_prediction(json.dumps(d)) == want
    assert parse_prediction("Here you go:\n```json\n" + json.dumps(d) + "\n```") == want
    assert parse_prediction("prefix " + json.dumps(d) + " suffix") == want


@pytest.mark.parametrize("bad", [None, "no json here", {"entities": "x"}, {"relations": []}, 42])
def test_parse_prediction_returns_none_when_no_answer_recoverable(bad):
    assert parse_prediction(bad) is None


def test_parse_prediction_is_lenient_about_out_of_scheme_items():
    out = {"entities": [{"text": "pump", "type": "PhysicalObject"}, {"text": "a", "type": "Banana"}, {"type": "State"}],
           "relations": [{"head": "a", "tail": "b", "type": "located_at"}, {"head": "pump", "tail": "x", "type": "hasPart"}]}
    p = parse_prediction(out)
    assert [e["text"] for e in p["entities"]] == ["pump"]
    assert [r["type"] for r in p["relations"]] == ["hasPart"]
    assert p["n_invalid"] == 3
    s = score_prediction(parse_record(REC_B, "x"), out)
    assert not s["valid"] and s["entity"]["tp"] == 1 and s["entity"]["n_pred"] == 1  # the valid entity is still scored


def test_silver_style_relation_names_are_accepted_as_aliases():
    out = {"entities": [], "relations": [{"head": "a", "tail": "b", "type": "hasPatient"},
                                           {"head": "a", "tail": "b", "type": "hasAgent"}]}
    p = parse_prediction(out)
    assert [r["type"] for r in p["relations"]] == ["hasParticipant/hasPatient", "hasParticipant/hasAgent"] and p["n_invalid"] == 0


def test_parse_record_normalises_silver_relation_names():
    rec = dict(REC_B, relations=[{"head": 0, "tail": 2, "type": "hasPatient"}])
    assert parse_record(rec, "s0").relations[0].type == "hasParticipant/hasPatient"


def test_json_is_recovered_after_free_text_reasoning_and_draft_json():
    final = {"entities": [{"text": "pump", "type": "PhysicalObject"}], "relations": []}
    draft = '{"entities": [{"text": "draft", "type": "State"}], "relations": []} // not final'
    text = "Reasoning: let me think {with braces}.\n" + draft + "\nso the answer is:\n" + json.dumps(final, indent=2)
    p = parse_prediction(text)
    assert [e["text"] for e in p["entities"]] == ["pump"]


def test_score_prediction_perfect_partial_wrong_class_and_invalid(items):
    a = items[0]
    perfect = {"entities": [{"text": e.text, "type": e.coarse} for e in a.entities],
               "relations": [{"head": r.head, "tail": r.tail, "type": r.type} for r in a.relations]}
    s = score_prediction(a, perfect)
    assert s["entity"]["f1"] == 1.0 and s["relation"]["f1"] == 1.0 and s["valid"]

    wrong = {"entities": [{"text": "thermostat", "type": "State"}], "relations": []}
    s = score_prediction(a, wrong)
    assert s["entity"]["tp"] == 0 and s["entity"]["n_gold"] == 3

    s = score_prediction(a, "garbage")
    assert not s["valid"] and s["entity"]["recall"] == 0.0 and s["entity"]["n_gold"] == 3
    assert score_prediction(a, perfect)["valid"] is True


def test_matching_is_case_and_whitespace_insensitive(items):
    out = {"entities": [{"text": "  Thermostat ", "type": "PhysicalObject"}], "relations": []}
    assert score_prediction(items[0], out)["entity"]["tp"] == 1


def test_task_definition_oracle_and_subgoals(items):
    a = items[0]
    task = build_task(a, optimal_steps=3)
    full = {"entities": [{"text": e.text, "type": e.coarse} for e in a.entities], "relations": []}
    assert task.oracle_fn(full) is True
    assert task.oracle_fn({"entities": full["entities"][:2], "relations": []}) is False
    assert task.oracle_fn(None) is False
    assert len(task.subgoal_checks) == 3 + 2  # entities + relations
    assert sum(c(full) for c in task.subgoal_checks) == 3
    assert task.optimal_steps == 3


def test_select_eval_items_is_deterministic_and_filters(items):
    many = [parse_record(REC_A if i % 2 else REC_B, f"g{i:05d}") for i in range(20)]
    a, b = select_eval_items(many, 5, seed=1), select_eval_items(many, 5, seed=1)
    assert [i.item_id for i in a] == [i.item_id for i in b] and len(a) == 5
    assert select_eval_items(many, 5, seed=2) != a


def test_retrieval_never_returns_excluded_eval_texts(items):
    index = BM25Index(items, exclude_texts={items[0].text})
    hits = index.search("air conditioner thermostat", k=5)
    assert all(h.text != items[0].text for h in hits)
    assert "pump seal" in BM25Index(items).retrieve("pump seal")
    assert BM25Index(items).retrieve("zzzz") .startswith("No similar")


def test_ontology_lookup_uses_silver_only(items):
    lk = OntologyLookup(items, exclude_texts={items[1].text})
    assert lk.lookup("thermostat").startswith("PhysicalObject")
    assert lk.lookup("pump") == "unknown"  # only seen in the excluded item
    assert OntologyLookup(items).lookup("replace").startswith("Activity")


def test_adapter_falls_back_to_last_ai_text_when_no_structured_response():
    msgs = [{"type": "human", "content": "q"}, {"type": "ai", "content": "", "tool_calls": [{"name": "t"}]},
            {"type": "ai", "content": '{"entities": []}'}]
    traj = from_wp4_pattern_result({"messages": msgs, "context": ["c"]}, "q")
    assert traj.output == '{"entities": []}'


def test_serialize_messages_handles_objects_and_dicts():
    class M:
        type, content, tool_calls, usage_metadata = "ai", "hi", [{"name": "x", "args": {}, "id": "1"}], {"input_tokens": 3}
    out = serialize_messages([M(), {"type": "human", "content": "q"}])
    assert out[0]["tool_calls"][0]["name"] == "x" and out[0]["usage_metadata"]["input_tokens"] == 3
    json.dumps(out)


def test_capture_resumes_retries_and_records_errors(tmp_path, items):
    calls = {"n": 0}
    good = make_fake_agent("react", items)

    def flaky(text):
        calls["n"] += 1
        if text == items[1].text:
            raise RuntimeError("TooManyRequests")
        return good(text)

    kw = dict(model="m/free:x", pattern="react", runs=2, out_dir=tmp_path, retries=2, backoff=0, sleep=lambda s: None, log=lambda s: None)
    counts = capture(items, flaky, **kw)
    assert counts == {"done": 2, "skipped": 0, "errors": 2}
    assert calls["n"] == 2 + 2 * 3  # item0 twice, item1 twice with 3 attempts each
    rec = json.loads(run_path(tmp_path, "m/free:x", "react", items[1].item_id, 0).read_text())
    assert "TooManyRequests" in rec["error"] and rec["attempts"] == 3

    before = calls["n"]
    assert capture(items, flaky, **kw)["skipped"] == 4 and calls["n"] == before  # resume: nothing re-run
    assert capture(items, good, retry_errors=True, **kw) == {"done": 2, "skipped": 2, "errors": 0}


def test_end_to_end_evaluation_and_dashboard(tmp_path, items):
    for pattern in ("react", "basic_rag"):
        capture(items, make_fake_agent(pattern, items, drop=0.0, wrong_class=0.0), model="m/a", pattern=pattern, runs=3,
                out_dir=tmp_path, log=lambda s: None, provenance={"wp4_commit": "abc"})
    bad = lambda t: (_ for _ in ()).throw(RuntimeError("boom"))  # noqa: E731
    capture(items, bad, model="m/b", pattern="react", runs=3, out_dir=tmp_path, retries=0, backoff=0, log=lambda s: None)

    res = evaluate_all(load_runs(tmp_path), {i.item_id: i for i in items}, prices={"m/a": (0.001, 0.002)})
    cfg = {(c["model"], c["pattern"]): c for c in res["configs"]}
    perfect = cfg[("m/a", "react")]
    assert perfect["task_success_rate"] == 1.0 and perfect["entity"]["f1"] == 1.0 and perfect["relation"]["f1"] == 1.0
    assert perfect["pass_hat_k"]["score"] == 1.0
    assert perfect["cross_run_consistency"]["band_level"] == 4
    assert perfect["schema_conformance"]["score"] == 1.0
    assert perfect["cost_mean_usd"] > 0 and perfect["tool_calls_mean"] > 1
    assert cfg[("m/a", "basic_rag")]["step_efficiency"] == 1.0

    broken = cfg[("m/b", "react")]
    assert broken["n_errors"] == broken["n_runs"] == 6
    assert broken["task_success_rate"] == 0.0 and broken["schema_conformance"]["score"] == 0.0

    html = build_html(res)
    assert "AIXPERT WP3" in html and '"m/a"' in html
    assert "</script><script" not in html.split('id="data"')[1].split("</script>")[0]


# ---- robustness: auth abort, timeout, request guard, evaluate filters ----

def _kw(tmp_path, **extra):
    return dict(model="m/free:x", pattern="react", runs=1, out_dir=tmp_path, retries=2, backoff=0,
                sleep=lambda s: None, log=lambda s: None, **extra)


def test_auth_error_aborts_without_retry_and_writes_nothing(tmp_path, items):
    from harness.maintie_demo.capture import CaptureAborted, is_fatal_error
    calls = {"n": 0}

    def unauthorized(text):
        calls["n"] += 1
        raise RuntimeError("UnauthorizedResponseError: User not found.")

    with pytest.raises(CaptureAborted):
        capture(items, unauthorized, **_kw(tmp_path))
    assert calls["n"] == 1  # no retries on an error that cannot succeed
    assert not list(tmp_path.rglob("*.json"))  # resume starts exactly here
    assert is_fatal_error("HTTP 401 from provider") and not is_fatal_error("TooManyRequests: 4012 ms")


def test_timeout_is_recorded_as_error_and_not_retried(tmp_path, items):
    import time as _t
    calls = {"n": 0}

    def slow(text):
        calls["n"] += 1
        _t.sleep(5)

    counts = capture(items[:1], slow, **_kw(tmp_path, timeout=0.2))
    assert counts == {"done": 0, "skipped": 0, "errors": 1} and calls["n"] == 1
    rec = json.loads(run_path(tmp_path, "m/free:x", "react", items[0].item_id, 0).read_text())
    assert rec["error"].startswith("RunTimeout") and rec["wall_seconds"] < 2


def test_request_guard_stops_cleanly_and_resumes(tmp_path, items):
    good = make_fake_agent("react", items)
    left = iter([100, 3])  # first check passes, second falls below the guard
    counts = capture(items, good, **_kw(tmp_path, min_free_requests=10, remaining_fn=lambda: next(left), check_every=1))
    assert counts["done"] == 1 and "only 3 free requests" in counts["stopped"]
    assert capture(items, good, **_kw(tmp_path))["skipped"] == 1  # the finished run is kept, the rest resumes


def test_free_requests_remaining_parses_openrouter_payload(monkeypatch):
    from harness.maintie_demo.capture import free_requests_remaining
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)  # an empty key falls back to the environment; a real key there must not leak in
    body = {"data": {"free_model_daily_requests": {"used": 47, "limit": 50, "remaining": 3}}}
    assert free_requests_remaining("k", fetch=lambda url, headers: body) == 3
    assert free_requests_remaining("k", fetch=lambda url, headers: {"data": {}}) is None
    assert free_requests_remaining("", fetch=lambda url, headers: body) is None

    def boom(url, headers):
        raise OSError("offline")
    assert free_requests_remaining("k", fetch=boom) is None


def test_filter_records_puts_configs_on_the_same_items_and_runs(tmp_path, items):
    from harness.maintie_demo.evaluate import coverage_warning, filter_records
    capture(items, make_fake_agent("react", items), model="m/a", pattern="react", runs=3, out_dir=tmp_path, log=lambda s: None)
    capture(items[:1], make_fake_agent("react", items), model="m/b", pattern="react", runs=2, out_dir=tmp_path, log=lambda s: None)
    recs = load_runs(tmp_path)
    assert len(recs) == 6 + 2
    assert {r["run"] for r in filter_records(recs, max_runs=2)} == {0, 1}
    assert {r["model"] for r in filter_records(recs, models=["M/B"])} == {"m/b"}
    assert {r["item_id"] for r in filter_records(recs, item_ids={items[1].item_id})} == {items[1].item_id}
    common = filter_records(recs, common=True)
    assert {(r["item_id"], r["run"]) for r in common} == {(items[0].item_id, 0), (items[0].item_id, 1)}
    gold = {i.item_id: i for i in items}
    assert coverage_warning(evaluate_all(recs, gold)["configs"]) is not None
    assert coverage_warning(evaluate_all(common, gold)["configs"]) is None


def test_cost_is_priced_free_or_not_measured(tmp_path, items):
    for model in ("m/table", "m/paid", "m/other:free"):
        capture(items, make_fake_agent("react", items, drop=0.0, wrong_class=0.0), model=model, pattern="react", runs=2,
                out_dir=tmp_path, log=lambda s: None)
    res = evaluate_all(load_runs(tmp_path), {i.item_id: i for i in items}, prices={"m/table": (0.001, 0.002)})
    cfg = {c["model"]: c for c in res["configs"]}
    table, paid, free = cfg["m/table"], cfg["m/paid"], cfg["m/other:free"]
    assert table["price_basis"] == "table" and table["cost_mean_usd"] > 0 and table["cost"] is not None
    assert free["price_basis"] == "free" and free["cost_mean_usd"] == 0.0 and free["cost"]["band_level"] == 4
    # a paid model with no price entry must not pass as free: its cost is not measured, not banded
    assert paid["price_basis"] == "none" and paid["cost_mean_usd"] is None and paid["cost"] is None
    assert "not priced" in build_html(res)
