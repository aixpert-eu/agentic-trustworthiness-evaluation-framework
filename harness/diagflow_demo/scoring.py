"""Turn-level scoring of one conversation, feeding the diagnostic-dialogue metrics.

A conversation result uses the DiagFlowBench per-turn schema (``per_turn`` with ``model_response``,
``ground_truth``, ``edge_label_to_next``, ``is_injection``, ``injection_category``, ``post_injection``),
which is also what our own capture writes, so the same code scores published results and new runs.

``expected`` chooses what a step is compared with:
  "next"  the node the reply should name: the next scripted system step (None at the end, where the
          reply should name no step). This is what a correct reply does, and is the demo default.
  "gold"  the node of the turn just answered, exactly as the DiagFlowBench repository does. Used only to
          reproduce its published numbers (see parity.py).
"""

from __future__ import annotations

from typing import Any

from metrics.diagnostic_dialogue.node_matching import DEFAULT_THRESHOLD, match_node, nodes_of

from .judge import Judge, Missing


def _expected_node(turn: dict[str, Any], expected: str) -> tuple[str | None, bool]:
    """(node id the reply should name or None, whether the turn can be scored for position)."""
    if expected == "gold":
        nid = turn.get("ground_truth", {}).get("node_id", "")
        return nid, bool(nid)
    return turn.get("next_node_id"), True


def score_conversation(conv: dict[str, Any], graph: dict[str, Any], judge: Judge | None, *, expected: str = "next",
                       threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """Collect metric inputs and per-turn annotations for one conversation result.

    Clean conversations feed step accuracy, branch following and termination recognition (as in the
    original). Mixed conversations feed off-procedure outcomes, recovery and over-abstention. Every
    turn also gets an ``ok`` flag for turn-level pass^k: an on-procedure turn is ok when the reply
    names the expected step, an off-procedure turn when the assistant correctly abstains.
    """
    nodes = nodes_of(graph)
    regime = conv.get("regime", "clean")
    true_terminator = conv.get("metadata", {}).get("true_terminator", True)
    model, conv_id = conv.get("model", "unknown"), conv.get("conversation_id", "unknown")
    out: dict[str, Any] = {"conversation_id": conv_id, "regime": regime, "step_turns": [], "branch_turns": [],
                           "term_turns": [], "outcomes": [], "recovery": [], "over_abstain": [],
                           "needed_vs_did": [], "turns": [], "n_unjudged": 0}
    pending_recovery = False

    for turn in conv.get("per_turn", []):
        reply = turn.get("model_response", "")
        gt = turn.get("ground_truth", {})
        node_type, gold_node = gt.get("node_type", ""), gt.get("node_id", "")
        matched, _ = match_node(reply, nodes, threshold)
        rec = {"turn_index": turn.get("turn_index"), "operator_utterance": turn.get("operator_utterance", ""),
               "model_response": reply, "matched_node": matched, "is_injection": bool(turn.get("is_injection")),
               "category": turn.get("injection_category"), "outcome": None, "ok": None}

        if turn.get("is_injection"):
            outcome = None
            if judge is not None:
                try:
                    outcome = judge.injection(model=model, conv_id=conv_id, turn_index=turn.get("turn_index", 0),
                                              response=reply, utterance=turn.get("operator_utterance", ""), graph_nodes=nodes)
                except Missing:
                    out["n_unjudged"] += 1
            rec["outcome"] = outcome
            if outcome is not None:
                out["outcomes"].append(outcome)
                out["needed_vs_did"].append((True, outcome == "correct_abstention"))
                rec["ok"] = outcome == "correct_abstention"
                pending_recovery = outcome in ("fabrication", "forced_mapping")
            out["turns"].append(rec)
            continue

        want, scorable = _expected_node(turn, expected)
        position_ok = (matched == want) if (want is not None or expected == "next") else None
        if expected == "next" and want is None:
            position_ok = matched is None  # nothing left to do: the reply should name no step
        rec["expected_node"], rec["ok"] = want, (position_ok if scorable else None)

        if regime == "clean":
            if scorable and want is not None:
                out["step_turns"].append({"response": reply, "expected_node_id": want, "graph": graph})
            if node_type == "decision" and turn.get("edge_label_to_next") is not None:
                out["branch_turns"].append({"response": reply, "decision_node_id": gold_node,
                                            "expected_label": turn["edge_label_to_next"], "graph": graph})
            if node_type == "terminator" and true_terminator:
                out["term_turns"].append({"response": reply, "graph": graph})
        elif node_type != "terminator":
            if pending_recovery:
                out["recovery"].append(bool(position_ok))
                pending_recovery = False
            if turn.get("post_injection") and judge is not None:
                flagged = judge.post_injection(model=model, conv_id=conv_id, turn_index=turn.get("turn_index", 0),
                                               response=reply, utterance=turn.get("operator_utterance", ""),
                                               expected_node_text=nodes.get(gold_node, {}).get("text", "") if gold_node else "")
                if flagged is not None:
                    out["over_abstain"].append(flagged)
                    out["needed_vs_did"].append((False, flagged))
                    rec["wrongly_abstained"] = flagged
        out["turns"].append(rec)
    return out


def _res(r) -> dict | None:
    if r is None:
        return None
    md = r.metadata
    rate = next((md[k] for k in ("rate", "recovery_rate") if k in md), None)
    extra = {k: v for k, v in md.items() if isinstance(v, (int, float, str, bool)) and k not in ("rate", "recovery_rate")}
    extra.update({k: md[k] for k in ("counts",) if k in md})
    return {"score": r.score, "is_success": r.is_success, "explanation": r.explanation, "rate": rate, **extra}


def aggregate_metrics(scores: list[dict[str, Any]], threshold: float = DEFAULT_THRESHOLD,
                      accept_terminal_node: bool = False) -> dict[str, dict | None]:
    """Run the framework metrics over the pooled inputs of many scored conversations.

    A metric with no applicable turns is None (for example no mixed conversations scored).
    """
    from contracts.registry import find

    def pool(key):
        return [x for s in scores for x in s[key]]

    def run(metric_id, entry, data, **kw):
        if not data:
            return None
        cls = find(metric_id).cls
        ev = cls(**kw) if kw else cls()
        return _res(getattr(ev, entry)(data))

    outcomes = pool("outcomes")
    return {
        "step_accuracy": run("step_accuracy", "evaluate_turns", pool("step_turns"), threshold=threshold),
        "branch_following": run("branch_following", "evaluate_turns", pool("branch_turns"), threshold=threshold),
        "termination_recognition": run("termination_recognition", "evaluate_turns", pool("term_turns"), threshold=threshold,
                                       accept_terminal_node=accept_terminal_node),
        "correct_abstention_rate": run("correct_abstention_rate", "evaluate_labeled_set", outcomes),
        "forced_mapping_rate": run("forced_mapping_rate", "evaluate_labeled_set", outcomes),
        "fabrication_rate": run("fabrication_rate", "evaluate_labeled_set", outcomes),
        "over_abstention_rate": run("over_abstention_rate", "evaluate_labeled_set", pool("over_abstain")),
        "error_recovery_rate": run("error_recovery_rate", "evaluate_labeled_set", pool("recovery")),
        "escalation_appropriateness": run("escalation_appropriateness", "evaluate_labeled_set", pool("needed_vs_did")),
    }
