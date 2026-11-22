"""On-procedure capability metrics: does the assistant follow the documented procedure?

All three are Level 2 metrics under Accuracy, higher is better (fixed H bands),
computed over a list of scored turns. Each turn is a dict carrying the
assistant's reply, the expected position and the graph it belongs to, so turns
from many conversations and flowcharts can be pooled.
"""

from __future__ import annotations

from typing import Any

from contracts.banding import higher_is_better_rate_band
from contracts.evaluator import EvaluationResult, WP3Evaluator

from .node_matching import (
    DEFAULT_THRESHOLD,
    adjacency,
    best_node,
    forward_reachable,
    jaccard,
    match_node,
    nodes_of,
)


def _result(rate: float, n: int, what: str, extra: dict[str, Any]) -> EvaluationResult:
    band = higher_is_better_rate_band(rate)
    return EvaluationResult(
        score=band.level / 4.0,
        explanation=f"{what}: {rate:.3f} over {n} turns; higher is better -- band {band.level} ({band.label}).",
        is_success=band.level >= 2,
        metadata={"rate": rate, "n_turns": n, "band_level": band.level, "band_label": band.label, **extra},
    )


class _TurnMetric(WP3Evaluator):
    dimensions = ("Accuracy",)
    applicability = "aggregate"

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError(f"{type(self).__name__} operates on a list of scored turns. Call evaluate_turns(turns) instead.")


class StepAccuracyEvaluator(_TurnMetric):
    """Share of on-procedure turns where the reply names exactly the expected next node.

    turn keys: ``response`` (str), ``expected_node_id`` (str), ``graph`` (graph dict).
    Metadata also carries the exact-or-forward-reachable rate (right node, or a node within two hops ahead)
    and the off-graph rate (reply resolves to no node at all).
    """

    metric_id = "step_accuracy"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold

    def evaluate_turns(self, turns: list[dict[str, Any]]) -> EvaluationResult:
        if not turns:
            raise ValueError("evaluate_turns requires at least one turn.")
        exact = exact_or_forward = off_graph = 0
        for t in turns:
            g = t["graph"]
            matched, _ = match_node(t["response"], nodes_of(g), self.threshold)
            if matched is None:
                off_graph += 1
                continue
            if matched == t["expected_node_id"]:
                exact += 1
                exact_or_forward += 1
            elif matched in forward_reachable(t["expected_node_id"], adjacency(g)):
                exact_or_forward += 1
        n = len(turns)
        return _result(exact / n, n, "Step accuracy", {
            "threshold": self.threshold, "n_exact": exact,
            "forward_reachable_rate": exact_or_forward / n, "off_graph_rate": off_graph / n,
        })


class BranchFollowingEvaluator(_TurnMetric):
    """Share of decision-follow turns where the reply goes down the branch the operator's answer selects.

    turn keys: ``response``, ``decision_node_id``, ``expected_label`` (e.g. "Yes"), ``graph``.
    The reply is matched against the successors of the decision node only.
    """

    metric_id = "branch_following"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold

    def evaluate_turns(self, turns: list[dict[str, Any]]) -> EvaluationResult:
        if not turns:
            raise ValueError("evaluate_turns requires at least one turn.")
        correct = 0
        for t in turns:
            g = t["graph"]
            nodes = nodes_of(g)
            best_label, best_score = None, 0.0
            for target, label in adjacency(g).get(t["decision_node_id"], []):
                s = jaccard(t["response"], nodes.get(target, {}).get("text", ""))
                if s > best_score:
                    best_label, best_score = label, s
            if best_score >= self.threshold and (best_label or "") == (t["expected_label"] or ""):
                correct += 1
        return _result(correct / len(turns), len(turns), "Branch following", {"threshold": self.threshold, "n_correct": correct})


class TerminationRecognitionEvaluator(_TurnMetric):
    """Share of turns at a terminal node where the reply proposes no further step.

    turn keys: ``response``, ``graph``. By default a reply is correct when it resolves to no graph node,
    exactly as in the DiagFlowBench code. That rule also fails a reply such as "the procedure has ended at
    'End, Notify maintenance team'", which quotes the terminal node while proposing nothing further.
    ``accept_terminal_node=True`` counts a reply that resolves to a terminal node as correct too, so only
    a reply that proposes a non-terminal step fails.
    """

    metric_id = "termination_recognition"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, accept_terminal_node: bool = False):
        self.threshold = threshold
        self.accept_terminal_node = accept_terminal_node

    def evaluate_turns(self, turns: list[dict[str, Any]]) -> EvaluationResult:
        if not turns:
            raise ValueError("evaluate_turns requires at least one turn.")
        correct = 0
        for t in turns:
            nodes = nodes_of(t["graph"])
            matched, _ = match_node(t["response"], nodes, self.threshold)
            if matched is None or (self.accept_terminal_node and nodes[matched].get("type") == "terminator"):
                correct += 1
        return _result(correct / len(turns), len(turns), "Termination recognition",
                       {"threshold": self.threshold, "n_correct": correct, "accept_terminal_node": self.accept_terminal_node})


__all__ = ["StepAccuracyEvaluator", "BranchFollowingEvaluator", "TerminationRecognitionEvaluator", "best_node"]
