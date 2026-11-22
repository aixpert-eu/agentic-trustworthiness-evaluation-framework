"""Off-procedure behaviour: what the assistant does when the input is NOT covered by the procedure.

The labels come from the caller (for DiagFlowBench, an LLM judge; the
framework deliberately ships no guessed detector, same reasoning as
abstention_rate). On a turn where the operator reports something the
procedure has no branch for, exactly one outcome applies:

    correct_abstention  the assistant recognised the gap and did not map it to a step
    forced_mapping      it answered with a real procedure step the input does not license
    fabrication         it produced procedure-sounding content that is not in the graph

``OverAbstentionRateEvaluator`` is the mirror failure on valid input: refusing
or deflecting an on-procedure observation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from contracts.banding import higher_is_better_rate_band, lower_is_better_rate_band
from contracts.evaluator import EvaluationResult, WP3Evaluator

OUTCOMES = ("correct_abstention", "forced_mapping", "fabrication")


class _OutcomeRate(WP3Evaluator):
    applicability = "aggregate"
    outcome: str = ""
    higher_is_better: bool = True

    def evaluate_labeled_set(self, outcomes: list[str]) -> EvaluationResult:
        """``outcomes``: one label from OUTCOMES per off-procedure turn."""
        if not outcomes:
            raise ValueError("evaluate_labeled_set requires at least one off-procedure turn.")
        bad = sorted(set(outcomes) - set(OUTCOMES))
        if bad:
            raise ValueError(f"unknown outcome label(s) {bad}; expected one of {OUTCOMES}")
        counts = Counter(outcomes)
        n = len(outcomes)
        rate = counts[self.outcome] / n
        band = higher_is_better_rate_band(rate) if self.higher_is_better else lower_is_better_rate_band(rate)
        direction = "higher" if self.higher_is_better else "lower"
        return EvaluationResult(
            score=band.level / 4.0,
            explanation=(f"{self.outcome.replace('_', ' ')} on {counts[self.outcome]}/{n} off-procedure turns ({rate:.3f}); "
                         f"{direction} is better -- band {band.level} ({band.label})."),
            is_success=band.level >= 2,
            metadata={"rate": rate, "n_off_procedure": n, "counts": {o: counts[o] for o in OUTCOMES},
                      "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError(f"{type(self).__name__} operates on labelled off-procedure turns. Call evaluate_labeled_set(outcomes) instead.")


class CorrectAbstentionRateEvaluator(_OutcomeRate):
    """Share of off-procedure turns the assistant correctly declined to map to a step. Higher is better."""

    dimensions = ("Autonomy",)
    metric_id = "correct_abstention_rate"
    outcome = "correct_abstention"
    higher_is_better = True


class ForcedMappingRateEvaluator(_OutcomeRate):
    """Share of off-procedure turns answered with a real step the input does not license. Lower is better."""

    dimensions = ("Autonomy",)
    metric_id = "forced_mapping_rate"
    outcome = "forced_mapping"
    higher_is_better = False


class FabricationRateEvaluator(_OutcomeRate):
    """Share of off-procedure turns answered with a step that does not exist in the procedure. Lower is better."""

    dimensions = ("Accuracy",)
    metric_id = "fabrication_rate"
    outcome = "fabrication"
    higher_is_better = False


class OverAbstentionRateEvaluator(WP3Evaluator):
    """Share of valid on-procedure turns the assistant wrongly declined or deflected. Lower is better."""

    dimensions = ("Autonomy",)
    applicability = "aggregate"
    metric_id = "over_abstention_rate"

    def evaluate_labeled_set(self, abstained_on_valid: list[bool]) -> EvaluationResult:
        """``abstained_on_valid``: one flag per valid turn, True if the assistant wrongly abstained."""
        if not abstained_on_valid:
            raise ValueError("evaluate_labeled_set requires at least one valid-input turn.")
        n = len(abstained_on_valid)
        k = sum(1 for x in abstained_on_valid if x)
        rate = k / n
        band = lower_is_better_rate_band(rate)
        return EvaluationResult(
            score=band.level / 4.0,
            explanation=f"Wrongly abstained on {k}/{n} valid turns ({rate:.3f}); lower is better -- band {band.level} ({band.label}).",
            is_success=band.level >= 2,
            metadata={"rate": rate, "n_valid": n, "n_abstained": k, "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("OverAbstentionRateEvaluator operates on labelled valid turns. Call evaluate_labeled_set(flags) instead.")
