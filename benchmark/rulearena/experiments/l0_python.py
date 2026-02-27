"""
L0: Python Oracle Experiment

Passes ground truth parameters directly to Python calculators.
Zero LLM calls - establishes the accuracy ceiling for each domain.

Supported domains:
- airline: compute_airline_fee(instance.metadata)
- tax: compute_tax_fee(instance.metadata)
- nba: not supported (no deterministic checker)
"""

import time

from benchmark.rulearena.dataset.loader import RuleArenaInstance
from benchmark.rulearena.experiments.base import BaseExperiment, ExperimentResult


class L0PythonExperiment(BaseExperiment):
    """
    L0: Python oracle baseline.

    Feeds ground truth params straight into the deterministic calculator.
    No LLM involvement - this measures calculator correctness only.
    """

    def __init__(self):
        super().__init__(
            experiment_name="l0_python",
            model_id="none",
        )

    def run_instance(self, instance: RuleArenaInstance) -> ExperimentResult:
        start_time = time.time()

        if instance.domain == "nba":
            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=None,
                expected=instance.ground_truth_answer,
                is_correct_exact=False,
                is_correct_tolerance=False,
                latency_ms=0.0,
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                calculator_name="nba",
                error="L0 not supported for NBA domain",
                metadata={
                    "domain": instance.domain,
                    "complexity_level": instance.complexity_level,
                },
            )

        try:
            predicted = self._compute(instance)
            elapsed_ms = (time.time() - start_time) * 1000

            exact, tolerance = self.compare_answers(
                predicted, instance.ground_truth_answer
            )

            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=predicted,
                expected=instance.ground_truth_answer,
                is_correct_exact=exact,
                is_correct_tolerance=tolerance,
                latency_ms=elapsed_ms,
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                calculator_name=instance.domain,
                metadata={
                    "domain": instance.domain,
                    "complexity_level": instance.complexity_level,
                },
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=None,
                expected=instance.ground_truth_answer,
                is_correct_exact=False,
                is_correct_tolerance=False,
                latency_ms=elapsed_ms,
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                calculator_name=instance.domain,
                error=str(e),
                metadata={
                    "domain": instance.domain,
                    "complexity_level": instance.complexity_level,
                },
            )

    def _compute(self, instance: RuleArenaInstance):
        if instance.domain == "airline":
            from benchmark.rulearena.calculators.airline import compute_airline_fee
            return compute_airline_fee(instance.metadata)

        if instance.domain == "tax":
            from benchmark.rulearena.calculators.tax import compute_tax_fee
            result = compute_tax_fee(instance.metadata)
            if result is None:
                raise RuntimeError("Tax calculator returned None")
            return result

        raise ValueError(f"Unsupported domain: {instance.domain}")
