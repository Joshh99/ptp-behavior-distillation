"""
Base Experiment Classes

Defines the abstract base class for experiments and result dataclasses.
Schema MUST match MedCalc's ExperimentResult for cross-benchmark compatibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from benchmark.rulearena.dataset.loader import RuleArenaInstance


@dataclass
class ExperimentResult:
    """
    Per-instance result.

    Schema aligned with AgentProject for future merge into
    AgentProject/benchmark/rulearena/.
    """
    instance_id: str
    predicted: Any                   # Model's answer
    expected: Any                    # Ground truth
    is_correct_exact: bool           # Whether predicted matches expected (exact)
    is_correct_tolerance: bool       # Whether predicted matches within tolerance
    latency_ms: float                # Wall-clock time in milliseconds
    cost_usd: float                  # API cost for this instance
    input_tokens: int
    output_tokens: int
    calculator_name: str = ""        # Domain name (e.g. "airline", "tax", "nba")
    category: str = ""               # Category placeholder
    timestamp: str = ""              # ISO-format creation time
    error: Optional[str] = None      # Error message if failed
    raw_response: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        """Convert to dict, handling numpy types for JSON serialization."""
        def convert_value(val):
            """Convert numpy types to Python native types."""
            # Handle numpy types
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            elif isinstance(val, dict):
                return {k: convert_value(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [convert_value(v) for v in val]
            return val

        return {
            "instance_id": self.instance_id,
            "predicted": convert_value(self.predicted),
            "expected": convert_value(self.expected),
            "is_correct_exact": self.is_correct_exact,
            "is_correct_tolerance": self.is_correct_tolerance,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calculator_name": self.calculator_name,
            "category": self.category,
            "timestamp": self.timestamp,
            "error": self.error,
            "raw_response": self.raw_response,
            "metadata": convert_value(self.metadata),
        }


class BaseExperiment(ABC):
    """
    Abstract base class for all experiments.

    Each experiment level (L0, L0F, L1, L1-TA, L3) extends this class.
    """

    def __init__(self, experiment_name: str, model_id: str = "deepseek-ai/DeepSeek-V3"):
        """
        Initialize experiment.

        Args:
            experiment_name: Name of the experiment (e.g., "l1_ptool")
            model_id: Model identifier for API calls
        """
        self.experiment_name = experiment_name
        self.model_id = model_id

    @abstractmethod
    def run_instance(self, instance: RuleArenaInstance) -> ExperimentResult:
        """
        Run experiment on a single instance.

        Args:
            instance: RuleArenaInstance to evaluate

        Returns:
            ExperimentResult with predictions and metrics

        Raises:
            Should NOT raise exceptions - catch and record in ExperimentResult.error
        """
        pass

    def run_batch(
        self, instances: List[RuleArenaInstance]
    ) -> List[ExperimentResult]:
        """
        Run experiment on a batch of instances.

        Args:
            instances: List of RuleArenaInstances

        Returns:
            List of ExperimentResults (one per instance)
        """
        results = []
        for instance in instances:
            try:
                result = self.run_instance(instance)
                results.append(result)
            except Exception as e:
                # Catch any unhandled errors and record them
                error_result = ExperimentResult(
                    instance_id=instance.instance_id,
                    predicted=None,
                    expected=instance.ground_truth_answer,
                    is_correct_exact=False,
                    is_correct_tolerance=False,
                    latency_ms=0.0,
                    cost_usd=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    calculator_name=instance.domain,
                    error=f"Unhandled error: {str(e)}",
                    raw_response=None,
                    metadata={"domain": instance.domain, "complexity_level": instance.complexity_level},
                )
                results.append(error_result)

        return results

    def compare_answers(self, predicted: Any, expected: Any) -> tuple[bool, bool]:
        """
        Compare predicted answer to expected answer.

        Returns:
            (exact_match, tolerance_match)

        For RuleArena:
        - exact_match: predicted == expected (after rounding)
        - tolerance_match: within 1% relative tolerance
        """
        if predicted is None or expected is None:
            return False, False

        try:
            pred_num = self._parse_number(predicted)
            exp_num = self._parse_number(expected)

            if pred_num is None or exp_num is None:
                return False, False

            # Exact match (rounded to 2 decimal places for currency)
            exact = round(pred_num, 2) == round(exp_num, 2)

            # Tolerance match (1% relative tolerance)
            if abs(exp_num) < 1e-9:
                # Avoid division by zero
                tolerance = abs(pred_num - exp_num) < 0.01
            else:
                relative_error = abs(pred_num - exp_num) / abs(exp_num)
                tolerance = relative_error <= 0.01

            return exact, tolerance

        except Exception:
            return False, False

    def _parse_number(self, value: Any) -> Optional[float]:
        """
        Parse a number from various formats.

        Handles:
        - Numeric types (int, float, numpy types)
        - Strings with currency symbols, commas
        - Strings with "ANSWER: " prefix
        """
        # Handle numpy types
        if hasattr(value, 'item'):  # numpy scalar
            return float(value.item())

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Remove common prefixes
            value = value.replace("ANSWER:", "").replace("$", "").strip()
            # Remove commas
            value = value.replace(",", "")
            # Extract first number
            import re
            match = re.search(r'-?\d+\.?\d*', value)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    pass

        return None
