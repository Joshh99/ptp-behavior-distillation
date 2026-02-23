"""
Metrics Aggregation

Aggregates per-instance results into summary metrics.
Schema MUST match MedCalc's AggregatedMetrics for cross-benchmark compatibility.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict

from benchmark.rulearena.experiments.base import ExperimentResult


@dataclass
class CategoryMetrics:
    """
    Metrics for a specific category (domain in RuleArena).

    Matches MedCalc's CategoryMetrics schema.
    """
    category: str                    # Domain name
    total_instances: int
    correct_exact: int
    accuracy_exact: float
    correct_tolerance: int
    accuracy_tolerance: float
    total_cost_usd: float
    avg_cost_usd: float
    avg_latency_ms: float
    error_count: int
    error_rate: float

    def to_dict(self) -> Dict:
        return {
            "category": self.category,
            "total_instances": self.total_instances,
            "correct_exact": self.correct_exact,
            "accuracy_exact": self.accuracy_exact,
            "correct_tolerance": self.correct_tolerance,
            "accuracy_tolerance": self.accuracy_tolerance,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_usd": self.avg_cost_usd,
            "avg_latency_ms": self.avg_latency_ms,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
        }


@dataclass
class CalculatorMetrics:
    """
    Metrics for a specific subcategory (domain_level_N in RuleArena).

    Matches MedCalc's CalculatorMetrics schema.
    """
    calculator_name: str             # "{domain}_level_{complexity}"
    total_instances: int
    correct_exact: int
    accuracy_exact: float
    correct_tolerance: int
    accuracy_tolerance: float
    avg_cost_usd: float
    error_count: int

    def to_dict(self) -> Dict:
        return {
            "calculator_name": self.calculator_name,
            "total_instances": self.total_instances,
            "correct_exact": self.correct_exact,
            "accuracy_exact": self.accuracy_exact,
            "correct_tolerance": self.correct_tolerance,
            "accuracy_tolerance": self.accuracy_tolerance,
            "avg_cost_usd": self.avg_cost_usd,
            "error_count": self.error_count,
        }


@dataclass
class AggregatedMetrics:
    """
    Aggregated metrics for an entire experiment.

    Schema MUST match MedCalc's AggregatedMetrics exactly.
    This is critical for cross-benchmark analysis scripts.
    """
    experiment_name: str
    experiment_level: str            # "L0", "L0F", "L1", "L1-TA", "L3"

    # Overall metrics
    total_instances: int
    correct_exact: int
    accuracy_exact: float
    correct_tolerance: int
    accuracy_tolerance: float

    # Cost metrics
    total_cost_usd: float
    avg_cost_usd: float

    # Latency metrics
    avg_latency_ms: float

    # Token metrics
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int

    # Error metrics
    error_count: int
    error_rate: float

    # Breakdowns
    by_category: Dict[str, CategoryMetrics] = field(default_factory=dict)
    by_calculator: Dict[str, CalculatorMetrics] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "experiment_name": self.experiment_name,
            "experiment_level": self.experiment_level,
            "total_instances": self.total_instances,
            "correct_exact": self.correct_exact,
            "accuracy_exact": self.accuracy_exact,
            "correct_tolerance": self.correct_tolerance,
            "accuracy_tolerance": self.accuracy_tolerance,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_usd": self.avg_cost_usd,
            "avg_latency_ms": self.avg_latency_ms,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "by_category": {
                k: v.to_dict() for k, v in self.by_category.items()
            },
            "by_calculator": {
                k: v.to_dict() for k, v in self.by_calculator.items()
            },
        }


class MetricsAggregator:
    """Aggregates per-instance results into summary metrics."""

    @staticmethod
    def aggregate(
        experiment_name: str,
        experiment_level: str,
        results: List[ExperimentResult],
    ) -> AggregatedMetrics:
        """
        Aggregate per-instance results into summary metrics.

        Args:
            experiment_name: Name of the experiment
            experiment_level: Level string (e.g., "L1")
            results: List of per-instance results

        Returns:
            AggregatedMetrics with all summary statistics
        """
        if not results:
            return MetricsAggregator._empty_metrics(experiment_name, experiment_level)

        # Overall metrics
        total_instances = len(results)
        correct_exact = sum(1 for r in results if r.is_correct_exact and not r.error)
        correct_tolerance = sum(
            1 for r in results
            if r.is_correct_tolerance and not r.error
        )

        total_cost = sum(r.cost_usd for r in results)
        total_time_ms = sum(r.latency_ms for r in results)

        total_input_tokens = sum(r.input_tokens for r in results)
        total_output_tokens = sum(r.output_tokens for r in results)

        error_count = sum(1 for r in results if r.error is not None)

        # Calculate by-category metrics (domain)
        by_category = MetricsAggregator._calculate_by_category(results)

        # Calculate by-calculator metrics (domain_level_N)
        by_calculator = MetricsAggregator._calculate_by_calculator(results)

        return AggregatedMetrics(
            experiment_name=experiment_name,
            experiment_level=experiment_level,
            total_instances=total_instances,
            correct_exact=correct_exact,
            accuracy_exact=correct_exact / total_instances if total_instances > 0 else 0.0,
            correct_tolerance=correct_tolerance,
            accuracy_tolerance=correct_tolerance / total_instances if total_instances > 0 else 0.0,
            total_cost_usd=total_cost,
            avg_cost_usd=total_cost / total_instances if total_instances > 0 else 0.0,
            avg_latency_ms=total_time_ms / total_instances if total_instances > 0 else 0.0,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
            error_count=error_count,
            error_rate=error_count / total_instances if total_instances > 0 else 0.0,
            by_category=by_category,
            by_calculator=by_calculator,
        )

    @staticmethod
    def _calculate_by_category(
        results: List[ExperimentResult]
    ) -> Dict[str, CategoryMetrics]:
        """Calculate per-domain metrics."""
        by_domain = defaultdict(list)
        for r in results:
            domain = r.metadata.get('domain', 'unknown')
            by_domain[domain].append(r)

        category_metrics = {}
        for domain, domain_results in by_domain.items():
            total = len(domain_results)
            correct_exact = sum(1 for r in domain_results if r.is_correct_exact and not r.error)
            correct_tolerance = sum(
                1 for r in domain_results
                if r.is_correct_tolerance and not r.error
            )
            total_cost = sum(r.cost_usd for r in domain_results)
            total_time_ms = sum(r.latency_ms for r in domain_results)
            error_count = sum(1 for r in domain_results if r.error is not None)

            category_metrics[domain] = CategoryMetrics(
                category=domain,
                total_instances=total,
                correct_exact=correct_exact,
                accuracy_exact=correct_exact / total if total > 0 else 0.0,
                correct_tolerance=correct_tolerance,
                accuracy_tolerance=correct_tolerance / total if total > 0 else 0.0,
                total_cost_usd=total_cost,
                avg_cost_usd=total_cost / total if total > 0 else 0.0,
                avg_latency_ms=total_time_ms / total if total > 0 else 0.0,
                error_count=error_count,
                error_rate=error_count / total if total > 0 else 0.0,
            )

        return category_metrics

    @staticmethod
    def _calculate_by_calculator(
        results: List[ExperimentResult]
    ) -> Dict[str, CalculatorMetrics]:
        """Calculate per-subcategory metrics (domain_level_N)."""
        by_calc = defaultdict(list)
        for r in results:
            domain = r.metadata.get('domain', 'unknown')
            complexity = r.metadata.get('complexity_level', 0)
            calc_name = f"{domain}_level_{complexity}"
            by_calc[calc_name].append(r)

        calc_metrics = {}
        for calc_name, calc_results in by_calc.items():
            total = len(calc_results)
            correct_exact = sum(1 for r in calc_results if r.is_correct_exact and not r.error)
            correct_tolerance = sum(
                1 for r in calc_results
                if r.is_correct_tolerance and not r.error
            )
            total_cost = sum(r.cost_usd for r in calc_results)
            error_count = sum(1 for r in calc_results if r.error is not None)

            calc_metrics[calc_name] = CalculatorMetrics(
                calculator_name=calc_name,
                total_instances=total,
                correct_exact=correct_exact,
                accuracy_exact=correct_exact / total if total > 0 else 0.0,
                correct_tolerance=correct_tolerance,
                accuracy_tolerance=correct_tolerance / total if total > 0 else 0.0,
                avg_cost_usd=total_cost / total if total > 0 else 0.0,
                error_count=error_count,
            )

        return calc_metrics

    @staticmethod
    def _empty_metrics(experiment_name: str, experiment_level: str) -> AggregatedMetrics:
        """Return empty metrics for edge case of no results."""
        return AggregatedMetrics(
            experiment_name=experiment_name,
            experiment_level=experiment_level,
            total_instances=0,
            correct_exact=0,
            accuracy_exact=0.0,
            correct_tolerance=0,
            accuracy_tolerance=0.0,
            total_cost_usd=0.0,
            avg_cost_usd=0.0,
            avg_latency_ms=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_tokens=0,
            error_count=0,
            error_rate=0.0,
        )
