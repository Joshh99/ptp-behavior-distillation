"""
Regenerate HTML report from existing metrics.json

This allows rebuilding the report without re-running experiments.
Useful for tweaking visualizations or updating report format.

Usage:
    python -m benchmark.rulearena.reports.regenerate_report --results-dir benchmark_results/rulearena
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

from benchmark.rulearena.reports.generator import ReportGenerator
from benchmark.rulearena.metrics.aggregator import (
    AggregatedMetrics,
    CategoryMetrics,
    CalculatorMetrics,
)


def load_metrics_from_json(metrics_file: Path) -> Dict[str, AggregatedMetrics]:
    """
    Load AggregatedMetrics from metrics.json.

    Args:
        metrics_file: Path to metrics.json

    Returns:
        Dict mapping experiment names to AggregatedMetrics
    """
    with open(metrics_file, 'r') as f:
        data = json.load(f)

    all_metrics = {}
    for exp_name, metrics_dict in data.items():
        # Reconstruct CategoryMetrics
        by_category = {}
        for cat_name, cat_dict in metrics_dict.get('by_category', {}).items():
            by_category[cat_name] = CategoryMetrics(**cat_dict)

        # Reconstruct CalculatorMetrics
        by_calculator = {}
        for calc_name, calc_dict in metrics_dict.get('by_calculator', {}).items():
            by_calculator[calc_name] = CalculatorMetrics(**calc_dict)

        # Reconstruct AggregatedMetrics
        metrics = AggregatedMetrics(
            experiment_name=metrics_dict['experiment_name'],
            experiment_level=metrics_dict['experiment_level'],
            total_instances=metrics_dict['total_instances'],
            correct_exact=metrics_dict['correct_exact'],
            accuracy_exact=metrics_dict['accuracy_exact'],
            correct_tolerance=metrics_dict['correct_tolerance'],
            accuracy_tolerance=metrics_dict['accuracy_tolerance'],
            total_cost_usd=metrics_dict['total_cost_usd'],
            avg_cost_usd=metrics_dict['avg_cost_usd'],
            avg_latency_ms=metrics_dict['avg_latency_ms'],
            total_input_tokens=metrics_dict['total_input_tokens'],
            total_output_tokens=metrics_dict['total_output_tokens'],
            total_tokens=metrics_dict['total_tokens'],
            error_count=metrics_dict['error_count'],
            error_rate=metrics_dict['error_rate'],
            by_category=by_category,
            by_calculator=by_calculator,
        )
        all_metrics[exp_name] = metrics

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate HTML report from metrics.json"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing metrics.json"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="deepseek-ai/DeepSeek-V3",
        help="Model ID to display in report (default: deepseek-ai/DeepSeek-V3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to display in report (default: 42)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics_file = results_dir / "metrics.json"

    if not metrics_file.exists():
        print(f"Error: metrics.json not found at {metrics_file}")
        return

    print(f"Loading metrics from {metrics_file}...")
    all_metrics = load_metrics_from_json(metrics_file)

    print(f"Loaded {len(all_metrics)} experiments:")
    for exp_name, metrics in all_metrics.items():
        print(f"  - {exp_name}: {metrics.total_instances} instances, "
              f"{metrics.accuracy_tolerance*100:.1f}% accuracy, "
              f"${metrics.total_cost_usd:.4f} cost")

    print("\nRegenerating report...")
    generator = ReportGenerator(results_dir)
    generator.generate(all_metrics, model_id=args.model_id, seed=args.seed)

    print("\nDone! Report regenerated successfully.")
    print(f"Open {results_dir / 'report.html'} to view.")


if __name__ == "__main__":
    main()
