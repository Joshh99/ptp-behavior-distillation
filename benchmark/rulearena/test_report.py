"""
Test report generation with mock data.

This allows testing the report generator without running expensive experiments.

Usage:
    python -m benchmark.rulearena.test_report
"""

from pathlib import Path
from benchmark.rulearena.metrics.aggregator import AggregatedMetrics, CategoryMetrics, CalculatorMetrics
from benchmark.rulearena.reports.generator import ReportGenerator


def create_mock_metrics():
    """Create mock metrics for testing report generation."""

    # Mock L0F CoT experiment
    l0f_metrics = AggregatedMetrics(
        experiment_name="l0f_cot",
        experiment_level="L0F",
        total_instances=30,
        correct_exact=2,
        accuracy_exact=0.067,
        correct_tolerance=3,
        accuracy_tolerance=0.10,
        total_cost_usd=0.045,
        avg_cost_usd=0.0015,
        avg_latency_ms=2340,
        total_input_tokens=37500,
        total_output_tokens=12300,
        total_tokens=49800,
        error_count=1,
        error_rate=0.033,
        by_category={
            'airline': CategoryMetrics(
                category='airline',
                total_instances=30,
                correct_exact=2,
                accuracy_exact=0.067,
                correct_tolerance=3,
                accuracy_tolerance=0.10,
                total_cost_usd=0.045,
                avg_cost_usd=0.0015,
                avg_latency_ms=2340,
                error_count=1,
                error_rate=0.033,
            )
        },
        by_calculator={
            'airline_level_0': CalculatorMetrics(
                calculator_name='airline_level_0',
                total_instances=10,
                correct_exact=1,
                accuracy_exact=0.10,
                correct_tolerance=2,
                accuracy_tolerance=0.20,
                avg_cost_usd=0.0015,
                error_count=0,
            ),
            'airline_level_1': CalculatorMetrics(
                calculator_name='airline_level_1',
                total_instances=10,
                correct_exact=1,
                accuracy_exact=0.10,
                correct_tolerance=1,
                accuracy_tolerance=0.10,
                avg_cost_usd=0.0015,
                error_count=0,
            ),
            'airline_level_2': CalculatorMetrics(
                calculator_name='airline_level_2',
                total_instances=10,
                correct_exact=0,
                accuracy_exact=0.0,
                correct_tolerance=0,
                accuracy_tolerance=0.0,
                avg_cost_usd=0.0015,
                error_count=1,
            ),
        }
    )

    # Mock L1 PTool experiment
    l1_metrics = AggregatedMetrics(
        experiment_name="l1_ptool",
        experiment_level="L1",
        total_instances=30,
        correct_exact=25,
        accuracy_exact=0.833,
        correct_tolerance=27,
        accuracy_tolerance=0.90,
        total_cost_usd=0.038,
        avg_cost_usd=0.00127,
        avg_latency_ms=1850,
        total_input_tokens=32500,
        total_output_tokens=5600,
        total_tokens=38100,
        error_count=0,
        error_rate=0.0,
        by_category={
            'airline': CategoryMetrics(
                category='airline',
                total_instances=30,
                correct_exact=25,
                accuracy_exact=0.833,
                correct_tolerance=27,
                accuracy_tolerance=0.90,
                total_cost_usd=0.038,
                avg_cost_usd=0.00127,
                avg_latency_ms=1850,
                error_count=0,
                error_rate=0.0,
            )
        },
        by_calculator={
            'airline_level_0': CalculatorMetrics(
                calculator_name='airline_level_0',
                total_instances=10,
                correct_exact=9,
                accuracy_exact=0.90,
                correct_tolerance=10,
                accuracy_tolerance=1.0,
                avg_cost_usd=0.00127,
                error_count=0,
            ),
            'airline_level_1': CalculatorMetrics(
                calculator_name='airline_level_1',
                total_instances=10,
                correct_exact=9,
                accuracy_exact=0.90,
                correct_tolerance=9,
                accuracy_tolerance=0.90,
                avg_cost_usd=0.00127,
                error_count=0,
            ),
            'airline_level_2': CalculatorMetrics(
                calculator_name='airline_level_2',
                total_instances=10,
                correct_exact=7,
                accuracy_exact=0.70,
                correct_tolerance=8,
                accuracy_tolerance=0.80,
                avg_cost_usd=0.00127,
                error_count=0,
            ),
        }
    )

    # Mock L1-TA Tool-Augmented experiment
    l1ta_metrics = AggregatedMetrics(
        experiment_name="l1ta_tool_augmented",
        experiment_level="L1-TA",
        total_instances=30,
        correct_exact=12,
        accuracy_exact=0.40,
        correct_tolerance=15,
        accuracy_tolerance=0.50,
        total_cost_usd=0.052,
        avg_cost_usd=0.00173,
        avg_latency_ms=2850,
        total_input_tokens=35200,
        total_output_tokens=15800,
        total_tokens=51000,
        error_count=3,
        error_rate=0.10,
        by_category={
            'airline': CategoryMetrics(
                category='airline',
                total_instances=30,
                correct_exact=12,
                accuracy_exact=0.40,
                correct_tolerance=15,
                accuracy_tolerance=0.50,
                total_cost_usd=0.052,
                avg_cost_usd=0.00173,
                avg_latency_ms=2850,
                error_count=3,
                error_rate=0.10,
            )
        },
        by_calculator={
            'airline_level_0': CalculatorMetrics(
                calculator_name='airline_level_0',
                total_instances=10,
                correct_exact=5,
                accuracy_exact=0.50,
                correct_tolerance=6,
                accuracy_tolerance=0.60,
                avg_cost_usd=0.00173,
                error_count=1,
            ),
            'airline_level_1': CalculatorMetrics(
                calculator_name='airline_level_1',
                total_instances=10,
                correct_exact=4,
                accuracy_exact=0.40,
                correct_tolerance=5,
                accuracy_tolerance=0.50,
                avg_cost_usd=0.00173,
                error_count=1,
            ),
            'airline_level_2': CalculatorMetrics(
                calculator_name='airline_level_2',
                total_instances=10,
                correct_exact=3,
                accuracy_exact=0.30,
                correct_tolerance=4,
                accuracy_tolerance=0.40,
                avg_cost_usd=0.00173,
                error_count=1,
            ),
        }
    )

    return {
        'l0f_cot': l0f_metrics,
        'l1_ptool': l1_metrics,
        'l1ta_tool_augmented': l1ta_metrics,
    }


def main():
    print("=" * 80)
    print("Testing Report Generator with Mock Data")
    print("=" * 80)

    # Create mock metrics
    all_metrics = create_mock_metrics()

    print("\nCreated mock metrics for experiments:")
    for exp_name, metrics in all_metrics.items():
        print(f"  - {exp_name}: {metrics.total_instances} instances, "
              f"{metrics.accuracy_tolerance*100:.1f}% accuracy, "
              f"${metrics.total_cost_usd:.4f} cost")

    # Generate report
    output_dir = Path("benchmark_results") / "rulearena_test"
    print(f"\nGenerating report in {output_dir}...")

    generator = ReportGenerator(output_dir)
    generator.generate(
        all_metrics=all_metrics,
        model_id="deepseek-ai/DeepSeek-V3",
        seed=42,
    )

    print("\n" + "=" * 80)
    print("SUCCESS! Test report generated.")
    print("=" * 80)
    print(f"\nReport: {output_dir / 'report.html'}")
    print(f"Metrics: {output_dir / 'metrics.json'}")
    print("\nOpen the report in your browser to verify:")
    print(f"  start {output_dir / 'report.html'}  (Windows)")
    print(f"  open {output_dir / 'report.html'}   (macOS)")
    print(f"  xdg-open {output_dir / 'report.html'}  (Linux)")


if __name__ == "__main__":
    main()
