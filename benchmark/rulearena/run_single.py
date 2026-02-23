"""
Simple runner for individual experiments.

Usage:
    python -m benchmark.rulearena.run_single --experiment l0f_cot --n 3
    python -m benchmark.rulearena.run_single --experiment l0f_cot --n 10 --domain airline
    python -m benchmark.rulearena.run_single --experiment l0f_cot l1_ptool --n 10 --report
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List

from benchmark.rulearena.dataset.loader import RuleArenaDataset
from benchmark.rulearena.config import get_experiment_config
from benchmark.rulearena.metrics.aggregator import MetricsAggregator
from benchmark.rulearena.reports.generator import ReportGenerator


EXPERIMENT_LEVELS = {
    'l0_python': 'L0',
    'l0f_cot': 'L0F',
    'l1_ptool': 'L1',
    'l1ta_tool_augmented': 'L1-TA',
    'l3_react': 'L3',
}


def run_experiment(
    experiment_name: str,
    n: int = 3,
    domain: Optional[str] = None,
    seed: int = 42,
):
    """
    Run a single experiment on n instances.

    Args:
        experiment_name: Name of experiment (e.g., "l0f_cot")
        n: Number of instances to run
        domain: Optional domain filter (airline, nba, tax)
        seed: Random seed for sampling
    """
    print("=" * 60)
    print(f"Running {experiment_name} on {n} instances")
    if domain:
        print(f"Domain filter: {domain}")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = RuleArenaDataset()

    # Sample instances
    if domain:
        instances = [inst for inst in dataset.instances if inst.domain == domain]
        if len(instances) < n:
            print(f"Warning: Only {len(instances)} {domain} instances available")
            n = len(instances)
        instances = instances[:n]
    else:
        instances = dataset.stratified_sample(n, seed=seed)

    print(f"Sampled {len(instances)} instances")

    # Load experiment
    if experiment_name == "l0f_cot":
        from benchmark.rulearena.experiments.l0f_cot import L0F_CoT_Experiment
        experiment = L0F_CoT_Experiment()
    elif experiment_name == "l1_ptool":
        from benchmark.rulearena.experiments.l1_ptool import L1_PTool_Experiment
        experiment = L1_PTool_Experiment()
    elif experiment_name == "l1ta_tool_augmented":
        from benchmark.rulearena.experiments.l1ta_tool_augmented import L1TA_ToolAugmented_Experiment
        experiment = L1TA_ToolAugmented_Experiment()
    elif experiment_name == "l3_react":
        from benchmark.rulearena.experiments.l3_react import L3_ReAct_Experiment
        experiment = L3_ReAct_Experiment()
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    # Run experiment
    print(f"\nRunning {experiment_name}...\n")
    results = []
    for i, instance in enumerate(instances, 1):
        print(f"[{i}/{len(instances)}] {instance.instance_id}...", end=" ")
        result = experiment.run_instance(instance)

        if result.error:
            print(f"ERROR: {result.error}")
        else:
            status = "CORRECT" if result.is_correct_exact else "INCORRECT"
            print(f"{status} predicted={result.predicted}, expected={result.expected}, "
                  f"cost=${result.cost_usd:.4f}, time={result.latency_ms:.0f}ms")

        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = len(results)
    correct = sum(1 for r in results if r.is_correct_exact and not r.error)
    tolerance = sum(1 for r in results if r.is_correct_tolerance and not r.error)
    errors = sum(1 for r in results if r.error)
    total_cost = sum(r.cost_usd for r in results)

    print(f"Total instances: {total}")
    print(f"Correct (exact): {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Correct (tolerance): {tolerance}/{total} ({tolerance/total*100:.1f}%)")
    print(f"Errors: {errors}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Avg cost: ${total_cost/total:.4f}")

    # Save results
    output_dir = Path("benchmark_results") / "rulearena"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{experiment_name}_debug.json"
    results_data = [r.to_dict() for r in results]

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return results


def run_multiple_experiments(
    experiment_names: List[str],
    n: int = 3,
    domain: Optional[str] = None,
    seed: int = 42,
    generate_report: bool = False,
):
    """
    Run multiple experiments and optionally generate a report.

    Args:
        experiment_names: List of experiment names to run
        n: Number of instances to run for each
        domain: Optional domain filter
        seed: Random seed
        generate_report: Whether to generate HTML report
    """
    all_experiment_results = {}

    for exp_name in experiment_names:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {exp_name}")
        print("=" * 80)

        results = run_experiment(
            experiment_name=exp_name,
            n=n,
            domain=domain,
            seed=seed,
        )
        all_experiment_results[exp_name] = results

    if generate_report and all_experiment_results:
        print("\n" + "=" * 80)
        print("GENERATING REPORT")
        print("=" * 80)

        # Aggregate metrics for each experiment
        all_metrics = {}
        for exp_name, results in all_experiment_results.items():
            exp_level = EXPERIMENT_LEVELS.get(exp_name, "Unknown")
            metrics = MetricsAggregator.aggregate(
                experiment_name=exp_name,
                experiment_level=exp_level,
                results=results,
            )
            all_metrics[exp_name] = metrics

        # Generate report
        output_dir = Path("benchmark_results") / "rulearena"
        generator = ReportGenerator(output_dir)
        generator.generate(all_metrics, model_id="deepseek-ai/DeepSeek-V3", seed=seed)

        print(f"\nReport generated: {output_dir / 'report.html'}")
        print(f"Metrics saved: {output_dir / 'metrics.json'}")


def main():
    parser = argparse.ArgumentParser(description="Run experiments and generate reports")
    parser.add_argument(
        "--experiment",
        type=str,
        nargs='+',
        required=True,
        help="Experiment name(s) (l0f_cot, l1_ptool, l1ta_tool_augmented, l3_react)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of instances to run (default: 3)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["airline", "nba", "tax"],
        help="Optional domain filter"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML report after running experiments"
    )

    args = parser.parse_args()

    if len(args.experiment) == 1:
        # Single experiment
        run_experiment(
            experiment_name=args.experiment[0],
            n=args.n,
            domain=args.domain,
            seed=args.seed,
        )
    else:
        # Multiple experiments - always generate report
        run_multiple_experiments(
            experiment_names=args.experiment,
            n=args.n,
            domain=args.domain,
            seed=args.seed,
            generate_report=True,
        )

    # Generate report if requested for single experiment
    if len(args.experiment) == 1 and args.report:
        exp_name = args.experiment[0]
        output_dir = Path("benchmark_results") / "rulearena"
        results_file = output_dir / f"{exp_name}_debug.json"

        if results_file.exists():
            with open(results_file, 'r') as f:
                results_data = json.load(f)

            # Convert back to ExperimentResult objects
            from benchmark.rulearena.experiments.base import ExperimentResult
            results = [
                ExperimentResult(
                    instance_id=r['instance_id'],
                    predicted=r['predicted'],
                    expected=r['expected'],
                    is_correct_exact=r.get('is_correct_exact', r.get('correct', False)),
                    is_correct_tolerance=r.get('is_correct_tolerance', r.get('correct', False)),
                    latency_ms=r.get('latency_ms', r.get('time_seconds', 0) * 1000),
                    cost_usd=r['cost_usd'],
                    input_tokens=r['input_tokens'],
                    output_tokens=r['output_tokens'],
                    error=r.get('error'),
                    raw_response=r.get('raw_response'),
                    metadata=r.get('metadata', {}),
                )
                for r in results_data
            ]

            exp_level = EXPERIMENT_LEVELS.get(exp_name, "Unknown")
            metrics = MetricsAggregator.aggregate(
                experiment_name=exp_name,
                experiment_level=exp_level,
                results=results,
            )

            generator = ReportGenerator(output_dir)
            generator.generate({exp_name: metrics}, model_id="deepseek-ai/DeepSeek-V3", seed=args.seed)

            print(f"\nReport generated: {output_dir / 'report.html'}")
            print(f"Metrics saved: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
