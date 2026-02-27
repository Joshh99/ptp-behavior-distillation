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
    debug: bool = False,
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
    if experiment_name == "l0_python":
        from benchmark.rulearena.experiments.l0_python import L0PythonExperiment
        experiment = L0PythonExperiment()
    elif experiment_name == "l0f_cot":
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

        if debug:
            print("\n" + "-" * 60)
            print(f"DEBUG [{instance.instance_id}]")
            print("-" * 60)
            print(f"RAW LLM RESPONSE:\n{result.raw_response}")
            print("-" * 60)
            print(f"EXTRACTED ANSWER : {result.predicted!r}")
            print(f"GROUND TRUTH     : {result.expected!r}")
            print(f"EXACT MATCH      : {result.is_correct_exact}")
            print(f"TOLERANCE MATCH  : {result.is_correct_tolerance}")
            print("-" * 60 + "\n")

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

    # Failure mode breakdown
    from collections import Counter
    failure_counts = Counter(getattr(r, 'failure_mode', 'none') for r in results)
    print(f"\nFailure modes:")
    for mode in ["none", "extraction_failure", "calculation_error", "scope_error"]:
        count = failure_counts.get(mode, 0)
        if count > 0 or mode == "none":
            print(f"  {mode + ':':25s} {count}/{total}")
    for mode in sorted(failure_counts):
        if mode not in ["none", "extraction_failure", "calculation_error", "scope_error"]:
            print(f"  {mode + ':':25s} {failure_counts[mode]}/{total}")

    # Compute F1 macro for NBA domain (boolean classification)
    nba_results = [r for r in results if r.metadata.get("domain") == "nba" and not r.error]
    f1_macro = None
    if nba_results:
        y_true = [r.expected for r in nba_results]
        y_pred = [r.predicted for r in nba_results]
        if all(isinstance(v, bool) for v in y_true + y_pred):
            from sklearn.metrics import f1_score
            f1_macro = float(f1_score(y_true, y_pred, average="macro"))
            print(f"F1 macro (NBA): {f1_macro:.4f}")

    # Save results
    output_dir = Path("benchmark_results") / "rulearena"
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_suffix = domain if domain else "all"
    output_file = output_dir / f"{experiment_name}_{domain_suffix}.json"
    results_data = [r.to_dict() for r in results]

    # Wrap in envelope with run-level summary
    failure_modes_summary = dict(failure_counts)
    output_envelope = {
        "run_summary": {
            "experiment_name": experiment_name,
            "n": total,
            "accuracy_exact": correct / total if total > 0 else 0.0,
            "accuracy_tolerance": tolerance / total if total > 0 else 0.0,
            "f1_macro": f1_macro,
            "error_count": errors,
            "failure_modes": failure_modes_summary,
            "total_cost_usd": total_cost,
            "avg_cost_usd": total_cost / total if total > 0 else 0.0,
        },
        "results": results_data,
    }

    with open(output_file, 'w') as f:
        json.dump(output_envelope, f, indent=2)

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
    parser.add_argument(
        "--debug-n",
        type=int,
        default=None,
        metavar="N",
        help="Run N instances with verbose debug output (raw response, extraction, ground truth)"
    )

    args = parser.parse_args()

    debug = args.debug_n is not None
    n = args.debug_n if debug else args.n

    if len(args.experiment) == 1:
        # Single experiment
        run_experiment(
            experiment_name=args.experiment[0],
            n=n,
            domain=args.domain,
            seed=args.seed,
            debug=debug,
        )
    else:
        # Multiple experiments - always generate report
        run_multiple_experiments(
            experiment_names=args.experiment,
            n=n,
            domain=args.domain,
            seed=args.seed,
            generate_report=True,
        )

    # Generate report if requested for single experiment
    if len(args.experiment) == 1 and args.report:
        exp_name = args.experiment[0]
        output_dir = Path("benchmark_results") / "rulearena"
        domain_suffix = args.domain if args.domain else "all"
        results_file = output_dir / f"{exp_name}_{domain_suffix}.json"

        if results_file.exists():
            with open(results_file, 'r') as f:
                raw_data = json.load(f)

            # Handle envelope format or legacy flat list
            if isinstance(raw_data, dict) and "results" in raw_data:
                results_data = raw_data["results"]
            else:
                results_data = raw_data

            # Convert back to ExperimentResult objects
            from benchmark.rulearena.experiments.base import ExperimentResult
            results = [
                ExperimentResult(
                    instance_id=r['instance_id'],
                    predicted=r['predicted'],
                    expected=r['expected'],
                    is_correct_exact=r.get('is_correct_exact', False),
                    is_correct_tolerance=r.get('is_correct_tolerance', False),
                    latency_ms=r.get('latency_ms', 0.0),
                    cost_usd=r['cost_usd'],
                    input_tokens=r['input_tokens'],
                    output_tokens=r['output_tokens'],
                    error=r.get('error'),
                    failure_mode=r.get('failure_mode', 'none'),
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
