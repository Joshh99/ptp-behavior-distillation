#!/usr/bin/env python
"""
Unified RuleArena Baseline Experiment Runner

Runs both L1 (PTool) and L3 (ReAct) baselines on the same RuleArena problems
and generates a comparison report.

Usage:
    python scripts/run_rulearena_baselines.py
    python scripts/run_rulearena_baselines.py --num-problems 5  # Quick test
    python scripts/run_rulearena_baselines.py --skip-l3  # Only run L1
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

# Import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

# Import experiment modules
from src.experiments.config import DEFAULT_MODEL, MODELS, RESULTS_DIR
from src.dataset.rulearena_loader import RuleArenaLoader

# Import usage tracker
from usage_tracker import UsageTracker


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    name: str
    accuracy: float
    correct: int
    total: int
    avg_llm_calls: float
    total_llm_calls: int
    cost_per_problem: float
    total_cost: float
    total_time: float
    avg_time: float
    success_rate: float  # 1 - failure_rate
    failed_count: int
    error_breakdown: Dict[str, int]
    raw_results: List[Dict]
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# WRAPPED EXPERIMENT RUNNERS WITH PROGRESS
# =============================================================================

def run_l1_with_progress(
    loader: RuleArenaLoader,
    problems: List[Dict],
    model: str,
    usage_tracker: Optional[UsageTracker] = None,
) -> ExperimentResult:
    """
    Run L1 experiment with progress bar and detailed tracking.
    """
    from src.experiments.l1_baggage import baggage_allowance_l1, CostTracker
    
    cost_tracker = CostTracker(model=model)
    results = []
    correct = 0
    total_time = 0.0
    error_breakdown = {
        "extraction_error": 0,
        "calculation_error": 0,
        "other_error": 0,
    }
    
    # Progress bar
    iterator = tqdm(problems, desc="L1 PTool", unit="problem") if HAS_TQDM else problems
    
    for problem in iterator:
        query = problem["query"]
        expected = problem["ground_truth"]
        
        try:
            result, input_tokens, output_tokens = baggage_allowance_l1(
                query=query,
                loader=loader,
                model=model,
                verbose=False,  # Disable verbose for clean progress bar
            )
            
            # Track costs
            cost_tracker.add_call(input_tokens, output_tokens)
            
            # Log to usage tracker
            if usage_tracker:
                call_cost = 0.0
                if model in MODELS:
                    config = MODELS[model]
                    call_cost = (input_tokens / 1_000_000) * config.cost_per_m_input
                    call_cost += (output_tokens / 1_000_000) * config.cost_per_m_output
                usage_tracker.log_call(model, input_tokens, output_tokens, call_cost)
            
            # Check correctness
            predicted = result["answer"]
            is_correct = predicted == expected
            
            if is_correct:
                correct += 1
            
            total_time += result["metrics"]["total_time"]
            
            results.append({
                "problem_id": problem["id"],
                "predicted": predicted,
                "expected": expected,
                "correct": is_correct,
                "time": result["metrics"]["total_time"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "error": None,
            })
            
        except Exception as e:
            error_type = "extraction_error" if "extract" in str(e).lower() else "other_error"
            error_breakdown[error_type] += 1
            
            results.append({
                "problem_id": problem["id"],
                "predicted": None,
                "expected": expected,
                "correct": False,
                "time": 0,
                "error": str(e),
            })
    
    # Calculate summary
    n = len(problems)
    accuracy = correct / n if n > 0 else 0
    avg_time = total_time / n if n > 0 else 0
    avg_cost = cost_tracker.total_cost_usd / n if n > 0 else 0
    failed = sum(error_breakdown.values())
    success_rate = (n - failed) / n if n > 0 else 0
    
    return ExperimentResult(
        name="L1 (PTool)",
        accuracy=accuracy,
        correct=correct,
        total=n,
        avg_llm_calls=1.0,  # L1 always uses exactly 1 call
        total_llm_calls=cost_tracker.total_calls,
        cost_per_problem=avg_cost,
        total_cost=cost_tracker.total_cost_usd,
        total_time=total_time,
        avg_time=avg_time,
        success_rate=success_rate,
        failed_count=failed,
        error_breakdown=error_breakdown,
        raw_results=results,
    )


def run_l3_with_progress(
    loader: RuleArenaLoader,
    problems: List[Dict],
    model: str,
    max_steps: int = 10,
    usage_tracker: Optional[UsageTracker] = None,
) -> ExperimentResult:
    """
    Run L3 experiment with progress bar and detailed tracking.
    """
    from src.experiments.l3_baggage_react import baggage_allowance_l3, CostTracker
    
    cost_tracker = CostTracker(model=model)
    results = []
    correct = 0
    total_time = 0.0
    total_llm_calls = 0
    error_breakdown = {
        "max_steps_exceeded": 0,
        "timeout": 0,
        "extraction_error": 0,
        "other_error": 0,
    }
    
    # Progress bar
    iterator = tqdm(problems, desc="L3 ReAct", unit="problem") if HAS_TQDM else problems
    
    for problem in iterator:
        query = problem["query"]
        expected = problem["ground_truth"]
        context = problem.get("passenger_context", {})
        
        try:
            result, input_tokens, output_tokens = baggage_allowance_l3(
                query=query,
                passenger_context=context,
                model=model,
                max_steps=max_steps,
                verbose=False,  # Disable verbose for clean progress bar
            )
            
            # Track costs
            cost_tracker.add_call(input_tokens, output_tokens)
            llm_calls = result["metrics"]["llm_calls"]
            total_llm_calls += llm_calls
            
            # Log to usage tracker (log total for this problem)
            if usage_tracker:
                call_cost = 0.0
                if model in MODELS:
                    config = MODELS[model]
                    call_cost = (input_tokens / 1_000_000) * config.cost_per_m_input
                    call_cost += (output_tokens / 1_000_000) * config.cost_per_m_output
                usage_tracker.log_call(model, input_tokens, output_tokens, call_cost)
            
            # Check correctness
            predicted = result["answer"]
            is_correct = predicted == expected
            is_failed = result.get("failed", False)
            
            if is_correct:
                correct += 1
            
            if is_failed:
                reason = result.get("failure_reason", "")
                if "max_steps" in reason.lower():
                    error_breakdown["max_steps_exceeded"] += 1
                elif "timeout" in reason.lower():
                    error_breakdown["timeout"] += 1
                else:
                    error_breakdown["other_error"] += 1
            
            total_time += result["metrics"]["total_time"]
            
            results.append({
                "problem_id": problem["id"],
                "predicted": predicted,
                "expected": expected,
                "correct": is_correct,
                "failed": is_failed,
                "failure_reason": result.get("failure_reason"),
                "llm_calls": llm_calls,
                "time": result["metrics"]["total_time"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "error": None,
            })
            
        except Exception as e:
            error_breakdown["other_error"] += 1
            
            results.append({
                "problem_id": problem["id"],
                "predicted": None,
                "expected": expected,
                "correct": False,
                "failed": True,
                "error": str(e),
            })
    
    # Calculate summary
    n = len(problems)
    accuracy = correct / n if n > 0 else 0
    avg_time = total_time / n if n > 0 else 0
    avg_cost = cost_tracker.total_cost_usd / n if n > 0 else 0
    avg_calls = total_llm_calls / n if n > 0 else 0
    failed = sum(error_breakdown.values())
    success_rate = (n - failed) / n if n > 0 else 0
    
    return ExperimentResult(
        name="L3 (ReAct)",
        accuracy=accuracy,
        correct=correct,
        total=n,
        avg_llm_calls=avg_calls,
        total_llm_calls=total_llm_calls,
        cost_per_problem=avg_cost,
        total_cost=cost_tracker.total_cost_usd,
        total_time=total_time,
        avg_time=avg_time,
        success_rate=success_rate,
        failed_count=failed,
        error_breakdown=error_breakdown,
        raw_results=results,
    )


# =============================================================================
# COMPARISON TABLE GENERATION
# =============================================================================

def generate_comparison_table(
    l1_result: Optional[ExperimentResult],
    l3_result: Optional[ExperimentResult],
) -> str:
    """
    Generate markdown comparison table.
    """
    lines = [
        "# RuleArena Baseline Comparison",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Results Summary",
        "",
        "| Metric | L1 (PTool) | L3 (ReAct) |",
        "|--------|------------|------------|",
    ]
    
    def fmt_or_na(result: Optional[ExperimentResult], field: str, fmt: str = "{}") -> str:
        if result is None:
            return "N/A"
        value = getattr(result, field)
        return fmt.format(value)
    
    # Accuracy
    l1_acc = f"{l1_result.accuracy*100:.1f}%" if l1_result else "N/A"
    l3_acc = f"{l3_result.accuracy*100:.1f}%" if l3_result else "N/A"
    lines.append(f"| **Accuracy** | {l1_acc} | {l3_acc} |")
    
    # Avg LLM Calls
    l1_calls = "1.0" if l1_result else "N/A"
    l3_calls = f"{l3_result.avg_llm_calls:.1f}" if l3_result else "N/A"
    lines.append(f"| **Avg LLM Calls** | {l1_calls} | {l3_calls} |")
    
    # Cost per Problem
    l1_cost = f"${l1_result.cost_per_problem:.6f}" if l1_result else "N/A"
    l3_cost = f"${l3_result.cost_per_problem:.6f}" if l3_result else "N/A"
    lines.append(f"| **Cost per Problem** | {l1_cost} | {l3_cost} |")
    
    # Total Cost
    l1_total = f"${l1_result.total_cost:.6f}" if l1_result else "N/A"
    l3_total = f"${l3_result.total_cost:.6f}" if l3_result else "N/A"
    lines.append(f"| **Total Cost** | {l1_total} | {l3_total} |")
    
    # Total Time
    l1_time = f"{l1_result.total_time:.1f}s" if l1_result else "N/A"
    l3_time = f"{l3_result.total_time:.1f}s" if l3_result else "N/A"
    lines.append(f"| **Total Time** | {l1_time} | {l3_time} |")
    
    # Avg Time per Problem
    l1_avg_time = f"{l1_result.avg_time:.2f}s" if l1_result else "N/A"
    l3_avg_time = f"{l3_result.avg_time:.2f}s" if l3_result else "N/A"
    lines.append(f"| **Avg Time/Problem** | {l1_avg_time} | {l3_avg_time} |")
    
    # Success Rate
    l1_success = f"{l1_result.success_rate*100:.1f}%" if l1_result else "N/A"
    l3_success = f"{l3_result.success_rate*100:.1f}%" if l3_result else "N/A"
    lines.append(f"| **Success Rate** | {l1_success} | {l3_success} |")
    
    # Correct / Total
    l1_correct = f"{l1_result.correct}/{l1_result.total}" if l1_result else "N/A"
    l3_correct = f"{l3_result.correct}/{l3_result.total}" if l3_result else "N/A"
    lines.append(f"| **Correct/Total** | {l1_correct} | {l3_correct} |")
    
    # Error Breakdown Section
    lines.extend([
        "",
        "## Error Breakdown",
        "",
    ])
    
    if l1_result:
        lines.append("### L1 (PTool) Errors")
        lines.append("")
        if l1_result.error_breakdown:
            for error_type, count in l1_result.error_breakdown.items():
                if count > 0:
                    lines.append(f"- {error_type}: {count}")
            if all(c == 0 for c in l1_result.error_breakdown.values()):
                lines.append("- No errors")
        lines.append("")
    
    if l3_result:
        lines.append("### L3 (ReAct) Errors")
        lines.append("")
        if l3_result.error_breakdown:
            for error_type, count in l3_result.error_breakdown.items():
                if count > 0:
                    lines.append(f"- {error_type}: {count}")
            if all(c == 0 for c in l3_result.error_breakdown.values()):
                lines.append("- No errors")
        lines.append("")
    
    # Key Insights
    lines.extend([
        "## Key Insights",
        "",
    ])
    
    if l1_result and l3_result:
        # Cost comparison
        if l3_result.cost_per_problem > 0 and l1_result.cost_per_problem > 0:
            cost_ratio = l3_result.cost_per_problem / l1_result.cost_per_problem
            lines.append(f"- **Cost Efficiency:** L3 costs {cost_ratio:.1f}x more per problem than L1")
        
        # Accuracy comparison
        acc_diff = l1_result.accuracy - l3_result.accuracy
        if acc_diff > 0:
            lines.append(f"- **Accuracy:** L1 is {acc_diff*100:.1f}pp more accurate than L3")
        elif acc_diff < 0:
            lines.append(f"- **Accuracy:** L3 is {-acc_diff*100:.1f}pp more accurate than L1")
        else:
            lines.append("- **Accuracy:** Both approaches have equal accuracy")
        
        # Time comparison
        if l3_result.avg_time > 0 and l1_result.avg_time > 0:
            time_ratio = l3_result.avg_time / l1_result.avg_time
            lines.append(f"- **Speed:** L3 takes {time_ratio:.1f}x longer per problem than L1")
        
        # LLM calls
        lines.append(f"- **LLM Calls:** L1 uses 1 call/problem, L3 uses {l3_result.avg_llm_calls:.1f} calls/problem")
    
    lines.extend([
        "",
        "---",
        "",
        "*This report was auto-generated by `scripts/run_rulearena_baselines.py`*",
    ])
    
    return "\n".join(lines)


def generate_json_results(
    l1_result: Optional[ExperimentResult],
    l3_result: Optional[ExperimentResult],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate JSON-serializable results dictionary.
    """
    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "config": config,
        },
        "l1_ptool": l1_result.to_dict() if l1_result else None,
        "l3_react": l3_result.to_dict() if l3_result else None,
        "comparison": {
            "accuracy_diff": (
                (l1_result.accuracy - l3_result.accuracy) 
                if l1_result and l3_result else None
            ),
            "cost_ratio": (
                (l3_result.cost_per_problem / l1_result.cost_per_problem)
                if l1_result and l3_result and l1_result.cost_per_problem > 0 else None
            ),
            "time_ratio": (
                (l3_result.avg_time / l1_result.avg_time)
                if l1_result and l3_result and l1_result.avg_time > 0 else None
            ),
        }
    }


# =============================================================================
# MAIN RUNNER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run RuleArena baseline experiments (L1 vs L3)"
    )
    parser.add_argument(
        "--num-problems", type=int, default=30,
        help="Number of problems to evaluate (default: 30)"
    )
    parser.add_argument(
        "--complexity", type=int, default=0, choices=[0, 1, 2],
        help="Problem complexity level (default: 0)"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--max-steps", type=int, default=10,
        help="Max steps for L3 agent (default: 10)"
    )
    parser.add_argument(
        "--skip-l1", action="store_true",
        help="Skip L1 experiment"
    )
    parser.add_argument(
        "--skip-l3", action="store_true",
        help="Skip L3 experiment"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to files"
    )
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "num_problems": args.num_problems,
        "complexity_level": args.complexity,
        "model": args.model,
        "max_steps": args.max_steps,
    }
    
    print("=" * 80)
    print("RuleArena Baseline Experiment Runner")
    print("=" * 80)
    print(f"Problems:        {args.num_problems}")
    print(f"Complexity:      {args.complexity}")
    print(f"Model:           {args.model}")
    print(f"Max Steps (L3):  {args.max_steps}")
    print(f"Run L1:          {'Yes' if not args.skip_l1 else 'No'}")
    print(f"Run L3:          {'Yes' if not args.skip_l3 else 'No'}")
    print("=" * 80)
    print()
    
    # Initialize loader
    try:
        loader = RuleArenaLoader("external/RuleArena")
        print("[OK] RuleArenaLoader initialized")
    except FileNotFoundError as e:
        print(f"[ERROR] Error: {e}")
        print("\nMake sure to clone RuleArena first:")
        print("  git clone https://github.com/SkyRiver-2000/RuleArena external/RuleArena")
        return 1
    
    # Load problems once (same problems for both experiments)
    print(f"\nLoading {args.num_problems} problems (complexity={args.complexity})...")
    try:
        problems = loader.load_airline_problems(
            complexity_level=args.complexity,
            max_problems=args.num_problems,
        )
        print(f"[OK] Loaded {len(problems)} problems")
    except Exception as e:
        print(f"[ERROR] Failed to load problems: {e}")
        return 1
    
    # Initialize usage tracker
    tracker_path = os.path.join(PROJECT_ROOT, "scripts", "usage_log.json")
    try:
        usage_tracker = UsageTracker(tracker_path)
        print(f"[OK] Usage tracker initialized (current cost: ${usage_tracker.data['total_cost']:.4f})")
    except Exception as e:
        print(f"[WARN] Warning: Could not initialize usage tracker: {e}")
        usage_tracker = None
    
    # Results storage
    l1_result = None
    l3_result = None
    
    # Run L1 experiment
    if not args.skip_l1:
        print("\n" + "=" * 80)
        print("Running L1 (PTool) Experiment")
        print("=" * 80 + "\n")
        
        try:
            l1_result = run_l1_with_progress(
                loader=loader,
                problems=problems,
                model=args.model,
                usage_tracker=usage_tracker,
            )
            print(f"\n[OK] L1 Complete: {l1_result.accuracy*100:.1f}% accuracy ({l1_result.correct}/{l1_result.total})")
        except Exception as e:
            print(f"\n[ERROR] L1 Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run L3 experiment
    if not args.skip_l3:
        print("\n" + "=" * 80)
        print("Running L3 (ReAct) Experiment")
        print("=" * 80 + "\n")
        
        try:
            l3_result = run_l3_with_progress(
                loader=loader,
                problems=problems,
                model=args.model,
                max_steps=args.max_steps,
                usage_tracker=usage_tracker,
            )
            print(f"\n[OK] L3 Complete: {l3_result.accuracy*100:.1f}% accuracy ({l3_result.correct}/{l3_result.total})")
        except Exception as e:
            print(f"\n[ERROR] L3 Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comparison
    print("\n" + "=" * 80)
    print("Comparison Results")
    print("=" * 80 + "\n")
    
    comparison_table = generate_comparison_table(l1_result, l3_result)
    
    # Print to console
    print(comparison_table)
    
    # Save results
    if not args.no_save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Save markdown
        md_path = os.path.join(RESULTS_DIR, "baseline_comparison.md")
        with open(md_path, "w") as f:
            f.write(comparison_table)
        print(f"\n[OK] Saved comparison table to: {md_path}")
        
        # Save JSON
        json_results = generate_json_results(l1_result, l3_result, config)
        
        # Remove raw_results from JSON to keep file size manageable
        if json_results["l1_ptool"]:
            json_results["l1_ptool"]["raw_results"] = json_results["l1_ptool"]["raw_results"][:5]
            json_results["l1_ptool"]["raw_results_truncated"] = True
        if json_results["l3_react"]:
            json_results["l3_react"]["raw_results"] = json_results["l3_react"]["raw_results"][:5]
            json_results["l3_react"]["raw_results_truncated"] = True
        
        json_path = os.path.join(RESULTS_DIR, "baseline_comparison.json")
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"[OK] Saved JSON results to: {json_path}")
    
    # Final usage report
    if usage_tracker:
        print("\n" + "=" * 80)
        print("Usage Summary")
        print("=" * 80)
        usage_tracker.report()
    
    print("\n[OK] Experiment complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
