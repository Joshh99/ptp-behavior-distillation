"""Run all 4 baselines on same problems"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dataset.airline_loader import AirlineLoader
from src.experiments.l1_ptool_extraction import (
    baggage_allowance_l1_ptool,
    baggage_allowance_l1_transparent
)
from src.experiments.cot_baseline import run_cot_baseline
from src.experiments.tool_aug_baseline import run_tool_augmented


def run_single_experiment(
    experiment_name: str,
    complexity_level: int,
    num_problems: int,
    loader: AirlineLoader,
    experiment_fn,
    verbose: bool = True,
):
    """Run one experiment configuration."""
    
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_name} | Level {complexity_level}")
    print(f"{'='*80}")
    
    problems = loader.load_problems(complexity_level, num_problems)
    
    results = []
    correct = 0
    total_cost = 0.0
    total_time = 0.0
    
    for i, problem in enumerate(problems):
        if verbose:
            print(f"\n[{i+1}/{len(problems)}]")
        
        import time
        start = time.time()
        
        # All experiment functions return (result_dict, input_tokens, output_tokens, cost)
        result, input_tokens, output_tokens, cost = experiment_fn(
            query=problem.query,
            loader=loader,
            verbose=verbose,
        )
        
        elapsed = time.time() - start
        total_time += elapsed
        
        predicted = result["answer"]
        expected = problem.ground_truth
        is_correct = predicted == expected
        
        if is_correct:
            correct += 1
        
        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"{status} Expected: ${expected}, Got: ${predicted}")
            print(f"   Tokens: {input_tokens + output_tokens} total, Cost: ${cost:.6f}")
        
        total_cost += cost
        
        results.append({
            "problem_id": problem.id,
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
            "cost": cost,
            "time": elapsed,
        })
    
    accuracy = correct / len(problems) if problems else 0
    avg_cost = total_cost / len(problems) if problems else 0
    avg_time = total_time / len(problems) if problems else 0
    
    return {
        "experiment": experiment_name,
        "complexity_level": complexity_level,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "total_cost": total_cost,
        "avg_cost": avg_cost,
        "total_time": total_time,
        "avg_time": avg_time,
        "results": results,
    }


def run_all_baselines(
    complexity_level: int = 0,
    num_problems: int = 5,
):
    """Run all 4 baselines on the same problems."""
    
    loader = AirlineLoader("external/RuleArena")
    
    experiments = {
        "l1_pure": baggage_allowance_l1_ptool,
        "l1_transparent": baggage_allowance_l1_transparent,
        "cot": run_cot_baseline,
        "tool_aug": run_tool_augmented,
    }
    
    all_results = []
    
    for exp_name, exp_fn in experiments.items():
        result = run_single_experiment(
            experiment_name=exp_name,
            complexity_level=complexity_level,
            num_problems=num_problems,
            loader=loader,
            experiment_fn=exp_fn,
            verbose=True,
        )
        all_results.append(result)
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Experiment':<20} {'Accuracy':<12} {'Avg Cost':<12} {'Avg Time':<12} {'Total Cost':<12}")
    print("-" * 80)
    
    for res in all_results:
        print(f"{res['experiment']:<20} "
              f"{res['accuracy']*100:>6.1f}%     "
              f"${res['avg_cost']:>9.6f}  "
              f"{res['avg_time']:>8.2f}s    "
              f"${res['total_cost']:>9.6f}")
    
    print(f"\n{'='*80}")
    print(f"Test complete on {num_problems} problems (complexity level {complexity_level})")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    # Test on 5 problems first
    NUM_PROBLEMS = 5
    COMPLEXITY = 0
    
    print("=" * 80)
    print("Running All 4 Baselines")
    print("=" * 80)
    print(f"Problems: {NUM_PROBLEMS} per experiment")
    print(f"Complexity: Level {COMPLEXITY}")
    print("=" * 80)
    
    results = run_all_baselines(
        complexity_level=COMPLEXITY,
        num_problems=NUM_PROBLEMS,
    )