"""
Run full baseline experiments and save results to JSON for report.html

Features:
- Runs all 4 experiment types (l1_pure, l1_transparent, cot, tool_aug)
- Handles API failures gracefully (continues with next problem)
- Saves checkpoints after each experiment
- Can resume from interrupted runs
- Estimates cost before running
"""
import sys
import math
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

from src.dataset.airline_loader import AirlineLoader
from src.experiments.l1_ptool_extraction import (
    baggage_allowance_l1_ptool,
    baggage_allowance_l1_transparent
)
from src.experiments.cot_baseline import run_cot_baseline
from src.experiments.tool_aug_baseline import run_tool_augmented


# Model pricing for cost estimation (USD per 1M tokens)
# Prices from Together.ai as of Feb 2026
MODEL_PRICING = {
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {"input": 0.88, "output": 0.88},
    "Qwen/Qwen2.5-7B-Instruct-Turbo": {"input": 1.20, "output": 1.20},
    "deepseek-ai/DeepSeek-V3": {"input": 0.60, "output": 1.25},
}

# Average tokens per call (estimated from previous runs)
AVG_TOKENS_PER_CALL = {
    "l1_pure": {"input": 800, "output": 200},
    "l1_transparent": {"input": 3000, "output": 200},
    "cot": {"input": 3000, "output": 1000},
    "tool_aug": {"input": 3000, "output": 500},
}


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and special float values."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle NaN and Inf
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, float):
            # Handle Python float NaN/Inf
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        return super().default(obj)


def estimate_cost(
    num_problems: int,
    complexity_levels: List[int],
    model: str,
) -> Dict[str, Any]:
    """Estimate total cost before running experiments."""
    pricing = MODEL_PRICING.get(model, {"input": 1.0, "output": 1.0})
    
    total_calls = num_problems * len(complexity_levels) * 4  # 4 experiments
    estimates = {}
    total_cost = 0.0
    
    for exp_name, tokens in AVG_TOKENS_PER_CALL.items():
        calls = num_problems * len(complexity_levels)
        input_cost = (tokens["input"] * calls / 1_000_000) * pricing["input"]
        output_cost = (tokens["output"] * calls / 1_000_000) * pricing["output"]
        exp_cost = input_cost + output_cost
        estimates[exp_name] = {
            "calls": calls,
            "estimated_cost": exp_cost,
        }
        total_cost += exp_cost
    
    return {
        "total_calls": total_calls,
        "total_estimated_cost": total_cost,
        "by_experiment": estimates,
        "model": model,
        "pricing": pricing,
    }


def run_experiment(
    experiment_name: str,
    experiment_fn,
    loader: AirlineLoader,
    complexity_level: int,
    num_problems: int,
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run single experiment with error handling for individual problems."""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Experiment: {experiment_name} | Level {complexity_level} | Model: {model}")
        print(f"{'='*80}")
    
    problems = loader.load_problems(complexity_level, num_problems)
    
    results = []
    correct = 0
    total_cost = 0.0
    total_time = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    failed_problems = 0
    
    for i, problem in enumerate(problems):
        if verbose:
            print(f"\n[{i+1}/{len(problems)}]", end=" ")
        
        start = time.time()
        
        try:
            result, input_tokens, output_tokens, cost = experiment_fn(
                query=problem.query,
                loader=loader,
                model=model,
                verbose=verbose,
            )
            
            elapsed = time.time() - start
            
            predicted = result.get("answer", 0)
            expected = problem.ground_truth
            is_correct = predicted == expected
            
            if is_correct:
                correct += 1
            
            if verbose:
                status = "CORRECT" if is_correct else "WRONG"
                print(f"{status} Expected: ${expected}, Got: ${predicted}")
                print(f"   Tokens: {input_tokens + output_tokens} total, Cost: ${cost:.6f}")
            
            total_cost += cost
            total_time += elapsed
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            
            results.append({
                "problem_id": problem.id,
                "complexity_level": complexity_level,
                "query": problem.query[:100] + "..." if len(problem.query) > 100 else problem.query,
                "predicted": predicted,
                "expected": expected,
                "correct": is_correct,
                "cost": cost,
                "time": elapsed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "success": result.get("success", True),
            })
            
        except Exception as e:
            elapsed = time.time() - start
            failed_problems += 1
            
            if verbose:
                print(f"ERROR: {e}")
            
            results.append({
                "problem_id": problem.id,
                "complexity_level": complexity_level,
                "query": problem.query[:100] + "..." if len(problem.query) > 100 else problem.query,
                "predicted": 0,
                "expected": problem.ground_truth,
                "correct": False,
                "cost": 0.0,
                "time": elapsed,
                "input_tokens": 0,
                "output_tokens": 0,
                "success": False,
                "error": str(e),
            })
    
    num_completed = len(problems) - failed_problems
    accuracy = correct / len(problems) if problems else 0
    
    return {
        "experiment": experiment_name,
        "model": model,
        "complexity_level": complexity_level,
        "num_problems": len(problems),
        "num_completed": num_completed,
        "num_failed": failed_problems,
        "accuracy": accuracy,
        "correct": correct,
        "total_cost": total_cost,
        "avg_cost": total_cost / num_completed if num_completed else 0,
        "total_time": total_time,
        "avg_time": total_time / num_completed if num_completed else 0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "avg_tokens": (total_input_tokens + total_output_tokens) / num_completed if num_completed else 0,
        "results": results,
    }


def save_checkpoint(
    output_path: Path,
    all_results: List[Dict],
    metadata: Dict[str, Any],
) -> None:
    """Save current progress to checkpoint file."""
    checkpoint_path = output_path.with_suffix('.checkpoint.json')
    
    output_data = {
        "metadata": {
            **metadata,
            "checkpoint_timestamp": datetime.now().isoformat(),
            "is_checkpoint": True,
        },
        "results": all_results,
    }
    
    with open(checkpoint_path, "w") as f:
        json.dump(output_data, f, indent=2, cls=NumpyEncoder)


def load_checkpoint(output_path: Path) -> Optional[List[Dict]]:
    """Load results from checkpoint file if it exists."""
    checkpoint_path = output_path.with_suffix('.checkpoint.json')
    
    if not checkpoint_path.exists():
        return None
    
    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        return data.get('results', [])
    except Exception:
        return None


def run_full_baseline(
    complexity_levels: List[int] = [0],
    num_problems_per_level: int = 100,
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    output_file: str = "results/baseline_results.json",
    resume: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run all 4 baselines across complexity levels and save to JSON.
    
    Args:
        complexity_levels: List of complexity levels (0, 1, 2)
        num_problems_per_level: Problems per level (100 max per level)
        model: Model to use for experiments
        output_file: Where to save JSON results
        resume: If True, skip experiments that are already in checkpoint
    """
    
    loader = AirlineLoader("external/RuleArena")
    
    experiments = {
        "l1_pure": baggage_allowance_l1_ptool,
        "l1_transparent": baggage_allowance_l1_transparent,
        "cot": run_cot_baseline,
        "tool_aug": run_tool_augmented,
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoint
    all_results = []
    completed_keys = set()
    
    if resume:
        checkpoint_results = load_checkpoint(output_path)
        if checkpoint_results:
            all_results = checkpoint_results
            for r in all_results:
                key = (r.get('experiment'), r.get('model'), r.get('complexity_level'))
                completed_keys.add(key)
            print(f"\nResuming from checkpoint: {len(all_results)} experiments already completed")
    
    # Run experiments
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "complexity_levels": complexity_levels,
        "num_problems_per_level": num_problems_per_level,
        "model": model,
    }
    
    for complexity_level in complexity_levels:
        for exp_name, exp_fn in experiments.items():
            key = (exp_name, model, complexity_level)
            
            if key in completed_keys:
                print(f"\nSkipping {exp_name} level {complexity_level} (already completed)")
                continue
            
            try:
                result = run_experiment(
                    experiment_name=exp_name,
                    experiment_fn=exp_fn,
                    loader=loader,
                    complexity_level=complexity_level,
                    num_problems=num_problems_per_level,
                    model=model,
                    verbose=True,
                )
                all_results.append(result)
                
                # Save checkpoint after each experiment
                save_checkpoint(output_path, all_results, metadata)
                print(f"\n[Checkpoint saved]")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted! Saving checkpoint...")
                save_checkpoint(output_path, all_results, metadata)
                print(f"Checkpoint saved to {output_path.with_suffix('.checkpoint.json')}")
                print("Run again with same parameters to resume.")
                raise
            except Exception as e:
                print(f"\n\nExperiment {exp_name} failed with error: {e}")
                print("Saving checkpoint and continuing with next experiment...")
                save_checkpoint(output_path, all_results, metadata)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Experiment':<20} {'Level':<7} {'Accuracy':<12} {'Avg Cost':<12} {'Total Cost':<12}")
    print("-" * 80)
    
    for res in all_results:
        if res.get('model') == model:  # Only show current model
            print(f"{res['experiment']:<20} "
                  f"{res['complexity_level']:<7} "
                  f"{res['accuracy']*100:>6.1f}%     "
                  f"${res['avg_cost']:>9.6f}  "
                  f"${res['total_cost']:>9.6f}")
    
    # Save final results
    output_data = {
        "metadata": {
            **metadata,
            "completed_timestamp": datetime.now().isoformat(),
            "total_experiments": len(all_results),
            "total_api_calls": sum(r.get('num_problems', 0) for r in all_results),
        },
        "results": all_results,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, cls=NumpyEncoder)
    
    # Remove checkpoint file after successful completion
    checkpoint_path = output_path.with_suffix('.checkpoint.json')
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"Total API calls: {output_data['metadata']['total_api_calls']}")
    print(f"Total cost: ${sum(r.get('total_cost', 0) for r in all_results):.2f}")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument(
        "--num-problems",
        type=int,
        default=5,
        help="Number of problems per experiment (default: 5 for testing, use 100 for full run)"
    )
    parser.add_argument(
        "--complexity",
        type=int,
        nargs="+",
        default=[0],
        help="Complexity levels to test (default: 0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        choices=[
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3",
        ],
        help="Model to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline_results.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint, start fresh"
    )
    
    args = parser.parse_args()
    
    # Estimate cost
    estimate = estimate_cost(args.num_problems, args.complexity, args.model)
    
    print("=" * 80)
    print("BASELINE EXPERIMENTS - FULL RUN")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Complexity levels: {args.complexity}")
    print(f"Problems per level: {args.num_problems}")
    print(f"Total experiments: {len(args.complexity) * 4}")
    print(f"Total API calls: {estimate['total_calls']}")
    print(f"Estimated cost: ${estimate['total_estimated_cost']:.2f}")
    print(f"Output: {args.output}")
    print("=" * 80)
    
    print("\nCost breakdown by experiment:")
    for exp_name, exp_data in estimate['by_experiment'].items():
        print(f"  {exp_name}: {exp_data['calls']} calls, ~${exp_data['estimated_cost']:.2f}")
    
    if not args.yes:
        try:
            response = input("\nPress Enter to start (or Ctrl+C to cancel)...")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
    
    run_full_baseline(
        complexity_levels=args.complexity,
        num_problems_per_level=args.num_problems,
        model=args.model,
        output_file=args.output,
        resume=not args.no_resume,
    )
