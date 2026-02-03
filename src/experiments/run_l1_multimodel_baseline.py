"""
Multi-Model L1 PTP Baseline Experiment

Compares L1 PTP performance across multiple models, complexity levels, and shot settings.
This replicates RuleArena's CoT experiments but using PTP instead.

Models tested:
- Qwen-2.5 72B
- DeepSeek-R1 Distilled Llama 70B
- Llama-3.1 70B
- Llama-3.1 405B

Complexity levels: 0 (5 bags), 1 (8 bags), 2 (11 bags)
Shot settings: 0-shot, 1-shot
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dataset.airline_loader import AirlineLoader
from src.experiments.config import call_llm
from scripts.usage_tracker import UsageTracker
from src.experiments.l1_baggage_v2 import extract_rulearena_params, baggage_allowance_l1


# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================

MODELS = {
    "qwen-72b": {
        "name": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "cost_per_m_input": 0.88,
        "cost_per_m_output": 0.88,
        "display_name": "Qwen-2.5 72B",
    },
    # "deepseek-r1-70b": {
    #     "name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    #     "cost_per_m_input": 2.0,
    #     "cost_per_m_output": 2.0,
    #     "display_name": "DeepSeek-R1 Distilled Llama 70B",
    # },
    "llama-70b": {
        "name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "cost_per_m_input": 0.88,
        "cost_per_m_output": 0.88,
        "display_name": "Llama-3.1 70B",
    },
    "llama-405b": {
        "name": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "cost_per_m_input": 3.5,
        "cost_per_m_output": 3.5,
        "display_name": "Llama-3.1 405B",
    },
}

COMPLEXITY_LEVELS = [0, 1, 2]  # 5, 8, 11 bags
SHOT_SETTINGS = [0, 1]  # 0-shot, 1-shot


# ==============================================================================
# 1-SHOT EXAMPLE (Perfect PTP L1 Pattern)
# ==============================================================================

ONE_SHOT_EXAMPLE = """Example of parameter extraction:

Query: "Sarah is a Main Cabin Class passenger flying from Orlando to Philadelphia with ticket price $180. She has 2 items: (1) backpack (22 x 13 x 6 in, 10 lbs), (2) luggage box (44 x 22 x 20 in, 69 lbs)"

Extracted Parameters:
{
  "base_price": 180,
  "customer_class": "Main Cabin",
  "routine": "U.S.",
  "direction": 0,
  "bag_list": [
    {"id": 1, "name": "backpack", "size": [22, 13, 6], "weight": 10},
    {"id": 2, "name": "luggage box", "size": [44, 22, 20], "weight": 69}
  ]
}
"""

def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj

# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

def run_single_experiment(
    model_key: str,
    complexity_level: int,
    num_shots: int,
    num_problems: int,
    loader: AirlineLoader,
    usage_tracker: UsageTracker,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run L1 baseline on a single model/complexity/shot combination.
    
    Returns:
        Dict with accuracy, cost, latency metrics
    """
    model_config = MODELS[model_key]
    model_name = model_config["name"]
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"Experiment: {model_config['display_name']} | Level {complexity_level} | {num_shots}-shot")
        print("=" * 80)
    
    # Load problems
    problems = loader.load_problems(
        complexity_level=complexity_level,
        max_problems=num_problems,
    )
    
    if not problems:
        return {"error": "No problems loaded"}
    
    # Track metrics
    results = []
    correct = 0
    total_time = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, problem in enumerate(problems):
        if verbose:
            print(f"\n[{i+1}/{len(problems)}] ", end="")
        
        # Add 1-shot example to query if needed
        query = problem.query
        if num_shots == 1:
            query = ONE_SHOT_EXAMPLE + "\n\nNow extract from this query:\n" + query
        
        # Run L1 pipeline
        start_time = time.time()
        result, input_tokens, output_tokens = baggage_allowance_l1(
            query=query,
            loader=loader,
            model=model_name,
            verbose=False,  # Reduce noise
        )
        elapsed_time = time.time() - start_time
        
        # Track tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        # Calculate cost
        cost = (
            (input_tokens / 1_000_000) * model_config["cost_per_m_input"] +
            (output_tokens / 1_000_000) * model_config["cost_per_m_output"]
        )
        
        # Log to usage tracker
        usage_tracker.log_call(
            model=model_name,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            cost=cost,
        )
        
        # Check correctness
        predicted = result["answer"]
        expected = problem.ground_truth
        is_correct = predicted == expected
        
        if is_correct:
            correct += 1
            if verbose:
                print("✓", end="")
        else:
            if verbose:
                print(f"✗ (exp ${expected}, got ${predicted})", end="")
        
        total_time += elapsed_time
        
        results.append({
            "problem_id": problem.id,
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
            "time": elapsed_time,
            "cost": cost,
        })
    
    # Calculate summary metrics
    accuracy = correct / len(problems) if problems else 0
    avg_time = total_time / len(problems) if problems else 0
    total_cost = sum(r["cost"] for r in results)
    avg_cost = total_cost / len(problems) if problems else 0
    
    if verbose:
        print(f"\n\nAccuracy: {accuracy*100:.1f}% ({correct}/{len(problems)})")
        print(f"Avg Time: {avg_time:.2f}s")
        print(f"Avg Cost: ${avg_cost:.6f}")
        print(f"Total Cost: ${total_cost:.6f}")
    
    return {
        "model": model_key,
        "model_display": model_config["display_name"],
        "complexity_level": complexity_level,
        "num_shots": num_shots,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "avg_time": avg_time,
        "total_time": total_time,
        "avg_cost": avg_cost,
        "total_cost": total_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "results": results,
    }


def run_all_experiments(
    num_problems: int = 5,
    save_results: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run all model/complexity/shot combinations.
    
    Args:
        num_problems: Number of problems per experiment (start with 5 for testing)
        save_results: Save detailed results to JSON
    
    Returns:
        List of all experiment results
    """
    print("=" * 80)
    print("L1 PTP Multi-Model Baseline Experiments")
    print("=" * 80)
    print(f"Models: {len(MODELS)}")
    print(f"Complexity Levels: {len(COMPLEXITY_LEVELS)}")
    print(f"Shot Settings: {len(SHOT_SETTINGS)}")
    print(f"Problems per experiment: {num_problems}")
    print(f"Total experiments: {len(MODELS) * len(COMPLEXITY_LEVELS) * len(SHOT_SETTINGS)}")
    print(f"Total API calls: {len(MODELS) * len(COMPLEXITY_LEVELS) * len(SHOT_SETTINGS) * num_problems}")
    print("=" * 80)
    
    # Initialize
    loader = AirlineLoader("external/RuleArena")
    usage_tracker = UsageTracker("scripts/usage_log.json")
    
    all_results = []
    
    # Run all combinations
    for model_key in MODELS.keys():
        for complexity_level in COMPLEXITY_LEVELS:
            for num_shots in SHOT_SETTINGS:
                try:
                    result = run_single_experiment(
                        model_key=model_key,
                        complexity_level=complexity_level,
                        num_shots=num_shots,
                        num_problems=num_problems,
                        loader=loader,
                        usage_tracker=usage_tracker,
                        verbose=True,
                    )
                    all_results.append(result)
                    
                    # Rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"\n✗ Error in experiment: {e}")
                    continue
    
    # Generate summary table
    print("\n" + "=" * 120)
    print("SUMMARY TABLE (for report.html)")
    print("=" * 120)
    print(f"{'Model':<35} {'Level':<7} {'Shot':<6} {'Accuracy':<12} {'Avg Cost':<12} {'Avg Latency':<14} {'Total Cost':<12}")
    print("-" * 120)
    
    for result in all_results:
        if "error" not in result:
            print(f"{result['model_display']:<35} "
                  f"{result['complexity_level']:<7} "
                  f"{result['num_shots']:<6} "
                  f"{result['accuracy']*100:>6.1f}%     "
                  f"${result['avg_cost']:>9.6f}  "
                  f"{result['avg_time']:>8.2f}s       "
                  f"${result['total_cost']:>9.6f}")
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/l1_multimodel_baseline_{timestamp}.json"
        os.makedirs("results", exist_ok=True)
        
        summary = {
            "metadata": {
                "experiment": "l1_ptp_multimodel_baseline",
                "timestamp": datetime.now().isoformat(),
                "num_problems_per_experiment": num_problems,
                "total_experiments": len(all_results),
            },
            "results": all_results,
        }

        summary_clean = convert_to_native_types(summary)
        with open(output_path, "w") as f:
            json.dump(summary_clean, f, indent=2)
        
        print(f"\n Results saved to: {output_path}")
    
    return all_results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Start with 5 problems for testing
    # Change to 100 after verifying everything works
    NUM_PROBLEMS = 5  # TODO: Change to 100 for full run
    
    print("\n TESTING MODE: Using 5 problems per experiment")
    print("    Change NUM_PROBLEMS to 100 for full baseline\n")
    
    results = run_all_experiments(
        num_problems=NUM_PROBLEMS,
        save_results=True,
    )
    
    print("\n" + "=" * 80)
    print("✓ All experiments complete!")
    print("=" * 80)
    print(f"Total experiments run: {len(results)}")
    print(f"Check usage_log.json for detailed cost tracking")