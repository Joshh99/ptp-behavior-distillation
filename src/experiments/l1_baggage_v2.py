"""
L1 Baggage Allowance Evaluation - Using New AirlineLoader

This is an updated version of l1_baggage.py that uses the new AirlineLoader
with RuleArena's reference implementation for ground truth.

Key improvements:
- Simpler code (no manual ground truth computation)
- Guaranteed correctness (uses benchmark's reference implementation)
- Better accuracy on complex problems
"""

import time
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
import json
import os

# Add project root to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import new loader
from src.dataset.airline_loader import AirlineLoader

# ==============================================================================
# EXTRACTION PTOOL (UNCHANGED)
# ==============================================================================

def extract_rulearena_params(query: str, model: str = "gpt-4o-mini") -> Tuple[Dict[str, Any], int, int]:
    """
    Extract structured parameters from natural language query using PTP.
    
    This is your L1 extraction step - uses LLM to convert NL → structured data.
    
    Returns:
        Tuple of (params_dict, input_tokens, output_tokens)
    """
    # TODO: Implement your actual extraction logic here
    # This is a placeholder - replace with your secretagent PTP code
    
    # Mock response for testing
    params = {
        "base_price": 180,
        "customer_class": "Main Cabin",
        "routine": "U.S.",
        "direction": 0,
        "bag_list": [
            {"id": 1, "name": "backpack", "size": [22, 13, 6], "weight": 10},
            {"id": 2, "name": "luggage box", "size": [44, 22, 20], "weight": 69},
        ]
    }
    
    return params, 100, 50  # Mock token counts


# ==============================================================================
# L1 WORKFLOW (SIMPLIFIED)
# ==============================================================================

def baggage_allowance_l1(
    query: str,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> Tuple[Dict[str, Any], int, int]:
    """
    L1 PTool pattern for baggage allowance queries.
    
    Architecture:
        1. Extract parameters (1 LLM call via PTool)
        2. Use extracted params as answer
    
    NOTE: Ground truth is computed separately by the loader using RuleArena's
          reference implementation. Your job is to extract params correctly.
    
    Args:
        query: Natural language query from RuleArena problem
        model: Model to use for extraction
        verbose: Print progress
    
    Returns:
        Tuple of (result_dict, input_tokens, output_tokens)
    """
    if verbose:
        print(f"\n[L1 PTool] Processing: {query[:80]}...")
    
    # Step 1: Extract parameters (LLM call)
    start_time = time.time()
    params, input_tokens, output_tokens = extract_rulearena_params(query, model=model)
    extraction_time = time.time() - start_time
    
    if verbose:
        print(f"  1. Extracted in {extraction_time:.2f}s:")
        print(f"     Class: {params.get('customer_class')}, Route: {params.get('routine')}")
        print(f"     Direction: {params.get('direction')}, Bags: {len(params.get('bag_list', []))}")
    
    result = {
        "extracted_params": params,
        "metrics": {
            "extraction_time": extraction_time,
            "llm_calls": 1,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    }
    
    return result, input_tokens, output_tokens


# ==============================================================================
# EVALUATION WITH NEW LOADER
# ==============================================================================

def evaluate_l1_on_airline(
    loader: AirlineLoader,
    complexity_level: int = 0,
    max_problems: Optional[int] = None,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate L1 PTool approach on RuleArena airline dataset.
    
    Args:
        loader: AirlineLoader instance
        complexity_level: 0=easy (5 bags), 1=medium (8 bags), 2=hard (11 bags)
        max_problems: Maximum problems to evaluate (None = all)
        model: Model to use for extraction
        verbose: Print progress
    
    Returns:
        Dict with metrics: accuracy, cost, latency, detailed results
    """
    # Load problems (ground truth computed automatically)
    problems = loader.load_problems(
        complexity_level=complexity_level,
        max_problems=max_problems,
    )
    
    if not problems:
        print("No problems loaded!")
        return {"error": "No problems loaded"}
    
    # Initialize tracking
    results = []
    correct = 0
    total_time = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    
    print("=" * 80)
    print(f"L1 PTool Evaluation - RuleArena Airline (Complexity {complexity_level})")
    print(f"Model: {model}")
    print(f"Problems: {len(problems)}")
    print("=" * 80)
    
    for i, problem in enumerate(problems):
        query = problem.query
        expected = problem.ground_truth  # From reference implementation
        
        if verbose:
            print(f"\n[{i+1}/{len(problems)}]")
        
        # Run L1 pipeline
        result, input_tokens, output_tokens = baggage_allowance_l1(
            query=query,
            model=model,
            verbose=verbose,
        )
        
        # Track tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        # Compute predicted answer from extracted params
        # TODO: Implement proper fee calculation from params
        # For now, use a placeholder
        predicted = 0  # Replace with actual calculation
        
        # Check correctness
        is_correct = predicted == expected
        
        if is_correct:
            correct += 1
            status = "✓ CORRECT"
        else:
            status = f"✗ WRONG (expected ${expected}, got ${predicted})"
        
        if verbose:
            print(f"  Status: {status}")
        
        # Store result
        results.append({
            "problem_id": problem.id,
            "query": query,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "extraction_time": result["metrics"]["extraction_time"],
        })
        
        total_time += result["metrics"]["extraction_time"]
    
    # Calculate metrics
    accuracy = correct / len(problems)
    avg_time = total_time / len(problems)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Accuracy:        {correct}/{len(problems)} ({accuracy*100:.1f}%)")
    print(f"Avg Time:        {avg_time:.2f}s per problem")
    print(f"Total Time:      {total_time:.2f}s")
    print(f"Total Tokens:    {total_input_tokens + total_output_tokens}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "avg_time": avg_time,
        "total_time": total_time,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "model": model,
        "complexity_level": complexity_level,
        "results": results,
    }


# ==============================================================================
# BASELINE RUN
# ==============================================================================

def run_l1_baseline(
    num_problems: int = 30,
    complexity_level: int = 0,
    model: str = "gpt-4o-mini",
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Run L1 baseline evaluation on RuleArena Airline.
    
    Args:
        num_problems: Number of problems to evaluate
        complexity_level: 0=easy, 1=medium, 2=hard
        model: Model to use
        save_results: Whether to save results to JSON file
    
    Returns:
        Evaluation results dictionary
    """
    print("=" * 80)
    print("L1 RuleArena Airline Baseline (New Loader)")
    print("=" * 80)
    print(f"Problems: {num_problems}")
    print(f"Complexity: {complexity_level}")
    print(f"Model: {model}")
    print()
    
    # Initialize loader (uses RuleArena reference implementation)
    try:
        loader = AirlineLoader("external/RuleArena")
        print()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    
    # Run evaluation
    results = evaluate_l1_on_airline(
        loader=loader,
        complexity_level=complexity_level,
        max_problems=num_problems,
        model=model,
        verbose=True,
    )
    
    # Add metadata
    results["metadata"] = {
        "experiment": "l1_airline_baseline_v2",
        "timestamp": datetime.now().isoformat(),
        "uses_reference_implementation": True,
        "attribution": "RuleArena benchmark (Zhou et al., ACL 2025)",
    }
    
    # Save results
    if save_results:
        os.makedirs("results", exist_ok=True)
        output_path = os.path.join("results", "l1_airline_baseline_v2.json")
        
        # Convert numpy types to native Python types
        def convert_numpy(obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Make results JSON serializable
        results_to_save = {k: v for k, v in results.items() if k != "results"}
        results_to_save["sample_results"] = results["results"][:5]
        results_to_save = convert_numpy(results_to_save)
        
        with open(output_path, "w") as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    return results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    results = run_l1_baseline(
        num_problems=10,  # Start small for testing
        complexity_level=0,
        model="gpt-4o-mini",
        save_results=True,
    )
    
    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print("=" * 80)