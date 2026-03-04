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
import re

# Add project root to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import new loader
from src.dataset.airline_loader import AirlineLoader

# Import config for LLM calls
from src.experiments.config import call_llm, DEFAULT_MODEL


# ==============================================================================
# EXTRACTION PTOOL (IMPLEMENTED)
# ==============================================================================

# Extraction prompt with routing logic
EXTRACTION_PROMPT = """You are an expert at extracting structured information from airline baggage queries.

Given a passenger scenario, extract the following parameters as a JSON object:

FIELDS:
- base_price: integer (ticket price in USD)
- customer_class: string (one of: "Basic Economy", "Main Cabin", "Main Plus", "Premium Economy", "Business", "First")
- routine: string (destination region - see ROUTING RULES below)
- direction: integer (0 or 1 - see ROUTING RULES below)
- bag_list: array of objects, each with:
  - id: integer (bag number starting from 1)
  - name: string ("backpack" or "luggage box")
  - size: array of 3 integers [length, width, height] in inches
  - weight: integer in lbs

ROUTING RULES:
1. Identify Origin and Destination cities
2. Determine if flight is:
   - Domestic (US -> US): routine="U.S.", direction=0
   - Outbound (US -> World): routine=<destination_region>, direction=0
   - Inbound (World -> US): routine=<origin_region>, direction=1

Valid regions: "U.S.", "Puerto Rico", "Canada", "Mexico", "Cuba", "Haiti", "Panama", "Colombia", "Ecuador", "Peru", "South America", "Israel", "Qatar", "Europe", "India", "China", "Japan", "South Korea", "Hong Kong", "Australia", "New Zealand"

US cities include: Orlando, Philadelphia, Charlotte, Phoenix, Las Vegas, Atlanta, Boston, New York, Los Angeles, San Francisco, Miami, etc.

EXAMPLES:
- "Orlando to Tokyo" -> routine="Japan", direction=0 (US->World)
- "Paris to Atlanta" -> routine="Europe", direction=1 (World->US)
- "Phoenix to Charlotte" -> routine="U.S.", direction=0 (US->US)

Query:
{query}

Respond with ONLY a valid JSON object, no other text.
"""


def extract_rulearena_params(query: str, model: str = DEFAULT_MODEL) -> Tuple[Dict[str, Any], int, int]:
    """
    Extract structured parameters from natural language query using PTP.
    
    This is your L1 extraction step - uses LLM to convert NL → structured data.
    
    Returns:
        Tuple of (params_dict, input_tokens, output_tokens)
    """
    prompt = EXTRACTION_PROMPT.format(query=query)
    
    # Estimate input tokens (rough: ~4 chars per token)
    input_tokens = len(prompt) // 4
    
    try:
        response = call_llm(prompt, model=model, max_tokens=512)
        output_tokens = len(response) // 4
        
        # Parse JSON from response
        json_text = response.strip()
        
        # Handle markdown code blocks
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0]
        
        # Find JSON object
        json_match = re.search(r'\{[\s\S]*\}', json_text)
        if json_match:
            params = json.loads(json_match.group())
        else:
            params = json.loads(json_text)
        
        # Post-processing: Sort bag dimensions (L >= W >= H) for consistency
        for bag in params.get("bag_list", []):
            if "size" in bag and len(bag["size"]) == 3:
                bag["size"] = sorted(bag["size"], reverse=True)
        
        # Validate required fields
        defaults = {
            "base_price": 0,
            "customer_class": "Main Cabin",
            "routine": "U.S.",
            "direction": 0,
            "bag_list": [],
        }
        
        for key, default_value in defaults.items():
            if key not in params or params[key] is None:
                params[key] = default_value
        
        return params, input_tokens, output_tokens
        
    except Exception as e:
        print(f"  Warning: LLM extraction failed ({e})")
        return {
            "base_price": 0,
            "customer_class": "Main Cabin",
            "routine": "U.S.",
            "direction": 0,
            "bag_list": [],
        }, input_tokens, 0


# ==============================================================================
# L1 WORKFLOW (SIMPLIFIED)
# ==============================================================================

def baggage_allowance_l1(
    query: str,
    loader: AirlineLoader,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], int, int]:
    """
    L1 PTool pattern for baggage allowance queries.
    
    Architecture:
        1. Extract parameters (1 LLM call via PTool)
        2. Calculate fees (Pure Python using loader's reference implementation)
    
    NOTE: Ground truth is computed separately by the loader using RuleArena's
          reference implementation. Your job is to extract params correctly.
    
    Args:
        query: Natural language query from RuleArena problem
        loader: AirlineLoader instance for fee calculation
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
    
    # Step 2: Calculate fees using loader's reference implementation
    calc_start = time.time()
    try:
        result_tuple = loader._compute_answer_fn(
            base_price=params['base_price'],
            direction=params['direction'],
            routine=params['routine'],
            customer_class=params['customer_class'],
            bag_list=params['bag_list'],
            check_base_tables=loader._fee_tables,
        )
        # Unpack tuple (cost, details)
        total_cost = int(result_tuple[0]) if isinstance(result_tuple, tuple) else int(result_tuple)
        calc_time = time.time() - calc_start
        
        if verbose:
            print(f"  2. Calculated fees in {calc_time:.3f}s: ${total_cost}")
    
    except Exception as e:
        print(f"  Error in fee calculation: {e}")
        total_cost = params['base_price']  # Fallback to just base price
        calc_time = time.time() - calc_start
    
    result = {
        "answer": total_cost,
        "extracted_params": params,
        "metrics": {
            "extraction_time": extraction_time,
            "calculation_time": calc_time,
            "total_time": extraction_time + calc_time,
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
    model: str = DEFAULT_MODEL,
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
            loader=loader,
            model=model,
            verbose=verbose,
        )
        
        # Track tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        # Get predicted answer
        predicted = result["answer"]
        
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
            "calculation_time": result["metrics"]["calculation_time"],
            "total_time": result["metrics"]["total_time"],
        })
        
        total_time += result["metrics"]["total_time"]
    
    # Calculate metrics
    accuracy = correct / len(problems)
    avg_time = total_time / len(problems)
    
    # Estimate cost (rough estimate based on token counts)
    # Using Together.ai pricing for Qwen2.5-72B: $0.88/$0.88 per M tokens
    total_tokens = total_input_tokens + total_output_tokens
    estimated_cost = (total_tokens / 1_000_000) * 0.88
    avg_cost = estimated_cost / len(problems)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Accuracy:        {correct}/{len(problems)} ({accuracy*100:.1f}%)")
    print(f"Avg Time:        {avg_time:.2f}s per problem")
    print(f"Total Time:      {total_time:.2f}s")
    print(f"Total Tokens:    {total_tokens}")
    print(f"Estimated Cost:  ${estimated_cost:.6f}")
    print(f"Avg Cost:        ${avg_cost:.6f} per problem")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "avg_time": avg_time,
        "total_time": total_time,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": estimated_cost,
        "avg_cost_per_problem": avg_cost,
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
    model: str = DEFAULT_MODEL,
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
        print("AirlineLoader initialized successfully")
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
            elif isinstance(obj, np.bool_): 
                return bool(obj)
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
        model=DEFAULT_MODEL,
        save_results=True,
    )
    
    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print("=" * 80)