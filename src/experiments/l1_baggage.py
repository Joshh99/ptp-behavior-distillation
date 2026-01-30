"""
L1 Baggage Allowance - PTool Pattern (The "SecretAgent" Sweet Spot)

This is the target reliability level for behavior distillation.
LLM extracts parameters → Python calculates fees.

Target reliability: 95%+
Target cost: Minimal (1 LLM call per query)

Architecture:
    1. LLM extracts: customer_class, routine, direction, bag_list
    2. Python calculates: fee based on RuleArena's deterministic rules
    3. Return total cost (ticket + baggage fees)

This demonstrates Prof. Cohen's key insight:
    "Python does maximum heavy lifting before handing to LLMs"
"""

import sys
import os
import time
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.rulearena_loader import RuleArenaLoader
from experiments.config import (
    get_experiment_config,
    get_model_config,
    ExperimentLevel,
    RESULTS_DIR,
    MODELS,
    call_llm,
    DEFAULT_MODEL,
)


# =============================================================================
# COST TRACKING
# =============================================================================

@dataclass
class CostTracker:
    """Track API costs for experiment runs."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    model: str = DEFAULT_MODEL
    
    def add_call(self, input_tokens: int, output_tokens: int):
        """Record an API call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1
    
    @property
    def total_cost_usd(self) -> float:
        """Calculate total cost in USD."""
        if self.model not in MODELS:
            return 0.0
        config = MODELS[self.model]
        input_cost = (self.total_input_tokens / 1_000_000) * config.cost_per_m_input
        output_cost = (self.total_output_tokens / 1_000_000) * config.cost_per_m_output
        return input_cost + output_cost
    
    def to_dict(self) -> Dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_calls": self.total_calls,
            "model": self.model,
            "total_cost_usd": self.total_cost_usd,
        }


# =============================================================================
# PTOOLS - LLM-powered parameter extraction (Together.ai)
# =============================================================================

# Extraction prompt tailored for RuleArena airline baggage format
EXTRACTION_PROMPT = """You are an expert at extracting structured information from airline baggage queries.

Given a passenger scenario, extract the following parameters as a JSON object:
- customer_class: string (one of: "Basic Economy", "Main Cabin", "Main Plus", "Premium Economy", "Business", "First")
- routine: string (destination region, e.g., "U.S.", "Canada", "Mexico", "Europe", "China", etc.)
- direction: integer (0 = departing from U.S. to destination, 1 = returning to U.S. from destination)
- base_price: integer (ticket price in USD)
- bag_list: array of objects, each with:
  - id: integer (bag number starting from 1)
  - name: string ("backpack" or "luggage box")
  - size: array of 3 integers [length, width, height] in inches
  - weight: integer in lbs

IMPORTANT:
- The first item in bag_list is typically a carry-on (small backpack)
- Parse ALL items listed in the query
- Look for phrases like "flying from X to Y" to determine direction
- "from [international city] to [US city]" means direction=1 (arriving to US)
- "from [US city] to [international city]" means direction=0 (departing from US)
- US cities include: Orlando, Philadelphia, Charlotte, Phoenix, Las Vegas, Atlanta, etc.

Query:
{query}

Respond with ONLY a valid JSON object, no other text.
"""


def extract_rulearena_params(query: str, model: str = DEFAULT_MODEL) -> Tuple[Dict, int, int]:
    """
    Extract RuleArena-specific parameters from a passenger query.
    
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
        # Handle markdown code blocks
        json_text = response.strip()
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
        
        # Validate and set defaults
        defaults = {
            "customer_class": "Main Cabin",
            "routine": "U.S.",
            "direction": 0,
            "base_price": 0,
            "bag_list": [],
        }
        
        for key, default_value in defaults.items():
            if key not in params or params[key] is None:
                params[key] = default_value
        
        return params, input_tokens, output_tokens
        
    except Exception as e:
        print(f"  Warning: LLM extraction failed ({e})")
        return {
            "customer_class": "Main Cabin",
            "routine": "U.S.",
            "direction": 0,
            "base_price": 0,
            "bag_list": [],
        }, input_tokens, 0


# =============================================================================
# PYTHON RULE ENGINE - Using RuleArena's fee calculation
# =============================================================================

class RuleArenaFeeCalculator:
    """
    Fee calculator using RuleArena's exact logic.
    
    This wraps the loader's calculation methods for use in L1 pipeline.
    """
    
    def __init__(self, loader: RuleArenaLoader):
        self.loader = loader
    
    def calculate_total_cost(
        self,
        base_price: int,
        direction: int,
        routine: str,
        customer_class: str,
        bag_list: List[Dict[str, Any]],
    ) -> int:
        """
        Calculate total cost using RuleArena's fee logic.
        
        This is the same calculation as the loader's _compute_answer method.
        """
        return self.loader._compute_answer(
            base_price=base_price,
            direction=direction,
            routine=routine,
            customer_class=customer_class,
            bag_list=bag_list,
        )


# =============================================================================
# L1 WORKFLOW - The main pipeline
# =============================================================================

def baggage_allowance_l1(
    query: str,
    loader: RuleArenaLoader,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], int, int]:
    """
    L1 PTool pattern for baggage allowance queries.
    
    Architecture:
        1. Extract parameters (1 LLM call via PTool)
        2. Calculate fees (Pure Python - deterministic, using RuleArena logic)
        3. Return total cost
    
    Args:
        query: Natural language query from RuleArena problem
        loader: RuleArenaLoader instance for fee calculation
        model: Model to use for extraction
        verbose: Print progress
    
    Returns:
        Tuple of (result_dict, input_tokens, output_tokens)
    """
    calculator = RuleArenaFeeCalculator(loader)
    
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
    
    # Step 2: Calculate total cost (Pure Python - no LLM)
    calc_start = time.time()
    try:
        total_cost = calculator.calculate_total_cost(
            base_price=params.get("base_price", 0),
            direction=params.get("direction", 0),
            routine=params.get("routine", "U.S."),
            customer_class=params.get("customer_class", "Main Cabin"),
            bag_list=params.get("bag_list", []),
        )
    except Exception as e:
        if verbose:
            print(f"  Warning: Fee calculation failed ({e}), returning base price")
        total_cost = params.get("base_price", 0)
    
    calc_time = time.time() - calc_start
    
    if verbose:
        print(f"  2. Calculated total: ${total_cost} in {calc_time:.4f}s")
    
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


# =============================================================================
# EVALUATION WITH RULEARENA
# =============================================================================

def evaluate_l1_on_rulearena(
    loader: RuleArenaLoader,
    complexity_level: int = 0,
    max_problems: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate L1 PTool approach on RuleArena dataset.
    
    Args:
        loader: RuleArenaLoader instance
        complexity_level: 0=easy (5 bags), 1=medium (8 bags), 2=hard (11 bags)
        max_problems: Maximum problems to evaluate (None = all)
        model: Model to use for extraction
        verbose: Print progress
    
    Returns:
        Dict with metrics: accuracy, cost, latency, detailed results
    """
    # Load problems
    problems = loader.load_airline_problems(
        complexity_level=complexity_level,
        max_problems=max_problems,
    )
    
    if not problems:
        print("No problems loaded!")
        return {"error": "No problems loaded"}
    
    # Initialize tracking
    cost_tracker = CostTracker(model=model)
    results = []
    correct = 0
    total_time = 0.0
    
    print("=" * 80)
    print(f"L1 PTool Evaluation - RuleArena Complexity {complexity_level}")
    print(f"Model: {model}")
    print(f"Problems: {len(problems)}")
    print("=" * 80)
    
    for i, problem in enumerate(problems):
        query = problem["query"]
        expected = problem["ground_truth"]
        
        if verbose:
            print(f"\n[{i+1}/{len(problems)}]")
        
        # Run L1 pipeline
        result, input_tokens, output_tokens = baggage_allowance_l1(
            query=query,
            loader=loader,
            model=model,
            verbose=verbose,
        )
        
        # Track costs
        cost_tracker.add_call(input_tokens, output_tokens)
        
        # Check correctness
        predicted = result["answer"]
        is_correct = predicted == expected
        
        if is_correct:
            correct += 1
            status = "CORRECT"
        else:
            status = "WRONG"
        
        if verbose:
            print(f"  Result: ${predicted} | Expected: ${expected} | {status}")
        
        total_time += result["metrics"]["total_time"]
        
        results.append({
            "problem_id": problem["id"],
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
            "extracted_params": result["extracted_params"],
            "time": result["metrics"]["total_time"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })
    
    # Calculate summary metrics
    accuracy = correct / len(problems) if problems else 0
    avg_time = total_time / len(problems) if problems else 0
    avg_cost = cost_tracker.total_cost_usd / len(problems) if problems else 0
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Accuracy:        {accuracy*100:.1f}% ({correct}/{len(problems)})")
    print(f"Avg Time:        {avg_time:.2f}s per problem")
    print(f"Total Time:      {total_time:.2f}s")
    print(f"Total Cost:      ${cost_tracker.total_cost_usd:.6f}")
    print(f"Avg Cost:        ${avg_cost:.6f} per problem")
    print(f"LLM Calls:       {cost_tracker.total_calls}")
    print(f"Total Tokens:    {cost_tracker.total_input_tokens + cost_tracker.total_output_tokens}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "avg_time": avg_time,
        "total_time": total_time,
        "cost": cost_tracker.to_dict(),
        "avg_cost_per_problem": avg_cost,
        "model": model,
        "complexity_level": complexity_level,
        "results": results,
    }


# =============================================================================
# BASELINE RUN
# =============================================================================

def run_l1_baseline_rulearena(
    num_problems: int = 30,
    complexity_level: int = 0,
    model: str = DEFAULT_MODEL,
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Run L1 baseline evaluation on RuleArena.
    
    Args:
        num_problems: Number of problems to evaluate
        complexity_level: 0=easy, 1=medium, 2=hard
        model: Model to use
        save_results: Whether to save results to JSON file
    
    Returns:
        Evaluation results dictionary
    """
    print("=" * 80)
    print("L1 RuleArena Baseline")
    print("=" * 80)
    print(f"Problems: {num_problems}")
    print(f"Complexity: {complexity_level}")
    print(f"Model: {model}")
    print()
    
    # Initialize loader
    try:
        loader = RuleArenaLoader("external/RuleArena")
        print("RuleArenaLoader initialized successfully")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    
    # Run evaluation
    results = evaluate_l1_on_rulearena(
        loader=loader,
        complexity_level=complexity_level,
        max_problems=num_problems,
        model=model,
        verbose=True,
    )
    
    # Add metadata
    results["metadata"] = {
        "experiment": "l1_rulearena_baseline",
        "timestamp": datetime.now().isoformat(),
        "rulearena_path": "external/RuleArena",
    }
    
    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(RESULTS_DIR, "l1_rulearena_baseline.json")
        
        # Make results JSON serializable
        results_to_save = {k: v for k, v in results.items() if k != "results"}
        results_to_save["sample_results"] = results["results"][:5]  # Save first 5 detailed results
        results_to_save["num_detailed_results"] = len(results["results"])
        
        with open(output_path, "w") as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("FINAL RESULTS TABLE")
    print("=" * 80)
    print(f"| Metric                | Value                    |")
    print(f"|----------------------|--------------------------|")
    print(f"| Accuracy             | {results['accuracy']*100:.1f}%                    |")
    print(f"| Avg Cost/Problem     | ${results['avg_cost_per_problem']:.6f}           |")
    print(f"| Total Time           | {results['total_time']:.2f}s                   |")
    print(f"| Total Cost           | ${results['cost']['total_cost_usd']:.6f}           |")
    print(f"| Model                | {model}              |")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("L1 Baggage Allowance - PTool Pattern with RuleArena")
    print("=" * 80)
    print("\nThis demonstrates the 'SecretAgent sweet spot':")
    print("  LLM extracts parameters -> Python calculates deterministically")
    print()
    
    # First, test on 5 problems
    print("\n--- Test Run: 5 problems ---\n")
    test_results = run_l1_baseline_rulearena(
        num_problems=5,
        complexity_level=0,
        model=DEFAULT_MODEL,
        save_results=False,
    )
    
    if "error" not in test_results:
        print(f"\nTest accuracy: {test_results['accuracy']*100:.1f}%")
        
        # If test looks reasonable, run the full 30
        print("\n" + "=" * 80)
        print("--- Full Baseline: 30 problems ---")
        print("=" * 80 + "\n")
        
        full_results = run_l1_baseline_rulearena(
            num_problems=30,
            complexity_level=0,
            model=DEFAULT_MODEL,
            save_results=True,
        )
        
        print(f"\nFinal Accuracy: {full_results['accuracy']*100:.1f}%")
    else:
        print(f"\nTest failed: {test_results['error']}")
