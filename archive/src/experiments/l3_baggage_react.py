"""
L3 Baggage Allowance - ReAct/Agentic Pattern (Together.ai)

Autonomous agent approach - flexible but unreliable.
Target reliability: 60-75%
Use for trace collection and distillation research.

This uses the ReAct (Reasoning + Acting) pattern where the LLM:
1. Thinks about what to do next
2. Takes an action (calls a tool or produces final answer)
3. Observes the result
4. Repeats until done or max_steps reached

Key difference from L1:
- L1: 1 LLM call (extraction) → Python calculates
- L3: Multiple LLM calls (agent loop) → LLM reasons through the problem
"""

import sys
import os
import re
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.rulearena_loader import RuleArenaLoader
from experiments.config import (
    call_llm,
    DEFAULT_MODEL,
    MODELS,
    RESULTS_DIR,
)


# =============================================================================
# COST TRACKING (same as L1 for comparison)
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
# REACT TRACE - Capture agent reasoning
# =============================================================================

@dataclass
class ReActTrace:
    """Captures the full ReAct agent execution trace."""
    query: str
    steps: List[Dict] = field(default_factory=list)
    final_answer: Optional[Any] = None
    success: bool = False
    total_time: float = 0.0
    llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    failed: bool = False
    failure_reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "steps": self.steps,
            "final_answer": self.final_answer,
            "success": self.success,
            "total_time": self.total_time,
            "llm_calls": self.llm_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
        }


# =============================================================================
# REACT AGENT - Autonomous reasoning with RuleArena
# =============================================================================

# ReAct prompt for airline baggage fee reasoning
REACT_SYSTEM_PROMPT = """You are an expert at calculating airline baggage fees.

Given a passenger scenario with ticket information and bags, you need to calculate the total cost (ticket price + all baggage fees).

IMPORTANT RULES:
1. The first item in the bag list is usually a carry-on (backpack) - it's FREE and doesn't count
2. Only checked bags (items after the first one) incur fees
3. Base fees depend on: cabin class, destination region, direction (departing/arriving US), and bag number
4. Additional fees apply for overweight (>50 lbs) and oversize (>62 inches total) bags
5. First/Business class often get free bags and higher weight limits
6. The total cost = ticket price + sum of all baggage fees

Think step by step:
1. Extract the ticket price
2. Identify which bags are checked vs carry-on
3. For each checked bag, calculate: base fee + overweight fee + oversize fee
4. Sum everything up

Your final answer MUST be a single integer representing the total cost in dollars.
Format your final answer as: ANSWER: [number]"""


class ReActAgent:
    """
    ReAct agent for airline baggage fee calculation.
    
    The agent uses reasoning and acting patterns to solve problems:
    - Reasoning: LLM thinks about what to do
    - Acting: LLM produces intermediate results or final answer
    
    Unlike L1 which uses 1 LLM call for extraction, L3 may use
    multiple calls as the agent reasons through the problem.
    """
    
    def __init__(
        self, 
        model: str = DEFAULT_MODEL, 
        max_steps: int = 10, 
        verbose: bool = True,
        timeout_seconds: float = 120.0,
    ):
        """
        Initialize ReAct agent.
        
        Args:
            model: Model to use for reasoning
            max_steps: Maximum reasoning steps before failing
            verbose: Print progress
            timeout_seconds: Maximum time per problem
        """
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        self.timeout_seconds = timeout_seconds
    
    def run(self, query: str, passenger_context: Dict = None) -> ReActTrace:
        """
        Run the ReAct agent on a baggage fee problem.
        
        Args:
            query: Natural language problem description
            passenger_context: Optional structured context (for debugging)
        
        Returns:
            ReActTrace with full execution history and result.
        """
        trace = ReActTrace(query=query)
        start = time.time()
        
        if self.verbose:
            print(f"[L3 ReAct] Processing: {query[:80]}...")
        
        # Build the full prompt with context
        full_prompt = f"""{REACT_SYSTEM_PROMPT}

Problem:
{query}

Calculate the total cost step by step, then provide your final answer as: ANSWER: [number]"""
        
        # Estimate input tokens
        input_tokens = len(full_prompt) // 4
        
        step = 0
        accumulated_reasoning = ""
        
        try:
            while step < self.max_steps:
                # Check timeout
                elapsed = time.time() - start
                if elapsed > self.timeout_seconds:
                    trace.failed = True
                    trace.failure_reason = f"Timeout after {elapsed:.1f}s"
                    break
                
                step += 1
                
                # Build prompt (initial or continuation)
                if step == 1:
                    prompt = full_prompt
                else:
                    # Continue reasoning
                    prompt = f"""{full_prompt}

Your previous reasoning:
{accumulated_reasoning}

Continue your analysis and provide your final answer as: ANSWER: [number]"""
                    input_tokens = len(prompt) // 4
                
                # Make LLM call
                try:
                    response = call_llm(prompt, model=self.model, max_tokens=1024)
                    output_tokens = len(response) // 4
                    
                    trace.llm_calls += 1
                    trace.total_input_tokens += input_tokens
                    trace.total_output_tokens += output_tokens
                    
                    if self.verbose:
                        print(f"  Step {step}: {len(response)} chars response")
                    
                    trace.steps.append({
                        "step": step,
                        "response": response[:500],  # Truncate for storage
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    })
                    
                    accumulated_reasoning += f"\n{response}"
                    
                except Exception as e:
                    trace.steps.append({"step": step, "error": str(e)})
                    if self.verbose:
                        print(f"  Step {step} error: {e}")
                    continue
                
                # Try to extract final answer
                answer_match = re.search(r'ANSWER:\s*\$?(\d+)', response)
                if answer_match:
                    trace.final_answer = int(answer_match.group(1))
                    trace.success = True
                    if self.verbose:
                        print(f"  Found answer: ${trace.final_answer}")
                    break
                
                # Also try common answer patterns
                patterns = [
                    r'total(?:\s+cost)?(?:\s+is)?[:\s]*\$?(\d+)',
                    r'final(?:\s+cost)?(?:\s+is)?[:\s]*\$?(\d+)',
                    r'=\s*\$?(\d+)\s*(?:dollars?)?\s*$',
                    r'\$(\d+)\s*(?:total|in total)',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                    if match:
                        trace.final_answer = int(match.group(1))
                        trace.success = True
                        if self.verbose:
                            print(f"  Found answer (pattern): ${trace.final_answer}")
                        break
                
                if trace.success:
                    break
            
            # Check if we exceeded max steps without answer
            if not trace.success and step >= self.max_steps:
                trace.failed = True
                trace.failure_reason = f"Exceeded max_steps ({self.max_steps})"
                
                # Last-ditch: try to find any reasonable number in the last response
                if trace.steps:
                    last_response = trace.steps[-1].get("response", "")
                    numbers = re.findall(r'\$?(\d{2,})', last_response)
                    if numbers:
                        # Take the last large number as a guess
                        trace.final_answer = int(numbers[-1])
                        if self.verbose:
                            print(f"  Fallback answer: ${trace.final_answer}")
                
        except Exception as e:
            trace.failed = True
            trace.failure_reason = f"Exception: {str(e)}"
            if self.verbose:
                print(f"  Fatal error: {e}")
        
        trace.total_time = time.time() - start
        return trace


# =============================================================================
# L3 PIPELINE - Main entry point
# =============================================================================

def baggage_allowance_l3(
    query: str,
    passenger_context: Dict = None,
    model: str = DEFAULT_MODEL,
    max_steps: int = 10,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], int, int]:
    """
    L3 ReAct pattern for baggage allowance queries.
    
    Architecture:
        1. Agent reasons about the problem (multiple LLM calls)
        2. Agent produces final answer
    
    Key difference from L1:
        - L1: 1 LLM call (extraction) → Python calculates = reliable
        - L3: Multiple LLM calls (reasoning) → LLM calculates = flexible but unreliable
    
    Args:
        query: Natural language query from RuleArena problem
        passenger_context: Optional structured context
        model: Model to use for reasoning
        max_steps: Maximum agent steps
        verbose: Print progress
    
    Returns:
        Tuple of (result_dict, total_input_tokens, total_output_tokens)
    """
    agent = ReActAgent(model=model, max_steps=max_steps, verbose=verbose)
    trace = agent.run(query, passenger_context)
    
    result = {
        "answer": trace.final_answer,
        "success": trace.success,
        "failed": trace.failed,
        "failure_reason": trace.failure_reason,
        "trace": trace.to_dict(),
        "metrics": {
            "total_time": trace.total_time,
            "llm_calls": trace.llm_calls,
            "input_tokens": trace.total_input_tokens,
            "output_tokens": trace.total_output_tokens,
        }
    }
    
    return result, trace.total_input_tokens, trace.total_output_tokens


# =============================================================================
# EVALUATION WITH RULEARENA
# =============================================================================

def evaluate_l3_on_rulearena(
    loader: RuleArenaLoader,
    complexity_level: int = 0,
    max_problems: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    max_steps: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate L3 ReAct approach on RuleArena dataset.
    
    Args:
        loader: RuleArenaLoader instance
        complexity_level: 0=easy (5 bags), 1=medium (8 bags), 2=hard (11 bags)
        max_problems: Maximum problems to evaluate (None = all)
        model: Model to use for reasoning
        max_steps: Maximum steps per problem
        verbose: Print progress
    
    Returns:
        Dict with metrics: accuracy, cost, latency, llm_calls, detailed results
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
    failed = 0
    total_time = 0.0
    total_llm_calls = 0
    
    print("=" * 80)
    print(f"L3 ReAct Evaluation - RuleArena Complexity {complexity_level}")
    print(f"Model: {model}")
    print(f"Max Steps: {max_steps}")
    print(f"Problems: {len(problems)}")
    print("=" * 80)
    
    for i, problem in enumerate(problems):
        query = problem["query"]
        expected = problem["ground_truth"]
        context = problem.get("passenger_context", {})
        
        if verbose:
            print(f"\n[{i+1}/{len(problems)}]")
        
        # Run L3 pipeline
        result, input_tokens, output_tokens = baggage_allowance_l3(
            query=query,
            passenger_context=context,
            model=model,
            max_steps=max_steps,
            verbose=verbose,
        )
        
        # Track costs
        cost_tracker.add_call(input_tokens, output_tokens)
        
        # Track LLM calls
        llm_calls = result["metrics"]["llm_calls"]
        total_llm_calls += llm_calls
        
        # Check correctness
        predicted = result["answer"]
        is_correct = predicted == expected
        is_failed = result.get("failed", False)
        
        if is_failed:
            failed += 1
            status = f"FAILED ({result.get('failure_reason', 'unknown')})"
        elif is_correct:
            correct += 1
            status = "CORRECT"
        else:
            status = "WRONG"
        
        if verbose:
            print(f"  Result: ${predicted} | Expected: ${expected} | Calls: {llm_calls} | {status}")
        
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
        })
    
    # Calculate summary metrics
    accuracy = correct / len(problems) if problems else 0
    failure_rate = failed / len(problems) if problems else 0
    avg_time = total_time / len(problems) if problems else 0
    avg_cost = cost_tracker.total_cost_usd / len(problems) if problems else 0
    avg_calls = total_llm_calls / len(problems) if problems else 0
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Accuracy:        {accuracy*100:.1f}% ({correct}/{len(problems)})")
    print(f"Failure Rate:    {failure_rate*100:.1f}% ({failed}/{len(problems)})")
    print(f"Avg Time:        {avg_time:.2f}s per problem")
    print(f"Total Time:      {total_time:.2f}s")
    print(f"Total Cost:      ${cost_tracker.total_cost_usd:.6f}")
    print(f"Avg Cost:        ${avg_cost:.6f} per problem")
    print(f"Total LLM Calls: {cost_tracker.total_calls}")
    print(f"Avg Calls:       {avg_calls:.1f} per problem")
    print(f"Total Tokens:    {cost_tracker.total_input_tokens + cost_tracker.total_output_tokens}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "failed_count": failed,
        "failure_rate": failure_rate,
        "avg_time": avg_time,
        "total_time": total_time,
        "cost": cost_tracker.to_dict(),
        "avg_cost_per_problem": avg_cost,
        "total_llm_calls": total_llm_calls,
        "avg_llm_calls": avg_calls,
        "model": model,
        "max_steps": max_steps,
        "complexity_level": complexity_level,
        "results": results,
    }


# =============================================================================
# BASELINE RUN
# =============================================================================

def run_l3_baseline_rulearena(
    num_problems: int = 30,
    complexity_level: int = 0,
    model: str = DEFAULT_MODEL,
    max_steps: int = 10,
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Run L3 baseline evaluation on RuleArena.
    
    Args:
        num_problems: Number of problems to evaluate
        complexity_level: 0=easy, 1=medium, 2=hard
        model: Model to use
        max_steps: Maximum agent steps per problem
        save_results: Whether to save results to JSON file
    
    Returns:
        Evaluation results dictionary
    """
    print("=" * 80)
    print("L3 RuleArena Baseline (ReAct Agent)")
    print("=" * 80)
    print(f"Problems: {num_problems}")
    print(f"Complexity: {complexity_level}")
    print(f"Model: {model}")
    print(f"Max Steps: {max_steps}")
    print()
    
    # Initialize loader
    try:
        loader = RuleArenaLoader("external/RuleArena")
        print("RuleArenaLoader initialized successfully")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    
    # Run evaluation
    results = evaluate_l3_on_rulearena(
        loader=loader,
        complexity_level=complexity_level,
        max_problems=num_problems,
        model=model,
        max_steps=max_steps,
        verbose=True,
    )
    
    # Add metadata
    results["metadata"] = {
        "experiment": "l3_rulearena_baseline",
        "timestamp": datetime.now().isoformat(),
        "rulearena_path": "external/RuleArena",
    }
    
    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(RESULTS_DIR, "l3_rulearena_baseline.json")
        
        # Make results JSON serializable
        results_to_save = {k: v for k, v in results.items() if k != "results"}
        results_to_save["sample_results"] = results["results"][:5]  # Save first 5 detailed
        results_to_save["num_detailed_results"] = len(results["results"])
        
        with open(output_path, "w") as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("L3 RESULTS TABLE")
    print("=" * 80)
    print(f"| Metric                | Value                    |")
    print(f"|----------------------|--------------------------|")
    print(f"| Accuracy             | {results['accuracy']*100:.1f}%                    |")
    print(f"| Failure Rate         | {results['failure_rate']*100:.1f}%                    |")
    print(f"| Avg Cost/Problem     | ${results['avg_cost_per_problem']:.6f}           |")
    print(f"| Avg LLM Calls        | {results['avg_llm_calls']:.1f}                      |")
    print(f"| Total Time           | {results['total_time']:.2f}s                   |")
    print(f"| Total Cost           | ${results['cost']['total_cost_usd']:.6f}           |")
    print(f"| Model                | {model}              |")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("L3 Baggage Allowance - ReAct Agent with RuleArena")
    print("=" * 80)
    print("\nThis is the autonomous agent baseline for comparison with L1:")
    print("  L1: 1 LLM call → Python calculates = reliable, cheap")
    print("  L3: N LLM calls → LLM reasons = flexible, expensive, unreliable")
    print()
    
    # First, test on 3 problems
    print("\n--- Test Run: 3 problems ---\n")
    test_results = run_l3_baseline_rulearena(
        num_problems=3,
        complexity_level=0,
        model=DEFAULT_MODEL,
        max_steps=10,
        save_results=False,
    )
    
    if "error" not in test_results:
        print(f"\nTest accuracy: {test_results['accuracy']*100:.1f}%")
        print(f"Test avg calls: {test_results['avg_llm_calls']:.1f}")
        
        # If test looks reasonable, run the full 30
        user_input = input("\nRun full 30 problems? (y/n): ").strip().lower()
        if user_input == 'y':
            print("\n" + "=" * 80)
            print("--- Full Baseline: 30 problems ---")
            print("=" * 80 + "\n")
            
            full_results = run_l3_baseline_rulearena(
                num_problems=30,
                complexity_level=0,
                model=DEFAULT_MODEL,
                max_steps=10,
                save_results=True,
            )
            
            print(f"\nFinal Accuracy: {full_results['accuracy']*100:.1f}%")
            print(f"Final Avg LLM Calls: {full_results['avg_llm_calls']:.1f}")
    else:
        print(f"\nTest failed: {test_results['error']}")
