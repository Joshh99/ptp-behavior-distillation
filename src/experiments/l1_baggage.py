"""
L1 Baggage Allowance - PTool Pattern (The "SecretAgent" Sweet Spot)

This is the target reliability level for behavior distillation.
LLM extracts parameters → Python calculates fees.

Target reliability: 95%+
Target cost: Minimal (1 LLM call per query)

Architecture:
    1. LLM extracts: airline, class, route, weights, membership
    2. Python calculates: fee based on deterministic rules
    3. LLM interprets: natural language explanation (optional)

This demonstrates Prof. Cohen's key insight:
    "Python does maximum heavy lifting before handing to LLMs"
"""

import sys
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.rule_arena_loader import (
    RuleArenaDataset,
    RuleArenaInstance,
    BaggageRule,
    TaskCategory,
    load_baggage_rules,
)
from experiments.config import (
    get_experiment_config,
    get_model_config,
    ExperimentLevel,
    RESULTS_DIR,
    call_llm,
    DEFAULT_MODEL,
)

import json
import re


# =============================================================================
# PTOOLS - LLM-powered parameter extraction (Together.ai)
# =============================================================================

EXTRACTION_PROMPT = """You are an expert at extracting structured information from airline baggage queries.

Given a passenger query and context, extract the following parameters as a JSON object:
- airline: string (e.g., "united", "delta", "american") 
- travel_class: string ("economy", "business", or "first")
- route_type: string ("domestic" or "international")
- num_bags: integer
- bag_weights_kg: list of floats
- membership_status: string or null
- special_items: list of strings (e.g., ["golf_clubs", "skis"])

Query: {query}

Passenger Context:
{passenger_info}

Respond with ONLY a valid JSON object, no other text. Example:
{{"airline": "united", "travel_class": "economy", "route_type": "domestic", "num_bags": 1, "bag_weights_kg": [20.0], "membership_status": null, "special_items": []}}
"""


def extract_baggage_params(query: str, passenger_info: str, model: str = DEFAULT_MODEL) -> dict:
    """
    Extract baggage-related parameters from a passenger query using Together.ai.
    
    This is the L1 "PTool" pattern: LLM extracts, Python calculates.
    """
    prompt = EXTRACTION_PROMPT.format(query=query, passenger_info=passenger_info)
    
    try:
        response = call_llm(prompt, model=model, max_tokens=256)
        
        # Parse JSON from response
        # Try to find JSON in the response (handle markdown code blocks)
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            params = json.loads(json_match.group())
        else:
            params = json.loads(response.strip())
        
        # Ensure all required fields exist with defaults
        defaults = {
            "airline": "united",
            "travel_class": "economy",
            "route_type": "domestic", 
            "num_bags": 1,
            "bag_weights_kg": [20.0],
            "membership_status": None,
            "special_items": [],
        }
        
        for key, default_value in defaults.items():
            if key not in params or params[key] is None:
                params[key] = default_value
        
        return params
        
    except Exception as e:
        print(f"  Warning: LLM extraction failed ({e}), using defaults")
        return {
            "airline": "united",
            "travel_class": "economy", 
            "route_type": "domestic",
            "num_bags": 1,
            "bag_weights_kg": [20.0],
            "membership_status": None,
            "special_items": [],
        }


# =============================================================================
# PYTHON RULE ENGINE - Deterministic fee calculation
# =============================================================================

@dataclass
class FeeCalculation:
    """Result of baggage fee calculation."""
    total_fee: float
    breakdown: Dict[str, float]
    explanation: str
    applicable_rules: List[str]
    warnings: List[str]


class BaggageRuleEngine:
    """
    Pure Python rule engine for baggage fee calculation.
    
    This is the "distilled" knowledge - deterministic, fast, testable.
    No LLM calls needed once parameters are extracted.
    """
    
    def __init__(self, rules: List[BaggageRule] = None):
        self.rules = rules or load_baggage_rules()
        self._build_rule_index()
    
    def _build_rule_index(self):
        """Index rules for fast lookup."""
        self.rule_index = {}
        for rule in self.rules:
            key = (
                rule.conditions.get("airline", "").lower(),
                rule.conditions.get("class", "").lower(),
                rule.conditions.get("route", "").lower(),
            )
            if key not in self.rule_index:
                self.rule_index[key] = []
            self.rule_index[key].append(rule)
    
    def find_applicable_rule(
        self, 
        airline: str, 
        travel_class: str, 
        route_type: str
    ) -> Optional[BaggageRule]:
        """Find the most specific rule for given parameters."""
        # Try exact match first
        key = (airline.lower(), travel_class.lower(), route_type.lower())
        if key in self.rule_index:
            return self.rule_index[key][0]
        
        # Try partial matches
        for rule in self.rules:
            conditions = rule.conditions
            if (conditions.get("airline", "").lower() == airline.lower() and
                conditions.get("class", "").lower() == travel_class.lower() and
                conditions.get("route", "").lower() == route_type.lower()):
                return rule
        
        return None
    
    def calculate_fee(
        self,
        airline: str,
        travel_class: str,
        route_type: str,
        num_bags: int,
        bag_weights_kg: List[float],
        membership_status: Optional[str] = None,
        special_items: List[str] = None,
    ) -> FeeCalculation:
        """
        Calculate baggage fees using deterministic rules.
        
        This is the core L1 logic - pure Python, no LLM.
        """
        total_fee = 0.0
        breakdown = {}
        warnings = []
        applicable_rules = []
        
        # Find applicable rule
        rule = self.find_applicable_rule(airline, travel_class, route_type)
        
        if rule is None:
            # Default to generic economy domestic if no specific rule
            warnings.append(f"No specific rule found for {airline}/{travel_class}/{route_type}. Using defaults.")
            base_fees = {"first_bag": 35, "second_bag": 45}
            max_weight = 23
            free_bags = 0
        else:
            applicable_rules.append(rule.rule_id)
            base_fees = rule.fees
            max_weight = rule.thresholds.get("max_weight_kg", 23)
            free_bags = rule.thresholds.get("max_pieces", 0)
            
            # Check for membership exceptions
            if membership_status:
                for exception in rule.exceptions:
                    if any(status in exception.lower() for status in [membership_status.lower(), "premier", "medallion", "gold"]):
                        free_bags += 1
                        breakdown["membership_benefit"] = f"+1 free bag ({membership_status})"
        
        # Calculate fees for each bag
        for i, weight in enumerate(bag_weights_kg):
            bag_num = i + 1
            bag_fee = 0.0
            
            # Check if this bag is free
            if bag_num <= free_bags:
                breakdown[f"bag_{bag_num}"] = 0.0
            else:
                # Get base fee for this bag position
                if bag_num == 1:
                    bag_fee = base_fees.get("first_bag", 35)
                elif bag_num == 2:
                    bag_fee = base_fees.get("second_bag", 45)
                else:
                    bag_fee = base_fees.get("third_bag", 150)
                
                breakdown[f"bag_{bag_num}_base"] = bag_fee
            
            # Check overweight
            if weight > max_weight:
                if weight <= 32:
                    overweight_fee = base_fees.get("overweight_23_32kg", 100)
                    breakdown[f"bag_{bag_num}_overweight"] = overweight_fee
                    bag_fee += overweight_fee
                elif weight <= 45:
                    overweight_fee = base_fees.get("overweight_32_45kg", 200)
                    breakdown[f"bag_{bag_num}_overweight"] = overweight_fee
                    bag_fee += overweight_fee
                else:
                    warnings.append(f"Bag {bag_num} ({weight}kg) exceeds 45kg limit - may not be accepted")
                    overweight_fee = 400  # Penalty or rejection
                    breakdown[f"bag_{bag_num}_overweight"] = overweight_fee
                    bag_fee += overweight_fee
            
            total_fee += bag_fee
        
        # Handle special items
        special_items = special_items or []
        for item in special_items:
            if item.lower() in ["golf_clubs", "skis", "ski_equipment"]:
                # Sports equipment usually counts as one bag
                if f"bag_1_base" not in breakdown:
                    breakdown[f"{item}_fee"] = 35
                    total_fee += 35
        
        # Generate explanation
        explanation = self._generate_explanation(
            airline, travel_class, route_type, 
            num_bags, bag_weights_kg, 
            total_fee, breakdown, warnings
        )
        
        return FeeCalculation(
            total_fee=total_fee,
            breakdown=breakdown,
            explanation=explanation,
            applicable_rules=applicable_rules,
            warnings=warnings,
        )
    
    def _generate_explanation(
        self,
        airline: str,
        travel_class: str,
        route_type: str,
        num_bags: int,
        bag_weights_kg: List[float],
        total_fee: float,
        breakdown: Dict[str, float],
        warnings: List[str],
    ) -> str:
        """Generate human-readable explanation of fee calculation."""
        lines = [f"{airline.title()} {travel_class.title()} {route_type.title()} - {num_bags} bag(s)"]
        
        for key, value in breakdown.items():
            if isinstance(value, (int, float)):
                lines.append(f"  {key.replace('_', ' ').title()}: ${value:.0f}")
            else:
                lines.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        lines.append(f"  TOTAL: ${total_fee:.0f}")
        
        if warnings:
            lines.append("  Warnings:")
            for warning in warnings:
                lines.append(f"    - {warning}")
        
        return "\n".join(lines)


# =============================================================================
# L1 WORKFLOW - The main pipeline
# =============================================================================

def baggage_allowance_l1(
    query: str,
    passenger_context: Dict[str, Any],
    rule_engine: BaggageRuleEngine = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    L1 PTool pattern for baggage allowance queries.
    
    Architecture:
        1. Extract parameters (1 LLM call via PTool)
        2. Calculate fees (Pure Python - deterministic)
        3. Return structured result
    
    Args:
        query: Natural language query from passenger
        passenger_context: Additional context (from dataset)
        rule_engine: Optional pre-initialized rule engine
        verbose: Print progress
    
    Returns:
        Dict with answer, explanation, extracted_params, fee_breakdown
    """
    if rule_engine is None:
        rule_engine = BaggageRuleEngine()
    
    if verbose:
        print(f"\n[L1 PTool] Processing: {query[:60]}...")
    
    # Step 1: Extract parameters (LLM call)
    passenger_info = "\n".join([f"{k}: {v}" for k, v in passenger_context.items()])
    
    start_time = time.time()
    try:
        params = extract_baggage_params(query, passenger_info)
        extraction_time = time.time() - start_time
        if verbose:
            print(f"  1. Extracted params in {extraction_time:.2f}s: {params}")
    except Exception as e:
        if verbose:
            print(f"  1. Extraction failed: {e}")
        # Fallback to context-based extraction
        params = {
            "airline": passenger_context.get("airline", "united"),
            "travel_class": passenger_context.get("class", "economy"),
            "route_type": passenger_context.get("route", "domestic"),
            "num_bags": passenger_context.get("num_bags", 1),
            "bag_weights_kg": passenger_context.get("bag_weights_kg", [20.0]),
            "membership_status": passenger_context.get("membership_status"),
            "special_items": passenger_context.get("items", []),
        }
        extraction_time = 0
    
    # Step 2: Calculate fees (Pure Python - no LLM)
    calc_start = time.time()
    result = rule_engine.calculate_fee(
        airline=params.get("airline", "united"),
        travel_class=params.get("travel_class", "economy"),
        route_type=params.get("route_type", "domestic"),
        num_bags=params.get("num_bags", 1),
        bag_weights_kg=params.get("bag_weights_kg", [20.0]),
        membership_status=params.get("membership_status"),
        special_items=params.get("special_items", []),
    )
    calc_time = time.time() - calc_start
    
    if verbose:
        print(f"  2. Calculated fee in {calc_time:.4f}s: ${result.total_fee:.0f}")
        print(f"  3. Explanation: {result.explanation}")
    
    return {
        "answer": result.total_fee,
        "explanation": result.explanation,
        "extracted_params": params,
        "fee_breakdown": result.breakdown,
        "applicable_rules": result.applicable_rules,
        "warnings": result.warnings,
        "metrics": {
            "extraction_time": extraction_time,
            "calculation_time": calc_time,
            "total_time": extraction_time + calc_time,
            "llm_calls": 1,
        }
    }


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_l1_on_dataset(
    dataset: RuleArenaDataset = None,
    num_samples: int = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate L1 PTool approach on RuleArena dataset.
    
    Returns metrics: accuracy, avg_time, total_cost
    """
    if dataset is None:
        dataset = RuleArenaDataset()
    
    instances = dataset.load("test")
    if num_samples:
        instances = instances[:num_samples]
    
    rule_engine = BaggageRuleEngine()
    
    results = []
    correct = 0
    total_time = 0
    
    print("=" * 80)
    print(f"L1 PTool Evaluation - {len(instances)} instances")
    print("=" * 80)
    
    for i, instance in enumerate(instances):
        print(f"\n[{i+1}/{len(instances)}] {instance.query[:50]}...")
        
        result = baggage_allowance_l1(
            query=instance.query,
            passenger_context=instance.passenger_context,
            rule_engine=rule_engine,
            verbose=verbose,
        )
        
        # Check correctness
        predicted = result["answer"]
        expected = instance.ground_truth_answer
        
        # Handle different answer types
        if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
            is_correct = abs(predicted - expected) < 0.01
        else:
            is_correct = str(predicted).lower() == str(expected).lower()
        
        if is_correct:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"  Result: {predicted} | Expected: {expected} | {status}")
        
        total_time += result["metrics"]["total_time"]
        
        results.append({
            "instance_id": instance.instance_id,
            "query": instance.query,
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
            "time": result["metrics"]["total_time"],
        })
    
    accuracy = correct / len(instances) if instances else 0
    avg_time = total_time / len(instances) if instances else 0
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Accuracy: {accuracy*100:.1f}% ({correct}/{len(instances)})")
    print(f"Avg Time: {avg_time:.2f}s per query")
    print(f"Total Time: {total_time:.2f}s")
    print(f"LLM Calls: {len(instances)} (1 per query)")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(instances),
        "avg_time": avg_time,
        "total_time": total_time,
        "results": results,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("L1 Baggage Allowance - PTool Pattern")
    print("=" * 80)
    print("\nThis demonstrates the 'SecretAgent sweet spot':")
    print("  LLM extracts parameters → Python calculates deterministically")
    print()
    
    # Load dataset
    dataset = RuleArenaDataset()
    
    # Run on debug subset
    print("\n--- Running on 3 sample instances ---\n")
    results = evaluate_l1_on_dataset(
        dataset=dataset,
        num_samples=3,
        verbose=True,
    )
    
    print(f"\nFinal Accuracy: {results['accuracy']*100:.1f}%")
