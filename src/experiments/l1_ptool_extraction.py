"""
L1 PTP-based extraction - Pure and Transparent approaches.

L1 Pure: LLM extracts parameters, deterministic code calculates answer.
L1 Transparent: Same, but LLM sees the rules during extraction.

Both approaches use direct LLM calls (not @ptool) to allow dynamic model selection.
"""
import sys
import time
import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from external.ptool_framework.llm_backend import call_llm, calculate_cost


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

EXTRACTION_PROMPT_PURE = """Extract structured baggage parameters from airline query.

Extract these fields from the passenger scenario:

REQUIRED FIELDS:
{{
    "base_price": <integer ticket price in USD>,
    "customer_class": <one of: "Basic Economy", "Main Cabin", "Main Plus", "Premium Economy", "Business", "First">,
    "routine": <destination region>,
    "direction": <0 or 1>,
    "bag_list": [
        {{"id": 1, "name": "backpack", "size": [22, 13, 6], "weight": 10}},
        {{"id": 2, "name": "luggage box", "size": [44, 22, 20], "weight": 69}}
    ]
}}

ROUTING RULES (CRITICAL):
1. Identify Origin and Destination cities from the query
2. Determine flight direction:
   - Domestic (US -> US): routine="U.S.", direction=0
   - Outbound (US -> World): routine=<destination_region>, direction=0
   - Inbound (World -> US): routine=<origin_region>, direction=1

Valid regions: "U.S.", "Puerto Rico", "Canada", "Mexico", "Cuba", "Haiti", 
"Panama", "Colombia", "Ecuador", "Peru", "South America", "Israel", "Qatar", 
"Europe", "India", "China", "Japan", "South Korea", "Hong Kong", "Australia", 
"New Zealand"

US cities: Orlando, Philadelphia, Charlotte, Phoenix, Las Vegas, Atlanta, 
Boston, New York, Los Angeles, San Francisco, Miami

EXAMPLES:
Input: "Sarah is a Main Cabin passenger flying from Orlando to Tokyo..."
Output: {{"base_price": 500, "customer_class": "Main Cabin", "routine": "Japan", "direction": 0, "bag_list": [...]}}

Input: "John is a Business passenger flying from Paris to Atlanta..."
Output: {{"base_price": 800, "customer_class": "Business", "routine": "Europe", "direction": 1, "bag_list": [...]}}

QUERY:
{query}

Return ONLY a valid JSON object with these exact fields. No markdown, no explanation."""


EXTRACTION_PROMPT_WITH_RULES = """Extract structured baggage parameters using the complete airline rules.

AIRLINE BAGGAGE RULES:
{rules_text}

PASSENGER QUERY:
{query}

CRITICAL - EXACT VALUE REQUIREMENTS:

customer_class - Must be EXACTLY one of these strings (case-sensitive):
- "Basic Economy"
- "Main Cabin"
- "Main Plus"
- "Premium Economy"
- "Business"
- "First"

DO NOT add "Class" to the end. Use "Business" not "Business Class".

routine - Must be EXACTLY one of these region strings:
- "U.S." (for domestic US flights, use the period!)
- "Puerto Rico"
- "Canada"
- "Mexico"
- "Cuba"
- "Haiti"
- "Panama"
- "Colombia"
- "Ecuador"
- "Peru"
- "South America"
- "Israel"
- "Qatar"
- "Europe"
- "India"
- "China"
- "Japan"
- "South Korea"
- "Hong Kong"
- "Australia"
- "New Zealand"

DO NOT use "US", "Domestic", or "United States" - use "U.S." with the period.

direction - Must be exactly 0 or 1 (integer):
- 0 = Departing FROM US
- 1 = Arriving TO US

bag_list - Array of objects, each with:
- id: integer (1, 2, 3, ...)
- name: string ("backpack" or "luggage box")
- size: array of 3 integers [length, width, height] in inches
- weight: integer in pounds

EXAMPLE OUTPUT FORMAT:
{{
    "base_price": 180,
    "customer_class": "Main Cabin",
    "routine": "U.S.",
    "direction": 0,
    "bag_list": [
        {{"id": 1, "name": "backpack", "size": [22, 13, 6], "weight": 10}},
        {{"id": 2, "name": "luggage box", "size": [44, 22, 20], "weight": 69}}
    ]
}}

Return ONLY a valid JSON object matching this exact format."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON object from LLM response, handling markdown fences."""
    # Try to find JSON in markdown code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # Try to find raw JSON object
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group()
        else:
            raise ValueError(f"No JSON found in response: {response[:200]}...")
    
    return json.loads(json_str)


# Valid regions in RuleArena fee tables
VALID_REGIONS = {
    "U.S.", "Puerto Rico", "Canada", "Mexico", "Cuba", "Haiti", "Panama", 
    "Colombia", "Ecuador", "Peru", "South America", "Israel", "Qatar", 
    "Europe", "India", "China", "Japan", "South Korea", "Hong Kong", 
    "Australia", "New Zealand"
}

# Map common LLM mistakes to valid regions
REGION_FIXES = {
    "Asia": "China",
    "United States": "U.S.",
    "US": "U.S.",
    "USA": "U.S.",
    "Domestic": "U.S.",
    "UK": "Europe",
    "United Kingdom": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Spain": "Europe",
    "Italy": "Europe",
    "Tokyo": "Japan",
    "Beijing": "China",
    "Shanghai": "China",
    "Seoul": "South Korea",
    "Sydney": "Australia",
    "Melbourne": "Australia",
    "Auckland": "New Zealand",
    "Mumbai": "India",
    "Delhi": "India",
    "Tel Aviv": "Israel",
    "Doha": "Qatar",
}


def normalize_region(routine: str) -> str:
    """Normalize region to valid RuleArena value."""
    if routine in VALID_REGIONS:
        return routine
    return REGION_FIXES.get(routine, "U.S.")


def compute_baggage_cost(
    params: Dict[str, Any],
    loader,
) -> Tuple[int, bool, Optional[str]]:
    """
    Compute baggage cost using extracted parameters.
    
    Returns:
        (total_cost, success, error_message)
    """
    try:
        # Normalize region to handle LLM mistakes like "Asia"
        routine = normalize_region(params.get('routine', 'U.S.'))
        
        result_tuple = loader._compute_answer_fn(
            base_price=params['base_price'],
            direction=params['direction'],
            routine=routine,
            customer_class=params['customer_class'],
            bag_list=params['bag_list'],
            check_base_tables=loader._fee_tables,
        )
        total_cost = int(result_tuple[0]) if isinstance(result_tuple, tuple) else int(result_tuple)
        return total_cost, True, None
    except Exception as e:
        return 0, False, str(e)


# =============================================================================
# L1 (PURE) PIPELINE
# =============================================================================

def baggage_allowance_l1_ptool(
    query: str,
    loader,
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    verbose: bool = False,
) -> Tuple[Dict[str, Any], int, int, float]:
    """
    L1 Pure: Extract parameters without showing rules, then calculate.
    
    Args:
        query: Natural language problem
        loader: AirlineLoader for calculation
        model: Model to use for extraction
        verbose: Print progress
        
    Returns:
        (result_dict, input_tokens, output_tokens, cost)
    """
    if verbose:
        print(f"\n[L1 Pure] Processing: {query[:80]}...")
        print(f"  Model: {model}")
    
    # Format prompt
    prompt = EXTRACTION_PROMPT_PURE.format(query=query)
    
    # Call LLM
    start = time.time()
    try:
        response, input_tokens, output_tokens = call_llm(
            prompt=prompt,
            model=model,
        )
    except Exception as e:
        if verbose:
            print(f"  LLM call failed: {e}")
        return {
            "answer": 0,
            "params": {},
            "success": False,
            "error": f"LLM call failed: {e}",
        }, 0, 0, 0.0
    
    extraction_time = time.time() - start
    cost = calculate_cost(model, input_tokens, output_tokens)
    
    if verbose:
        print(f"  Extracted in {extraction_time:.2f}s:")
        print(f"     Tokens: {input_tokens} in, {output_tokens} out")
        print(f"     Cost: ${cost:.6f}")
    
    # Parse response
    try:
        params = extract_json_from_response(response)
        # Unwrap if needed
        if isinstance(params, dict) and "result" in params:
            params = params["result"]
    except Exception as e:
        if verbose:
            print(f"  JSON parse failed: {e}")
            print(f"  Raw response: {response[:300]}...")
        return {
            "answer": 0,
            "params": {},
            "success": False,
            "error": f"JSON parse failed: {e}",
            "raw_response": response[:500],
        }, input_tokens, output_tokens, cost
    
    if verbose:
        print(f"     Class: {params.get('customer_class')}, Route: {params.get('routine')}")
    
    # Calculate answer
    calc_start = time.time()
    total_cost, calc_success, calc_error = compute_baggage_cost(params, loader)
    calc_time = time.time() - calc_start
    
    if calc_success:
        if verbose:
            print(f"  Calculated ${total_cost} in {calc_time:.2f}s")
        return {
            "answer": total_cost,
            "params": params,
            "success": True,
        }, input_tokens, output_tokens, cost
    else:
        if verbose:
            print(f"  Calculation failed: {calc_error}")
            print(f"  Extracted params: {params}")
        return {
            "answer": 0,
            "params": params,
            "success": False,
            "error": calc_error,
        }, input_tokens, output_tokens, cost


# =============================================================================
# L1 (TRANSPARENT) PIPELINE
# =============================================================================

def baggage_allowance_l1_transparent(
    query: str,
    loader,
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    verbose: bool = False,
) -> Tuple[Dict[str, Any], int, int, float]:
    """
    L1 Transparent: Show rules to LLM during extraction, then calculate.
    
    Args:
        query: Natural language problem
        loader: AirlineLoader for rules and calculation
        model: Model to use for extraction
        verbose: Print progress
        
    Returns:
        (result_dict, input_tokens, output_tokens, cost)
    """
    if verbose:
        print(f"\n[L1 Transparent] Processing: {query[:80]}...")
        print(f"  Model: {model}")
    
    # Load rules
    rules_text = loader.load_rules(textual=True)
    
    # Format prompt with rules
    prompt = EXTRACTION_PROMPT_WITH_RULES.format(
        rules_text=rules_text,
        query=query
    )
    
    # Call LLM
    start = time.time()
    try:
        response, input_tokens, output_tokens = call_llm(
            prompt=prompt,
            model=model,
        )
    except Exception as e:
        if verbose:
            print(f"  LLM call failed: {e}")
        return {
            "answer": 0,
            "params": {},
            "success": False,
            "error": f"LLM call failed: {e}",
        }, 0, 0, 0.0
    
    extraction_time = time.time() - start
    cost = calculate_cost(model, input_tokens, output_tokens)
    
    if verbose:
        print(f"  Extracted in {extraction_time:.2f}s:")
        print(f"     Tokens: {input_tokens} in, {output_tokens} out (rules added ~2K tokens)")
        print(f"     Cost: ${cost:.6f}")
    
    # Parse response
    try:
        params = extract_json_from_response(response)
        # Unwrap if needed
        if isinstance(params, dict) and "result" in params:
            params = params["result"]
    except Exception as e:
        if verbose:
            print(f"  JSON parse failed: {e}")
            print(f"  Raw response: {response[:300]}...")
        return {
            "answer": 0,
            "params": {},
            "success": False,
            "error": f"JSON parse failed: {e}",
            "raw_response": response[:500],
        }, input_tokens, output_tokens, cost
    
    if verbose:
        print(f"     Class: {params.get('customer_class')}, Route: {params.get('routine')}")
    
    # Calculate answer
    calc_start = time.time()
    total_cost, calc_success, calc_error = compute_baggage_cost(params, loader)
    calc_time = time.time() - calc_start
    
    if calc_success:
        if verbose:
            print(f"  Calculated ${total_cost} in {calc_time:.2f}s")
        return {
            "answer": total_cost,
            "params": params,
            "success": True,
        }, input_tokens, output_tokens, cost
    else:
        if verbose:
            print(f"  Calculation failed: {calc_error}")
            print(f"  Extracted params: {params}")
        return {
            "answer": 0,
            "params": params,
            "success": False,
            "error": calc_error,
        }, input_tokens, output_tokens, cost
