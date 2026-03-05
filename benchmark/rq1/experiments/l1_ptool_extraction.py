import time
import json
import re
from typing import Dict, Any, Tuple, Optional

from external.ptool_framework.llm_backend import call_llm, calculate_cost


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

EXAMPLE:
Input: "Sarah is a Main Cabin passenger flying from Orlando to Tokyo..."
Output: {{"base_price": 500, "customer_class": "Main Cabin", "routine": "Japan", "direction": 0, "bag_list": [...]}}

QUERY:
{query}

Return ONLY a valid JSON object with these exact fields. No markdown, no explanation."""


EXTRACTION_PROMPT_WITH_RULES = """Extract structured baggage parameters using the complete airline rules.

EXAMPLE:
Input: "Sarah is a Main Cabin passenger flying from Orlando to Tokyo..."
Output: {{"base_price": 500, "customer_class": "Main Cabin", "routine": "Japan", "direction": 0, "bag_list": [...]}}

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


VALID_REGIONS = {
    "U.S.", "Puerto Rico", "Canada", "Mexico", "Cuba", "Haiti", "Panama",
    "Colombia", "Ecuador", "Peru", "South America", "Israel", "Qatar",
    "Europe", "India", "China", "Japan", "South Korea", "Hong Kong",
    "Australia", "New Zealand"
}

REGION_FIXES = {
    "asia": "China",
    "north america": "U.S.",
    "central america": "Mexico",
    "united states": "U.S.",
    "us": "U.S.",
    "usa": "U.S.",
    "domestic": "U.S.",
    "america": "U.S.",
    "uk": "Europe",
    "united kingdom": "Europe",
    "britain": "Europe",
    "great britain": "Europe",
    "england": "Europe",
    "france": "Europe",
    "germany": "Europe",
    "spain": "Europe",
    "italy": "Europe",
    "netherlands": "Europe",
    "belgium": "Europe",
    "switzerland": "Europe",
    "portugal": "Europe",
    "greece": "Europe",
    "ireland": "Europe",
    "scotland": "Europe",
    "austria": "Europe",
    "poland": "Europe",
    "denmark": "Europe",
    "norway": "Europe",
    "sweden": "Europe",
    "finland": "Europe",
    "tokyo": "Japan",
    "osaka": "Japan",
    "beijing": "China",
    "shanghai": "China",
    "guangzhou": "China",
    "shenzhen": "China",
    "wuhan": "China",
    "seoul": "South Korea",
    "busan": "South Korea",
    "sydney": "Australia",
    "melbourne": "Australia",
    "brisbane": "Australia",
    "perth": "Australia",
    "auckland": "New Zealand",
    "wellington": "New Zealand",
    "mumbai": "India",
    "delhi": "India",
    "bangalore": "India",
    "chennai": "India",
    "tel aviv": "Israel",
    "jerusalem": "Israel",
    "doha": "Qatar",
    "bogota": "Colombia",
    "lima": "Peru",
    "santiago": "South America",
    "buenos aires": "South America",
    "sao paulo": "South America",
    "rio de janeiro": "South America",
    "havana": "Cuba",
    "toronto": "Canada",
    "vancouver": "Canada",
    "montreal": "Canada",
    "mexico city": "Mexico",
    "cancun": "Mexico",
    "london": "Europe",
    "paris": "Europe",
    "rome": "Europe",
    "madrid": "Europe",
    "berlin": "Europe",
    "amsterdam": "Europe",
    "barcelona": "Europe",
    "frankfurt": "Europe",
    "munich": "Europe",
    "milan": "Europe",
    "zurich": "Europe",
    "vienna": "Europe",
}


def extract_json_from_response(response: str) -> Dict[str, Any]:
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group()
        else:
            raise ValueError(f"No JSON found in response: {response[:200]}...")

    return json.loads(json_str)


def normalize_region(routine: str) -> str:
    if not routine:
        return "U.S."

    if routine in VALID_REGIONS:
        return routine

    routine_lower = routine.lower().strip()
    if routine_lower in REGION_FIXES:
        return REGION_FIXES[routine_lower]

    return "U.S."


def compute_baggage_cost(
    params: Dict[str, Any],
    loader,
) -> Tuple[int, bool, Optional[str]]:
    try:
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


def baggage_allowance_l1_ptool(
    query: str,
    loader,
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    verbose: bool = False,
) -> Tuple[Dict[str, Any], int, int, float]:
    if verbose:
        print(f"\n[L1 Pure] Processing: {query[:80]}...")

    prompt = EXTRACTION_PROMPT_PURE.format(query=query)

    start = time.time()
    try:
        response, input_tokens, output_tokens = call_llm(prompt=prompt, model=model)
    except Exception as e:
        return {"answer": 0, "params": {}, "success": False, "error": str(e)}, 0, 0, 0.0

    cost = calculate_cost(model, input_tokens, output_tokens)

    try:
        params = extract_json_from_response(response)
        if isinstance(params, dict) and "result" in params:
            params = params["result"]
    except Exception as e:
        return {
            "answer": 0, "params": {}, "success": False,
            "error": f"JSON parse failed: {e}", "raw_response": response[:500],
        }, input_tokens, output_tokens, cost

    total_cost, calc_success, calc_error = compute_baggage_cost(params, loader)

    if calc_success:
        return {"answer": total_cost, "params": params, "success": True}, input_tokens, output_tokens, cost
    else:
        return {
            "answer": 0, "params": params, "success": False, "error": calc_error,
        }, input_tokens, output_tokens, cost


def baggage_allowance_l1_transparent(
    query: str,
    loader,
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    verbose: bool = False,
) -> Tuple[Dict[str, Any], int, int, float]:
    if verbose:
        print(f"\n[L1 Transparent] Processing: {query[:80]}...")

    rules_text = loader.load_rules(textual=True)

    prompt = EXTRACTION_PROMPT_WITH_RULES.format(rules_text=rules_text, query=query)

    start = time.time()
    try:
        response, input_tokens, output_tokens = call_llm(prompt=prompt, model=model)
    except Exception as e:
        return {"answer": 0, "params": {}, "success": False, "error": str(e)}, 0, 0, 0.0

    cost = calculate_cost(model, input_tokens, output_tokens)

    try:
        params = extract_json_from_response(response)
        if isinstance(params, dict) and "result" in params:
            params = params["result"]
    except Exception as e:
        return {
            "answer": 0, "params": {}, "success": False,
            "error": f"JSON parse failed: {e}", "raw_response": response[:500],
        }, input_tokens, output_tokens, cost

    total_cost, calc_success, calc_error = compute_baggage_cost(params, loader)

    if calc_success:
        return {"answer": total_cost, "params": params, "success": True}, input_tokens, output_tokens, cost
    else:
        return {
            "answer": 0, "params": params, "success": False, "error": calc_error,
        }, input_tokens, output_tokens, cost
