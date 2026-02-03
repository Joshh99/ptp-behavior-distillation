"""
Test extraction ptool standalone before integrating into l1_baggage_v2.py

This tests the extract_rulearena_params function in isolation.
"""

import sys
import os
import json
import re
from typing import Dict, Any, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import call_llm, DEFAULT_MODEL


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
    Extract structured parameters from natural language query.
    
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
        print(f"Error during extraction: {e}")
        print(f"Response was: {response[:200]}...")
        
        # Return default structure on failure
        return {
            "base_price": 0,
            "customer_class": "Main Cabin",
            "routine": "U.S.",
            "direction": 0,
            "bag_list": [],
        }, input_tokens, 0


# Test cases
TEST_QUERIES = [
    # Domestic
    """I'm buying a ticket from Orlando to Charlotte. The ticket is $180, and I'm flying Main Cabin.
I have 2 items to bring:
1. backpack (22 x 13 x 6 in, 10 lbs)
2. luggage box (44 x 22 x 20 in, 69 lbs)
How much will I pay in total?""",
    
    # International Outbound
    """Flying from New York to Tokyo in Business class. Ticket is $2500.
Bags:
1. Small backpack (18 x 12 x 8 in, 8 lbs)
2. Large suitcase (62 x 28 x 20 in, 50 lbs)
What's the total cost?""",
    
    # International Inbound
    """Returning to Los Angeles from Paris. Economy ticket ($800).
Carrying:
1. Carry-on backpack (20 x 14 x 9 in, 12 lbs)
2. Checked luggage (58 x 24 x 18 in, 48 lbs)
Total cost?""",
]


def test_extraction():
    """Test extraction on sample queries."""
    print("=" * 80)
    print("EXTRACTION PTOOL TEST")
    print("=" * 80)
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n--- Test {i} ---")
        print(f"Query: {query[:80]}...")
        
        try:
            params, input_tok, output_tok = extract_rulearena_params(query)
            
            print(f"\nExtracted Parameters:")
            print(f"  Base Price: ${params['base_price']}")
            print(f"  Class: {params['customer_class']}")
            print(f"  Routine: {params['routine']}")
            print(f"  Direction: {params['direction']}")
            print(f"  Bags: {len(params['bag_list'])}")
            
            for j, bag in enumerate(params['bag_list'], 1):
                print(f"    Bag {j}: {bag['name']} - {bag['size']} in, {bag['weight']} lbs")
            
            print(f"\nTokens: {input_tok} in, {output_tok} out")
            print(f"JSON: {json.dumps(params, indent=2)}")
            
        except Exception as e:
            print(f"ERROR: {e}")
        
        print()
    
    print("=" * 80)
    print("Test complete!")


if __name__ == "__main__":
    test_extraction()