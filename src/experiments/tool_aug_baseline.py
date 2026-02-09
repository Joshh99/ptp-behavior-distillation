"""
Tool-Augmented baseline - RuleArena's approach where LLM generates Python code.

Reference: RuleArena paper mentions tool-augmented approach achieved 19-44% accuracy.
The LLM generates code using the reference implementation as a tool.

WARNING: This executes LLM-generated code. We sandbox it carefully.
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# from src.experiments.config import call_llm
from external.ptool_framework.llm_backend import call_llm, calculate_cost
import re


TOOL_AUG_PROMPT = """Calculate airline baggage fees using the provided function.

RULES:
{rules}

QUERY:
{query}

You have this function available (already imported):
compute_baggage_cost(base_price, customer_class, routine, direction, bag_list, check_base_tables)

The variable 'check_base_tables' is already available.

INSTRUCTIONS:
Extract parameters from the query and call the function directly.
- base_price: ticket price (integer)
- customer_class: EXACTLY one of: "Basic Economy", "Main Cabin", "Main Plus", "Premium Economy", "Business", "First"
- routine: EXACTLY one of these regions (use specific region, NOT "Asia"):
  "U.S.", "Puerto Rico", "Canada", "Mexico", "Cuba", "Haiti", "Panama", "Colombia", 
  "Ecuador", "Peru", "South America", "Israel", "Qatar", "Europe", "India", "China", 
  "Japan", "South Korea", "Hong Kong", "Australia", "New Zealand"
  NOTE: For Hong Kong use "Hong Kong", for mainland China use "China", for Tokyo use "Japan", etc.
- direction: 0 (departing FROM US) or 1 (arriving TO US from international)
- bag_list: list of dicts, each bag needs: {{"id": 0, "name": "backpack", "size": [L, W, H], "weight": W}}
  IMPORTANT: First bag (carry-on) should have id=0, second bag id=1, etc.

Example:
base_price = 180
customer_class = "Main Cabin"
routine = "U.S."
direction = 0
bag_list = [
    {{"id": 0, "name": "backpack", "size": [22, 13, 6], "weight": 10}},
    {{"id": 1, "name": "luggage box", "size": [44, 22, 20], "weight": 69}}
]
result = compute_baggage_cost(base_price, customer_class, routine, direction, bag_list, check_base_tables)
print(f"Total: ${{result[0]}}")

Write Python code following this pattern. Do NOT define functions - write code directly.
"""


def run_tool_augmented(
    query: str,
    loader,
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    verbose: bool = False,
) -> tuple[dict, int, int, float]:
    """Tool-augmented approach: LLM generates code, we execute it safely."""
    
    if verbose:
        print(f"\n[Tool-Aug] Processing: {query[:80]}...")
    
    rules = loader.load_rules(textual=True)
    
    prompt = TOOL_AUG_PROMPT.format(
        rules=rules,
        query=query
    )
    
    # Generate code
    import time
    start = time.time()
    
    response, input_tokens, output_tokens = call_llm(
        prompt=prompt,
        model=model,
    )
    
    execution_time = time.time() - start
    
    # from ptool_framework.llm_backend import calculate_cost
    cost = calculate_cost(model, input_tokens, output_tokens)
    
    if verbose:
        print(f"  Generated code in {execution_time:.2f}s:")
        print(f"     Tokens: {input_tokens} in, {output_tokens} out")
        print(f"     Cost: ${cost:.6f}")
    
    # Extract code (handle markdown fences)
    code = response
    match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
    if match:
        code = match.group(1)
    else:
        # Try without 'python' keyword
        match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
        if match:
            code = match.group(1)
    
    # DEBUG: Show generated code
    if verbose:
        print(f"  Generated code ({len(code.split(chr(10)))} lines):")
        for i, line in enumerate(code.split('\n'), 1):  # Show ALL lines
            print(f"    {i}: {line}")
        if len(code.split('\n')) > 10:
            print(f"    ... ({len(code.split('\n'))-10} more lines)")
    
    # Execute code in sandboxed environment
    answer = 0
    success = False
    error_msg = None
    
    try:
        # Valid regions in RuleArena fee tables
        VALID_REGIONS = {
            "U.S.", "Puerto Rico", "Canada", "Mexico", "Cuba", "Haiti", "Panama", 
            "Colombia", "Ecuador", "Peru", "South America", "Israel", "Qatar", 
            "Europe", "India", "China", "Japan", "South Korea", "Hong Kong", 
            "Australia", "New Zealand"
        }
        
        # Map common LLM mistakes to valid regions
        REGION_FIXES = {
            "Asia": "China",  # Default Asia to China
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
        
        # Wrapper function that fixes bag_list indexing and normalizes regions
        def compute_baggage_cost_wrapper(base_price, customer_class, routine, direction, bag_list, check_base_tables):
            """Wrapper that re-indexes bags and normalizes invalid regions."""
            # Fix region if invalid
            fixed_routine = routine
            if routine not in VALID_REGIONS:
                fixed_routine = REGION_FIXES.get(routine, "U.S.")  # Default to U.S. if unknown
            
            # Re-index bags to start at 0 (carry-on is bag 0)
            fixed_bag_list = []
            for i, bag in enumerate(bag_list):
                fixed_bag = bag.copy()
                fixed_bag['id'] = i  # Force 0-indexed
                fixed_bag_list.append(fixed_bag)
            
            return loader._compute_answer_fn(
                base_price=base_price,
                customer_class=customer_class,
                routine=fixed_routine,
                direction=direction,
                bag_list=fixed_bag_list,
                check_base_tables=check_base_tables,
            )
        
        # Provide the wrapped implementation
        namespace = {
            'compute_baggage_cost': compute_baggage_cost_wrapper,
            'check_base_tables': loader._fee_tables,
        }
        
        # Capture stdout
        import io
        from contextlib import redirect_stdout
        
        output_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer):
            exec(code, namespace)
        
        output = output_buffer.getvalue()
        
        if verbose:
            print(f"  Code output: {output[:200] if output else '(no output)'}")
        
        # Extract answer from output
        patterns = [
            r'Total:\s*\$?(\d+)',
            r'total_cost\s*=\s*(\d+)',
            r'\$(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                answer = int(match.group(1))
                success = True
                break
        
        if not success and verbose:
            print(f"  ✗ No answer found in output")
        elif verbose:
            print(f"  Executed: ${answer}")
        
    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"  ✗ Execution error: {e}")
            import traceback
            print(f"  Traceback:")
            traceback.print_exc()
    
    return {
        "answer": answer,
        "code": code,
        "success": success,
        "error": error_msg,
    }, input_tokens, output_tokens, cost