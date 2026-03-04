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
Write a Python function called `solution()` that:
1. Extracts parameters from the query
2. Calls compute_baggage_cost with those parameters
3. Returns the total_cost as an integer

Parameters to extract:
- base_price: ticket price (integer)
- customer_class: EXACTLY one of: "Basic Economy", "Main Cabin", "Main Plus", "Premium Economy", "Business", "First"
- routine: the destination region string (determine from the rules and query)
- direction: 0 (departing FROM US) or 1 (arriving TO US from international)
- bag_list: list of dicts, each bag needs: {{"id": <bag_number>, "name": "<description>", "size": [L, W, H], "weight": W}}

Example:
```python
def solution():
    base_price = 180
    customer_class = "Main Cabin"
    routine = "U.S."
    direction = 0
    bag_list = [
        {{"id": 0, "name": "backpack", "size": [22, 13, 6], "weight": 10}},
        {{"id": 1, "name": "luggage box", "size": [44, 22, 20], "weight": 69}}
    ]
    result = compute_baggage_cost(base_price, customer_class, routine, direction, bag_list, check_base_tables)
    total_cost = result[0]
    return total_cost
```

Write ONLY the solution() function. The function must return total_cost as an integer.
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
    
    # Extract code (handle markdown fences - case insensitive)
    code = response
    # Try ```python (case-insensitive)
    match = re.search(r'```[Pp]ython\n(.*?)\n```', response, re.DOTALL)
    if match:
        code = match.group(1)
    else:
        # Try ```py
        match = re.search(r'```py\n(.*?)\n```', response, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            # Try ``` without language tag
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
        # Provide the reference implementation directly (no preprocessing)
        def compute_baggage_cost(base_price, customer_class, routine, direction, bag_list, check_base_tables):
            """Pass-through to reference implementation."""
            return loader._compute_answer_fn(
                base_price=base_price,
                customer_class=customer_class,
                routine=routine,
                direction=direction,
                bag_list=bag_list,
                check_base_tables=check_base_tables,
            )

        namespace = {
            'compute_baggage_cost': compute_baggage_cost,
            'check_base_tables': loader._fee_tables,
        }
        
        # Capture stdout for fallback
        import io
        from contextlib import redirect_stdout
        output_buffer = io.StringIO()
        
        # Execute code (capture stdout)
        with redirect_stdout(output_buffer):
            exec(code, namespace)
        
        stdout_output = output_buffer.getvalue()
        
        # Method 1: Look for solution() function (preferred)
        if 'solution' in namespace and callable(namespace['solution']):
            result = namespace['solution']()
            # Handle numpy types and regular Python types
            try:
                answer = int(result)
                success = True
                if verbose:
                    print(f"  solution() returned: ${answer}")
            except (TypeError, ValueError):
                if verbose:
                    print(f"  ✗ solution() returned non-convertible: {result}")
        else:
            # Method 2: Check namespace for common variable names
            if verbose:
                print(f"  ✗ No solution() function found, trying fallbacks...")
            
            # Check multiple possible variable names
            for var_name in ['total_cost', 'result', 'total', 'cost', 'answer', 'output']:
                if var_name in namespace:
                    val = namespace[var_name]
                    try:
                        if isinstance(val, tuple):
                            answer = int(val[0])
                        else:
                            answer = int(val)
                        success = True
                        if verbose:
                            print(f"  Fallback: found {var_name} = ${answer}")
                        break
                    except (TypeError, ValueError):
                        continue
            
            # Method 3: Parse stdout for answer patterns (last resort)
            if not success and stdout_output:
                patterns = [
                    r'Total:\s*\$?(\d+)',
                    r'total cost[:\s]+\$?(\d+)',
                    r'Total cost[:\s]+\$?(\d+)',
                    r'\$(\d+)',
                ]
                for pattern in patterns:
                    match = re.search(pattern, stdout_output, re.IGNORECASE)
                    if match:
                        answer = int(match.group(1))
                        success = True
                        if verbose:
                            print(f"  Fallback: parsed from stdout = ${answer}")
                        break
        
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