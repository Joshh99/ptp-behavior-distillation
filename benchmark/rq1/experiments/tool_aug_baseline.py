import re
import time
import io
from contextlib import redirect_stdout

from external.ptool_framework.llm_backend import call_llm, calculate_cost


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
    if verbose:
        print(f"\n[Tool-Aug] Processing: {query[:80]}...")

    rules = loader.load_rules(textual=True)

    prompt = TOOL_AUG_PROMPT.format(rules=rules, query=query)

    start = time.time()

    response, input_tokens, output_tokens = call_llm(prompt=prompt, model=model)

    cost = calculate_cost(model, input_tokens, output_tokens)

    code = response
    match = re.search(r'```[Pp]ython\n(.*?)\n```', response, re.DOTALL)
    if match:
        code = match.group(1)
    else:
        match = re.search(r'```py\n(.*?)\n```', response, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
            if match:
                code = match.group(1)

    answer = 0
    success = False
    error_msg = None

    try:
        def compute_baggage_cost(base_price, customer_class, routine, direction, bag_list, check_base_tables):
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

        output_buffer = io.StringIO()

        with redirect_stdout(output_buffer):
            exec(code, namespace)

        stdout_output = output_buffer.getvalue()

        if 'solution' in namespace and callable(namespace['solution']):
            result = namespace['solution']()
            try:
                answer = int(result)
                success = True
            except (TypeError, ValueError):
                pass
        else:
            for var_name in ['total_cost', 'result', 'total', 'cost', 'answer', 'output']:
                if var_name in namespace:
                    val = namespace[var_name]
                    try:
                        answer = int(val[0]) if isinstance(val, tuple) else int(val)
                        success = True
                        break
                    except (TypeError, ValueError):
                        continue

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
                        break

    except Exception as e:
        error_msg = str(e)

    return {
        "answer": answer,
        "code": code,
        "success": success,
        "error": error_msg,
    }, input_tokens, output_tokens, cost
