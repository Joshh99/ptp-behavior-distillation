import re
import time

from external.ptool_framework.llm_backend import call_llm, calculate_cost


COT_PROMPT_TEMPLATE = """System Prompt: You are a helpful assistant at American Airlines.

User Prompt: You are given the information of a passenger, his / her items, his / her special needs, and the policies of American Airlines. You should compute the total cost (including the flight ticket fee, checked bag fees, cost of special needs) according to the policies for the passenger. The policies of American Airlines are as follows:

<reference_rules>
{rules}
</reference_rules>

<example>
<user_query>
David is a Main Cabin passenger flying from Miami to Chicago with the following items:
- Ticket price: $250
- 1 backpack (size: 20x14x8 inches, weight: 12 lbs)
- 1 luggage box (size: 24x16x10 inches, weight: 35 lbs)
Compute the total cost step by step.
</user_query>

<assistant_response>
Let me calculate the total cost step by step.

Step 1: Identify the flight details
- Route: Miami to Chicago (domestic U.S. flight)
- Customer class: Main Cabin
- Ticket price: $250

Step 2: Evaluate the carry-on bag (backpack)
- Size: 20x14x8 inches
- The carry-on size limit is 22x14x9 inches
- 20 <= 22, 14 <= 14, 8 <= 9, so the backpack fits within carry-on limits
- Carry-on fee for Main Cabin: $0 (included)
- Carry-on bag cost: $0

Step 3: Evaluate the first checked bag (luggage box)
- Size: 24x16x10 inches
- Weight: 35 lbs
- For Main Cabin domestic U.S. flights, first checked bag fee: $40
- Size check: 24+16+10 = 50 inches total, which is under 62 inches limit (not oversized)
- Weight check: 35 lbs is under 50 lbs limit (not overweight)
- First checked bag cost: $40

Step 4: Calculate total cost
- Ticket price: $250
- Carry-on bag: $0
- First checked bag: $40
- Total: $250 + $0 + $40 = $290

The total cost is $290.
</assistant_response>
</example>

<user_query> Compute the total cost for him step by step (don't omit any bag) and end your response with "The total cost is $xxx." (xxx is a number)
{query}
</user_query>

Your response:"""


def run_cot_baseline(
    query: str,
    loader,
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    verbose: bool = False,
) -> tuple[dict, int, int, float]:
    if verbose:
        print(f"\n[CoT Baseline] Processing: {query[:80]}...")

    rules = loader.load_rules(textual=True)

    prompt = COT_PROMPT_TEMPLATE.format(rules=rules, query=query)

    start = time.time()

    response, input_tokens, output_tokens = call_llm(prompt=prompt, model=model)

    execution_time = time.time() - start

    cost = calculate_cost(model, input_tokens, output_tokens)

    patterns = [
        r'The total cost is \$(\d+)',
        r'total cost is \$(\d+)',
        r'Total cost: \$(\d+)',
        r'Total: \$(\d+)',
        r'\$(\d+)(?:\s|\.|\n|$)',
    ]

    answer = 0
    success = False

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            answer = int(match.group(1))
            success = True
            break

    return {
        "answer": answer,
        "reasoning": response,
        "success": success,
    }, input_tokens, output_tokens, cost
