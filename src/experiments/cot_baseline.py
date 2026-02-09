"""
Chain-of-Thought baseline - Direct replication of RuleArena's CoT approach.

Reference: RuleArena paper Appendix E (Airline prompt template)
This does NOT use @ptool - it's a raw LLM call following their exact format.
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# CHANGE THIS LINE:
# from src.experiments.config import call_llm
# TO:
from external.ptool_framework.llm_backend import call_llm, calculate_cost

import re


# RuleArena's CoT prompt from Appendix E
COT_PROMPT_TEMPLATE = """System Prompt: You are a helpful assistant at American Airlines.

User Prompt: You are given the information of a passenger, his / her items, his / her special needs, and the policies of American Airlines. You should compute the total cost (including the flight ticket fee, checked bag fees, cost of special needs) according to the policies for the passenger. The policies of American Airlines are as follows:

<reference_rules>
{rules}
</reference_rules>

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
    """
    Run CoT baseline exactly as RuleArena did.
    
    Args:
        query: Natural language problem
        loader: AirlineLoader for getting rules
        model: Model to use
        verbose: Print progress
        
    Returns:
        (result_dict, input_tokens, output_tokens, cost)
    """
    if verbose:
        print(f"\n[CoT Baseline] Processing: {query[:80]}...")
    
    # Load rules (textual format, same as they used)
    rules = loader.load_rules(textual=True)
    
    # Format prompt exactly as they did
    prompt = COT_PROMPT_TEMPLATE.format(
        rules=rules,
        query=query
    )
    
    # Call LLM
    import time
    start = time.time()
    
    response, input_tokens, output_tokens = call_llm(
        prompt=prompt,
        model=model,
    )
    
    execution_time = time.time() - start
    
    # Calculate cost
    cost = calculate_cost(model, input_tokens, output_tokens)
    
    if verbose:
        print(f"  Reasoned in {execution_time:.2f}s:")
        print(f"     Tokens: {input_tokens} in, {output_tokens} out")
        print(f"     Cost: ${cost:.6f}")
    
    # Extract answer from "The total cost is $xxx"
    # Try multiple patterns to be robust
    patterns = [
        r'The total cost is \$(\d+)',
        r'total cost is \$(\d+)',
        r'Total cost: \$(\d+)',
        r'Total: \$(\d+)',
        r'\$(\d+)(?:\s|\.|\n|$)',  # Fallback: any $XXX pattern
    ]
    
    answer = 0
    success = False
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            answer = int(match.group(1))
            success = True
            if verbose:
                print(f"  Extracted: ${answer}")
            break
    
    if not success and verbose:
        print(f"  ✗ Failed to extract answer")
        print(f"  Response preview: {response[:200]}...")
    
    return {
        "answer": answer,
        "reasoning": response,
        "success": success,
    }, input_tokens, output_tokens, cost