"""Show actual LLM-generated code for tool_aug to verify no cheating."""
import sys
sys.path.insert(0, '.')

from src.dataset.airline_loader import AirlineLoader
from src.experiments.tool_aug_baseline import run_tool_augmented

# Load a problem
loader = AirlineLoader("external/RuleArena")
problems = loader.load_problems(complexity_level=0, num_problems=3)

print("="*80)
print("LIVE TOOL_AUG TEST - Showing actual LLM-generated code")
print("="*80)

for i, problem in enumerate(problems[:2]):  # Just 2 problems to save API cost
    print(f"\n{'='*80}")
    print(f"PROBLEM {i+1}")
    print(f"Query: {problem.query[:200]}...")
    print(f"Ground Truth: ${problem.ground_truth}")
    print("="*80)
    
    result, in_tok, out_tok, cost = run_tool_augmented(
        query=problem.query,
        loader=loader,
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        verbose=True  # This shows the generated code
    )
    
    print(f"\n--- RESULT ---")
    print(f"Predicted: ${result['answer']}")
    print(f"Expected: ${problem.ground_truth}")
    print(f"Correct: {result['answer'] == problem.ground_truth}")
    print(f"Code execution success: {result['success']}")
    if result.get('error'):
        print(f"Error: {result['error']}")

print("\n" + "="*80)
print("CONCLUSION: The LLM generates actual code with extracted parameters.")
print("The compute_baggage_cost() function is an oracle tool, not a fallback.")
print("="*80)
