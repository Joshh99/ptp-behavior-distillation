"""Test L1 Transparent on 3 problems"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dataset.airline_loader import AirlineLoader
from src.experiments.l1_ptool_extraction import baggage_allowance_l1_transparent

def test_l1_transparent():
    loader = AirlineLoader("external/RuleArena")
    problems = loader.load_problems(complexity_level=0, max_problems=3)
    
    print("Testing L1 Transparent (Rules Shown)\n" + "="*60)
    
    total_cost = 0.0
    correct = 0
    
    for i, problem in enumerate(problems):
        print(f"\n[Problem {i+1}]")
        result, in_tok, out_tok, cost = baggage_allowance_l1_transparent(
            query=problem.query,
            loader=loader,
            verbose=True,
        )
        
        predicted = result["answer"]
        expected = problem.ground_truth
        
        if predicted == expected:
            correct += 1
            status = "SUCCESS"
        else:
            status = "FAIL"
        
        print(f"{status} Expected: ${expected}, Got: ${predicted}")
        print(f"   Tokens: {in_tok + out_tok} total, Cost: ${cost:.6f}")
        
        total_cost += cost
    
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/3 ({correct/3*100:.1f}%)")
    print(f"Total Cost: ${total_cost:.6f}")
    print(f"Avg Cost: ${total_cost/3:.6f} per problem")
    print(f"\nNote: L1 Transparent uses ~2K more input tokens per call")

if __name__ == "__main__":
    test_l1_transparent()