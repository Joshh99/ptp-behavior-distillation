"""Test L1 with @ptool on 2-3 problems"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dataset.airline_loader import AirlineLoader
from src.experiments.l1_ptool_extraction import baggage_allowance_l1_ptool
from ptool_framework.trace_store import get_trace_store

def test_l1_ptool():
    loader = AirlineLoader("external/RuleArena")
    problems = loader.load_problems(complexity_level=0, max_problems=3)
    
    trace_store = get_trace_store()
    
    print("Testing L1 with @ptool decorator\n" + "="*60)
    
    total_cost = 0.0
    for i, problem in enumerate(problems):
        print(f"\n[Problem {i+1}]")
        result, in_tok, out_tok, cost = baggage_allowance_l1_ptool(
            query=problem.query,
            loader=loader,
            verbose=True,
        )
        
        predicted = result["answer"]
        expected = problem.ground_truth
        status = "✓" if predicted == expected else "✗"
        
        print(f"{status} Expected: ${expected}, Got: ${predicted}")
        print(f"   Tokens: {in_tok + out_tok} total, Cost: ${cost:.6f}")
        
        total_cost += cost
    
    print(f"\n{'='*60}")
    print(f"Total Cost: ${total_cost:.6f}")
    print(f"\nTrace Statistics:")
    stats = trace_store.get_stats()
    print(f"  Total traces: {stats['total_traces']}")
    print(f"  Session traces: {stats['session_traces']}")

if __name__ == "__main__":
    test_l1_ptool()