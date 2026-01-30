"""
Phase 2: Evaluation & Comparison
Run multiple test cases and generate metrics comparison

SETUP:
1. Run Phase 1 first to ensure everything works
2. Run: python phase2_evaluation.py
"""

import os
import json
import time
from typing import List, Dict, Tuple

import secretagent as sec
from phase1_l0_l3_implementations import (
    sports_understanding_L0,
    sports_understanding_L1,
    sports_understanding_L2,
    sports_understanding_L3
)

# Configure for GitHub Models
os.environ['OPENAI_API_KEY'] = os.environ.get('GITHUB_TOKEN', '')
os.environ['OPENAI_BASE_URL'] = 'https://models.github.ai/inference'

MODEL_NAME = "gpt-4o-mini"  # ← CHANGE THIS if needed
sec.configure(service="openai", model=MODEL_NAME, echo_call=False, echo_response=False)

print(f"✓ Evaluation configured with {MODEL_NAME}\n")


# ============================================================================
# TEST CASES
# ============================================================================

TEST_CASES = [
    # (sentence, expected_result, description)
    ("Santi Cazorla played soccer and scored a touchdown", False, "Inconsistent: soccer vs football"),
    ("The player kicked a goal and played soccer", True, "Consistent: both soccer"),
    ("Basketball player scored a basket and played basketball", True, "Consistent: both basketball"),
    ("Michael Jordan played basketball and scored a touchdown", False, "Inconsistent: basketball vs football"),
    ("The athlete made a basket and played soccer", False, "Inconsistent: basketball vs soccer"),
]


# ============================================================================
# EVALUATION RUNNER
# ============================================================================

def evaluate_level(level_name: str, level_func, test_cases: List[Tuple]) -> Dict:
    """
    Evaluate a single level (L0, L1, L2, or L3) on all test cases.
    
    Returns:
        Dict with accuracy, total calls, total time, and per-case results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {level_name}")
    print(f"{'='*80}")
    
    results = {
        'level': level_name,
        'correct': 0,
        'total': 0,
        'total_calls': 0,
        'total_time': 0.0,
        'cases': []
    }
    
    for sentence, expected, description in test_cases:
        print(f"\n[{level_name}] Test: {description}")
        print(f"  Input: {sentence[:60]}...")
        
        # Start recording
        with sec.recorder() as record:
            start_time = time.time()
            
            try:
                result = level_func(sentence, verbose=False)
                elapsed = time.time() - start_time
                
                is_correct = (result == expected)
                num_calls = len(record)
                
                print(f"  Result: {result} | Expected: {expected} | {'✓' if is_correct else '✗'}")
                print(f"  Calls: {num_calls} | Time: {elapsed:.2f}s")
                
                results['correct'] += is_correct
                results['total'] += 1
                results['total_calls'] += num_calls
                results['total_time'] += elapsed
                
                results['cases'].append({
                    'sentence': sentence,
                    'expected': expected,
                    'result': result,
                    'correct': is_correct,
                    'num_calls': num_calls,
                    'time': elapsed,
                    'description': description
                })
                
            except Exception as e:
                print(f"  ERROR: {e}")
                results['total'] += 1
                results['cases'].append({
                    'sentence': sentence,
                    'expected': expected,
                    'result': None,
                    'correct': False,
                    'error': str(e),
                    'description': description
                })
    
    # Calculate averages
    if results['total'] > 0:
        results['accuracy'] = results['correct'] / results['total']
        results['avg_calls'] = results['total_calls'] / results['total']
        results['avg_time'] = results['total_time'] / results['total']
    else:
        results['accuracy'] = 0.0
        results['avg_calls'] = 0.0
        results['avg_time'] = 0.0
    
    return results


# ============================================================================
# COMPARISON TABLE GENERATOR
# ============================================================================

def print_comparison_table(all_results: Dict[str, Dict]):
    """Print formatted comparison table."""
    
    print("\n" + "=" * 100)
    print("COMPARISON TABLE: L0-L3 Workflow Spectrum")
    print("=" * 100)
    
    # Header
    print(f"{'Level':<10} {'Pattern':<20} {'Accuracy':<12} {'Avg Calls':<12} {'Avg Time (s)':<15} {'Target Rel.':<15}")
    print("-" * 100)
    
    # Define patterns and target reliability
    patterns = {
        'L0': ('Fixed Pipeline', '98%'),
        'L1': ('Router', '95%'),
        'L2': ('State Machine', '90%'),
        'L3': ('ReAct/Agentic', '60-75%')
    }
    
    # Print each level
    for level in ['L0', 'L1', 'L2', 'L3']:
        if level in all_results:
            r = all_results[level]
            pattern, target = patterns[level]
            
            accuracy_pct = f"{r['accuracy']*100:.1f}%"
            avg_calls = f"{r['avg_calls']:.1f}"
            avg_time = f"{r['avg_time']:.2f}"
            
            print(f"{level:<10} {pattern:<20} {accuracy_pct:<12} {avg_calls:<12} {avg_time:<15} {target:<15}")
    
    print("=" * 100)
    
    # Summary statistics
    print("\nKEY FINDINGS:")
    if 'L0' in all_results and 'L3' in all_results:
        l0_acc = all_results['L0']['accuracy']
        l3_acc = all_results['L3']['accuracy']
        reliability_gain = (l0_acc - l3_acc) * 100
        
        l0_calls = all_results['L0']['avg_calls']
        l3_calls = all_results['L3']['avg_calls']
        call_reduction = ((l3_calls - l0_calls) / l3_calls * 100) if l3_calls > 0 else 0
        
        l0_time = all_results['L0']['avg_time']
        l3_time = all_results['L3']['avg_time']
        speedup = (l3_time / l0_time) if l0_time > 0 else 1
        
        print(f"  • Reliability improvement (L3→L0): +{reliability_gain:.1f}%")
        print(f"  • Call reduction (L3→L0): {call_reduction:.1f}% fewer calls")
        print(f"  • Speedup (L3→L0): {speedup:.1f}x faster")
    
    print()


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_full_evaluation():
    """Run complete evaluation across all levels."""
    
    print("=" * 100)
    print("PHASE 2: EVALUATION & COMPARISON")
    print(f"Model: {MODEL_NAME}")
    print(f"Test Cases: {len(TEST_CASES)}")
    print("=" * 100)
    
    all_results = {}
    
    # Evaluate each level
    all_results['L0'] = evaluate_level('L0', sports_understanding_L0, TEST_CASES)
    all_results['L1'] = evaluate_level('L1', sports_understanding_L1, TEST_CASES)
    all_results['L2'] = evaluate_level('L2', sports_understanding_L2, TEST_CASES)
    all_results['L3'] = evaluate_level('L3', sports_understanding_L3, TEST_CASES)
    
    # Print comparison table
    print_comparison_table(all_results)
    
    # Save results to JSON
    output_file = 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")
    
    return all_results


if __name__ == "__main__":
    results = run_full_evaluation()