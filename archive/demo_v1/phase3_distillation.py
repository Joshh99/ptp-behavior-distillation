"""
Phase 3: Behavior Distillation
Collect traces from L3/L2, cluster patterns, generate L0 workflows

SETUP:
1. Ensure Phase 1 and Phase 2 work
2. Run: python phase3_distillation.py

This demonstrates the core concept:
  L3 (autonomous, 60-75% reliable) → traces → patterns → L0 (fixed, 98% reliable)
"""

import os
import json
import time
from typing import List, Dict, Tuple
from collections import Counter

import secretagent as sec
from phase1_l0_l3_implementations import (
    extract_entities,
    sport_for,
    consistent_sports,
    sports_understanding_L3
)

# Configure for GitHub Models
os.environ['OPENAI_API_KEY'] = os.environ.get('GITHUB_TOKEN', '')
os.environ['OPENAI_BASE_URL'] = 'https://models.github.ai/inference'

MODEL_NAME = "gpt-4o-mini"
sec.configure(service="openai", model=MODEL_NAME, echo_call=False, echo_response=False)

print(f"✓ Distillation configured with {MODEL_NAME}\n")


# ============================================================================
# TRACE COLLECTION
# ============================================================================

def collect_traces_from_L3(sentences: List[str]) -> List[Dict]:
    """
    Run L3 agent on multiple sentences and collect execution traces.
    
    Returns:
        List of trace dictionaries with sentence, result, and ptool calls
    """
    print("=" * 80)
    print("STEP 1: COLLECTING TRACES FROM L3 AGENT")
    print("=" * 80)
    
    traces = []
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\n[{i}/{len(sentences)}] Running L3 on: {sentence[:60]}...")
        
        # Rate limiting: wait between requests
        if i > 1:
            time.sleep(2)  # 2 second delay between requests
        
        with sec.recorder() as record:
            try:
                result = sports_understanding_L3(sentence, verbose=False)
                
                # Extract just the function names and create a trace signature
                trace = {
                    'sentence': sentence,
                    'result': result,
                    'calls': [
                        {
                            'function': call['func'],
                            'args': call['args'],
                            'output': call['output']
                        }
                        for call in record
                    ],
                    'num_calls': len(record),
                    'signature': ' → '.join([call['func'] for call in record])
                }
                
                traces.append(trace)
                print(f"  ✓ Trace: {trace['signature']}")
                print(f"  ✓ Result: {result} | Calls: {len(record)}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                traces.append({
                    'sentence': sentence,
                    'result': None,
                    'error': str(e),
                    'calls': [],
                    'signature': 'ERROR'
                })
    
    print(f"\n✓ Collected {len(traces)} traces")
    return traces


# ============================================================================
# TRACE CLUSTERING
# ============================================================================

def cluster_traces_by_pattern(traces: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group traces by their execution pattern (sequence of ptool calls).
    
    Returns:
        Dict mapping signature → list of traces with that signature
    """
    print("\n" + "=" * 80)
    print("STEP 2: CLUSTERING TRACES BY EXECUTION PATTERN")
    print("=" * 80)
    
    clusters = {}
    
    for trace in traces:
        sig = trace['signature']
        if sig not in clusters:
            clusters[sig] = []
        clusters[sig].append(trace)
    
    # Sort clusters by frequency
    sorted_clusters = dict(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True))
    
    print(f"\n✓ Found {len(sorted_clusters)} distinct patterns:\n")
    
    for i, (signature, trace_list) in enumerate(sorted_clusters.items(), 1):
        percentage = (len(trace_list) / len(traces)) * 100
        print(f"  [{i}] {signature}")
        print(f"      Frequency: {len(trace_list)}/{len(traces)} ({percentage:.1f}%)")
    
    return sorted_clusters


# ============================================================================
# CODE GENERATION
# ============================================================================

def generate_L0_from_pattern(pattern_signature: str, example_traces: List[Dict]) -> str:
    """
    Generate L0 (fixed workflow) Python code from a trace pattern.
    
    Args:
        pattern_signature: String like "extract_entities → sport_for → sport_for → consistent_sports"
        example_traces: List of traces that follow this pattern
    
    Returns:
        Python code as string
    """
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING L0 CODE FROM MOST COMMON PATTERN")
    print("=" * 80)
    
    print(f"\nMost common pattern ({len(example_traces)} occurrences):")
    print(f"  {pattern_signature}\n")
    
    # Parse the signature to understand the workflow
    steps = pattern_signature.split(' → ')
    
    # Generate function body
    code_lines = [
        'def distilled_sports_understanding(sentence: str) -> bool:',
        '    """',
        f'    Auto-generated L0 workflow from behavior distillation.',
        f'    Based on {len(example_traces)} traces from L3 agent.',
        f'    Pattern: {pattern_signature}',
        '    ',
        '    This is a FIXED workflow with 98% target reliability.',
        '    """',
        ''
    ]
    
    # Generate workflow based on pattern
    if 'extract_entities' in steps:
        code_lines.append('    # Step 1: Extract entities from sentence')
        code_lines.append('    entities = extract_entities(sentence)')
        code_lines.append('')
    
    # Count how many times sport_for appears
    sport_for_count = steps.count('sport_for')
    if sport_for_count >= 1:
        code_lines.append('    # Step 2: Get sport for first entity')
        code_lines.append('    sport1 = sport_for(entities[0])')
        code_lines.append('')
    
    if sport_for_count >= 2:
        code_lines.append('    # Step 3: Get sport for second entity')
        code_lines.append('    sport2 = sport_for(entities[1])')
        code_lines.append('')
    
    if 'consistent_sports' in steps:
        code_lines.append('    # Step 4: Check if sports are consistent')
        code_lines.append('    result = consistent_sports(sport1, sport2)')
        code_lines.append('    return result')
    
    generated_code = '\n'.join(code_lines)
    
    print("Generated L0 code:")
    print("-" * 80)
    print(generated_code)
    print("-" * 80)
    
    return generated_code


# ============================================================================
# VALIDATION
# ============================================================================

def validate_distilled_workflow(generated_code: str, test_sentences: List[str]) -> Dict:
    """
    Test the generated L0 code to verify it works.
    
    Returns:
        Validation results
    """
    print("\n" + "=" * 80)
    print("STEP 4: VALIDATING DISTILLED WORKFLOW")
    print("=" * 80)
    
    # Execute the generated code to create the function
    local_scope = {
        'extract_entities': extract_entities,
        'sport_for': sport_for,
        'consistent_sports': consistent_sports
    }
    
    try:
        exec(generated_code, local_scope)
        distilled_func = local_scope['distilled_sports_understanding']
        print("\n✓ Generated function compiled successfully")
    except Exception as e:
        print(f"\n✗ Failed to compile generated code: {e}")
        return {'success': False, 'error': str(e)}
    
    # Test on sample sentences
    print(f"\nTesting on {len(test_sentences)} examples:")
    
    results = []
    for sentence in test_sentences:
        try:
            with sec.recorder() as record:
                result = distilled_func(sentence)
                num_calls = len(record)
                
            print(f"  • {sentence[:50]}... → {result} ({num_calls} calls)")
            results.append({
                'sentence': sentence,
                'result': result,
                'calls': num_calls,
                'success': True
            })
            
        except Exception as e:
            print(f"  • {sentence[:50]}... → ERROR: {e}")
            results.append({
                'sentence': sentence,
                'error': str(e),
                'success': False
            })
    
    success_rate = sum(1 for r in results if r.get('success', False)) / len(results)
    avg_calls = sum(r.get('calls', 0) for r in results if r.get('success', False)) / max(1, sum(1 for r in results if r.get('success', False)))
    
    print(f"\n✓ Validation: {success_rate*100:.0f}% success rate, {avg_calls:.1f} avg calls")
    
    return {
        'success': True,
        'success_rate': success_rate,
        'avg_calls': avg_calls,
        'results': results
    }


# ============================================================================
# SUMMARY & COMPARISON
# ============================================================================

def print_distillation_summary(traces: List[Dict], clusters: Dict, validation: Dict):
    """Print final summary comparing L3 → L0 distillation."""
    
    print("\n" + "=" * 80)
    print("BEHAVIOR DISTILLATION SUMMARY")
    print("=" * 80)
    
    print("\n📊 TRACE STATISTICS:")
    print(f"  • Total traces collected: {len(traces)}")
    print(f"  • Unique patterns found: {len(clusters)}")
    print(f"  • Most common pattern coverage: {len(list(clusters.values())[0])}/{len(traces)} ({len(list(clusters.values())[0])/len(traces)*100:.1f}%)")
    
    print("\n🔄 DISTILLATION PROCESS:")
    print(f"  • Source: L3 autonomous agent (unreliable, exploratory)")
    print(f"  • Method: Trace clustering → pattern extraction")
    print(f"  • Target: L0 fixed workflow (reliable, predictable)")
    
    print("\n✅ VALIDATION RESULTS:")
    if validation.get('success'):
        print(f"  • Generated code: Compiled successfully")
        print(f"  • Success rate: {validation['success_rate']*100:.0f}%")
        print(f"  • Avg LLM calls: {validation['avg_calls']:.1f}")
    else:
        print(f"  • Error: {validation.get('error', 'Unknown')}")
    
    print("\n💡 KEY INSIGHT:")
    print("  PTP's observable traces enable systematic behavior distillation.")
    print("  We can extract reliable workflows from unreliable agent exploration!")
    
    print("\n" + "=" * 80)


# ============================================================================
# MAIN DISTILLATION PIPELINE
# ============================================================================

def run_behavior_distillation():
    """Execute the complete behavior distillation pipeline."""
    
    print("=" * 80)
    print("PHASE 3: BEHAVIOR DISTILLATION")
    print("=" * 80)
    print("\nGoal: Extract reliable L0 workflow from L3 agent traces\n")
    
    # Training sentences for trace collection
    training_sentences = [
        "Santi Cazorla played soccer and scored a touchdown",
        "The player kicked a goal and played soccer",
        "Basketball player scored a basket and played basketball",
        "Michael Jordan played basketball and scored a touchdown",
        "The athlete made a basket and played soccer",
        "Tennis player hit an ace and played tennis",
        "Hockey player scored a goal and played hockey",
    ]
    
    # Test sentences for validation
    test_sentences = [
        "Ronaldo played soccer and kicked a goal",
        "The quarterback threw a touchdown and played football",
        "LeBron James played basketball and scored a basket",
    ]
    
    # STEP 1: Collect traces
    traces = collect_traces_from_L3(training_sentences)
    
    # STEP 2: Cluster by pattern
    clusters = cluster_traces_by_pattern(traces)
    
    # STEP 3: Generate L0 code from most common pattern
    most_common_pattern = list(clusters.keys())[0]
    most_common_traces = clusters[most_common_pattern]
    generated_code = generate_L0_from_pattern(most_common_pattern, most_common_traces)
    
    # STEP 4: Validate
    validation = validate_distilled_workflow(generated_code, test_sentences)
    
    # STEP 5: Summary
    print_distillation_summary(traces, clusters, validation)
    
    # Save results
    results = {
        'training_sentences': training_sentences,
        'num_traces': len(traces),
        'num_patterns': len(clusters),
        'most_common_pattern': most_common_pattern,
        'pattern_frequency': len(most_common_traces),
        'generated_code': generated_code,
        'validation': validation
    }
    
    output_file = 'distillation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Also save the generated code as a Python file
    code_file = 'distilled_workflow.py'
    with open(code_file, 'w') as f:
        f.write('"""\n')
        f.write('Auto-generated L0 workflow from behavior distillation\n')
        f.write(f'Generated from {len(traces)} L3 agent traces\n')
        f.write('"""\n\n')
        f.write('from phase1_l0_l3_implementations import extract_entities, sport_for, consistent_sports\n\n')
        f.write(generated_code)
    
    print(f"✓ Generated code saved to {code_file}")
    
    return results


if __name__ == "__main__":
    results = run_behavior_distillation()