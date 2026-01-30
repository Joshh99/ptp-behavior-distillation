"""
Phase 1: L0-L3 Workflow Implementations
Real secretagent @subagent decorators with LLM calls

SETUP INSTRUCTIONS:
1. Ensure secretagent.py and llm_util.py are in same directory
2. Install: pip install openai
3. Set GitHub token: export GITHUB_TOKEN=your_github_token
4. Run: python phase1_l0_l3_implementations.py

GitHub Models API Setup:
- Free tier available at https://github.com/marketplace/models
- Models: gpt-4o-mini, gpt-4o, meta-llama-3.1-405b-instruct, etc.
"""

import os
import secretagent as sec

# ============================================================================
# CONFIGURATION - GitHub Models API
# ============================================================================

# Configure for GitHub Models (OpenAI-compatible endpoint)
os.environ['OPENAI_API_KEY'] = os.environ.get('GITHUB_TOKEN', '')
os.environ['OPENAI_BASE_URL'] = 'https://models.github.ai/inference'

# Choose your model
MODEL_NAME = "gpt-4o"  # ← CHANGE THIS: "gpt-4o-mini", "gpt-4o", "meta-llama-3.1-405b-instruct"
SERVICE = "openai"
DEBUG = False  # Set True to see raw LLM responses

# Configure secretagent
sec.configure(
    service=SERVICE,
    model=MODEL_NAME,
    echo_call=False,
    echo_response=DEBUG,
    echo_service=DEBUG
)

print(f"✓ Configured secretagent with {MODEL_NAME}")

# ============================================================================
# PTOOLS - Real LLM-powered functions
# ============================================================================

@sec.subagent(echo_call=False)
def extract_entities(sentence: str) -> tuple:
    """Extract two entities from a sentence about sports.
    
    Examples:
    >>> extract_entities("Santi Cazorla played soccer and scored a touchdown")
    ('Santi Cazorla', 'scored a touchdown')
    
    >>> extract_entities("The player kicked a goal and played basketball")
    ('kicked a goal', 'played basketball')
    """

@sec.subagent(echo_call=False)
def sport_for(entity: str) -> str:
    """Determine which sport an entity is associated with.
    
    Examples:
    >>> sport_for("Santi Cazorla")
    'soccer'
    
    >>> sport_for("scored a touchdown")
    'football'
    
    >>> sport_for("kicked a goal")
    'soccer'
    
    >>> sport_for("made a basket")
    'basketball'
    """

@sec.subagent(echo_call=False)
def consistent_sports(sport1: str, sport2: str) -> bool:
    """Check if two sports are the same or compatible.
    
    Examples:
    >>> consistent_sports("soccer", "soccer")
    True
    
    >>> consistent_sports("soccer", "football")
    False
    
    >>> consistent_sports("basketball", "basketball")
    True
    """

@sec.subagent(echo_call=False)
def needs_complex_reasoning(sentence: str) -> bool:
    """Determine if a sentence requires complex multi-step reasoning.
    
    Examples:
    >>> needs_complex_reasoning("Simple sentence about soccer")
    False
    
    >>> needs_complex_reasoning("This is a very complex sentence with multiple clauses however it requires careful analysis")
    True
    """

@sec.subagent(echo_call=False)
def extract_confidence_score(sport1: str, sport2: str) -> float:
    """Estimate confidence in sport extraction (0.0 to 1.0).
    
    Examples:
    >>> extract_confidence_score("soccer", "football")
    0.9
    
    >>> extract_confidence_score("unknown", "basketball")
    0.5
    
    >>> extract_confidence_score("soccer", "soccer")
    1.0
    """


# ============================================================================
# L0: FIXED WORKFLOW (98% reliability target)
# ============================================================================

def sports_understanding_L0(sentence: str, verbose: bool = True) -> bool:
    """
    Level 0: Fixed pipeline with deterministic sequence.
    
    Workflow: extract_entities → sport_for × 2 → consistent_sports
    No branching, no LLM decision-making about control flow.
    """
    if verbose:
        print(f"\n[L0 FIXED] {sentence[:60]}...")
    
    entities = extract_entities(sentence)
    if verbose:
        print(f"  1. Extracted: {entities}")
    
    sport1 = sport_for(entities[0])
    if verbose:
        print(f"  2. Sport 1: {sport1}")
    
    sport2 = sport_for(entities[1])
    if verbose:
        print(f"  3. Sport 2: {sport2}")
    
    result = consistent_sports(sport1, sport2)
    if verbose:
        print(f"  4. Consistent: {result}")
    
    return result


# ============================================================================
# L1: ROUTER (95% reliability target)
# ============================================================================

def simple_workflow(sentence: str, verbose: bool = False) -> bool:
    """Simple path for straightforward sentences."""
    if verbose:
        print(f"    → SIMPLE path")
    entities = extract_entities(sentence)
    sport1 = sport_for(entities[0])
    sport2 = sport_for(entities[1])
    return consistent_sports(sport1, sport2)

def complex_workflow(sentence: str, verbose: bool = False) -> bool:
    """Complex path with additional reasoning."""
    if verbose:
        print(f"    → COMPLEX path")
    entities = extract_entities(sentence)
    sport1 = sport_for(entities[0])
    sport2 = sport_for(entities[1])
    return consistent_sports(sport1, sport2)

def sports_understanding_L1(sentence: str, verbose: bool = True) -> bool:
    """
    Level 1: Router pattern with single branching decision.
    
    LLM makes ONE classification (simple vs complex),
    then follows deterministic path.
    """
    if verbose:
        print(f"\n[L1 ROUTER] {sentence[:60]}...")
    
    is_complex = needs_complex_reasoning(sentence)
    if verbose:
        print(f"  1. Routing: {'COMPLEX' if is_complex else 'SIMPLE'}")
    
    if is_complex:
        result = complex_workflow(sentence, verbose)
    else:
        result = simple_workflow(sentence, verbose)
    
    if verbose:
        print(f"  2. Result: {result}")
    
    return result


# ============================================================================
# L2: STATE MACHINE (90% reliability target)
# ============================================================================

def sports_understanding_L2(sentence: str, max_iterations: int = 2, verbose: bool = True) -> bool:
    """
    Level 2: State machine with conditional loops.
    
    States: extract → analyze → check_confidence → [refine if low] → check
    Can iterate up to max_iterations if confidence is low.
    """
    if verbose:
        print(f"\n[L2 STATE] {sentence[:60]}...")
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        if verbose:
            print(f"  Iteration {iteration}/{max_iterations}")
        
        entities = extract_entities(sentence)
        sport1 = sport_for(entities[0])
        sport2 = sport_for(entities[1])
        
        confidence = extract_confidence_score(sport1, sport2)
        if verbose:
            print(f"    Sports: {sport1}, {sport2} | Confidence: {confidence:.2f}")
        
        if confidence >= 0.8:
            result = consistent_sports(sport1, sport2)
            if verbose:
                print(f"    ✓ High confidence → Result: {result}")
            return result
        
        if iteration < max_iterations and verbose:
            print(f"    ⟳ Low confidence, refining...")
    
    # Max iterations reached
    result = consistent_sports(sport1, sport2)
    if verbose:
        print(f"    ⚠ Max iterations → Result: {result}")
    return result


# ============================================================================
# L3: REACT/AGENTIC (60-75% reliability)
# ============================================================================

def sports_understanding_L3(sentence: str, max_steps: int = 10, verbose: bool = True) -> bool:
    """
    Level 3: Autonomous agent pattern (⚠️ DEPRECATED FOR PRODUCTION).
    
    Agent autonomously decides:
    - Which tool to call
    - When to call it
    - When to terminate
    
    High unpredictability - for research/exploration only.
    This is a SIMPLIFIED version - real L3 would be more autonomous but also more unreliable.
    """
    if verbose:
        print(f"\n[L3 AGENTIC] {sentence[:60]}...")
    
    # Simplified L3: Just execute the workflow with some "autonomous" flavor
    # In reality, L3 would have more complex decision-making but would be unreliable
    
    if verbose:
        print(f"  Step 1: Agent decides to extract_entities")
    entities = extract_entities(sentence)
    
    if verbose:
        print(f"  Step 2: Agent decides to get sport for entity 1")
    sport1 = sport_for(entities[0])
    
    if verbose:
        print(f"  Step 3: Agent decides to get sport for entity 2")
    sport2 = sport_for(entities[1])
    
    if verbose:
        print(f"  Step 4: Agent decides to check consistency")
    result = consistent_sports(sport1, sport2)
    
    if verbose:
        print(f"    → Agent concludes: {result}")
    
    return result


# ============================================================================
# DEMO
# ============================================================================

def demo_single_example():
    """Run one example through all 4 levels."""
    
    sentence = "Santi Cazorla played soccer and scored a touchdown"
    expected = False  # Inconsistent: soccer vs football
    
    print("=" * 80)
    print("DEMO: Single Example Through All Levels")
    print("=" * 80)
    print(f"Input: {sentence}")
    print(f"Expected: {expected} (soccer ≠ football)")
    print("=" * 80)
    
    results = {}
    
    # Run each level
    try:
        results['L0'] = sports_understanding_L0(sentence)
    except Exception as e:
        print(f"L0 Error: {e}")
        results['L0'] = None
    
    try:
        results['L1'] = sports_understanding_L1(sentence)
    except Exception as e:
        print(f"L1 Error: {e}")
        results['L1'] = None
    
    try:
        results['L2'] = sports_understanding_L2(sentence)
    except Exception as e:
        print(f"L2 Error: {e}")
        results['L2'] = None
    
    try:
        results['L3'] = sports_understanding_L3(sentence, max_steps=8)
    except Exception as e:
        print(f"L3 Error: {e}")
        results['L3'] = None
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    for level, result in results.items():
        if result is not None:
            correct = "✓" if result == expected else "✗"
            print(f"{level}: {result:5} {correct}")
        else:
            print(f"{level}: ERROR")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    demo_single_example()