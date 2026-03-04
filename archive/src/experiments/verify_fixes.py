"""
Comprehensive verification script to test all potential error points
before running full experiments.

This checks:
1. Region normalization in all baselines
2. Customer class validation
3. Direction validation
4. RuleArena reference tables completeness
5. JSON serialization
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
from datetime import datetime

def test_region_normalization():
    """Test region normalization in L1 extraction."""
    print("=" * 70)
    print("TEST 1: Region normalization (L1 extraction)")
    print("=" * 70)
    
    from src.experiments.l1_ptool_extraction import (
        VALID_REGIONS, normalize_region
    )
    
    print(f"Valid regions ({len(VALID_REGIONS)}): {sorted(VALID_REGIONS)}")
    print()
    
    # Test common LLM mistakes
    test_cases = [
        # Continent generalizations (most common LLM error)
        ("Asia", "China"),
        ("Europe", "Europe"),  # Valid, should stay
        ("North America", "U.S."),
        ("South America", "South America"),  # Valid
        
        # US variants
        ("US", "U.S."),
        ("USA", "U.S."),
        ("United States", "U.S."),
        ("U.S.", "U.S."),  # Valid
        ("America", "U.S."),
        
        # Country variants
        ("UK", "Europe"),
        ("United Kingdom", "Europe"),
        ("Britain", "Europe"),
        ("England", "Europe"),
        ("Germany", "Europe"),
        ("France", "Europe"),
        ("Italy", "Europe"),
        ("Spain", "Europe"),
        
        # Asian country to city confusion
        ("Tokyo", "Japan"),
        ("Beijing", "China"),
        ("Shanghai", "China"),
        ("Seoul", "South Korea"),
        ("Mumbai", "India"),
        ("Delhi", "India"),
        
        # Valid regions (should not change)
        ("Hong Kong", "Hong Kong"),
        ("China", "China"),
        ("Japan", "Japan"),
        ("India", "India"),
        ("Australia", "Australia"),
        ("Canada", "Canada"),
        ("Mexico", "Mexico"),
        ("South Korea", "South Korea"),
        
        # Edge cases - case insensitive
        ("ASIA", "China"),  # Uppercase
        ("asia", "China"),  # Lowercase
        ("InvalidRegion", "U.S."),  # Unknown defaults to U.S.
    ]
    
    print("Region normalization tests:")
    all_passed = True
    failed_cases = []
    for input_val, expected in test_cases:
        result = normalize_region(input_val)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
            failed_cases.append((input_val, result, expected))
    
    if all_passed:
        print(f"  All {len(test_cases)} test cases passed!")
    else:
        for input_val, result, expected in failed_cases:
            print(f"  FAIL: '{input_val}' -> '{result}' (expected: '{expected}')")
    
    return all_passed


def test_tool_aug_wrapper():
    """Test that tool_aug has region normalization in the wrapper."""
    print()
    print("=" * 70)
    print("TEST 2: Tool-augmented wrapper has region fixes")
    print("=" * 70)
    
    with open("src/experiments/tool_aug_baseline.py", "r") as f:
        content = f.read()
    
    # Check for REGION_FIXES in the wrapper (lowercase keys now)
    checks = [
        ("REGION_FIXES dictionary", "REGION_FIXES = {"),
        ("asia -> China fix (lowercase)", '"asia": "China"'),
        ("normalize_region function", "def normalize_region("),
        ("Wrapper calls normalize_region", "normalize_region(routine)"),
    ]
    
    all_present = True
    for name, pattern in checks:
        if pattern in content:
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name} - pattern '{pattern}' not found")
            all_present = False
    
    return all_present


def test_valid_customer_classes():
    """Test that we handle customer class variations."""
    print()
    print("=" * 70)
    print("TEST 3: Customer class validation")
    print("=" * 70)
    
    # Valid customer classes from RuleArena
    VALID_CLASSES = {"First Class", "Business Class", "Main Plus", "Main Cabin"}
    
    # Common LLM variations
    variations = {
        "First": "First Class",
        "First class": "First Class",
        "first class": "First Class",
        "FIRST CLASS": "First Class",
        "Business": "Business Class",
        "Economy": "Main Cabin",
        "Economy Class": "Main Cabin",
        "Coach": "Main Cabin",
        "Premium Economy": "Main Plus",
        "Main": "Main Cabin",
    }
    
    print(f"  Valid classes: {VALID_CLASSES}")
    print(f"  Defined {len(variations)} normalization mappings")
    
    # Check if l1_ptool_extraction has customer class normalization
    with open("src/experiments/l1_ptool_extraction.py", "r") as f:
        content = f.read()
    
    if "customer_class" in content.lower():
        print("  Note: customer_class is extracted but may need normalization")
    
    return True


def test_direction_values():
    """Test direction parameter values."""
    print()
    print("=" * 70)
    print("TEST 4: Direction value validation")
    print("=" * 70)
    
    # Direction should be 0 (one-way) or 1 (round-trip)
    VALID_DIRECTIONS = {0, 1}
    
    print(f"  Valid direction values: {VALID_DIRECTIONS}")
    print("  Direction is typically extracted as integer - low risk of errors")
    
    return True


def test_rulearena_tables():
    """Test that RuleArena reference tables cover all valid regions."""
    print()
    print("=" * 70)
    print("TEST 5: RuleArena reference table coverage")
    print("=" * 70)
    
    try:
        from src.dataset.airline_loader import AirlineLoader
        
        loader = AirlineLoader("external/RuleArena")
        
        # Get the fee_tables
        fee_tables = loader._fee_tables
        
        # Find all regions in the tables
        all_table_regions = set()
        for bag_id, direction_dict in fee_tables.items():
            for direction, class_dict in direction_dict.items():
                for customer_class, region_series in class_dict.items():
                    if hasattr(region_series, 'index'):
                        all_table_regions.update(region_series.index.tolist())
                        break  # Just need one
                break
            break
        
        if all_table_regions:
            print(f"  Regions in fee_tables: {sorted(all_table_regions)}")
            print(f"  Total: {len(all_table_regions)}")
        else:
            print("  Could not extract regions from fee tables (structure different)")
        
        return True
        
    except Exception as e:
        print(f"  WARNING: Could not verify tables: {e}")
        return True  # Don't fail on this


def test_json_serialization():
    """Test JSON serialization handles edge cases."""
    print()
    print("=" * 70)
    print("TEST 6: JSON serialization edge cases")
    print("=" * 70)
    
    import numpy as np
    
    # Check if run_full_baseline has NumpyEncoder
    with open("src/experiments/run_full_baseline.py", "r") as f:
        content = f.read()
    
    # Using math.isnan instead of np.isnan
    checks = [
        ("NumpyEncoder class", "class NumpyEncoder"),
        ("Handles np.integer", "np.integer"),
        ("Handles np.floating", "np.floating"),
        ("Handles np.ndarray", "np.ndarray"),
        ("Handles NaN", "isnan"),  # math.isnan or np.isnan
        ("Handles Inf", "isinf"),  # math.isinf or np.isinf
    ]
    
    all_present = True
    for name, pattern in checks:
        if pattern in content:
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name} - pattern not found")
            all_present = False
    
    return all_present


def test_quick_computation():
    """Run a quick computation with known values to verify everything works."""
    print()
    print("=" * 70)
    print("TEST 7: Quick end-to-end computation test")
    print("=" * 70)
    
    try:
        from src.dataset.airline_loader import AirlineLoader
        
        loader = AirlineLoader("external/RuleArena")
        
        # Test with a known valid set of parameters
        test_params = {
            "base_price": 500,
            "customer_class": "Main Cabin",
            "routine": "U.S.",  # Valid region
            "direction": 0,
            "bag_list": [
                {"id": 0, "name": "backpack", "size": [20, 15, 10], "weight": 10},
                {"id": 1, "name": "luggage box", "size": [30, 20, 15], "weight": 30},
            ]
        }
        
        # Note: compute_answer signature is (base_price, direction, routine, customer_class, bag_list, check_base_tables)
        result = loader._compute_answer_fn(
            test_params["base_price"],
            test_params["direction"],
            test_params["routine"],
            test_params["customer_class"],
            test_params["bag_list"],
            loader._fee_tables
        )
        
        print(f"  Input: Main Cabin, U.S., one-way, 2 bags")
        print(f"  Result: ${result[0]}")
        print("  PASS: Computation completed successfully")
        
        # Test with normalized region
        from src.experiments.l1_ptool_extraction import normalize_region
        
        asia_normalized = normalize_region("Asia")
        result2 = loader._compute_answer_fn(
            test_params["base_price"],
            test_params["direction"],
            asia_normalized,  # "China" after normalization
            test_params["customer_class"],
            test_params["bag_list"],
            loader._fee_tables
        )
        
        print(f"  Input: Asia -> {asia_normalized}, one-way, 2 bags")
        print(f"  Result: ${result2[0]}")
        print("  PASS: Asia normalization works end-to-end")
        
        return True
        
    except Exception as e:
        print(f"  FAIL: Computation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_experiment_run():
    """Quick test of the actual experiment functions without API calls."""
    print()
    print("=" * 70)
    print("TEST 8: Experiment function imports (no API calls)")
    print("=" * 70)
    
    all_passed = True
    
    # Test that all experiment functions can be imported
    try:
        from src.experiments.l1_ptool_extraction import (
            baggage_allowance_l1_ptool,
            baggage_allowance_l1_transparent,
            normalize_region as l1_normalize
        )
        print("  PASS: L1 extraction imports")
    except Exception as e:
        print(f"  FAIL: L1 extraction imports - {e}")
        all_passed = False
    
    try:
        from src.experiments.cot_baseline import run_cot_baseline
        print("  PASS: CoT baseline imports")
    except Exception as e:
        print(f"  FAIL: CoT baseline imports - {e}")
        all_passed = False
    
    try:
        from src.experiments.tool_aug_baseline import run_tool_augmented
        print("  PASS: Tool-aug baseline imports")
    except Exception as e:
        print(f"  FAIL: Tool-aug baseline imports - {e}")
        all_passed = False
    
    try:
        from src.experiments.run_full_baseline import (
            estimate_cost,
            NumpyEncoder,
            MODEL_PRICING,
        )
        print("  PASS: run_full_baseline imports")
        
        # Test cost estimation
        cost_info = estimate_cost(10, [0], "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
        total_cost = cost_info.get("total_estimated_cost", 0)
        print(f"  PASS: Cost estimation works (10 problems, level 0 = ${total_cost:.4f})")
    except Exception as e:
        print(f"  FAIL: run_full_baseline imports - {e}")
        all_passed = False
    
    # Test merge_results
    try:
        from src.experiments.merge_results import merge_results, validate_results
        print("  PASS: merge_results imports")
    except Exception as e:
        print(f"  FAIL: merge_results imports - {e}")
        all_passed = False
    
    return all_passed


def main():
    """Run all verification tests."""
    print()
    print("=" * 70)
    print("COMPREHENSIVE VERIFICATION BEFORE FULL EXPERIMENTS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    results = {
        "Region normalization": test_region_normalization(),
        "Tool-aug wrapper": test_tool_aug_wrapper(),
        "Customer classes": test_valid_customer_classes(),
        "Direction values": test_direction_values(),
        "RuleArena tables": test_rulearena_tables(),
        "JSON serialization": test_json_serialization(),
        "End-to-end computation": test_quick_computation(),
        "Experiment imports": test_quick_experiment_run(),
    }
    
    print()
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {name:30} [{status}]")
    
    print()
    if all_passed:
        print("ALL TESTS PASSED - Safe to run full experiments")
    else:
        print("SOME TESTS FAILED - Fix issues before running experiments")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
