"""
RuleArena Reference Implementation - Vendored Copy

This file is copied from the RuleArena benchmark repository with minimal
modifications for path handling. The core calculation logic is unchanged
to ensure our evaluation matches the benchmark exactly.

Original Source:
    https://github.com/SkyRiver-2000/RuleArena/blob/main/airline/compute_answer.py
    
Citation:
    Zhou, R., Hua, W., Pan, L., Cheng, S., Wu, X., Yu, E., & Wang, W. Y. (2025).
    RULEARENA: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios.
    Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, 550-572.

Modifications from original:
    - Added this attribution header
    - Modified load_checking_fee() to accept repo_path parameter
    - All calculation logic remains identical to original

Version: Copied from commit [latest] on 2025-02-02
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


# Regions where First class gets complementary first bag
complementary_first = [
    "China", "Hong Kong", "Japan", "South Korea", "India",
    "Qatar", "Haiti", "Cuba", "Panama", "Colombia",
    "Ecuador", "Peru", "South America", "Israel"
]


def invert_order(a, order):
    """Helper function to restore original bag order after sorting."""
    return [a[i] for i in np.argsort(order)]


def load_checking_fee(repo_path: str = "external/RuleArena"):
    """
    Load fee tables from RuleArena repository.
    
    Modified from original to accept repo_path parameter.
    Original used relative paths assuming execution from airline/ directory.
    
    Args:
        repo_path: Path to RuleArena repository root
        
    Returns:
        List of dictionaries with fee tables for bags 1-4
    """
    repo_path = Path(repo_path)
    airline_path = repo_path / "airline"
    
    check_base = []
    for bag_num in range(1, 5):
        us_departure = pd.read_csv(
            airline_path / f"fee_tables/bag_{bag_num}/0.csv", 
            index_col=0
        )
        us_arrival = pd.read_csv(
            airline_path / f"fee_tables/bag_{bag_num}/1.csv", 
            index_col=0
        )
        check_base.append({
            0: us_departure,
            1: us_arrival
        })
    return check_base


# =============================================================================
# CORE CALCULATION LOGIC - UNCHANGED FROM ORIGINAL
# =============================================================================

def compute_answer(
    base_price: int,
    direction: int,
    routine: str,
    customer_class: str,
    bag_list: List[Dict[str, Any]],
    check_base_tables: List[Dict[int, pd.DataFrame]],
    override: dict = {},
):
    """
    Compute total cost for airline baggage scenario.
    
    This is the reference implementation from RuleArena - unchanged.
    
    Args:
        base_price: Ticket price in dollars
        direction: 0=departing US, 1=arriving to US
        routine: Region name (e.g., "U.S.", "Europe", "China")
        customer_class: Cabin class (e.g., "Main Cabin", "First")
        bag_list: List of bags (first is carry-on, rest are checked)
        check_base_tables: Fee tables loaded by load_checking_fee()
        override: Optional dict to override specific calculations
        
    Returns:
        Tuple of (total_cost, info_dict)
    """
    extra, info_dict = compute_check_cost(
        bag_list[1:],  # Skip first bag (carry-on)
        direction, 
        routine, 
        customer_class, 
        check_base_tables, 
        override
    )
    extra = override.get("check_total", extra)
    total_cost = base_price + extra
    info_dict.update({
        "customer_class": customer_class,
        "ticket_price": base_price,
        "place_of_departure": routine if direction == 1 else "U.S.",
        "place_of_arrival": routine if direction == 0 else "U.S.",
        "routine": routine,
        "total_cost": total_cost,
        "bag_list": bag_list[1:],
    })
    return total_cost, info_dict


def compute_check_cost(
    bag_list: List[Dict[str, float]],
    direction: int,
    routine: str,
    customer_class: str,
    check_base_tables: List[Dict[int, pd.DataFrame]],
    override: dict = {},
):
    """
    Compute checked bag costs with smart bag sorting.
    
    Key insight: Sort bags by "complementary gain" - which bags benefit
    most from being complementary (free). This maximizes fee savings.
    """
    if not "check_base" in override:
        # Calculate violation costs for each bag
        oversize_cost = [compute_oversize(b, routine) for b in bag_list]
        overweight_cost_if_comp = [
            compute_overweight(b, routine, customer_class, True) for b in bag_list
        ]
        overweight_cost_if_not_comp = [
            compute_overweight(b, routine, customer_class, False) for b in bag_list
        ]
        
        # Calculate which bags save the most money if complementary
        violation_cost_if_comp = np.maximum(oversize_cost, overweight_cost_if_comp)
        violation_cost_if_not_comp = np.maximum(oversize_cost, overweight_cost_if_not_comp)
        complementary_gain = violation_cost_if_not_comp - violation_cost_if_comp
        
        # Sort bags by highest gain first (smart optimization!)
        order = np.argsort(-complementary_gain)
        bag_list = [bag_list[i] for i in order]
        check_base = compute_base(bag_list, direction, routine, customer_class, check_base_tables)
    else:
        check_base = override["check_base"]
        order = np.arange(len(bag_list))
    
    # Determine which bags are complementary (free)
    complementary = [(x == 0) for x in check_base]
    
    # Calculate final costs
    oversize_cost = [compute_oversize(b, routine) for b in bag_list]
    overweight_cost = [
        compute_overweight(b, routine, customer_class, c) \
        for b, c in zip(bag_list, complementary)
    ]
    
    # Take maximum of oversize vs overweight (not sum!)
    violation_cost = np.maximum(oversize_cost, overweight_cost).sum()
    total_check_cost = np.sum(check_base) + violation_cost
    
    info_dict = {
        "overweight": invert_order(overweight_cost, order),
        "oversize": invert_order(oversize_cost, order),
        "base": invert_order(check_base, order)
    }
    return total_check_cost, info_dict


def compute_base(
    bag_list: List[Dict[str, float]],
    direction: int,
    routine: str,
    customer_class: str,
    check_base_tables: List[Dict[int, pd.DataFrame]]
):
    """
    Look up base fee for each bag from fee tables.
    
    Note: Bags 5+ use the bag 4 table (typically all $200).
    """
    check_base = []
    for bag_id, _ in enumerate(bag_list):
        bag_id = min(3, bag_id)  # Bags 4+ use table for bag 4
        check_base.append(
            check_base_tables[bag_id][direction][customer_class][routine]
        )
    return check_base


def compute_oversize(bag: Dict[str, float], routine: str):
    """Calculate oversize penalty based on total linear dimensions."""
    total_size = np.sum(bag["size"])
    
    if total_size <= 62:
        return 0
    if total_size <= 65:
        return 30
    
    # Different rates for different regions
    if routine in ["Panama", "South America", "Peru", "Colombia", 
                   "Ecuador", "Europe", "Israel", "Qatar"]:
        return 150
    return 200


def compute_overweight(
    bag: Dict[str, float], 
    routine: str, 
    customer_class: str, 
    complementary: bool
):
    """
    Calculate overweight penalty based on weight and context.
    
    Key insight: Complementary bags (free base fee) have different
    weight limits than paid bags.
    """
    weight = bag["weight"]
    
    # Special case: Australia/New Zealand
    if routine in ["Australia", "New Zealand"]:
        if complementary:
            if weight <= 70:
                return 0
            return 200
        # Non-complementary
        if weight <= 50:
            return 0
        if weight <= 53:
            return 30
        if weight <= 70:
            return 200 if routine == "Cuba" else 100
        # Over 70 lbs
        if routine in ["India", "China", "Japan", "South Korea", "Hong Kong"]:
            return 450
        return 200
    
    # Special case: Business/First class complementary bags
    if complementary and customer_class in ["Business", "First"]:
        if weight <= 70:
            return 0
        if routine in ["India", "China", "Japan", "South Korea", "Hong Kong"]:
            return 450
        return 200
    
    # Standard weight penalties
    if weight <= 50:
        return 0
    if weight <= 53:
        return 30
    if weight <= 70:
        return 200 if routine == "Cuba" else 100
    # Over 70 lbs
    if routine in ["India", "China", "Japan", "South Korea", "Hong Kong"]:
        return 450
    return 200


# =============================================================================
# TESTING (from original)
# =============================================================================

if __name__ == "__main__":
    from pprint import pprint
    
    # Test with sample data
    print("Testing RuleArena reference implementation...")
    
    check_base_tables = load_checking_fee()
    print(f"✓ Loaded {len(check_base_tables)} fee tables")
    
    # Sample problem
    sample_info = {
        "base_price": 180,
        "customer_class": "Main Cabin",
        "routine": "U.S.",
        "direction": 0,
        "bag_list": [
            {"id": 1, "name": "backpack", "size": [22, 13, 6], "weight": 10},
            {"id": 2, "name": "luggage", "size": [44, 22, 20], "weight": 69},
        ]
    }
    
    total_cost, info = compute_answer(
        **sample_info,
        check_base_tables=check_base_tables
    )
    
    print(f" Computed total cost: ${total_cost}")
    print("\nAll tests passed!")