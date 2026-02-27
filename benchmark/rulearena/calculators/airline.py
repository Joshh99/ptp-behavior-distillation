"""
Airline Baggage Fee Calculator

Wrapper around RuleArena's reference implementation for computing ground truth.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Import vendored reference implementation (local copy in benchmark/rulearena/)
BENCHMARK_ROOT = Path(__file__).parent.parent
REPO_ROOT = BENCHMARK_ROOT.parent.parent

# Add benchmark root to path to import rulearena_reference
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

try:
    import rulearena_reference
    FEE_TABLES = rulearena_reference.load_checking_fee(str(REPO_ROOT / "external" / "RuleArena"))
    print(f"[OK] Loaded airline fee tables ({len(FEE_TABLES)} tables)")
except Exception as e:
    print(f"Warning: Could not load airline fee tables: {e}")
    FEE_TABLES = None


def compute_airline_fee(info: Dict[str, Any]) -> int:
    """
    Compute airline baggage fee from problem info dict.

    Args:
        info: Problem info dict with keys: base_price, direction, routine,
              customer_class, bag_list

    Returns:
        Total cost (ticket price + baggage fees)
    """
    if FEE_TABLES is None:
        raise RuntimeError("Fee tables not loaded")

    total_cost, _ = rulearena_reference.compute_answer(
        base_price=info['base_price'],
        direction=info['direction'],
        routine=info['routine'],
        customer_class=info['customer_class'],
        bag_list=info['bag_list'],
        check_base_tables=FEE_TABLES,
    )

    return total_cost
