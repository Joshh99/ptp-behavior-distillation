"""
Tax Fee Calculator

Wrapper around RuleArena's reference implementation for computing ground truth.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

REPO_ROOT = Path(__file__).parent.parent.parent.parent

# RuleArena tax modules use bare imports (e.g. "from structured_forms import ..."),
# so we need the tax directory on sys.path.
TAX_DIR = str(REPO_ROOT / "external" / "RuleArena" / "tax")
if TAX_DIR not in sys.path:
    sys.path.insert(0, TAX_DIR)

try:
    from structured_forms import TaxPayer
    from micro_evaluation import compute_answer
    print("[OK] Loaded RuleArena tax reference implementation")
except Exception as e:
    print(f"Warning: Could not load tax reference implementation: {e}")
    TaxPayer = None
    compute_answer = None


def compute_tax_fee(info: Dict[str, Any]) -> Optional[float]:
    """
    Compute tax amount from a RuleArena tax problem dict.

    Args:
        info: A single problem dict from the synthesized_problems JSON.
              Must contain a "pydantic" key whose value is the TaxPayer fields.

    Returns:
        The computed amount owed (positive) or overpaid (negative) as a float,
        or None if computation fails.
    """
    if TaxPayer is None or compute_answer is None:
        print("Error: Tax reference implementation not loaded")
        return None

    try:
        tp = TaxPayer(**info["pydantic"])
        amount, _ = compute_answer(tp)
        return float(amount)
    except Exception as e:
        print(f"Error computing tax fee: {e}")
        return None
