"""
Python Calculator Layer with backoff detection.

Wraps the existing calculators.py with backoff signals:
- Identifies when Python can handle the calculation
- Signals backoff for rule-based calculators or extraction failures
- Provides confidence estimates for routing decisions
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure benchmark experiments path is available
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from ..base import LayerResult, BackoffReason
from .config import MedCalcContinualConfig
from .pattern_store import RULE_BASED_CALCULATORS


class PythonCalculatorLayer:
    """
    Wraps calculators.py with backoff detection.

    This layer attempts pure Python calculation and signals when
    the LLM pipeline should take over.

    Backoff triggers:
    1. Rule-based calculator identified (needs medical reasoning)
    2. Python extraction returns None (regex failed)
    3. Calculator not identified
    4. Python computation error
    5. Official calculator validation fails (results don't match)
    """

    def __init__(self, config: Optional[MedCalcContinualConfig] = None):
        """
        Initialize Python layer.

        Args:
            config: Optional MedCalc continual config
        """
        self.config = config or MedCalcContinualConfig()

        # Import calculator functions (lazy to avoid circular imports)
        self._identify_calculator = None
        self._python_calculate = None
        self._official_compute = None
        self._validate_extracted = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of calculator imports."""
        if self._initialized:
            return

        try:
            from benchmark.medcalc.experiments.calculators import (
                identify_calculator,
                calculate as python_calculate,
            )
            self._identify_calculator = identify_calculator
            self._python_calculate = python_calculate

            # Import official calculator validation from L4
            from benchmark.medcalc.experiments.l4_pipeline import (
                compute_with_python,
                validate_extracted_values,
            )
            from benchmark.medcalc.experiments.official_calculators import (
                convert_extracted_to_official,
            )
            self._official_compute = compute_with_python
            self._validate_extracted = validate_extracted_values
            self._convert_to_official = convert_extracted_to_official

            self._initialized = True
        except ImportError as e:
            raise ImportError(
                f"Could not import calculators.py: {e}. "
                "Make sure benchmark/experiments/calculators.py is accessible."
            )

    def _validate_with_official(
        self,
        calculator_name: str,
        extracted_values: Dict[str, Any],
        python_result: float,
    ) -> Tuple[bool, Optional[float]]:
        """
        Validate Python result against official calculator.

        Args:
            calculator_name: Name of the calculator
            extracted_values: Values extracted by Python layer
            python_result: Result from Python calculator

        Returns:
            (is_valid, official_result) - is_valid is True if official calc agrees
        """
        try:
            # First convert Python layer's extracted values to official format
            # This adds units like [value, "unit"] that official calculators expect
            converted = self._convert_to_official(extracted_values, calculator_name)

            # Validate and clean extracted values
            is_valid, missing, cleaned = self._validate_extracted(
                extracted=converted,
                calculator_name=calculator_name,
            )

            if not is_valid or missing:
                return False, None

            # Compute with official calculator
            official_result = self._official_compute(calculator_name, cleaned)

            if official_result is None or official_result.result is None:
                return False, None

            # Check if results are close enough (within 1% or absolute 0.01)
            official_val = official_result.result
            if abs(python_result - official_val) < max(0.01, abs(official_val) * 0.01):
                return True, official_val

            # Results don't match - prefer official
            return False, official_val

        except Exception:
            # Official calc failed - Python result is unverified
            return False, None

    def _is_rule_based(self, calculator_name: str) -> bool:
        """Check if calculator requires medical reasoning."""
        if not calculator_name:
            return False

        calc_lower = calculator_name.lower()

        # Check against known rule-based calculators
        for rule_calc in RULE_BASED_CALCULATORS:
            if calc_lower in rule_calc.lower() or rule_calc.lower() in calc_lower:
                return True

        # Also check for keywords indicating scoring systems
        rule_keywords = [
            'score', 'criteria', 'index', 'risk', 'scale',
            'cha2ds2', 'heart', 'wells', 'curb', 'sofa',
            'apache', 'child-pugh', 'meld', 'centor', 'perc',
            'has-bled', 'rcri', 'charlson', 'caprini', 'gcs',
            'blatchford', 'feverpain', 'sirs', 'framingham'
        ]

        for keyword in rule_keywords:
            if keyword in calc_lower:
                return True

        return False

    def try_calculate(
        self,
        question: str,
        patient_note: str,
    ) -> LayerResult:
        """
        Try Python calculation, return backoff signal if needed.

        Args:
            question: The calculation question
            patient_note: Patient note with clinical data

        Returns:
            LayerResult with success or backoff information
        """
        self._ensure_initialized()

        # Step 1: Identify calculator
        try:
            calculator_name = self._identify_calculator(question)
        except Exception as e:
            return LayerResult(
                success=False,
                backoff_reason=BackoffReason.PYTHON_ERROR,
                backoff_details=f"Calculator identification error: {type(e).__name__}: {e}",
            )

        if calculator_name is None:
            return LayerResult(
                success=False,
                backoff_reason=BackoffReason.CALCULATOR_NOT_IDENTIFIED,
                backoff_details="Python layer could not identify which calculator is needed",
            )

        # Step 2: Check if rule-based (always back off if configured)
        if self.config.backoff_on_rule_based and self._is_rule_based(calculator_name):
            return LayerResult(
                success=False,
                identified_entity=calculator_name,
                backoff_reason=BackoffReason.RULE_BASED_CALCULATOR,
                backoff_details=f"'{calculator_name}' is a rule-based scoring system requiring medical reasoning",
            )

        # Step 3: Try Python extraction and calculation
        try:
            result = self._python_calculate(patient_note, question)

            if result is None:
                return LayerResult(
                    success=False,
                    identified_entity=calculator_name,
                    backoff_reason=BackoffReason.EXTRACTION_FAILED,
                    backoff_details="Python regex extraction could not find required values",
                )

            # Check confidence from result
            confidence = getattr(result, 'confidence', 1.0)
            if confidence < self.config.extraction_confidence_threshold:
                return LayerResult(
                    success=False,
                    identified_entity=calculator_name,
                    extracted_values=result.extracted_values if hasattr(result, 'extracted_values') else None,
                    backoff_reason=BackoffReason.LOW_CONFIDENCE,
                    backoff_details=f"Extraction confidence {confidence:.2f} below threshold {self.config.extraction_confidence_threshold}",
                )

            # Success! L2's calculator implementations are well-tested and match
            # the official formulas, so we trust the result directly.
            return LayerResult(
                success=True,
                identified_entity=calculator_name,
                result=result.result,
                extracted_values=result.extracted_values if hasattr(result, 'extracted_values') else {},
                method="python",
            )

        except Exception as e:
            return LayerResult(
                success=False,
                identified_entity=calculator_name,
                backoff_reason=BackoffReason.PYTHON_ERROR,
                backoff_details=f"Python calculation error: {type(e).__name__}: {e}",
            )

    def get_calculator_category(self, calculator_name: str) -> str:
        """Get category of calculator (equation_based or rule_based)."""
        if self._is_rule_based(calculator_name):
            return "rule_based"
        return "equation_based"
