"""
Guided Pipeline: L4 Pipeline with learned pattern injection.

Extends the L4 pipeline experiment to inject learned patterns
as guidance into each stage's prompts.

Pattern injection points:
- Stage 1 (Calculator ID): Inject keyword→calculator mappings
- Stage 2 (Extraction): Inject calculator-specific extraction examples
- Stage 2a (Reasoning): Inject condition→finding mappings for scoring systems
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from ..base import Pattern, PatternStage, LayerResult, BackoffReason
from .config import MedCalcContinualConfig
from .pattern_store import MedCalcPatternStore

from ptool_framework.llm_backend import call_llm, reset_token_accumulator, get_token_accumulator


class GuidedPipeline:
    """
    L4 Pipeline that accepts learned patterns as guidance.

    Unlike the standalone L4PipelineExperiment, this class focuses on
    the core pipeline stages and accepts pattern injection for each.
    """

    def __init__(
        self,
        config: MedCalcContinualConfig,
        pattern_store: MedCalcPatternStore,
        model: str = "deepseek-v3-0324",
    ):
        """
        Initialize guided pipeline.

        Args:
            config: MedCalc continual config
            pattern_store: Pattern store for retrieving learned patterns
            model: LLM model to use
        """
        self.config = config
        self.pattern_store = pattern_store
        self.model = model

    def reset_tokens(self) -> None:
        """Reset token counters."""
        reset_token_accumulator()

    def get_tokens(self) -> Tuple[int, int]:
        """Get input and output token counts from accumulator."""
        acc = get_token_accumulator()
        return acc.total_prompt_tokens, acc.total_completion_tokens

    def run(
        self,
        question: str,
        patient_note: str,
        calculator_hint: Optional[str] = None,
    ) -> LayerResult:
        """
        Run the full guided pipeline.

        Args:
            question: The calculation question
            patient_note: Patient note with clinical data
            calculator_hint: Optional calculator name from Python layer

        Returns:
            LayerResult with computation result
        """
        self.reset_tokens()

        try:
            # Stage 1: Identify calculator (or use hint)
            if calculator_hint:
                calculator_name = calculator_hint
            else:
                calculator_name = self._identify_calculator_with_guidance(question)

            if not calculator_name:
                return LayerResult(
                    success=False,
                    backoff_reason=BackoffReason.CALCULATOR_NOT_IDENTIFIED,
                    backoff_details="Pipeline could not identify calculator",
                    input_tokens=get_token_accumulator().total_prompt_tokens,
                    output_tokens=get_token_accumulator().total_completion_tokens,
                )

            # Stage 2: Extract values with guidance
            extracted = self._extract_values_with_guidance(
                patient_note=patient_note,
                calculator_name=calculator_name,
            )

            if not extracted:
                return LayerResult(
                    success=False,
                    identified_entity=calculator_name,
                    backoff_reason=BackoffReason.EXTRACTION_FAILED,
                    backoff_details="Pipeline extraction failed",
                    input_tokens=get_token_accumulator().total_prompt_tokens,
                    output_tokens=get_token_accumulator().total_completion_tokens,
                )

            # Stage 3: Compute result using official calculators
            result = self._compute_result(calculator_name, extracted)

            if result is None:
                return LayerResult(
                    success=False,
                    identified_entity=calculator_name,
                    extracted_values=extracted,
                    backoff_reason=BackoffReason.PYTHON_ERROR,
                    backoff_details="Pipeline computation failed",
                    input_tokens=get_token_accumulator().total_prompt_tokens,
                    output_tokens=get_token_accumulator().total_completion_tokens,
                )

            return LayerResult(
                success=True,
                identified_entity=calculator_name,
                result=result,
                extracted_values=extracted,
                method="pipeline",
                input_tokens=get_token_accumulator().total_prompt_tokens,
                output_tokens=get_token_accumulator().total_completion_tokens,
            )

        except Exception as e:
            return LayerResult(
                success=False,
                backoff_reason=BackoffReason.PYTHON_ERROR,
                backoff_details=f"Pipeline error: {type(e).__name__}: {e}",
                input_tokens=get_token_accumulator().total_prompt_tokens,
                output_tokens=get_token_accumulator().total_completion_tokens,
            )

    def _identify_calculator_with_guidance(
        self,
        question: str,
    ) -> Optional[str]:
        """
        Stage 1: Identify calculator using L4's proven identification function.

        Uses L4's identify_calculator_l4 ptool for consistency with L4 baseline.
        """
        try:
            # Use L4's identification function
            from benchmark.medcalc.experiments.l4_pipeline import identify_calculator_l4
            from benchmark.medcalc.experiments.calculator_simple import (
                get_calculator_signatures,
                CALCULATOR_REGISTRY,
                CalculatorSpec,
            )
            from benchmark.medcalc.experiments.calculators import identify_calculator as python_identify_calculator

            # Get all 55 calculator signatures from calculator_simple
            signatures = get_calculator_signatures()
            available = list(signatures.keys())

            # Call L4's identification ptool
            result = identify_calculator_l4(
                question=question,
                available_calculators=available,
            )

            if isinstance(result, dict):
                # Handle nested {"result": {...}} structure from ptool
                inner = result.get("result", result)
                if isinstance(inner, dict):
                    calc_name = inner.get("calculator_name")
                    confidence = inner.get("confidence", 0.5)
                else:
                    calc_name = result.get("calculator_name")
                    confidence = result.get("confidence", 0.5)

                # Validate calculator name exists
                if calc_name in signatures:
                    return calc_name

                # Also check CALCULATOR_REGISTRY which includes aliases
                if calc_name in CALCULATOR_REGISTRY:
                    # Get canonical name
                    spec = CALCULATOR_REGISTRY[calc_name]
                    return spec.name

                # Try fuzzy matching
                calc_lower = calc_name.lower() if calc_name else ""
                for sig_name in signatures:
                    if calc_lower in sig_name.lower() or sig_name.lower() in calc_lower:
                        return sig_name

                # Also check aliases for fuzzy match
                for name, spec in CALCULATOR_REGISTRY.items():
                    if isinstance(spec, CalculatorSpec):
                        for alias in spec.aliases:
                            if calc_lower in alias.lower() or alias.lower() in calc_lower:
                                return spec.name

            # LLM failed, try Python fallback
            calc_pattern = python_identify_calculator(question)
            if calc_pattern:
                # Map pattern to full name via registry
                for name, spec in CALCULATOR_REGISTRY.items():
                    if isinstance(spec, CalculatorSpec):
                        if calc_pattern.lower() in name.lower() or calc_pattern.lower() in spec.name.lower():
                            return spec.name
                        for alias in spec.aliases:
                            if calc_pattern.lower() in alias.lower():
                                return spec.name

            return None

        except Exception as e:
            # Fall back to our simple identification
            try:
                from benchmark.medcalc.experiments.calculators import identify_calculator as python_identify_calculator
                return python_identify_calculator(question)
            except Exception:
                return None

    def _extract_values_with_guidance(
        self,
        patient_note: str,
        calculator_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Stage 2: Extract values using L4's proven extraction function.

        Uses L4's extract_values_two_stage for consistency with L4 baseline.
        """
        try:
            # Use L4's extraction function directly for consistency
            from benchmark.medcalc.experiments.l4_pipeline import extract_values_two_stage
            from benchmark.medcalc.experiments.calculator_simple import get_calculator_signatures

            # Get required/optional values from signatures
            signatures = get_calculator_signatures()
            sig = signatures.get(calculator_name, {})
            required = sig.get("required", [])
            optional = sig.get("optional", [])

            # Use L4's proven two-stage extraction
            result = extract_values_two_stage(
                patient_note=patient_note,
                calculator_name=calculator_name,
                required_values=required,
                optional_values=optional,
            )

            if isinstance(result, dict):
                extracted = result.get("extracted", {})
                return extracted if extracted else None

            return None

        except Exception as e:
            # Fall back to our extraction if L4's fails
            # Get learned extraction patterns for this calculator
            extraction_patterns = self.pattern_store.get_patterns_for_entity(
                entity=calculator_name,
                stage=PatternStage.EXTRACTION,
                min_confidence=self.config.min_pattern_confidence,
            )

            # Also get reasoning patterns for scoring systems
            reasoning_patterns = self.pattern_store.get_patterns_for_entity(
                entity=calculator_name,
                stage=PatternStage.REASONING,
                min_confidence=self.config.min_pattern_confidence,
            )

            guidance_text = self._format_extraction_guidance(
                extraction_patterns,
                reasoning_patterns,
            )

            # Check if this is a scoring system (needs medical reasoning)
            is_scoring = self.pattern_store.is_rule_based(calculator_name)

            if is_scoring:
                # Two-stage extraction for scoring systems
                return self._extract_scoring_system(
                    patient_note,
                    calculator_name,
                    guidance_text,
                )
            else:
                # Direct extraction for equation-based calculators
                return self._extract_equation_values(
                    patient_note,
                    calculator_name,
                    guidance_text,
                )

    def _format_extraction_guidance(
        self,
        extraction_patterns: List[Pattern],
        reasoning_patterns: List[Pattern],
    ) -> str:
        """Format learned patterns as few-shot guidance text."""
        sections = []

        if extraction_patterns:
            sections.append("## Learned extraction patterns for this calculator:")
            for p in extraction_patterns[:5]:
                if p.examples:
                    for ex in p.examples[:2]:
                        sections.append(f"""
Parameter: {p.metadata.get('param_name', 'unknown')}
- Pattern: {p.content}
- Example: "{ex.get('input', '')[:100]}" → {ex.get('output', '')}
""")
                else:
                    sections.append(f"- {p.content}")

        if reasoning_patterns:
            sections.append("\n## Known clinical finding → condition mappings:")
            for p in reasoning_patterns[:5]:
                sections.append(f"- {p.content}")
                if p.examples:
                    for ex in p.examples[:1]:
                        sections.append(f"  Example: {ex.get('input', '')} → {ex.get('output', '')}")

        return "\n".join(sections) if sections else ""

    def _extract_equation_values(
        self,
        patient_note: str,
        calculator_name: str,
        guidance_text: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract values for equation-based calculators."""
        # Get official calculator source for parameter format guidance
        try:
            from benchmark.medcalc.experiments.official_calculators import (
                get_official_source,
                get_expected_params,
            )
            official_source = get_official_source(calculator_name)
            expected_params = get_expected_params(calculator_name)
        except Exception:
            official_source = None
            expected_params = []

        # Build official source reference
        source_reference = ""
        if official_source:
            source_reference = f"""
OFFICIAL CALCULATOR IMPLEMENTATION (use EXACT parameter names from this code):
```python
{official_source[:2000]}
```

EXPECTED PARAMETERS: {', '.join(expected_params) if expected_params else 'Check the code above'}
"""

        prompt = f"""Extract values from this patient note for the {calculator_name} calculator.

{guidance_text}
{source_reference}

PATIENT NOTE:
{patient_note}

EXTRACTION INSTRUCTIONS:
1. **Use EXACT parameter names from the official implementation** (e.g., heart_rate, qt_interval, creatinine)
2. **NUMERIC values ALWAYS with units** as [value, "unit"]:
   - age → [actual_age, "years"] e.g., [33, "years"]
   - weight → [value, "kg"] (convert lbs × 0.453592)
   - height → [value, "cm"] (convert feet × 30.48 + inches × 2.54)
   - heart_rate → [value, "bpm"]
   - qt_interval → [value, "msec"]
   - creatinine → [value, "mg/dL"]
   - sodium → [value, "mEq/L"]
   - bun → [value, "mg/dL"]
   - glucose → [value, "mg/dL"]
   - pH → [value, ""]
   - pao2 → [value, "mmHg"]
   - temperature → [value, "°F"] or [value, "°C"]
   - sys_bp, dia_bp → [value, "mmHg"]
   - ast, alt → [value, "U/L"]
   - platelets → [value, "10^9/L"]
   - hemoglobin → [value, "g/dL"]
   - inr → [value, ""]
   - albumin → [value, "g/dL"]
   - bilirubin → [value, "mg/dL"]

3. Extract demographic info:
   - sex → "Male" or "Female"

COMMON CONVERSIONS:
- Temperature: °C × 9/5 + 32 = °F, or °F - 32 × 5/9 = °C
- lbs × 0.453592 = kg
- feet × 30.48 + inches × 2.54 = cm

Return ONLY valid JSON:
{{"extracted": {{"param_name": [value, "unit"], ...}}}}
"""

        try:
            response = call_llm(prompt=prompt, model=self.model, max_tokens=800).content

            return self._parse_extraction_response(response)

        except Exception:
            return None

    def _extract_scoring_system(
        self,
        patient_note: str,
        calculator_name: str,
        guidance_text: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Two-stage extraction for scoring systems.

        Stage 2a: Medical reasoning to infer conditions
        Stage 2b: Extract values with inferred context
        """
        # Get official calculator source for parameter format guidance
        try:
            from benchmark.medcalc.experiments.official_calculators import (
                get_official_source,
                get_expected_params,
            )
            official_source = get_official_source(calculator_name)
            expected_params = get_expected_params(calculator_name)
        except Exception:
            official_source = None
            expected_params = []

        # Stage 2a: Medical reasoning
        reasoning_prompt = f"""You are a medical expert analyzing a patient note for the {calculator_name}.

{guidance_text}

PATIENT NOTE:
{patient_note}

TASK: Identify ALL conditions/criteria relevant to this calculator.

CRITICAL - Look for:
1. **Explicit mentions**: "history of diabetes", "has hypertension"
2. **Clinical findings that IMPLY conditions**:
   - DWI/MRI showing infarcts → STROKE
   - Neurological deficits (dysarthria, hemiparesis) → STROKE/TIA
   - Troponin elevation, ST changes → vascular disease
   - Edema, JVD, reduced EF → CHF
   - On anticoagulation → consider prior DVT/PE/AF
3. **Negations**: "no history of diabetes" → diabetes = false

Return JSON:
{{
    "reasoning": "Your step-by-step analysis",
    "conditions_present": ["list of conditions present"],
    "conditions_absent": ["list of conditions explicitly denied"],
    "demographics": {{"age": number, "sex": "male/female"}}
}}
"""

        try:
            reasoning_response = call_llm(prompt=reasoning_prompt, model=self.model, max_tokens=1000).content

            try:
                reasoning_result = json.loads(reasoning_response)
            except json.JSONDecodeError:
                match = re.search(r'\{[\s\S]*\}', reasoning_response, re.DOTALL)
                reasoning_result = json.loads(match.group()) if match else {}

        except Exception:
            reasoning_result = {}

        # Build context from reasoning
        conditions_present = reasoning_result.get("conditions_present", [])
        conditions_absent = reasoning_result.get("conditions_absent", [])
        demographics = reasoning_result.get("demographics", {})
        reasoning_text = reasoning_result.get("reasoning", "")

        # Build official source reference
        source_reference = ""
        if official_source:
            source_reference = f"""
OFFICIAL CALCULATOR IMPLEMENTATION (use EXACT parameter names from this code):
```python
{official_source[:2500]}
```

EXPECTED PARAMETERS: {', '.join(expected_params) if expected_params else 'Check the code above'}
"""

        # Stage 2b: Extract values with reasoning context
        reasoning_context = f"""
MEDICAL ANALYSIS (from reasoning stage):
- Conditions PRESENT: {', '.join(conditions_present) if conditions_present else 'None identified'}
- Conditions ABSENT: {', '.join(conditions_absent) if conditions_absent else 'None denied'}
- Demographics: Age={demographics.get('age', 'unknown')}, Sex={demographics.get('sex', 'unknown')}
- Reasoning: {reasoning_text[:500]}

Use this analysis to inform extraction. Inferred conditions should be reflected as boolean parameters.
"""

        extraction_prompt = f"""Extract values for the {calculator_name} scoring system.

{reasoning_context}
{source_reference}

PATIENT NOTE:
{patient_note}

INSTRUCTIONS:
1. **Use EXACT parameter names from the official implementation**
2. Map conditions to boolean parameters (true/false)
3. Extract age as [number, "years"]
4. Extract sex as "Male" or "Female"
5. For categorical scores (e.g., HEART history), use exact category names from the code

Return ONLY valid JSON:
{{"extracted": {{"param_name": value, ...}}, "inferred": ["conditions inferred from findings"]}}
"""

        try:
            response = call_llm(prompt=extraction_prompt, model=self.model, max_tokens=800).content

            extracted = self._parse_extraction_response(response)

            # Inject conditions from reasoning if extraction missed them
            if extracted and conditions_present:
                extracted = self._inject_inferred_conditions(
                    extracted,
                    conditions_present,
                    calculator_name,
                )

            return extracted

        except Exception:
            return None

    def _parse_extraction_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM extraction response into dict."""
        try:
            result = json.loads(response)
            return result.get("extracted", result)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', response, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    return result.get("extracted", result)
                except json.JSONDecodeError:
                    pass
            return None

    def _inject_inferred_conditions(
        self,
        extracted: Dict[str, Any],
        conditions_present: List[str],
        calculator_name: str,
    ) -> Dict[str, Any]:
        """Inject inferred conditions into extraction result."""
        # Condition mapping for scoring systems
        condition_mapping = {
            # CHA2DS2-VASc
            "stroke": ["stroke", "stroke_tia", "has_stroke_tia"],
            "tia": ["tia", "stroke_tia", "has_stroke_tia"],
            "hypertension": ["hypertension", "has_hypertension", "htn"],
            "diabetes": ["diabetes", "has_diabetes", "diabetes_mellitus"],
            "chf": ["chf", "has_chf", "heart_failure"],
            "heart failure": ["chf", "has_chf", "heart_failure"],
            "vascular disease": ["vascular_disease", "has_vascular_disease"],
            # RCRI
            "coronary artery disease": ["ischemic_heart_disease"],
            "renal insufficiency": ["creatinine_over_2"],
            "cerebrovascular disease": ["history_of_cerebrovascular_disease"],
        }

        for condition in conditions_present:
            cond_lower = condition.lower().strip()

            for pattern, param_names in condition_mapping.items():
                if pattern in cond_lower or cond_lower in pattern:
                    for param in param_names:
                        if param not in extracted or not extracted.get(param):
                            extracted[param] = True
                    break

        return extracted

    def _normalize_param_names(self, extracted: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parameter names to match official calculator expectations.

        Also extracts raw values from [value, "unit"] format when needed.
        """
        # Key mappings: extracted_name -> official_name
        key_mappings = {
            # Renal/metabolic
            'serum_creatinine': 'creatinine',
            'creatinine_mg_dl': 'creatinine',
            'creatinine_value': 'creatinine',
            'blood_urea_nitrogen': 'bun',
            'urea_nitrogen': 'bun',
            'serum_albumin': 'albumin',
            'serum_calcium': 'calcium',
            'total_bilirubin': 'bilirubin',
            'serum_bilirubin': 'bilirubin',
            'serum_sodium': 'sodium',
            'serum_potassium': 'potassium',

            # Vitals
            'systolic_bp': 'sys_bp',
            'systolic_blood_pressure': 'sys_bp',
            'systolic': 'sys_bp',
            'diastolic_bp': 'dia_bp',
            'diastolic_blood_pressure': 'dia_bp',
            'diastolic': 'dia_bp',
            'sbp': 'sys_bp',
            'dbp': 'dia_bp',
            'hr': 'heart_rate',
            'pulse': 'heart_rate',
            'pulse_rate': 'heart_rate',
            'temp': 'temperature',
            'body_temperature': 'temperature',

            # Cardiac
            'qt': 'qt_interval',
            'qt_ms': 'qt_interval',
            'qtc': 'qt_interval',
            'rr': 'rr_interval',
            'rr_ms': 'rr_interval',
            'rr_interval_ms': 'rr_interval',

            # Electrolytes
            'na': 'sodium',
            'na+': 'sodium',
            'k': 'potassium',
            'k+': 'potassium',
            'cl': 'chloride',
            'cl-': 'chloride',
            'hco3': 'bicarbonate',
            'hco3-': 'bicarbonate',
            'co2': 'bicarbonate',
            'bicarb': 'bicarbonate',
            'serum_bicarbonate': 'bicarbonate',

            # Blood glucose
            'blood_glucose': 'glucose',
            'fasting_glucose': 'glucose',
            'serum_glucose': 'glucose',
            'fasting_insulin': 'insulin',
            'serum_insulin': 'insulin',

            # Liver function
            'aspartate_aminotransferase': 'ast',
            'sgot': 'ast',
            'alanine_aminotransferase': 'alt',
            'sgpt': 'alt',

            # Body measurements
            'weight_kg': 'weight',
            'body_weight': 'weight',
            'height_cm': 'height',
            'body_height': 'height',

            # Arterial blood gas
            'partial_pressure_oxygen': 'pao2',
            'pao2_mmhg': 'pao2',
            'partial_pressure_co2': 'paco2',
            'paco2_mmhg': 'paco2',
            'arterial_ph': 'pH',
            'blood_ph': 'pH',

            # Lipids
            'total_cholesterol': 'cholesterol',
            'serum_cholesterol': 'cholesterol',
            'hdl_cholesterol': 'hdl',
            'ldl_cholesterol': 'ldl',
            'triglycerides_level': 'triglycerides',

            # Coagulation
            'international_normalized_ratio': 'inr',
            'prothrombin_time': 'pt',

            # Obstetric
            'last_menstrual_period': 'menstrual_date',
            'lmp': 'menstrual_date',
            'lmp_date': 'menstrual_date',

            # Misc
            'hemoglobin_level': 'hemoglobin',
            'hgb': 'hemoglobin',
            'hb': 'hemoglobin',
            'platelet_count': 'platelets',
            'plt': 'platelets',
            'white_blood_cell': 'wbc',
            'white_blood_cell_count': 'wbc',
            'respiratory_rate': 'resp_rate',
            'alcoholic_drinks_per_week': 'alcoholic_drinks',
            'alcohol_drinks': 'alcoholic_drinks',
            'drinks_per_week': 'alcoholic_drinks',
        }

        normalized = {}
        for key, value in extracted.items():
            if value is None:
                continue

            key_lower = key.lower().replace(' ', '_')

            # Map to official name if mapping exists
            official_key = key_mappings.get(key_lower, key_lower)

            # Extract raw value from [value, "unit"] format for numeric params
            # The convert_extracted_to_official will re-wrap them as needed
            if isinstance(value, list) and len(value) >= 2:
                # This is likely [value, "unit"] format
                raw_value = value[0]
                unit = value[1] if len(value) > 1 else ""
                # Keep as list - convert_extracted_to_official handles this
                normalized[official_key] = value
            else:
                normalized[official_key] = value

        return normalized

    def _compute_result(
        self,
        calculator_name: str,
        extracted: Dict[str, Any],
    ) -> Optional[float]:
        """Compute result using L4's full validation and compute pipeline."""
        try:
            # Use L4's validation and compute functions directly for consistency
            from benchmark.medcalc.experiments.l4_pipeline import (
                compute_with_python,
                validate_extracted_values,
            )

            # First normalize parameter names (our normalization)
            normalized = self._normalize_param_names(extracted)

            # Run L4's validation which does critical normalization
            is_valid, missing, cleaned_values = validate_extracted_values(
                extracted=normalized,
                calculator_name=calculator_name,
            )

            # Use L4's compute_with_python which handles all the edge cases
            result = compute_with_python(calculator_name, cleaned_values)

            if result is not None and result.result is not None:
                return result.result

            # Debug print for failures
            print(f"  [DEBUG] Official calculator error: compute returned None")
            return None

        except KeyError as e:
            print(f"  [DEBUG] Official calculator error: {e}")
            return None
        except Exception as e:
            print(f"  [DEBUG] Official calculator error: {type(e).__name__}: {e}")
            return None
