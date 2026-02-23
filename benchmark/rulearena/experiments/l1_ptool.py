"""
L1: PTool Extraction + Python Calculation

Two-phase approach:
1. LLM extracts structured parameters as JSON
2. Python deterministic calculator computes answer

This is the core PTP hypothesis test: separation of extraction (LLM) from
computation (Python) improves reliability over end-to-end LLM reasoning.
"""

import time
import json
import re
from typing import Optional, Dict, Any, Tuple

from together import Together

from benchmark.rulearena.dataset.loader import RuleArenaInstance
from benchmark.rulearena.experiments.base import BaseExperiment, ExperimentResult
from benchmark.rulearena.config import MODEL_CONFIG, calculate_cost


class L1_PTool_Experiment(BaseExperiment):
    """
    L1: PTool extraction experiment.

    Phase 1: LLM extracts structured parameters from problem text
    Phase 2: Python calculator computes answer from parameters
    """

    def __init__(self):
        super().__init__(
            experiment_name="l1_ptool",
            model_id=MODEL_CONFIG["model_id"]
        )
        self.client = Together(api_key=None)
        self.temperature = MODEL_CONFIG["default_params"]["temperature"]
        self.seed = MODEL_CONFIG["default_params"]["seed"]
        self.max_tokens = MODEL_CONFIG["default_params"]["max_tokens"]

    def run_instance(self, instance: RuleArenaInstance) -> ExperimentResult:
        """
        Run L1 extraction + calculation on a single instance.

        Returns:
            ExperimentResult with prediction and metrics
        """
        start_time = time.time()

        try:
            # Phase 1: Extract parameters
            prompt = self._build_extraction_prompt(instance)

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )

            raw_response = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            # Parse JSON parameters
            params = self._parse_json_response(raw_response)

            # Phase 2: Calculate answer using deterministic Python
            predicted, calc_error = self._calculate_answer(params, instance)

            # Compare with ground truth
            exact_match, tolerance_match = self.compare_answers(
                predicted, instance.ground_truth_answer
            )

            cost = calculate_cost(input_tokens, output_tokens)
            elapsed = time.time() - start_time

            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=predicted,
                expected=instance.ground_truth_answer,
                is_correct_exact=exact_match,
                is_correct_tolerance=tolerance_match,
                latency_ms=elapsed * 1000,
                cost_usd=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                calculator_name=instance.domain,
                error=calc_error,
                raw_response=raw_response,
                metadata={
                    "domain": instance.domain,
                    "complexity_level": instance.complexity_level,
                    "extraction_json": params,
                }
            )

        except Exception as e:
            elapsed = time.time() - start_time
            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=None,
                expected=instance.ground_truth_answer,
                is_correct_exact=False,
                is_correct_tolerance=False,
                latency_ms=elapsed * 1000,
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                calculator_name=instance.domain,
                error=str(e),
                raw_response=None,
                metadata={
                    "domain": instance.domain,
                    "complexity_level": instance.complexity_level,
                }
            )

    def _build_extraction_prompt(self, instance: RuleArenaInstance) -> str:
        """Build extraction prompt with domain-specific schema."""

        if instance.domain == "airline":
            return self._build_airline_extraction_prompt(instance)
        elif instance.domain == "nba":
            return self._build_nba_extraction_prompt(instance)
        elif instance.domain == "tax":
            return self._build_tax_extraction_prompt(instance)
        else:
            raise ValueError(f"Unknown domain: {instance.domain}")

    def _build_airline_extraction_prompt(self, instance: RuleArenaInstance) -> str:
        """Build extraction prompt for airline domain."""

        prompt = f"""Extract structured baggage parameters from this airline query.

RULES:
{instance.rules_text[:3000]}

QUERY:
{instance.problem_text}

Extract these exact fields as JSON:

{{
    "base_price": <integer ticket price in USD>,
    "customer_class": <EXACTLY one of: "Basic Economy", "Main Cabin", "Main Plus", "Premium Economy", "Business", "First">,
    "routine": <destination region - see rules for valid values, use "U.S." for domestic>,
    "direction": <0 for departing from US, 1 for arriving to US>,
    "bag_list": [
        {{"id": 1, "name": "backpack", "size": [length, width, height], "weight": pounds}},
        {{"id": 2, "name": "luggage box", "size": [length, width, height], "weight": pounds}}
    ]
}}

CRITICAL REQUIREMENTS:
- customer_class must be exact match (e.g., "Business" not "Business Class")
- routine must match rules (e.g., "U.S." with period for domestic, "Japan", "Europe", etc.)
- direction: 0=from US, 1=to US
- size: array of 3 integers [length, width, height] in inches
- weight: integer in pounds

Return ONLY valid JSON, no markdown fences, no explanation.
"""
        return prompt

    def _build_nba_extraction_prompt(self, instance: RuleArenaInstance) -> str:
        """Build extraction prompt for NBA domain (stub for now)."""
        return f"""Extract NBA transaction parameters as JSON.

RULES:
{instance.rules_text[:2000]}

QUERY:
{instance.problem_text}

Return JSON with relevant parameters.
"""

    def _build_tax_extraction_prompt(self, instance: RuleArenaInstance) -> str:
        """Build extraction prompt for tax domain (stub for now)."""
        return f"""Extract tax calculation parameters as JSON.

RULES:
{instance.rules_text[:2000]}

QUERY:
{instance.problem_text}

Return JSON with relevant parameters.
"""

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling markdown fences.

        Returns:
            Parsed JSON dict

        Raises:
            ValueError if JSON cannot be parsed
        """
        # Try to find JSON in markdown code block
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            # Try to find raw JSON object
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                json_str = match.group()
            else:
                raise ValueError(f"No JSON found in response")

        return json.loads(json_str)

    def _calculate_answer(
        self, params: Dict[str, Any], instance: RuleArenaInstance
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Calculate answer using domain-specific Python calculator.

        Args:
            params: Extracted parameters
            instance: Instance with metadata

        Returns:
            (answer, error_message)
        """
        try:
            if instance.domain == "airline":
                return self._calculate_airline_answer(params, instance)
            elif instance.domain == "nba":
                return None, "NBA calculator not implemented"
            elif instance.domain == "tax":
                return None, "Tax calculator not implemented"
            else:
                return None, f"Unknown domain: {instance.domain}"
        except Exception as e:
            return None, f"Calculation error: {str(e)}"

    def _calculate_airline_answer(
        self, params: Dict[str, Any], instance: RuleArenaInstance
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Calculate airline baggage fee using extracted parameters.

        Uses the airline calculator imported at module level.
        """
        try:
            # Import airline calculator
            from benchmark.rulearena.calculators.airline import compute_airline_fee

            # Normalize region (handle LLM mistakes like "Asia" -> "China")
            routine = self._normalize_region(params.get('routine', 'U.S.'))
            params['routine'] = routine

            # Compute answer
            total_cost = compute_airline_fee(params)

            return total_cost, None

        except Exception as e:
            return None, str(e)

    def _normalize_region(self, routine: str) -> str:
        """
        Normalize region to valid RuleArena value.

        Handles common LLM mistakes (e.g., "Asia" -> "China").
        """
        # Valid regions from RuleArena
        VALID_REGIONS = {
            "U.S.", "Puerto Rico", "Canada", "Mexico", "Cuba", "Haiti", "Panama",
            "Colombia", "Ecuador", "Peru", "South America", "Israel", "Qatar",
            "Europe", "India", "China", "Japan", "South Korea", "Hong Kong",
            "Australia", "New Zealand"
        }

        # Already valid
        if routine in VALID_REGIONS:
            return routine

        # Common mistakes -> valid regions (case-insensitive)
        REGION_FIXES = {
            "asia": "China",
            "north america": "U.S.",
            "us": "U.S.",
            "usa": "U.S.",
            "united states": "U.S.",
            "domestic": "U.S.",
            "tokyo": "Japan",
            "beijing": "China",
            "shanghai": "China",
            "seoul": "South Korea",
            "sydney": "Australia",
            "london": "Europe",
            "paris": "Europe",
            "berlin": "Europe",
        }

        routine_lower = routine.lower().strip()
        if routine_lower in REGION_FIXES:
            return REGION_FIXES[routine_lower]

        # Default to U.S. for unknown
        return "U.S."


if __name__ == "__main__":
    # Quick test
    from benchmark.rulearena.dataset.loader import RuleArenaDataset

    print("Testing L1 PTool Experiment...")
    dataset = RuleArenaDataset()

    # Get airline instances
    airline_instances = [inst for inst in dataset.instances if inst.domain == "airline"]
    sample = airline_instances[:1]

    experiment = L1_PTool_Experiment()
    result = experiment.run_instance(sample[0])

    print(f"\nResult:")
    print(f"  Instance: {result.instance_id}")
    print(f"  Predicted: {result.predicted}")
    print(f"  Expected: {result.expected}")
    print(f"  Correct: {result.is_correct_exact}")
    print(f"  Cost: ${result.cost_usd:.4f}")
    print(f"  Time: {result.latency_ms:.0f}ms")
    if result.error:
        print(f"  Error: {result.error}")
    if result.metadata.get('extraction_json'):
        print(f"  Extracted params: {result.metadata['extraction_json']}")
