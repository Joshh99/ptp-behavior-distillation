"""
MedCalc Pattern Miner: Extracts patterns from successful L4 backoffs.

Mining types:
1. Calculator ID patterns: Keywords that identify calculators
2. Extraction patterns: Value extraction hints (semi-regex)
3. Reasoning patterns: Clinical finding → condition mappings

The miner collects traces from successful backoffs and periodically
mines patterns using LLM assistance for generalization.
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional
from pathlib import Path

from ..base import Pattern, PatternStage, PatternMiner, TraceRecord, BackoffReason
from .config import MedCalcContinualConfig
from .pattern_store import MedCalcPatternStore

from ptool_framework.llm_backend import call_llm


class MedCalcPatternMiner(PatternMiner):
    """
    Mines patterns from successful L4 backoff traces.

    Collects traces when:
    - L2 Python backs off to L4 Pipeline
    - L4 Pipeline succeeds (correct answer)

    Mines patterns for:
    - Calculator identification (keyword→calculator)
    - Value extraction (text pattern→value)
    - Medical reasoning (clinical finding→condition)
    """

    def __init__(
        self,
        config: MedCalcContinualConfig,
        pattern_store: MedCalcPatternStore,
    ):
        """
        Initialize miner.

        Args:
            config: MedCalc continual config
            pattern_store: Store for saving mined patterns
        """
        self.config = config
        self.pattern_store = pattern_store

        # Pending traces buffer
        self._pending_traces: List[TraceRecord] = []
        self._traces_since_mine = 0

        # Trace file for persistence
        self._trace_file = config.get_store_path() / "pending_traces.jsonl"

        # Load any persisted traces
        self._load_pending_traces()

    def _load_pending_traces(self) -> None:
        """Load pending traces from disk."""
        if self._trace_file.exists():
            try:
                with open(self._trace_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self._pending_traces.append(TraceRecord.from_dict(data))
            except (json.JSONDecodeError, IOError):
                pass

    def _save_pending_traces(self) -> None:
        """Save pending traces to disk."""
        self._trace_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._trace_file, "w") as f:
            for trace in self._pending_traces:
                f.write(json.dumps(trace.to_dict()) + "\n")

    # =========================================================================
    # PatternMiner Interface
    # =========================================================================

    def add_trace(self, trace: TraceRecord) -> None:
        """Add a trace record for later mining."""
        self._pending_traces.append(trace)
        self._traces_since_mine += 1
        self._save_pending_traces()

        # Auto-mine if threshold reached
        if self._traces_since_mine >= self.config.mining_frequency:
            self.mine_pending()

    def get_pending_count(self) -> int:
        """Get count of traces pending mining."""
        return len(self._pending_traces)

    def clear_pending(self) -> None:
        """Clear pending traces."""
        self._pending_traces = []
        self._traces_since_mine = 0
        if self._trace_file.exists():
            self._trace_file.unlink()

    def mine_pending(self) -> List[Pattern]:
        """Mine patterns from pending traces."""
        if not self._pending_traces:
            return []

        mined_patterns = []

        # Group traces by calculator for more effective mining
        traces_by_calc: Dict[str, List[TraceRecord]] = {}
        for trace in self._pending_traces:
            calc = trace.entity
            if calc not in traces_by_calc:
                traces_by_calc[calc] = []
            traces_by_calc[calc].append(trace)

        # Mine patterns for each calculator
        for calculator, traces in traces_by_calc.items():
            if len(traces) >= self.config.min_examples_for_pattern:
                # Mine extraction patterns
                extraction_patterns = self._mine_extraction_patterns(calculator, traces)
                mined_patterns.extend(extraction_patterns)

                # Mine reasoning patterns (for scoring systems only)
                if self.pattern_store.is_rule_based(calculator):
                    reasoning_patterns = self._mine_reasoning_patterns(calculator, traces)
                    mined_patterns.extend(reasoning_patterns)

        # Mine calculator ID patterns (cross-calculator)
        calc_id_patterns = self._mine_calc_id_patterns(self._pending_traces)
        mined_patterns.extend(calc_id_patterns)

        # Store mined patterns
        for pattern in mined_patterns:
            self.pattern_store.store_pattern(pattern)

        # Clear mined traces
        self._pending_traces = []
        self._traces_since_mine = 0
        self._save_pending_traces()

        return mined_patterns

    # =========================================================================
    # Mining Methods
    # =========================================================================

    def _mine_extraction_patterns(
        self,
        calculator: str,
        traces: List[TraceRecord],
    ) -> List[Pattern]:
        """Mine extraction patterns for a calculator."""
        patterns = []

        # Group by parameter name
        params_examples: Dict[str, List[Dict]] = {}
        for trace in traces:
            for param, value in trace.extracted_values.items():
                if param not in params_examples:
                    params_examples[param] = []

                # Find where in context this value appears
                context_snippet = self._find_value_context(
                    trace.context,
                    param,
                    value,
                )
                if context_snippet:
                    params_examples[param].append({
                        "context": context_snippet,
                        "value": value,
                        "full_note": trace.context[:500],
                    })

        # Generate pattern for each parameter with enough examples
        for param, examples in params_examples.items():
            if len(examples) >= self.config.min_examples_for_pattern:
                pattern = self._generate_extraction_pattern(
                    calculator,
                    param,
                    examples[:self.config.max_examples_per_pattern],
                )
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _mine_reasoning_patterns(
        self,
        calculator: str,
        traces: List[TraceRecord],
    ) -> List[Pattern]:
        """Mine reasoning patterns for scoring systems."""
        patterns = []

        # Collect all condition→context mappings
        condition_contexts: Dict[str, List[str]] = {}

        for trace in traces:
            # Look for boolean parameters (conditions)
            for param, value in trace.extracted_values.items():
                if isinstance(value, bool) and value:
                    # Find what in the context led to this condition
                    context_snippet = self._find_condition_context(
                        trace.context,
                        param,
                    )
                    if context_snippet:
                        if param not in condition_contexts:
                            condition_contexts[param] = []
                        condition_contexts[param].append(context_snippet)

        # Generate pattern for each condition with enough examples
        for condition, contexts in condition_contexts.items():
            if len(contexts) >= self.config.min_examples_for_pattern:
                pattern = self._generate_reasoning_pattern(
                    calculator,
                    condition,
                    contexts[:self.config.max_examples_per_pattern],
                )
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _mine_calc_id_patterns(
        self,
        traces: List[TraceRecord],
    ) -> List[Pattern]:
        """Mine calculator identification patterns."""
        patterns = []

        # Group by calculator
        calc_questions: Dict[str, List[str]] = {}
        for trace in traces:
            calc = trace.entity
            if calc not in calc_questions:
                calc_questions[calc] = []
            calc_questions[calc].append(trace.question)

        # Extract keywords for each calculator
        for calculator, questions in calc_questions.items():
            if len(questions) >= self.config.min_examples_for_pattern:
                pattern = self._generate_calc_id_pattern(
                    calculator,
                    questions[:self.config.max_examples_per_pattern],
                )
                if pattern:
                    patterns.append(pattern)

        return patterns

    # =========================================================================
    # Pattern Generation (LLM-assisted)
    # =========================================================================

    def _generate_extraction_pattern(
        self,
        calculator: str,
        param: str,
        examples: List[Dict],
    ) -> Optional[Pattern]:
        """Generate an extraction pattern using LLM."""
        examples_text = "\n".join([
            f"Context: \"{ex['context'][:200]}\"\nExtracted {param}: {ex['value']}"
            for ex in examples
        ])

        prompt = f"""Analyze these examples of extracting "{param}" for {calculator}:

{examples_text}

Create a brief extraction hint (1-2 sentences) that describes:
1. What text patterns indicate this value
2. Common formats/units to look for
3. Any conversions needed

Format: A concise hint like "Look for 'weighs X kg' or 'X lbs' (convert lbs to kg)"

Extraction hint for {param}:"""

        try:
            response = call_llm(prompt=prompt, model=self.config.mining_model, max_tokens=200).content
            hint = response.strip()

            if hint and len(hint) > 10:
                pattern = Pattern.create(
                    stage=PatternStage.EXTRACTION,
                    entity=calculator,
                    content=hint,
                    examples=[{"input": ex["context"][:100], "output": ex["value"]} for ex in examples],
                    metadata={"param_name": param},
                )
                return pattern

        except Exception:
            pass

        return None

    def _generate_reasoning_pattern(
        self,
        calculator: str,
        condition: str,
        contexts: List[str],
    ) -> Optional[Pattern]:
        """Generate a reasoning pattern for condition inference."""
        contexts_text = "\n".join([
            f"- \"{ctx[:200]}\" → {condition} = true"
            for ctx in contexts
        ])

        prompt = f"""Analyze these examples where "{condition}" was inferred for {calculator}:

{contexts_text}

Create a brief clinical reasoning hint (1-2 sentences) that describes:
1. What clinical findings indicate this condition
2. Key phrases or lab values to look for

Format: "If note mentions [X, Y, Z], likely has {condition}"

Reasoning hint:"""

        try:
            response = call_llm(prompt=prompt, model=self.config.mining_model, max_tokens=200).content
            hint = response.strip()

            if hint and len(hint) > 10:
                pattern = Pattern.create(
                    stage=PatternStage.REASONING,
                    entity=calculator,
                    content=hint,
                    examples=[{"input": ctx[:100], "output": f"{condition}=true"} for ctx in contexts],
                    metadata={"condition": condition},
                )
                return pattern

        except Exception:
            pass

        return None

    def _generate_calc_id_pattern(
        self,
        calculator: str,
        questions: List[str],
    ) -> Optional[Pattern]:
        """Generate a calculator identification pattern."""
        questions_text = "\n".join([f"- \"{q}\"" for q in questions])

        prompt = f"""Analyze these questions that all refer to the {calculator}:

{questions_text}

Extract the KEY WORDS that reliably identify this calculator (comma-separated list).
Focus on specific medical terms, not generic words like "calculate" or "patient".

Keywords for {calculator}:"""

        try:
            response = call_llm(prompt=prompt, model=self.config.mining_model, max_tokens=100).content
            keywords = response.strip()

            if keywords and len(keywords) > 3:
                pattern = Pattern.create(
                    stage=PatternStage.CALC_ID,
                    entity=calculator,
                    content=keywords,
                    examples=[{"input": q, "output": calculator} for q in questions[:3]],
                )
                return pattern

        except Exception:
            pass

        return None

    # =========================================================================
    # Context Finding Helpers
    # =========================================================================

    def _find_value_context(
        self,
        text: str,
        param: str,
        value: Any,
    ) -> Optional[str]:
        """Find the context around where a value appears in text."""
        text_lower = text.lower()

        # Try to find the value in the text
        if isinstance(value, (int, float)):
            # Look for the number
            patterns = [
                rf'\b{value}\b',
                rf'\b{int(value)}\b' if isinstance(value, float) else None,
            ]
            for pattern in patterns:
                if pattern:
                    match = re.search(pattern, text)
                    if match:
                        start = max(0, match.start() - 50)
                        end = min(len(text), match.end() + 50)
                        return text[start:end]

        elif isinstance(value, str):
            if value.lower() in text_lower:
                idx = text_lower.find(value.lower())
                start = max(0, idx - 50)
                end = min(len(text), idx + len(value) + 50)
                return text[start:end]

        elif isinstance(value, list) and len(value) >= 2:
            # [value, unit] format
            return self._find_value_context(text, param, value[0])

        return None

    def _find_condition_context(
        self,
        text: str,
        condition: str,
    ) -> Optional[str]:
        """Find context that led to a condition being true."""
        text_lower = text.lower()
        condition_clean = condition.lower().replace("_", " ").replace("has ", "")

        # Common medical term mappings
        condition_keywords = {
            "stroke": ["stroke", "cva", "infarct", "dwi", "hemiparesis", "aphasia"],
            "tia": ["tia", "transient ischemic"],
            "hypertension": ["hypertension", "htn", "blood pressure", "antihypertensive"],
            "diabetes": ["diabetes", "dm", "diabetic", "glucose", "hba1c"],
            "chf": ["chf", "heart failure", "ef ", "ejection fraction", "edema", "jvd"],
            "vascular": ["mi", "cad", "cabg", "pci", "pad", "stent"],
        }

        # Get keywords for this condition
        keywords = []
        for key, kw_list in condition_keywords.items():
            if key in condition_clean:
                keywords.extend(kw_list)

        if not keywords:
            keywords = [condition_clean]

        # Find first matching keyword in text
        for keyword in keywords:
            if keyword in text_lower:
                idx = text_lower.find(keyword)
                start = max(0, idx - 30)
                end = min(len(text), idx + len(keyword) + 50)
                return text[start:end]

        return None

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def add_successful_backoff(
        self,
        question: str,
        patient_note: str,
        calculator: str,
        extracted_values: Dict[str, Any],
        ground_truth: Optional[float] = None,
        backoff_reason: Optional[BackoffReason] = None,
    ) -> None:
        """
        Convenience method to add a trace from successful backoff.

        Use this when L2→L4 backoff succeeds.
        """
        trace = TraceRecord(
            trace_id=str(uuid.uuid4())[:8],
            question=question,
            context=patient_note,
            entity=calculator,
            extracted_values=extracted_values,
            ground_truth=ground_truth,
            backoff_reason=backoff_reason,
        )
        self.add_trace(trace)
