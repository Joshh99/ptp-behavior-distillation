"""
MedCalc-specific pattern store with calculator-indexed storage.

Organizes patterns by calculator type:
- equation_based/: BMI, GFR, Creatinine Clearance, etc.
- rule_based/: CHA2DS2-VASc, HEART Score, Wells Criteria, etc.

Each calculator has its own JSON file with learned patterns.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import Pattern, PatternStage, PatternStore
from .config import MedCalcContinualConfig


# Categorization of calculators
RULE_BASED_CALCULATORS = {
    # Cardiac scoring
    "CHA2DS2-VASc Score",
    "CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk",
    "HEART Score",
    "HEART Score for Major Cardiac Events",
    "RCRI Score",
    "Revised Cardiac Risk Index (RCRI)",
    "Wells Criteria for Pulmonary Embolism",
    "Wells' Criteria for Pulmonary Embolism",

    # Hepatic scoring
    "Child-Pugh Score",
    "Child-Pugh Score for Cirrhosis Mortality",
    "MELD Score",
    "MELD-Na Score",
    "MELD Score (Original)",
    "Model For End-Stage Liver Disease (MELD) Score",

    # ICU scoring
    "APACHE II Score",
    "SOFA Score",
    "Sequential Organ Failure Assessment (SOFA) Score",

    # DVT/PE scoring
    "Wells Criteria for DVT",
    "Wells' Criteria for DVT",
    "Caprini Score",
    "Caprini Score for Venous Thromboembolism",

    # Infectious/Pulmonary
    "CURB-65 Score",
    "CURB-65 Score for Pneumonia Severity",
    "Centor Score",
    "Centor Score (Modified/McIsaac) for Strep Pharyngitis",
    "PSI/PORT Score",
    "Pneumonia Severity Index (PSI/PORT Score)",
    "PERC Rule",
    "PERC Rule for Pulmonary Embolism",

    # Other scoring
    "HAS-BLED Score",
    "HAS-BLED Score for Major Bleeding Risk",
    "Charlson Comorbidity Index",
    "Charlson Comorbidity Index (CCI)",
    "Glasgow Coma Scale (GCS)",
    "Glasgow Coma Scale",
    "GCS Score",
    "Glasgow-Blatchford Bleeding Score",
    "Glasgow-Blatchford Bleeding Score (GBS)",
    "FeverPAIN Score",
    "FeverPAIN Score for Strep Pharyngitis",
    "SIRS Criteria",
    "SIRS, Pair, and SOFA Scores",
    "Framingham Risk Score",
    "Framingham Risk Score (Hard Coronary Heart Disease)",
}


class MedCalcPatternStore(PatternStore):
    """
    MedCalc-specific pattern store indexed by calculator.

    Storage structure:
        {base_path}/
            equation_based/
                bmi.json
                creatinine_clearance.json
                ...
            rule_based/
                cha2ds2_vasc.json
                heart_score.json
                ...
            stage_patterns/
                calc_id.json       # Cross-calculator ID patterns
    """

    def __init__(self, config: MedCalcContinualConfig):
        """Initialize pattern store with configuration."""
        self.config = config
        self.base_path = config.get_store_path()
        self._ensure_structure()
        self._patterns: Dict[str, Pattern] = {}
        self._load_all_patterns()

    def _ensure_structure(self) -> None:
        """Create directory structure if it doesn't exist."""
        (self.base_path / "equation_based").mkdir(parents=True, exist_ok=True)
        (self.base_path / "rule_based").mkdir(parents=True, exist_ok=True)
        (self.base_path / "stage_patterns").mkdir(parents=True, exist_ok=True)

    def _get_category(self, calculator_name: str) -> str:
        """Determine if calculator is equation-based or rule-based."""
        # Check against known rule-based calculators
        for rule_calc in RULE_BASED_CALCULATORS:
            if rule_calc.lower() in calculator_name.lower():
                return "rule_based"
            if calculator_name.lower() in rule_calc.lower():
                return "rule_based"
        return "equation_based"

    def _sanitize_name(self, name: str) -> str:
        """Convert calculator name to safe filename."""
        # Replace spaces, slashes, parentheses with underscores
        safe = name.lower()
        for char in [" ", "/", "(", ")", "-", "'"]:
            safe = safe.replace(char, "_")
        # Remove double underscores
        while "__" in safe:
            safe = safe.replace("__", "_")
        # Strip leading/trailing underscores
        safe = safe.strip("_")
        return safe

    def _get_calculator_path(self, calculator_name: str) -> Path:
        """Get file path for a calculator's patterns."""
        category = self._get_category(calculator_name)
        filename = self._sanitize_name(calculator_name) + ".json"
        return self.base_path / category / filename

    def _load_all_patterns(self) -> None:
        """Load all patterns from disk into memory."""
        self._patterns = {}

        # Load from equation_based and rule_based directories
        for category in ["equation_based", "rule_based"]:
            category_path = self.base_path / category
            if category_path.exists():
                for filepath in category_path.glob("*.json"):
                    try:
                        with open(filepath) as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for pattern_data in data:
                                    pattern = Pattern.from_dict(pattern_data)
                                    self._patterns[pattern.pattern_id] = pattern
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Warning: Could not load {filepath}: {e}")

        # Load stage patterns (cross-calculator)
        stage_path = self.base_path / "stage_patterns"
        if stage_path.exists():
            for filepath in stage_path.glob("*.json"):
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for pattern_data in data:
                                pattern = Pattern.from_dict(pattern_data)
                                self._patterns[pattern.pattern_id] = pattern
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not load {filepath}: {e}")

    def _save_patterns_for_calculator(self, calculator_name: str) -> None:
        """Save all patterns for a specific calculator."""
        filepath = self._get_calculator_path(calculator_name)
        patterns = [
            p.to_dict() for p in self._patterns.values()
            if p.entity.lower() == calculator_name.lower()
        ]
        if patterns:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(patterns, f, indent=2)

    def _save_stage_patterns(self, stage: PatternStage) -> None:
        """Save all patterns for a specific stage."""
        filepath = self.base_path / "stage_patterns" / f"{stage.value}.json"
        patterns = [
            p.to_dict() for p in self._patterns.values()
            if p.stage == stage and not p.entity  # Stage-level, not entity-specific
        ]
        if patterns:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(patterns, f, indent=2)

    # =========================================================================
    # PatternStore Interface Implementation
    # =========================================================================

    def get_patterns_for_entity(
        self,
        entity: str,
        stage: Optional[PatternStage] = None,
        min_confidence: float = 0.0,
    ) -> List[Pattern]:
        """Get patterns for a specific calculator."""
        patterns = []
        entity_lower = entity.lower()

        for pattern in self._patterns.values():
            # Match entity (calculator name)
            if pattern.entity.lower() != entity_lower:
                continue

            # Filter by stage if specified
            if stage and pattern.stage != stage:
                continue

            # Filter by confidence
            if pattern.confidence < min_confidence:
                continue

            patterns.append(pattern)

        # Sort by relevance score descending
        patterns.sort(key=lambda p: p.relevance_score, reverse=True)
        return patterns

    def get_patterns_for_stage(
        self,
        stage: PatternStage,
        limit: int = 10,
    ) -> List[Pattern]:
        """Get all patterns for a pipeline stage (cross-calculator)."""
        patterns = [
            p for p in self._patterns.values()
            if p.stage == stage
        ]

        # Sort by relevance score descending
        patterns.sort(key=lambda p: p.relevance_score, reverse=True)
        return patterns[:limit]

    def get_patterns_for_calculator(
        self,
        calculator_name: str,
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Pattern]:
        """
        Get all learned patterns for a specific calculator.

        This is an alias for get_patterns_for_entity with MedCalc terminology.
        """
        stage = None
        if pattern_type == "extraction":
            stage = PatternStage.EXTRACTION
        elif pattern_type == "reasoning":
            stage = PatternStage.REASONING
        elif pattern_type == "calc_id":
            stage = PatternStage.CALC_ID

        return self.get_patterns_for_entity(
            entity=calculator_name,
            stage=stage,
            min_confidence=min_confidence,
        )

    def store_pattern(self, pattern: Pattern) -> None:
        """Store a new pattern."""
        self._patterns[pattern.pattern_id] = pattern

        # Save to appropriate file
        if pattern.entity:
            self._save_patterns_for_calculator(pattern.entity)
        else:
            self._save_stage_patterns(pattern.stage)

    def update_pattern(self, pattern_id: str, updates: Dict[str, Any]) -> None:
        """Update an existing pattern."""
        if pattern_id not in self._patterns:
            return

        pattern = self._patterns[pattern_id]
        for key, value in updates.items():
            if hasattr(pattern, key):
                setattr(pattern, key, value)

        # Re-save
        if pattern.entity:
            self._save_patterns_for_calculator(pattern.entity)
        else:
            self._save_stage_patterns(pattern.stage)

    def reinforce_pattern(self, pattern_id: str, was_helpful: bool) -> None:
        """Reinforce or decay a pattern based on usage."""
        if pattern_id not in self._patterns:
            return

        pattern = self._patterns[pattern_id]
        pattern.reinforce(was_helpful)

        # Re-save
        if pattern.entity:
            self._save_patterns_for_calculator(pattern.entity)
        else:
            self._save_stage_patterns(pattern.stage)

    def apply_decay(self, days: int = 1) -> int:
        """Apply decay to all patterns."""
        affected = 0
        calculators_to_save = set()
        stages_to_save = set()

        for pattern in self._patterns.values():
            old_confidence = pattern.confidence
            pattern.apply_decay(days, self.config.decay_rate)
            if pattern.confidence != old_confidence:
                affected += 1
                if pattern.entity:
                    calculators_to_save.add(pattern.entity)
                else:
                    stages_to_save.add(pattern.stage)

        # Save affected files
        for calc in calculators_to_save:
            self._save_patterns_for_calculator(calc)
        for stage in stages_to_save:
            self._save_stage_patterns(stage)

        return affected

    def prune_low_confidence(self, threshold: float = 0.1) -> int:
        """Remove patterns below confidence threshold."""
        to_remove = [
            pid for pid, p in self._patterns.items()
            if p.confidence < threshold
        ]

        calculators_to_save = set()
        stages_to_save = set()

        for pid in to_remove:
            pattern = self._patterns[pid]
            if pattern.entity:
                calculators_to_save.add(pattern.entity)
            else:
                stages_to_save.add(pattern.stage)
            del self._patterns[pid]

        # Save affected files
        for calc in calculators_to_save:
            self._save_patterns_for_calculator(calc)
        for stage in stages_to_save:
            self._save_stage_patterns(stage)

        return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        patterns_by_stage = {}
        patterns_by_category = {"equation_based": 0, "rule_based": 0}
        total_confidence = 0.0
        total_used = 0
        total_helpful = 0

        for pattern in self._patterns.values():
            # By stage
            stage_name = pattern.stage.value
            patterns_by_stage[stage_name] = patterns_by_stage.get(stage_name, 0) + 1

            # By category
            if pattern.entity:
                category = self._get_category(pattern.entity)
                patterns_by_category[category] += 1

            # Totals
            total_confidence += pattern.confidence
            total_used += pattern.times_used
            total_helpful += pattern.times_helpful

        return {
            "total_patterns": len(self._patterns),
            "patterns_by_stage": patterns_by_stage,
            "patterns_by_category": patterns_by_category,
            "avg_confidence": total_confidence / len(self._patterns) if self._patterns else 0.0,
            "total_uses": total_used,
            "total_helpful": total_helpful,
            "helpfulness_rate": total_helpful / total_used if total_used > 0 else 0.0,
        }

    def is_rule_based(self, calculator_name: str) -> bool:
        """Check if a calculator is rule-based (needs medical reasoning)."""
        return self._get_category(calculator_name) == "rule_based"
