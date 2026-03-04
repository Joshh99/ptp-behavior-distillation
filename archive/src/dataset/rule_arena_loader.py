"""
RuleArena Dataset Loader

Loads airline baggage rule-based reasoning tasks for L0-L3 workflow evaluation.
Designed for the "Reliability/Autonomy Spectrum" research project.

The RuleArena benchmark tests LLM's ability to:
1. Extract relevant parameters from natural language queries
2. Apply complex rule logic (if-then-else, thresholds, exceptions)
3. Produce correct answers that match ground truth

Domain: Airline Baggage Fees
- Perfect for L1 (Python calculates) vs L3 (LLM reasons) comparison
- Has ground truth logic (deterministic rules)
- Messy inputs (natural language passenger queries)

Usage:
    from src.dataset.rule_arena_loader import RuleArenaDataset, load_baggage_rules
    
    dataset = RuleArenaDataset()
    test_data = dataset.load("test")
    
    # Get a specific task
    instance = test_data[0]
    print(instance.query)
    print(instance.ground_truth_answer)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import json
import os


class TaskCategory(Enum):
    """Categories of rule-based reasoning tasks."""
    BAGGAGE_ALLOWANCE = "baggage_allowance"
    EXCESS_FEE = "excess_fee"
    PROHIBITED_ITEMS = "prohibited_items"
    SPECIAL_ITEMS = "special_items"
    UPGRADE_ELIGIBILITY = "upgrade_eligibility"


class DifficultyLevel(Enum):
    """Task difficulty based on rule complexity."""
    SIMPLE = "simple"           # Single rule lookup
    MODERATE = "moderate"       # 2-3 rules with conditions
    COMPLEX = "complex"         # Multiple rules with exceptions
    EXPERT = "expert"           # Nested rules with edge cases


@dataclass
class BaggageRule:
    """A single airline baggage rule."""
    rule_id: str
    airline: str
    category: TaskCategory
    description: str
    conditions: Dict[str, Any]      # e.g., {"class": "economy", "route": "domestic"}
    thresholds: Dict[str, float]    # e.g., {"max_weight_kg": 23, "max_pieces": 1}
    fees: Dict[str, float]          # e.g., {"first_bag": 0, "second_bag": 35}
    exceptions: List[str]           # e.g., ["Elite status members exempt"]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "airline": self.airline,
            "category": self.category.value,
            "description": self.description,
            "conditions": self.conditions,
            "thresholds": self.thresholds,
            "fees": self.fees,
            "exceptions": self.exceptions,
        }


@dataclass 
class RuleArenaInstance:
    """A single instance from the RuleArena dataset."""
    instance_id: int
    category: TaskCategory
    difficulty: DifficultyLevel
    query: str                          # Natural language question
    passenger_context: Dict[str, Any]   # e.g., {"class": "economy", "bags": 2, "weight": 25}
    applicable_rules: List[str]         # Rule IDs that apply
    ground_truth_answer: Any            # Could be: fee amount, yes/no, list of items
    ground_truth_explanation: str       # Step-by-step reasoning
    relevant_entities: Dict[str, Any]   # Extracted parameters for evaluation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "query": self.query,
            "passenger_context": self.passenger_context,
            "applicable_rules": self.applicable_rules,
            "ground_truth_answer": self.ground_truth_answer,
            "ground_truth_explanation": self.ground_truth_explanation,
            "relevant_entities": self.relevant_entities,
        }
    
    def get_prompt(self, include_rules: bool = False, rules: List[BaggageRule] = None) -> str:
        """Format as a prompt for the LLM."""
        prompt = f"""Passenger Query:{self.query} 
                    Passenger Information:"""
                    
        for key, value in self.passenger_context.items():
            prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        if include_rules and rules:
            prompt += "\nApplicable Airline Rules:\n"
            for rule in rules:
                if rule.rule_id in self.applicable_rules:
                    prompt += f"\n[{rule.rule_id}] {rule.description}\n"
                    prompt += f"  Conditions: {rule.conditions}\n"
                    prompt += f"  Thresholds: {rule.thresholds}\n"
                    prompt += f"  Fees: {rule.fees}\n"
        
        return prompt


# =============================================================================
# AIRLINE BAGGAGE RULES DATABASE
# =============================================================================

def get_baggage_rules() -> List[BaggageRule]:
    """
    Returns the complete set of airline baggage rules.
    
    These rules are designed to test:
    - Parameter extraction (weight, dimensions, class, route)
    - Conditional logic (if economy AND domestic, then...)
    - Fee calculations (tiered pricing, overweight fees)
    - Exception handling (elite status, military, etc.)
    """
    rules = [
        # === UNITED AIRLINES ===
        BaggageRule(
            rule_id="UA-DOM-ECON-001",
            airline="United Airlines",
            category=TaskCategory.BAGGAGE_ALLOWANCE,
            description="United Economy domestic baggage allowance",
            conditions={"airline": "united", "class": "economy", "route": "domestic"},
            thresholds={"max_weight_kg": 23, "max_pieces": 0, "max_dimension_cm": 157},
            fees={"first_bag": 35, "second_bag": 45, "overweight_23_32kg": 100, "overweight_32_45kg": 200},
            exceptions=["MileagePlus Premier members: 1 free bag", "Star Alliance Gold: 1 free bag"]
        ),
        BaggageRule(
            rule_id="UA-DOM-BUSI-001",
            airline="United Airlines", 
            category=TaskCategory.BAGGAGE_ALLOWANCE,
            description="United Business domestic baggage allowance",
            conditions={"airline": "united", "class": "business", "route": "domestic"},
            thresholds={"max_weight_kg": 32, "max_pieces": 2, "max_dimension_cm": 157},
            fees={"first_bag": 0, "second_bag": 0, "third_bag": 150, "overweight_32_45kg": 200},
            exceptions=[]
        ),
        BaggageRule(
            rule_id="UA-INT-ECON-001",
            airline="United Airlines",
            category=TaskCategory.BAGGAGE_ALLOWANCE,
            description="United Economy international baggage allowance",
            conditions={"airline": "united", "class": "economy", "route": "international"},
            thresholds={"max_weight_kg": 23, "max_pieces": 1, "max_dimension_cm": 157},
            fees={"first_bag": 0, "second_bag": 100, "overweight_23_32kg": 100, "overweight_32_45kg": 200},
            exceptions=["Transatlantic: 1 free bag included", "Asia routes: 2 free bags"]
        ),
        
        # === DELTA AIRLINES ===
        BaggageRule(
            rule_id="DL-DOM-ECON-001",
            airline="Delta Airlines",
            category=TaskCategory.BAGGAGE_ALLOWANCE,
            description="Delta Economy domestic baggage allowance",
            conditions={"airline": "delta", "class": "economy", "route": "domestic"},
            thresholds={"max_weight_kg": 23, "max_pieces": 0, "max_dimension_cm": 157},
            fees={"first_bag": 35, "second_bag": 45, "overweight_23_32kg": 100, "overweight_32_45kg": 200},
            exceptions=["SkyMiles Medallion members: 1-3 free bags based on tier", "Delta Reserve cardholders: 1 free bag"]
        ),
        BaggageRule(
            rule_id="DL-DOM-FIRST-001",
            airline="Delta Airlines",
            category=TaskCategory.BAGGAGE_ALLOWANCE,
            description="Delta First Class domestic baggage allowance",
            conditions={"airline": "delta", "class": "first", "route": "domestic"},
            thresholds={"max_weight_kg": 32, "max_pieces": 3, "max_dimension_cm": 157},
            fees={"first_bag": 0, "second_bag": 0, "third_bag": 0, "fourth_bag": 200},
            exceptions=[]
        ),
        
        # === AMERICAN AIRLINES ===
        BaggageRule(
            rule_id="AA-DOM-ECON-001",
            airline="American Airlines",
            category=TaskCategory.BAGGAGE_ALLOWANCE,
            description="American Economy domestic baggage allowance",
            conditions={"airline": "american", "class": "economy", "route": "domestic"},
            thresholds={"max_weight_kg": 23, "max_pieces": 0, "max_dimension_cm": 157},
            fees={"first_bag": 35, "second_bag": 45, "overweight_23_32kg": 100, "overweight_32_45kg": 200},
            exceptions=["AAdvantage Executive Platinum: 3 free bags", "Citi AAdvantage cardholders: 1 free bag"]
        ),
        
        # === EXCESS FEE RULES ===
        BaggageRule(
            rule_id="GENERIC-OVERWEIGHT-001",
            airline="Generic",
            category=TaskCategory.EXCESS_FEE,
            description="Standard overweight baggage fee structure",
            conditions={"applies_to": "all_airlines"},
            thresholds={"tier1_min_kg": 23, "tier1_max_kg": 32, "tier2_min_kg": 32, "tier2_max_kg": 45},
            fees={"tier1_fee": 100, "tier2_fee": 200, "over_45kg": "not_accepted"},
            exceptions=["Sports equipment may have different limits"]
        ),
        BaggageRule(
            rule_id="GENERIC-OVERSIZE-001",
            airline="Generic",
            category=TaskCategory.EXCESS_FEE,
            description="Standard oversize baggage fee structure",
            conditions={"applies_to": "all_airlines"},
            thresholds={"standard_max_cm": 157, "oversize_max_cm": 292},
            fees={"oversize_fee": 200, "over_292cm": "not_accepted"},
            exceptions=["Surfboards, skis may have special handling"]
        ),
        
        # === PROHIBITED ITEMS ===
        BaggageRule(
            rule_id="TSA-PROHIBITED-001",
            airline="TSA",
            category=TaskCategory.PROHIBITED_ITEMS,
            description="TSA prohibited items in carry-on",
            conditions={"bag_type": "carry_on"},
            thresholds={},
            fees={},
            exceptions=["Liquids under 100ml in clear bag allowed"]
        ),
        
        # === SPECIAL ITEMS ===
        BaggageRule(
            rule_id="SPORTS-GOLF-001",
            airline="Generic",
            category=TaskCategory.SPECIAL_ITEMS,
            description="Golf equipment baggage rules",
            conditions={"item_type": "golf_clubs"},
            thresholds={"max_weight_kg": 23, "counts_as_bags": 1},
            fees={"standard_fee": 35, "overweight_fee": 100},
            exceptions=["May count as one of your checked bags"]
        ),
        BaggageRule(
            rule_id="SPORTS-SKI-001",
            airline="Generic",
            category=TaskCategory.SPECIAL_ITEMS,
            description="Ski/Snowboard equipment baggage rules",
            conditions={"item_type": "ski_equipment"},
            thresholds={"max_weight_kg": 23, "counts_as_bags": 1},
            fees={"standard_fee": 35, "overweight_fee": 100},
            exceptions=["Boot bag + ski bag may count as one item"]
        ),
    ]
    return rules


# =============================================================================
# TEST INSTANCES (Synthetic Benchmark)
# =============================================================================

def get_test_instances() -> List[RuleArenaInstance]:
    """
    Returns test instances for the RuleArena benchmark.
    
    Each instance tests a specific aspect of rule-based reasoning:
    - Simple: Single rule lookup
    - Moderate: Multiple conditions
    - Complex: Calculations with exceptions
    - Expert: Edge cases and ambiguity
    """
    instances = [
        # === SIMPLE: Single rule lookup ===
        RuleArenaInstance(
            instance_id=1,
            category=TaskCategory.BAGGAGE_ALLOWANCE,
            difficulty=DifficultyLevel.SIMPLE,
            query="I'm flying United economy class domestically. How much does my first checked bag cost?",
            passenger_context={
                "airline": "united",
                "class": "economy", 
                "route": "domestic",
                "num_bags": 1,
                "bag_weight_kg": 20,
                "membership_status": None
            },
            applicable_rules=["UA-DOM-ECON-001"],
            ground_truth_answer=35,
            ground_truth_explanation="United Economy domestic first bag fee is $35. Bag is under 23kg weight limit.",
            relevant_entities={"airline": "united", "class": "economy", "route": "domestic", "bag_number": 1}
        ),
        
        RuleArenaInstance(
            instance_id=2,
            category=TaskCategory.BAGGAGE_ALLOWANCE,
            difficulty=DifficultyLevel.SIMPLE,
            query="What's the baggage fee for Delta first class on a domestic flight?",
            passenger_context={
                "airline": "delta",
                "class": "first",
                "route": "domestic",
                "num_bags": 2,
                "bag_weight_kg": 25
            },
            applicable_rules=["DL-DOM-FIRST-001"],
            ground_truth_answer=0,
            ground_truth_explanation="Delta First Class domestic includes 3 free checked bags up to 32kg each.",
            relevant_entities={"airline": "delta", "class": "first", "route": "domestic"}
        ),
        
        # === MODERATE: Multiple conditions ===
        RuleArenaInstance(
            instance_id=3,
            category=TaskCategory.BAGGAGE_ALLOWANCE,
            difficulty=DifficultyLevel.MODERATE,
            query="I'm flying United economy internationally with two bags, each weighing 22kg. What's my total baggage cost?",
            passenger_context={
                "airline": "united",
                "class": "economy",
                "route": "international",
                "num_bags": 2,
                "bag_weights_kg": [22, 22]
            },
            applicable_rules=["UA-INT-ECON-001"],
            ground_truth_answer=100,
            ground_truth_explanation="United Economy international: 1st bag free, 2nd bag $100. Both bags under 23kg limit. Total: $0 + $100 = $100.",
            relevant_entities={"airline": "united", "class": "economy", "route": "international", "bag1_weight": 22, "bag2_weight": 22}
        ),
        
        RuleArenaInstance(
            instance_id=4,
            category=TaskCategory.EXCESS_FEE,
            difficulty=DifficultyLevel.MODERATE,
            query="My bag weighs 28kg. I'm on a Delta domestic economy flight. What are my total fees for this one bag?",
            passenger_context={
                "airline": "delta",
                "class": "economy",
                "route": "domestic",
                "num_bags": 1,
                "bag_weight_kg": 28
            },
            applicable_rules=["DL-DOM-ECON-001", "GENERIC-OVERWEIGHT-001"],
            ground_truth_answer=135,
            ground_truth_explanation="Delta Economy domestic first bag: $35. Bag is 28kg (over 23kg limit). Overweight fee (23-32kg): $100. Total: $35 + $100 = $135.",
            relevant_entities={"airline": "delta", "bag_weight": 28, "standard_fee": 35, "overweight_fee": 100}
        ),
        
        # === COMPLEX: Calculations with exceptions ===
        RuleArenaInstance(
            instance_id=5,
            category=TaskCategory.BAGGAGE_ALLOWANCE,
            difficulty=DifficultyLevel.COMPLEX,
            query="I'm a United MileagePlus Premier member flying economy domestically with 2 bags (20kg and 25kg). What's my total cost?",
            passenger_context={
                "airline": "united",
                "class": "economy",
                "route": "domestic",
                "num_bags": 2,
                "bag_weights_kg": [20, 25],
                "membership_status": "MileagePlus Premier"
            },
            applicable_rules=["UA-DOM-ECON-001", "GENERIC-OVERWEIGHT-001"],
            ground_truth_answer=145,
            ground_truth_explanation="Premier member gets 1 free bag. First bag (20kg): $0. Second bag (25kg): $45 + $100 overweight (over 23kg) = $145. Total: $0 + $145 = $145.",
            relevant_entities={"membership": "MileagePlus Premier", "free_bags": 1, "bag1_weight": 20, "bag2_weight": 25}
        ),
        
        RuleArenaInstance(
            instance_id=6,
            category=TaskCategory.SPECIAL_ITEMS,
            difficulty=DifficultyLevel.COMPLEX,
            query="I'm bringing my golf clubs on an American Airlines economy domestic flight. I also have one regular suitcase. What will I pay?",
            passenger_context={
                "airline": "american",
                "class": "economy",
                "route": "domestic",
                "items": ["golf_clubs", "suitcase"],
                "golf_weight_kg": 18,
                "suitcase_weight_kg": 22
            },
            applicable_rules=["AA-DOM-ECON-001", "SPORTS-GOLF-001"],
            ground_truth_answer=70,
            ground_truth_explanation="Golf clubs count as 1 checked bag: $35. Regular suitcase as 2nd bag: $35. Total: $35 + $35 = $70.",
            relevant_entities={"golf_clubs": True, "suitcase": True, "first_bag_fee": 35, "second_bag_fee": 35}
        ),
        
        # === EXPERT: Edge cases ===
        RuleArenaInstance(
            instance_id=7,
            category=TaskCategory.EXCESS_FEE,
            difficulty=DifficultyLevel.EXPERT,
            query="I have a 35kg bag and a 48kg bag. I'm flying Delta business domestic. What fees do I owe and can I even check both?",
            passenger_context={
                "airline": "delta",
                "class": "business",
                "route": "domestic",
                "num_bags": 2,
                "bag_weights_kg": [35, 48]
            },
            applicable_rules=["DL-DOM-FIRST-001", "GENERIC-OVERWEIGHT-001"],
            ground_truth_answer="First bag (35kg): $200 overweight fee. Second bag (48kg): CANNOT BE CHECKED - exceeds 45kg limit.",
            ground_truth_explanation="Business class gets 2 free bags up to 32kg. Bag 1 (35kg): over 32kg = $200 tier 2 fee. Bag 2 (48kg): over 45kg maximum, airline will not accept. Only 1 bag can be checked.",
            relevant_entities={"bag1_weight": 35, "bag2_weight": 48, "max_acceptable_weight": 45}
        ),
        
        RuleArenaInstance(
            instance_id=8,
            category=TaskCategory.BAGGAGE_ALLOWANCE,
            difficulty=DifficultyLevel.EXPERT,
            query="I'm a Delta SkyMiles Diamond Medallion flying economy from New York to Tokyo with 3 bags at 22kg each. Cost?",
            passenger_context={
                "airline": "delta",
                "class": "economy",
                "route": "international",
                "origin": "JFK",
                "destination": "NRT",
                "num_bags": 3,
                "bag_weights_kg": [22, 22, 22],
                "membership_status": "Diamond Medallion"
            },
            applicable_rules=["DL-DOM-ECON-001"],
            ground_truth_answer=0,
            ground_truth_explanation="Diamond Medallion members get 3 free checked bags on all flights. All bags under weight limit. Total: $0.",
            relevant_entities={"membership": "Diamond Medallion", "free_bags": 3, "international": True}
        ),
        
        # === Additional test cases for diversity ===
        RuleArenaInstance(
            instance_id=9,
            category=TaskCategory.PROHIBITED_ITEMS,
            difficulty=DifficultyLevel.SIMPLE,
            query="Can I bring a bottle of water through security in my carry-on?",
            passenger_context={
                "item": "water_bottle",
                "container_size_ml": 500,
                "bag_type": "carry_on"
            },
            applicable_rules=["TSA-PROHIBITED-001"],
            ground_truth_answer=False,
            ground_truth_explanation="Liquids over 100ml are prohibited in carry-on bags. A 500ml water bottle must be emptied or placed in checked luggage.",
            relevant_entities={"item": "water", "size_ml": 500, "limit_ml": 100}
        ),
        
        RuleArenaInstance(
            instance_id=10,
            category=TaskCategory.SPECIAL_ITEMS,
            difficulty=DifficultyLevel.MODERATE,
            query="I want to bring my skis and a boot bag on United domestic economy. Do they count as one item or two?",
            passenger_context={
                "airline": "united",
                "class": "economy",
                "route": "domestic",
                "items": ["skis", "boot_bag"],
                "ski_weight_kg": 8,
                "boot_bag_weight_kg": 7
            },
            applicable_rules=["UA-DOM-ECON-001", "SPORTS-SKI-001"],
            ground_truth_answer="One item",
            ground_truth_explanation="Ski bag + boot bag together count as one checked bag. Combined weight (15kg) is under 23kg limit. Fee: $35 for first checked bag.",
            relevant_entities={"skis": True, "boot_bag": True, "combined_as_one": True}
        ),
    ]
    return instances


# =============================================================================
# DATASET CLASS
# =============================================================================

class RuleArenaDataset:
    """
    RuleArena dataset wrapper.
    
    Provides access to:
    - Airline baggage rules
    - Test instances with ground truth
    - Filtered access by category, difficulty
    
    Usage:
        dataset = RuleArenaDataset()
        
        # Load all test data
        test_data = dataset.load("test")
        
        # Get rules for a specific airline
        rules = dataset.get_rules_for_airline("united")
        
        # Filter by difficulty
        simple_tasks = dataset.get_by_difficulty(DifficultyLevel.SIMPLE)
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize dataset loader.
        
        Args:
            data_path: Optional path to external data file.
                      If None, uses built-in synthetic data.
        """
        self.data_path = data_path
        self._rules: Optional[List[BaggageRule]] = None
        self._test_data: Optional[List[RuleArenaInstance]] = None
    
    def load(self, split: str = "test") -> List[RuleArenaInstance]:
        """
        Load specified split.
        
        Args:
            split: "test" (only test split available for now)
        
        Returns:
            List of RuleArenaInstance objects
        """
        if split != "test":
            raise ValueError(f"Only 'test' split available. Got: {split}")
        
        if self._test_data is not None:
            return self._test_data
        
        # Load from external file if provided
        if self.data_path and os.path.exists(self.data_path):
            self._test_data = self._load_from_file(self.data_path)
        else:
            # Use built-in synthetic data
            self._test_data = get_test_instances()
        
        print(f"Loaded {len(self._test_data)} RuleArena instances")
        return self._test_data
    
    def _load_from_file(self, path: str) -> List[RuleArenaInstance]:
        """Load instances from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        instances = []
        for item in data:
            instances.append(RuleArenaInstance(
                instance_id=item['instance_id'],
                category=TaskCategory(item['category']),
                difficulty=DifficultyLevel(item['difficulty']),
                query=item['query'],
                passenger_context=item['passenger_context'],
                applicable_rules=item['applicable_rules'],
                ground_truth_answer=item['ground_truth_answer'],
                ground_truth_explanation=item['ground_truth_explanation'],
                relevant_entities=item['relevant_entities'],
            ))
        return instances
    
    def get_rules(self) -> List[BaggageRule]:
        """Get all baggage rules."""
        if self._rules is None:
            self._rules = get_baggage_rules()
        return self._rules
    
    def get_rules_for_airline(self, airline: str) -> List[BaggageRule]:
        """Get rules for a specific airline."""
        all_rules = self.get_rules()
        return [r for r in all_rules if r.airline.lower() == airline.lower() or r.airline == "Generic"]
    
    def get_rule_by_id(self, rule_id: str) -> Optional[BaggageRule]:
        """Get a specific rule by ID."""
        for rule in self.get_rules():
            if rule.rule_id == rule_id:
                return rule
        return None
    
    def get_by_category(self, category: TaskCategory) -> List[RuleArenaInstance]:
        """Filter instances by category."""
        test_data = self.load("test")
        return [i for i in test_data if i.category == category]
    
    def get_by_difficulty(self, difficulty: DifficultyLevel) -> List[RuleArenaInstance]:
        """Filter instances by difficulty level."""
        test_data = self.load("test")
        return [i for i in test_data if i.difficulty == difficulty]
    
    def get_debug_subset(self, n: int = 5) -> List[RuleArenaInstance]:
        """Get first n instances for debugging."""
        test_data = self.load("test")
        return test_data[:n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        test_data = self.load("test")
        rules = self.get_rules()
        
        category_counts = {}
        difficulty_counts = {}
        
        for instance in test_data:
            cat = instance.category.value
            diff = instance.difficulty.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        return {
            "total_instances": len(test_data),
            "total_rules": len(rules),
            "by_category": category_counts,
            "by_difficulty": difficulty_counts,
            "airlines": list(set(r.airline for r in rules)),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_baggage_rules() -> List[BaggageRule]:
    """Load all airline baggage rules."""
    return get_baggage_rules()


def load_rule_arena(
    split: str = "test",
    debug: bool = False,
    debug_n: int = 5,
) -> List[RuleArenaInstance]:
    """
    Convenience function to load RuleArena dataset.
    
    Args:
        split: "test" (only test available)
        debug: If True, return only first debug_n instances
        debug_n: Number of debug instances
    
    Returns:
        List of RuleArenaInstance objects
    """
    dataset = RuleArenaDataset()
    
    if debug:
        return dataset.get_debug_subset(debug_n)
    
    return dataset.load(split)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RuleArena Dataset Loader - Test")
    print("=" * 80)
    
    # Load dataset
    dataset = RuleArenaDataset()
    
    # Print statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total instances: {stats['total_instances']}")
    print(f"  Total rules: {stats['total_rules']}")
    print(f"  Airlines: {stats['airlines']}")
    print(f"\n  By Category:")
    for cat, count in stats['by_category'].items():
        print(f"    {cat}: {count}")
    print(f"\n  By Difficulty:")
    for diff, count in stats['by_difficulty'].items():
        print(f"    {diff}: {count}")
    
    # Show sample instance
    print("\n" + "=" * 80)
    print("Sample Instance:")
    print("=" * 80)
    
    instance = dataset.load("test")[0]
    print(f"ID: {instance.instance_id}")
    print(f"Category: {instance.category.value}")
    print(f"Difficulty: {instance.difficulty.value}")
    print(f"Query: {instance.query}")
    print(f"Ground Truth: {instance.ground_truth_answer}")
    print(f"Explanation: {instance.ground_truth_explanation}")
    
    # Show prompt format
    print("\n" + "=" * 80)
    print("Prompt Format (for LLM):")
    print("=" * 80)
    rules = dataset.get_rules_for_airline("united")
    print(instance.get_prompt(include_rules=True, rules=rules))
