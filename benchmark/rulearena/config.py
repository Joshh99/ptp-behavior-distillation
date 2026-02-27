"""
Experiment Configuration for RuleArena Benchmark

Defines experiment levels, model configuration, and ablation study configs
following MedCalc conventions.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class ExperimentLevel(Enum):
    """
    Experiment levels for the Reliability/Autonomy Spectrum.

    Levels are defined according to rq2_experiment_prompt.md:
    - L0: Pure Python baseline (no LLM)
    - L0F: Chain-of-Thought baseline (pure LLM, no structure)
    - L1: PTool extraction + Python calculation
    - L1-TA: Tool-Augmented (LLM generates code)
    - L3: ReAct autonomous agent
    """
    L0 = "L0"
    L0F = "L0F"
    L1 = "L1"
    L1_TA = "L1-TA"
    L3 = "L3"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    level: ExperimentLevel
    description: str

    # Model settings (single model per constraints)
    model_id: str = "deepseek-ai/DeepSeek-V3"
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 4096

    # Execution settings
    num_samples: Optional[int] = None  # None = all instances
    debug: bool = False

    # Output settings
    save_traces: bool = True
    save_raw_responses: bool = True

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "level": self.level.value,
            "description": self.description,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "num_samples": self.num_samples,
            "debug": self.debug,
        }


# Model Configuration (DeepSeek-V3 via Together.ai)
MODEL_CONFIG = {
    "model_id": "deepseek-ai/DeepSeek-V3",
    "provider": "together",
    "api_base": "https://api.together.xyz/v1",
    "pricing": {
        "input_per_million": 0.30,   # $0.30/M input tokens
        "output_per_million": 0.90,  # $0.90/M output tokens
    },
    "default_params": {
        "temperature": 0.0,
        "max_tokens": 4096,
        "seed": 42,
    }
}


# Ablation Study Configurations
ABLATION_CONFIGS = {
    "l0_python": ExperimentConfig(
        name="l0_python",
        level=ExperimentLevel.L0,
        description="Pure Python baseline - no LLM, oracle parameters",
    ),

    "l0f_cot": ExperimentConfig(
        name="l0f_cot",
        level=ExperimentLevel.L0F,
        description="Chain-of-Thought baseline - pure LLM reasoning",
    ),

    "l1_ptool": ExperimentConfig(
        name="l1_ptool",
        level=ExperimentLevel.L1,
        description="PTool extraction + Python calculation",
    ),

    "l1ta_tool_augmented": ExperimentConfig(
        name="l1ta_tool_augmented",
        level=ExperimentLevel.L1_TA,
        description="Tool-augmented - LLM generates and executes code",
    ),

    "l3_react": ExperimentConfig(
        name="l3_react",
        level=ExperimentLevel.L3,
        description="ReAct autonomous agent with tool access",
    ),
}


# Helper functions
def get_experiment_config(name: str) -> ExperimentConfig:
    """Get experiment configuration by name."""
    if name not in ABLATION_CONFIGS:
        available = ", ".join(ABLATION_CONFIGS.keys())
        raise ValueError(f"Unknown experiment: {name}. Available: {available}")
    return ABLATION_CONFIGS[name]


def list_experiments():
    """List all available experiments."""
    return list(ABLATION_CONFIGS.keys())


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for given token counts."""
    pricing = MODEL_CONFIG["pricing"]
    cost = (input_tokens * pricing["input_per_million"] / 1_000_000 +
            output_tokens * pricing["output_per_million"] / 1_000_000)
    return cost


# Environment
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY environment variable not set")
