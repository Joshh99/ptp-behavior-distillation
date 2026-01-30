"""
Experiment Configuration

Shared configuration for RuleArena experiments on the Reliability/Autonomy Spectrum.

Research Question:
    Can we replace unreliable, expensive autonomous agents (L3) with predictable,
    cheaper, code-driven workflows (L0/L1) by "distilling" their behavior?

Levels:
    L0 (Code): Pure Python. Fast, free, rigid.
    L1 (PTool): LLM extracts parameters → Python calculates. (The "SecretAgent" sweet spot)
    L3 (ReAct): Autonomous loop. Flexible but expensive/unstable.
    
Goal: Distill L3 → L1 behavior for the airline baggage domain.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


# =============================================================================
# ENVIRONMENT SETUP - Together.ai
# =============================================================================

# Together.ai API (primary provider)
TOGETHER_API_KEY = "tgp_v1_ltHhNSf3rT-bfZKM6wJpoDIZxEK9KfByeRuuHNnu2GA"
os.environ.setdefault('TOGETHER_API_KEY', TOGETHER_API_KEY)


# =============================================================================
# MODEL CONFIGURATION - Together.ai
# =============================================================================

class ModelProvider(Enum):
    """Supported LLM providers."""
    TOGETHER = "together"               # Together.ai (primary)
    OPENAI = "openai"                   # Direct OpenAI API (backup)
    ANTHROPIC = "anthropic"             # Claude models


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""
    name: str                       # Model ID for API calls
    provider: ModelProvider
    cost_per_m_input: float         # USD per 1M input tokens
    cost_per_m_output: float        # USD per 1M output tokens
    max_tokens: int = 4096
    temperature: float = 0.0        # Deterministic for reproducibility
    
    @property
    def cost_per_1k_input(self) -> float:
        return self.cost_per_m_input / 1000
    
    @property
    def cost_per_1k_output(self) -> float:
        return self.cost_per_m_output / 1000


# Available models on Together.ai
# Pricing: https://www.together.ai/pricing
MODELS: Dict[str, ModelConfig] = {
    # Qwen 2.5 (RECOMMENDED - good balance of cost/quality)
    "qwen-72b": ModelConfig(
        name="Qwen/Qwen2.5-72B-Instruct-Turbo",
        provider=ModelProvider.TOGETHER,
        cost_per_m_input=0.88,
        cost_per_m_output=0.88,
    ),
    "qwen-7b": ModelConfig(
        name="Qwen/Qwen2.5-7B-Instruct-Turbo",
        provider=ModelProvider.TOGETHER,
        cost_per_m_input=0.30,
        cost_per_m_output=0.30,
    ),
    # Llama 3.1 (Meta)
    "llama-70b": ModelConfig(
        name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        provider=ModelProvider.TOGETHER,
        cost_per_m_input=0.88,
        cost_per_m_output=0.88,
    ),
    "llama-8b": ModelConfig(
        name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        provider=ModelProvider.TOGETHER,
        cost_per_m_input=0.18,
        cost_per_m_output=0.18,
    ),
    "llama-405b": ModelConfig(
        name="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        provider=ModelProvider.TOGETHER,
        cost_per_m_input=3.50,
        cost_per_m_output=3.50,
    ),
    # DeepSeek (cheapest for complex reasoning)
    "deepseek-v3": ModelConfig(
        name="deepseek-ai/DeepSeek-V3",
        provider=ModelProvider.TOGETHER,
        cost_per_m_input=0.50,
        cost_per_m_output=1.50,
        max_tokens=8192,
    ),
    # Mixtral (good for structured output)
    "mixtral-8x22b": ModelConfig(
        name="mistralai/Mixtral-8x22B-Instruct-v0.1",
        provider=ModelProvider.TOGETHER,
        cost_per_m_input=1.20,
        cost_per_m_output=1.20,
    ),
}

# Default model for experiments (cost-effective)
DEFAULT_MODEL = "qwen-72b"


# =============================================================================
# EXPERIMENT LEVELS
# =============================================================================

class ExperimentLevel(Enum):
    """
    The Reliability/Autonomy Spectrum.
    
    Lower levels = more reliable, less flexible
    Higher levels = more flexible, less reliable
    """
    L0 = "L0"   # Pure Python (baseline, no LLM)
    L1 = "L1"   # PTool: LLM extracts → Python calculates
    L2 = "L2"   # Distilled: Python-first with LLM fallback  
    L3 = "L3"   # ReAct: Autonomous agent loop


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    level: ExperimentLevel
    
    # Model settings
    model: str = DEFAULT_MODEL
    
    # L3 ReAct settings
    max_steps: int = 10
    
    # Evaluation settings
    num_samples: Optional[int] = None   # None = all samples
    debug: bool = False
    
    # Output settings
    save_traces: bool = True
    output_dir: str = "results"
    
    # Rate limiting (for API calls)
    rate_limit_delay: float = 1.0       # Seconds between API calls
    max_retries: int = 3
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "level": self.level.value,
            "model": self.model,
            "max_steps": self.max_steps,
            "num_samples": self.num_samples,
            "debug": self.debug,
            "save_traces": self.save_traces,
            "output_dir": self.output_dir,
        }


# =============================================================================
# PRE-DEFINED EXPERIMENT CONFIGS
# =============================================================================

EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    # L0: Pure Python baseline
    "l0_baseline": ExperimentConfig(
        name="l0_baseline",
        description="Pure Python rule engine - no LLM calls",
        level=ExperimentLevel.L0,
    ),
    
    # L1: PTool (secretagent) - THE SWEET SPOT
    "l1_baggage": ExperimentConfig(
        name="l1_baggage",
        description="PTool extraction: LLM extracts parameters, Python calculates fees",
        level=ExperimentLevel.L1,
        model="qwen-72b",  # Together.ai
    ),
    
    # L2: Distilled (Python-first with LLM fallback)
    "l2_distilled": ExperimentConfig(
        name="l2_distilled",
        description="Distilled workflow: regex/rules first, LLM fallback for edge cases",
        level=ExperimentLevel.L2,
        model="qwen-72b",
    ),
    
    # L3: ReAct Agent (autonomous)
    "l3_baggage_react": ExperimentConfig(
        name="l3_baggage_react",
        description="ReAct agent: autonomous reasoning with tool use",
        level=ExperimentLevel.L3,
        model="qwen-72b",
        max_steps=10,
    ),
    
    # Debug configs (use cheaper model)
    "debug_l1": ExperimentConfig(
        name="debug_l1",
        description="Debug L1 with 3 samples",
        level=ExperimentLevel.L1,
        model="llama-8b",  # Cheapest for testing
        num_samples=3,
        debug=True,
    ),
    "debug_l3": ExperimentConfig(
        name="debug_l3", 
        description="Debug L3 with 3 samples",
        level=ExperimentLevel.L3,
        model="llama-8b",
        num_samples=3,
        debug=True,
        max_steps=5,
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_experiment_config(name: str) -> ExperimentConfig:
    """Get experiment configuration by name."""
    if name not in EXPERIMENT_CONFIGS:
        available = ", ".join(EXPERIMENT_CONFIGS.keys())
        raise ValueError(f"Unknown experiment: {name}. Available: {available}")
    return EXPERIMENT_CONFIGS[name]


def get_model_config(name: str) -> ModelConfig:
    """Get model configuration by name."""
    if name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return MODELS[name]


def list_experiments() -> List[str]:
    """List all available experiment names."""
    return list(EXPERIMENT_CONFIGS.keys())


def list_models() -> List[str]:
    """List all available model names."""
    return list(MODELS.keys())


def get_together_client():
    """Get configured Together.ai client."""
    from together import Together
    return Together(api_key=TOGETHER_API_KEY)


def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """
    Call LLM via Together.ai.
    
    Args:
        prompt: The prompt to send
        model: Model shortname (e.g., "qwen-72b") or full name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = deterministic)
    
    Returns:
        Generated text response
    """
    client = get_together_client()
    
    # Get model config if using shortname
    if model in MODELS:
        model_id = MODELS[model].name
    else:
        model_id = model
    
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    return response.choices[0].message.content


# =============================================================================
# PATHS
# =============================================================================

# Project root (relative to this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Results directory
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Data directory  
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Experiment Configuration - Together.ai")
    print("=" * 60)
    
    print("\nAvailable Models (Together.ai):")
    print("-" * 60)
    for name, config in MODELS.items():
        cost_str = f"${config.cost_per_m_input:.2f}/${config.cost_per_m_output:.2f} per 1M tokens"
        print(f"  {name:15} | {cost_str}")
        print(f"  {'':15} | {config.name}")
    
    print(f"\nDefault Model: {DEFAULT_MODEL}")
    
    print("\nAvailable Experiments:")
    print("-" * 60)
    for name, config in EXPERIMENT_CONFIGS.items():
        print(f"  {name}: {config.level.value} - {config.description}")
    
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Results Dir: {RESULTS_DIR}")
    
    # Test Together.ai connection
    print("\n" + "=" * 60)
    print("Testing Together.ai connection...")
    try:
        response = call_llm("What is 2+2? Reply with just the number.", model="llama-8b", max_tokens=10)
        print(f"✓ Together.ai working! Response: {response.strip()}")
    except Exception as e:
        print(f"✗ Together.ai error: {e}")