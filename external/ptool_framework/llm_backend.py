"""
LLM Backend: Handles actual LLM calls and response parsing for ptools.

This module:
1. Loads LLM configuration from LLMS.json
2. Routes requests to appropriate providers
3. Handles response parsing (structured JSON or freeform)
4. Supports local and cloud-hosted models
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type, TypeVar, Union, get_args, get_origin

import logging

# Try to use loguru, fall back to standard logging
try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)

# Import LLM providers - these are optional
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    import together
    HAS_TOGETHER = True
except ImportError:
    HAS_TOGETHER = False

try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

from .ptool import PToolSpec

T = TypeVar("T")

# Flag to enable/disable trace collection globally
TRACE_ENABLED = True


# ============================================================================
# Exceptions
# ============================================================================

class LLMError(Exception):
    """Error during LLM execution."""
    pass


class ParseError(Exception):
    """Error parsing LLM response."""
    pass


class ConfigError(Exception):
    """Error in LLM configuration."""
    pass


# ============================================================================
# LLM Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""
    name: str
    provider: str
    model_id: str
    api_key_env: Optional[str] = None
    endpoint: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    description: str = ""
    cost: Union[Dict[str, Any], str] = field(default_factory=dict)
    context_window: int = 8192
    max_output: int = 4096

    def get_api_key(self) -> Optional[str]:
        """Get API key from environment variable."""
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    @property
    def is_local(self) -> bool:
        """Check if this is a local model."""
        return self.provider == "local" or self.cost == "local"


@dataclass
class LLMConfig:
    """Configuration for all available LLMs."""
    default_model: str
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    provider_defaults: Dict[str, Dict[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> LLMConfig:
        """
        Load LLM configuration from LLMS.json.

        Args:
            config_path: Path to LLMS.json. If None, searches in:
                1. Current directory
                2. Parent directory
                3. Package directory
        """
        if config_path is None:
            # Search for LLMS.json
            search_paths = [
                Path.cwd() / "LLMS.json",
                Path.cwd().parent / "LLMS.json",
                Path(__file__).parent.parent / "LLMS.json",
            ]
            for path in search_paths:
                if path.exists():
                    config_path = str(path)
                    break

        if config_path is None or not Path(config_path).exists():
            logger.warning("LLMS.json not found, using default configuration")
            return cls._default_config()

        logger.info(f"Loading LLM config from {config_path}")

        with open(config_path) as f:
            data = json.load(f)

        # Parse models
        models = {}
        for name, model_data in data.get("models", {}).items():
            models[name] = ModelConfig(
                name=name,
                provider=model_data.get("provider", "unknown"),
                model_id=model_data.get("model_id", name),
                api_key_env=model_data.get("api_key_env"),
                endpoint=model_data.get("endpoint"),
                capabilities=model_data.get("capabilities", []),
                description=model_data.get("description", ""),
                cost=model_data.get("cost", {}),
                context_window=model_data.get("context_window", 8192),
                max_output=model_data.get("max_output", 4096),
            )

        return cls(
            default_model=data.get("default_model", "deepseek-v3-0324"),
            models=models,
            provider_defaults=data.get("provider_defaults", {}),
        )

    @classmethod
    def _default_config(cls) -> LLMConfig:
        """Return default configuration when LLMS.json is not found."""
        return cls(
            default_model="deepseek-v3-0324",
            models={
                "deepseek-v3-0324": ModelConfig(
                    name="deepseek-v3-0324",
                    provider="together",
                    model_id="deepseek-ai/DeepSeek-V3",
                    api_key_env="TOGETHER_API_KEY",
                    capabilities=["reasoning", "coding"],
                    description="DeepSeek V3 via Together.ai",
                ),
            },
        )

    def get_model(self, name: Optional[str] = None) -> ModelConfig:
        """Get model configuration by name, or default if not specified."""
        if name is None:
            name = self.default_model

        # Check if name is in our models
        if name in self.models:
            return self.models[name]

        # Try to find by model_id
        for model in self.models.values():
            if model.model_id == name:
                return model

        raise ConfigError(f"Unknown model: {name}. Available: {list(self.models.keys())}")


# Global config instance
_CONFIG: Optional[LLMConfig] = None


def get_config() -> LLMConfig:
    """Get the global LLM configuration."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = LLMConfig.load()
    return _CONFIG


def reload_config(config_path: Optional[str] = None) -> LLMConfig:
    """Reload the LLM configuration."""
    global _CONFIG
    _CONFIG = LLMConfig.load(config_path)
    return _CONFIG


# ============================================================================
# Provider-Specific Implementations
# ============================================================================

def _call_together(prompt: str, model_config: ModelConfig, max_tokens: int = 4096) -> tuple[str, int, int]:
    """Call Together.ai API."""
    if not HAS_TOGETHER:
        # Fall back to OpenAI client with Together endpoint
        if not HAS_OPENAI:
            raise LLMError("Neither 'together' nor 'openai' package installed")

        api_key = model_config.get_api_key()
        if not api_key:
            raise LLMError(f"API key not found in {model_config.api_key_env}")

        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
        )

        try:
            completion = client.chat.completions.create(
                model=model_config.model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise LLMError(f"Together API error: {e}")

    # Use native together client
    # api_key = model_config.get_api_key()
    api_key = os.getenv(model_config.api_key_env)
    if not api_key:
        raise LLMError(f"API key not found in {model_config.api_key_env}")

    client = together.Together(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model_config.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        # Extract response text
        content = response.choices[0].message.content
        
        # Extract token counts from usage
        input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else 0
        output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else 0
        
        return content, input_tokens, output_tokens
        
    except Exception as e:
        raise LLMError(f"Together API error: {e}")


def _call_anthropic(prompt: str, model_config: ModelConfig, max_tokens: int = 4096) -> str:
    """Call Anthropic Claude API."""
    if not HAS_ANTHROPIC:
        raise LLMError("anthropic package not installed")

    api_key = model_config.get_api_key()
    if not api_key:
        raise LLMError(f"API key not found in {model_config.api_key_env}")

    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model=model_config.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        raise LLMError(f"Anthropic API error: {e}")


def _call_openai(prompt: str, model_config: ModelConfig, max_tokens: int = 4096) -> str:
    """Call OpenAI API."""
    if not HAS_OPENAI:
        raise LLMError("openai package not installed")

    api_key = model_config.get_api_key()
    client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    try:
        completion = client.chat.completions.create(
            model=model_config.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise LLMError(f"OpenAI API error: {e}")


def _call_groq(prompt: str, model_config: ModelConfig, max_tokens: int = 4096) -> str:
    """Call Groq API."""
    if not HAS_GROQ:
        raise LLMError("groq package not installed")

    api_key = model_config.get_api_key()
    if not api_key:
        raise LLMError(f"API key not found in {model_config.api_key_env}")

    client = groq.Groq(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model=model_config.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise LLMError(f"Groq API error: {e}")


def _call_google(prompt: str, model_config: ModelConfig, max_tokens: int = 4096) -> str:
    """Call Google Gemini API."""
    if not HAS_GOOGLE:
        raise LLMError("google-generativeai package not installed")

    api_key = model_config.get_api_key()
    if not api_key:
        raise LLMError(f"API key not found in {model_config.api_key_env}")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_config.model_id)

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise LLMError(f"Google API error: {e}")


def _call_local(prompt: str, model_config: ModelConfig, max_tokens: int = 4096) -> str:
    """Call local LLM endpoint (Ollama, vLLM, etc.)."""
    if not HAS_OPENAI:
        raise LLMError("openai package not installed (needed for local endpoint)")

    endpoint = model_config.endpoint
    if not endpoint:
        raise LLMError("No endpoint specified for local model")

    # Use OpenAI client with custom endpoint
    client = openai.OpenAI(
        api_key="not-needed",  # Local endpoints often don't need auth
        base_url=endpoint,
    )

    try:
        completion = client.chat.completions.create(
            model=model_config.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise LLMError(f"Local LLM error ({endpoint}): {e}")


# ============================================================================
# Main LLM Call Function
# ============================================================================

def call_llm(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 4096,
) -> tuple[str, int, int]:
    """
    Call an LLM with the given prompt.

    Routes to the appropriate provider based on LLMS.json configuration.

    Args:
        prompt: The prompt to send to the LLM
        model: Model name (from LLMS.json) or None for default
        max_tokens: Maximum tokens in response

    Returns:
        (response_text, input_tokens, output_tokens)
    """
    config = get_config()
    model_config = config.get_model(model)

    logger.info(f"Calling LLM: {model_config.name} ({model_config.provider})")
    logger.debug(f"Model ID: {model_config.model_id}, prompt_len={len(prompt)}")

    # Route to provider
    provider = model_config.provider.lower()

    if provider == "together":
        return _call_together(prompt, model_config, max_tokens)
    elif provider == "anthropic":
        return _call_anthropic(prompt, model_config, max_tokens)
    elif provider == "openai":
        return _call_openai(prompt, model_config, max_tokens)
    elif provider == "groq":
        return _call_groq(prompt, model_config, max_tokens)
    elif provider == "google":
        return _call_google(prompt, model_config, max_tokens)
    elif provider == "local":
        return _call_local(prompt, model_config, max_tokens)
    else:
        raise LLMError(f"Unknown provider: {provider}")


# ============================================================================
# Response Parsing
# ============================================================================

def _extract_json_object(text: str) -> Optional[str]:
    """
    Extract the first valid JSON object from text using balanced brace matching.

    Unlike a greedy regex, this correctly handles nested braces and won't
    match from the first '{' to the last '}' across multiple JSON objects.
    """
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == '\\' and in_string:
            escape_next = True
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def parse_structured_response(response: str, expected_type: Type) -> Any:
    """
    Parse a structured JSON response from the LLM.
    
    The LLM is instructed to return {"result": <value>}, so we:
    1. Extract the JSON object
    2. Get the "result" field
    3. Validate it matches expected_type
    
    Args:
        response: Raw LLM response text
        expected_type: Expected Python type
        
    Returns:
        Parsed result (unwrapped from {"result": ...})
    """
    # Extract JSON from response (handles markdown fences)
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        raise ParseError(f"No JSON object found in response: {response[:200]}...")
    
    json_str = json_match.group()
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}\nResponse: {response[:200]}...")
    
    # Check if it's wrapped in {"result": ...}
    if isinstance(data, dict) and "result" in data:
        # Unwrap it
        result = data["result"]
    else:
        # No wrapper, use as-is
        result = data
    
    # Type validation (basic)
    if expected_type != Any:
        # For Dict types, check if it's a dict
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is dict:
            if not isinstance(result, dict):
                raise ParseError(f"Expected dict, got {type(result)}")
        # For List types, check if it's a list
        elif hasattr(expected_type, '__origin__') and expected_type.__origin__ is list:
            if not isinstance(result, list):
                raise ParseError(f"Expected list, got {type(result)}")
    
    return result

def parse_freeform_response(response: str, return_type: Type[T]) -> T:
    """
    Parse a freeform response from an LLM.

    Expects format with "ANSWER: <value>" on the last line.
    """
    # Look for ANSWER: pattern
    answer_match = re.search(r'ANSWER:\s*(.+?)$', response, re.MULTILINE | re.IGNORECASE)

    if answer_match:
        answer_str = answer_match.group(1).strip()
    else:
        # Use the last non-empty line
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        if lines:
            answer_str = lines[-1]
        else:
            raise ParseError(f"Could not extract answer from response: {response[:200]}...")

    # Try to parse as JSON first, then as literal
    try:
        result = json.loads(answer_str)
    except json.JSONDecodeError:
        result = answer_str

    # Coerce to expected type
    result = _coerce_type(result, return_type)
    return result


def _coerce_type(value: Any, target_type: Type[T]) -> T:
    """
    Attempt to coerce a value to the target type.

    Handles common cases like Literal types, Lists, etc.
    """
    origin = get_origin(target_type)
    args = get_args(target_type)

    # Handle Union/Optional types (Optional[X] is Union[X, None])
    if origin is Union:
        # Filter out NoneType from args
        non_none_args = [a for a in args if a is not type(None)]
        # If value is None and None is allowed, return None
        if value is None and type(None) in args:
            return None
        # Try coercing to each non-None type in order
        for arg_type in non_none_args:
            try:
                return _coerce_type(value, arg_type)
            except (ParseError, TypeError, ValueError):
                continue
        # If nothing worked, return as-is
        return value

    # Handle Literal types
    if origin is Literal:
        if value in args:
            return value
        # Try case-insensitive match for strings
        if isinstance(value, str):
            for arg in args:
                if isinstance(arg, str) and value.lower() == arg.lower():
                    return arg
        raise ParseError(f"Value {value!r} not in Literal{args}")

    # Handle List types
    if origin is list:
        if not isinstance(value, list):
            raise ParseError(f"Expected list, got {type(value).__name__}")
        if args:
            item_type = args[0]
            return [_coerce_type(item, item_type) for item in value]
        return value

    # Handle Dict types
    if origin is dict:
        if not isinstance(value, dict):
            raise ParseError(f"Expected dict, got {type(value).__name__}")
        return value

    # Handle Tuple types
    if origin is tuple:
        if isinstance(value, (list, tuple)):
            if args:
                return tuple(_coerce_type(v, t) for v, t in zip(value, args))
            return tuple(value)
        raise ParseError(f"Expected tuple, got {type(value).__name__}")

    # Handle basic types
    if target_type is str:
        return str(value)
    if target_type is int:
        if isinstance(value, int):
            return value
        # Extract integer from string like "42 points" or "Score: 3"
        match = re.search(r'[-+]?\d+', str(value))
        if match:
            return int(match.group())
        raise ParseError(f"Could not extract int from: {value}")
    if target_type is float:
        if isinstance(value, (int, float)):
            return float(value)
        # Extract number from string like "23.5 kg/m²" or "BMI: 26.12"
        match = re.search(r'[-+]?\d*\.?\d+', str(value))
        if match:
            return float(match.group())
        raise ParseError(f"Could not extract float from: {value}")
    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1')
        return bool(value)

    # Default: return as-is
    return value


# ============================================================================
# ptool Execution
# ============================================================================

def execute_ptool(
    spec: PToolSpec,
    kwargs: Dict[str, Any],
    llm_backend: Optional[Callable] = None,
) -> Any:
    """
    Execute a ptool via LLM with full tracking.
    
    This is the core execution function that:
    1. Formats the prompt from the ptool spec
    2. Calls the LLM
    3. Parses the response
    4. Tracks tokens, cost, and execution time
    5. Logs to trace store
    
    Args:
        spec: PToolSpec defining the ptool
        kwargs: Arguments to pass to the ptool
        llm_backend: Optional custom LLM backend
        
    Returns:
        Parsed result matching spec.return_type
    """
    from .trace_store import get_trace_store, ExecutionTrace
    import time
    import uuid
    
    trace_store = get_trace_store()
    trace_id = str(uuid.uuid4())[:8]
    
    # Emit start event
    trace_store.emit_ptool_start(
        ptool_name=spec.name,
        args=kwargs,
        trace_id=trace_id,
    )
    
    # Format prompt
    prompt = spec.format_prompt(**kwargs)
    
    # Emit LLM request
    trace_store.emit_llm_request(
        trace_id=trace_id,
        model=spec.model,
        prompt=prompt,
    )
    
    # Execute
    start_time = time.time()
    try:
        if llm_backend:
            # Custom backend - doesn't return tokens
            response = llm_backend(prompt, spec.model)
            input_tokens = 0
            output_tokens = 0
        else:
            # Standard backend - returns (response, input_tokens, output_tokens)
            response, input_tokens, output_tokens = call_llm(prompt, spec.model)
        
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Emit LLM response
        trace_store.emit_llm_response(
            trace_id=trace_id,
            response=response,
            latency_ms=execution_time_ms,
        )
        
        # Parse response based on output mode
        if spec.output_mode == "structured":
            result = parse_structured_response(response, spec.return_type)
        else:
            result = parse_freeform_response(response, spec.return_type)
        
        # Calculate cost
        estimated_cost = calculate_cost(
            model=spec.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        
        # Log execution trace
        trace = ExecutionTrace(
            ptool_name=spec.name,
            inputs=kwargs,
            output=result,
            success=True,
            execution_time_ms=execution_time_ms,
            model_used=spec.model,
            trace_id=trace_id,
            prompt=prompt,
            raw_response=response,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost=estimated_cost,
        )
        trace_store.log_execution(trace)
        
        return result
        
    except Exception as e:
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Log failed execution
        trace = ExecutionTrace(
            ptool_name=spec.name,
            inputs=kwargs,
            output=None,
            success=False,
            execution_time_ms=execution_time_ms,
            model_used=spec.model,
            trace_id=trace_id,
            error=str(e),
            prompt=prompt if 'prompt' in locals() else None,
            raw_response=response if 'response' in locals() else None,
            prompt_tokens=input_tokens if 'input_tokens' in locals() else 0,
            completion_tokens=output_tokens if 'output_tokens' in locals() else 0,
            total_tokens=0,
            estimated_cost=0.0,
        )
        trace_store.log_execution(trace)
        
        trace_store.emit_error(trace_id=trace_id, error=str(e), ptool_name=spec.name)
        raise


def enable_tracing(enabled: bool = True) -> None:
    """Enable or disable global trace collection."""
    global TRACE_ENABLED
    TRACE_ENABLED = enabled
    logger.info(f"Trace collection {'enabled' if enabled else 'disabled'}")


# ============================================================================
# Mock Backend for Testing
# ============================================================================

class MockLLMBackend:
    """
    Mock LLM backend for testing.

    Stores predefined responses for specific inputs.
    """

    def __init__(self):
        self.responses: Dict[str, str] = {}
        self.call_log: list = []

    def add_response(self, prompt_contains: str, response: str) -> None:
        """Add a mock response for prompts containing the given string."""
        self.responses[prompt_contains] = response

    def __call__(self, prompt: str, model: str) -> str:
        """Return a mock response."""
        self.call_log.append({"prompt": prompt, "model": model})

        for key, response in self.responses.items():
            if key in prompt:
                return response

        # Default: return a generic JSON response
        return '{"result": "mock_response"}'


# ============================================================================
# Utility Functions
# ============================================================================

def list_available_models() -> List[str]:
    """List all available models from LLMS.json."""
    config = get_config()
    return list(config.models.keys())


def get_model_info(model: str) -> Dict[str, Any]:
    """Get information about a specific model."""
    config = get_config()
    model_config = config.get_model(model)
    return {
        "name": model_config.name,
        "provider": model_config.provider,
        "model_id": model_config.model_id,
        "capabilities": model_config.capabilities,
        "description": model_config.description,
        "cost": model_config.cost,
        "context_window": model_config.context_window,
        "max_output": model_config.max_output,
        "is_local": model_config.is_local,
    }


def select_model_by_capability(capability: str) -> str:
    """Select the cheapest model that has a given capability."""
    config = get_config()

    matching = []
    for name, model in config.models.items():
        if capability in model.capabilities:
            matching.append(model)

    if not matching:
        raise ConfigError(f"No model found with capability: {capability}")

    # Sort by cost (local models are "free")
    def get_cost(m: ModelConfig) -> float:
        if m.cost == "local":
            return 0.0
        if isinstance(m.cost, dict):
            return m.cost.get("input", 0) + m.cost.get("output", 0)
        return float('inf')

    matching.sort(key=get_cost)
    return matching[0].name

# Model pricing (USD per 1M tokens)
# Prices from Together.ai as of Feb 2026
MODEL_PRICING = {
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {
        "input": 0.88,
        "output": 0.88,
    },
    "Qwen/Qwen2.5-7B-Instruct-Turbo": {
        "input": 1.20,
        "output": 1.20,
    },
    "deepseek-ai/DeepSeek-V3": {
        "input": 0.60,
        "output": 1.25,
    },
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost in USD."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        # Default pricing if model not found
        return (input_tokens + output_tokens) / 1_000_000 * 1.0
    
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost