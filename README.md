# Behavior Distillation via Program Trace Prompting

> **Research Question**: Can we replace unreliable, expensive autonomous agents (L3) with predictable, cheaper, code-driven workflows (L0/L1) by "distilling" their behavior?

**Independent Study Research Project**  
**Collaborator**: Professor William Cohen (CMU ML / Google DeepMind)

---

## Overview

This project investigates the **Reliability/Autonomy Spectrum** in LLM-powered systems, using airline baggage fee calculation as a testbed domain (RuleArena).

### The Spectrum

| Level | Name | Description | Trade-offs |
|-------|------|-------------|------------|
| **L0** | Pure Code | Deterministic Python logic | Fast, free, rigid |
| **L1** | PTool | LLM extracts parameters → Python calculates | Predictable, testable, cheap |
| **L3** | ReAct Agent | Autonomous reasoning loop | Flexible but expensive/unstable |

### Core Hypothesis

By analyzing execution traces from L3 agents, we can systematically **distill** their behavior into L1 workflows that are:
- More reliable (deterministic execution paths)
- Cheaper (fewer LLM calls)
- Debuggable (Python-native control flow)

---

## Project Structure

```
ptp-behavior-distillation/
├── src/                          # Main source code
│   ├── dataset/                  # Data loaders
│   │   └── rule_arena_loader.py  # RuleArena baggage fees dataset
│   ├── experiments/              # Research experiments
│   │   ├── config.py             # Model configs, Together.ai setup
│   │   ├── l1_baggage.py         # L1 PTool implementation
│   │   └── l3_baggage_react.py   # L3 ReAct agent implementation
│   ├── metrics/                  # Cost/accuracy tracking
│   ├── secretagent/              # PTP framework (forked)
│   └── utils/                    # Helper utilities
├── scripts/                      # Utility scripts
│   ├── list_models.py            # List available Together.ai models
│   ├── test_together.py          # Test API connection
│   └── usage_tracker.py          # Track API usage/costs
├── data/                         # Datasets
│   └── BBH/                      # BIG-Bench Hard benchmark
├── archive/                      # Previous iterations
│   ├── demo_v1/                  # Initial demo implementation
│   └── old_experiments/          # Tutorial-style experiments
├── docs/                         # Documentation
│   ├── research-log.md           # Progress notes
│   ├── meeting-notes.md          # Advisor meetings
│   └── decisions.md              # Design decisions
└── results/                      # Experimental outputs
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Access

This project uses [Together.ai](https://together.ai) for LLM inference.

```bash
# Set your API key
export TOGETHER_API_KEY="your_key_here"
```

### 3. Verify Setup

```bash
# Test API connection
python scripts/test_together.py

# List available models
python scripts/list_models.py
```

### 4. Run Experiments

```python
# Load the RuleArena dataset
from src.dataset.rule_arena_loader import RuleArenaDataset

dataset = RuleArenaDataset()
dataset.load_synthetic()  # Load synthetic test cases

# Get a test instance
instance = dataset.get_instance(1)
print(f"Query: {instance.query}")
print(f"Expected: ${instance.ground_truth_answer}")

# Run L1 (PTool) approach
from src.experiments.l1_baggage import calculate_baggage_fee
result = calculate_baggage_fee(instance.query)
```

---

## Available Models

Configured in `src/experiments/config.py`:

| Model | Provider | Use Case |
|-------|----------|----------|
| `qwen-72b` | Qwen | Default - strong reasoning |
| `llama-70b` | Meta | Alternative large model |
| `deepseek-chat` | DeepSeek | Cost-effective option |
| `mixtral-8x7b` | Mistral | Fast inference |

---

## Research Methodology

### Phase 1: Baseline Implementation
- [x] Repository restructuring
- [x] RuleArena data loader
- [x] Together.ai integration
- [ ] L1 PTool baseline
- [ ] L3 ReAct baseline

### Phase 2: Trace Collection
- [ ] Instrument L3 agent with `sec.recorder()`
- [ ] Collect execution traces on test set
- [ ] Analyze trace patterns

### Phase 3: Distillation
- [ ] Extract common decision paths from traces
- [ ] Generate L1 workflows from patterns
- [ ] Validate distilled workflows

### Phase 4: Evaluation
- [ ] Accuracy comparison (L1 vs L3)
- [ ] Cost analysis (tokens, latency)
- [ ] Reliability metrics (variance, failure modes)

---

## Key Concepts

### Program Trace Prompting (PTP)
Unlike traditional agents where LLMs control execution flow, PTP inverts this relationship:
- **Python controls the workflow** (predictable, testable)
- **LLMs handle the "thinking"** (extraction, reasoning)

### The `@subagent()` Decorator
Transforms Python function stubs into LLM-powered callables:

```python
@subagent()
def extract_parameters(query: str) -> dict:
    """Extract airline, class, route, and bag count from query."""
    pass  # LLM fills in the implementation
```

### Trace Recording
Capture execution traces for analysis:

```python
with sec.recorder() as trace:
    result = agent.run(query)
# trace now contains all LLM calls and decisions
```

---

## References

- [SecretAgent Framework](https://github.com/Joshh99/secretagent) - Program Trace Prompting implementation
- [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) - Evaluation benchmark
- [Together.ai Documentation](https://docs.together.ai/) - LLM API reference

---

## License

Research project for academic purposes.
