# Behavior Distillation via Program Trace Prompting

**Python programs calling LLMs, not LLMs calling tools.**

A research framework for investigating the trade-offs between control and autonomy in LLM-powered applications.

---

> **Research Question**: Can we replace unreliable, expensive autonomous agents (L3) with predictable, cheaper, code-driven workflows (L0/L1) by "distilling" their behavior?

**Independent Study Research Project**  
**Advisor**: Professor William Cohen (CMU Machine Learning / Google DeepMind)

---

## The Idea

Traditional agent frameworks have LLMs decide what tools to call. This is **unpredictable** and **hard to test**.

This framework **inverts the relationship**: Python controls the workflow, LLMs handle the "thinking" parts.

```python
from src.experiments.l1_baggage import baggage_allowance_l1

# Python controls the flow
query = "Calculate the total fee for a Gold member flying from NYC to LAX with 2 bags"
result = baggage_allowance_l1(query)

# LLM extracts parameters → Python calculates with actual fee tables
print(f"Fee: ${result['fee']}")  # Deterministic, testable, debuggable
```

---

## Overview

This project investigates the **Reliability/Autonomy Spectrum** using airline baggage fee calculation as a testbed domain ([RuleArena](https://github.com/SkyRiver-2000/RuleArena)).

### The Spectrum

| Level | Name | Description | Reliability | Cost |
|-------|------|-------------|-------------|------|
| **L0** | Pure Code | Deterministic Python logic | 100% | Free |
| **L1** | PTool | LLM extracts parameters → Python calculates | 95%+ target | Low |
| **L3** | ReAct Agent | Autonomous reasoning loop | 60-75% typical | High |

### Core Hypothesis

By analyzing execution traces from L3 agents, we can systematically **distill** their behavior into L1 workflows that are:
- **More reliable** (deterministic execution paths)
- **Cheaper** (fewer LLM calls)  
- **Debuggable** (Python-native control flow)

### Initial Results (Complexity Level 0)

| Metric | L1 (PTool) | L3 (ReAct) |
|--------|------------|------------|
| **Accuracy** | **80%** | 0% |
| **Cost/Problem** | **$0.0005** | $0.0008 |
| **Time/Problem** | **2.3s** | 7.7s |

📊 **[View Full Report](report.html)** for interactive charts and detailed analysis.

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

### 1. Setup

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/ptp-behavior-distillation
cd ptp-behavior-distillation
pip install -r requirements.txt

# Set API key
export TOGETHER_API_KEY="your-key-here"
```

### 2. Verify Setup

```bash
# Test API connection
python scripts/test_together.py

# List available models
python scripts/list_models.py
```

### 3. Run Baseline Experiments

```bash
# Run both L1 and L3 on 30 problems
python scripts/run_rulearena_baselines.py --num-problems 30 --complexity 0

# Run L1 only
python scripts/run_rulearena_baselines.py --num-problems 30 --skip-l3

# Run L3 only
python scripts/run_rulearena_baselines.py --num-problems 30 --skip-l1
```

Results are saved to:
- `results/baseline_comparison.md` - Markdown summary
- `results/baseline_comparison.json` - Raw data for analysis
- `report.html` - Interactive visualization

### 4. Use the Data Loader

```python
from src.dataset.rulearena_loader import RuleArenaLoader

# Load RuleArena problems
loader = RuleArenaLoader("external/RuleArena")
problems = loader.load_airline_problems(complexity_level=0, max_problems=10)

for p in problems:
    print(f"Query: {p['query'][:80]}...")
    print(f"Ground Truth: ${p['ground_truth']}")
```

### 5. Collect Traces (Coming Soon)

```python
from src.secretagent import sec

# Enable tracing for L3 agent
with sec.recorder() as trace:
    result = l3_agent.run(query)

# Analyze traces for distillation
trace.save("traces/experiment_001.json")
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

### Phase 1: Baseline Implementation ✅
- [x] Repository restructuring (src-layout)
- [x] RuleArena data loader with ground truth computation
- [x] Together.ai integration
- [x] L1 PTool baseline (80% accuracy)
- [x] L3 ReAct baseline (0% accuracy)
- [x] Baseline comparison report

### Phase 2: Trace Collection 🔄
- [ ] Instrument L3 agent with `sec.recorder()`
- [ ] Collect execution traces on test set
- [ ] Analyze trace patterns for distillation candidates

### Phase 3: Distillation
- [ ] Extract common decision paths from traces
- [ ] Generate L1 workflows from patterns
- [ ] Validate distilled workflows

### Phase 4: Evaluation
- [x] Accuracy comparison (L1 vs L3) - Initial results in
- [x] Cost analysis (tokens, latency) - L1 is 1.6× cheaper
- [ ] Higher complexity levels (1, 2)
- [ ] Reliability metrics (variance across runs)

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

## Troubleshooting

**"API key not found"**
```bash
export TOGETHER_API_KEY="your-key"
# On Windows PowerShell:
$env:TOGETHER_API_KEY="your-key"
```

**"Module not found"**
```bash
cd /path/to/ptp-behavior-distillation
python -c "from src.experiments.config import call_llm"
```

**"Connection timeout to Together.ai"**
- Check network/firewall settings
- Verify API key is valid at [api.together.ai](https://api.together.ai)

---

## References

- [RuleArena Benchmark](https://github.com/SkyRiver-2000/RuleArena) - Airline baggage fee calculation dataset
- [SecretAgent Framework](https://github.com/Joshh99/secretagent) - Program Trace Prompting implementation
- [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) - Evaluation benchmark
- [Together.ai Documentation](https://docs.together.ai/) - LLM API reference

---

## Research Context

This framework implements ideas from William Cohen's agent research:

- **ptools**: Prompt templates with type signatures
- **Program Trace Prompting**: Observable execution traces  
- **Behavior Distillation**: Converting LLM behavior to deterministic Python

The goal: Start with flexible LLM agents, progressively optimize to deterministic code.

---

## License

Research project for academic purposes. MIT License.
