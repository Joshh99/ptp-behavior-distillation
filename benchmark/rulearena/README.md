# PTP Behavior Distillation -- RuleArena Benchmark

## Overview

This project investigates the **Reliability/Autonomy Spectrum** using the [RuleArena](https://github.com/SkyRiver-2000/RuleArena) benchmark across three rule-guided reasoning domains. This implements the PTP (Parameter-Tool-Program) framework on the RuleArena benchmark.

### The Spectrum

| Level | Name | Description | Reliability | Cost |
|-------|------|-------------|-------------|------|
| **L0** | Pure Code | Deterministic Python logic | 100% | Free |
| **L0F** | CoT | Direct LLM chain-of-thought | Low | Low |
| **L1** | PTool | LLM extracts parameters → Python calculates | High | Low |
| **L1-TA** | Tool-Aug | LLM generates and executes code | Medium | Medium |
| **L3** | ReAct Agent | Autonomous reasoning loop | Variable | High |

---

## Domains

**Airline baggage fees** (300 instances): Given a passenger itinerary and baggage details, compute the total baggage fee by applying multi-tier airline rules (fare class, route, loyalty status, bag weight/size).

**Tax computation** (300 instances): Given a taxpayer's IRS form fields, compute federal tax liability by applying bracket rules, deductions, and credits.

**NBA transaction compliance** (216 instances): Given a proposed roster transaction, determine whether it violates NBA collective bargaining rules (salary cap, roster limits, contract restrictions). Binary classification; evaluated with F1 macro due to 82.9% class imbalance.

---


## Repository Structure
```
benchmark/rulearena/
├── experiments/        # L0/L1/L3 implementations
│   ├── l0_python.py    # Oracle (ground truth extraction)
│   ├── l0f_cot.py      # End-to-end CoT baseline
│   ├── l1_ptool.py     # PTP: LLM extract -> Python compute
│   └── l3_react.py     # ReAct agent with ptool loop
├── calculators/        # Deterministic Python calculators (airline, tax)
├── dataset/            # RuleArena data loader
├── metrics/            # Aggregation and scoring
├── run_single.py       # Main entry point
└── config.py           # Model and experiment configuration
external/
├── ptool_framework/    # Shared agent framework (ptool, ReActAgent)
└── RuleArena/          # Original benchmark reference implementation
```

## Results Summary
| Level | Airline | Tax | NBA (F1) |
|-------|---------|-----|----------|
| L0 oracle | ~100% | ~100% | -- |
| L0F CoT | 48.3% | 35.3% | 0.50 |
| L1 PTP | 77.0% | 99.7% | 0.44 |
| L3 ReAct | TBD | TBD | TBD |

## Running RQ1 Experiments

RQ1 evaluates four approaches on the airline baggage fee domain across multiple models and complexity levels.

```bash
# Run all experiments for a single model (100 problems x 3 complexity levels)
python -m benchmark.rq1.run_full_baseline --num-problems 100 --complexity 0 1 2 \
  --model "deepseek-ai/DeepSeek-V3" --output results/rq1/baseline_deepseek.json --yes

# Supported models: deepseek-ai/DeepSeek-V3, meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo, Qwen/Qwen2.5-7B-Instruct-Turbo
# To run a single complexity level or single experiment type, replace --complexity and add --experiment <cot|tool_aug|l1_pure|l1_transparent>
```

---

## Running RQ2 Experiments

RQ2 evaluates the full reliability/autonomy spectrum across all three domains.

```bash
# L1 PTool
python -m benchmark.rulearena.run_single --experiment l1_ptool --domain airline --n 300 --seed 42
python -m benchmark.rulearena.run_single --experiment l1_ptool --domain tax --n 300 --seed 42
python -m benchmark.rulearena.run_single --experiment l1_ptool --domain nba --n 216 --seed 42

# L0F CoT
python -m benchmark.rulearena.run_single --experiment l0f_cot --domain airline --n 300 --seed 42
python -m benchmark.rulearena.run_single --experiment l0f_cot --domain tax --n 300 --seed 42
python -m benchmark.rulearena.run_single --experiment l0f_cot --domain nba --n 216 --seed 42

# L3 ReAct
python -m benchmark.rulearena.run_single --experiment l3_react --domain airline --n 300 --seed 42
python -m benchmark.rulearena.run_single --experiment l3_react --domain tax --n 300 --seed 42
python -m benchmark.rulearena.run_single --experiment l3_react --domain nba --n 216 --seed 42

# L3 ReAct (Pydantic)
python -m benchmark.rulearena.run_single --experiment l3_pydantic --domain airline --n 300 --seed 42
python -m benchmark.rulearena.run_single --experiment l3_pydantic --domain tax --n 300 --seed 42
python -m benchmark.rulearena.run_single --experiment l3_pydantic --domain nba --n 216

# Aggregate all results
python -m benchmark.rulearena.aggregate_results

# Generate report

```
# Debug mode (shows Thought/Action/Observation trace for L3)
python -m benchmark.rulearena.run_single --experiment l3_react --domain airline --debug-n 3

# Run multiple experiments with report
python -m benchmark.rulearena.run_single --experiment l0f_cot l1_ptool --n 50 --report
```
```
---

## Configuration
```python
# benchmark/rulearena/config.py
MODEL_CONFIG = {
    "model_id": "deepseek-ai/DeepSeek-V3",  # change to swap models
    "debug": False,                           # True = verbose L3 traces
}
```
```bash
# Disable LLM call caching (required before full experiment runs)
export PTOOL_CACHE_ENABLED=false

# Results saved automatically to:
benchmark_results/rulearena/<experiment>_<domain>.json
```

## References

- [RuleArena Benchmark](https://github.com/SkyRiver-2000/RuleArena) — Zhou et al., ACL 2025
- [Together.ai Documentation](https://docs.together.ai/) — LLM API reference

