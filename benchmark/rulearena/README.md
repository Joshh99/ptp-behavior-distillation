# PTP Behavior Distillation -- RuleArena Benchmark

## Overview
This implements the PTP (Parameter-Tool-Program) framework on the RuleArena benchmark.
It evaluates four levels of LLM autonomy (L0 oracle, L0F CoT, L1 PTP, L3 ReAct) across
three domains: airline baggage fees, NBA transaction compliance, and tax computation.
Model: DeepSeek-V3 via Together.ai.

## Results Summary
| Level | Airline | Tax | NBA (F1) |
|-------|---------|-----|----------|
| L0 oracle | ~100% | ~100% | -- |
| L0F CoT | 48.3% | 35.3% | 0.50 |
| L1 PTP | 77.0% | 99.7% | 0.44 |
| L3 ReAct | TBD | TBD | TBD |

## Running Experiments
```bash
# Standard run (from repo root)
python -m benchmark.rulearena.run_single --experiment <name> --domain <domain> --n <N> --seed 42

# Experiment names: l0_python, l0f_cot, l1_ptool, l3_react
# Domains: airline, nba, tax

# Examples
python -m benchmark.rulearena.run_single --experiment l1_ptool --domain airline --n 50 --seed 42
python -m benchmark.rulearena.run_single --experiment l3_react --domain tax --n 50 --seed 42

# Debug mode (shows Thought/Action/Observation trace for L3)
python -m benchmark.rulearena.run_single --experiment l3_react --domain airline --debug-n 3

# Run multiple experiments with report
python -m benchmark.rulearena.run_single --experiment l0f_cot l1_ptool --n 50 --report
```

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

## Notes
- NBA domain uses F1 macro (82.9% class imbalance -- accuracy is misleading)
- Tax L1 99.7% accuracy is partly due to pre-structured IRS form input
- L3 trajectories stored in benchmark_results/ for future distillation work
