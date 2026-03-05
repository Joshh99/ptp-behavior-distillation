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

---

## Overview

This project investigates the **Reliability/Autonomy Spectrum** using the [RuleArena](https://github.com/SkyRiver-2000/RuleArena) benchmark across three rule-guided reasoning domains.

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

## Project Structure

```
ptp-behavior-distillation/
├── benchmark/
│   ├── rq1/                          # RQ1 experiments (airline domain, multi-model)
│   │   ├── dataset/airline_loader.py # Airline data loader
│   │   ├── experiments/              # l1_ptool, cot, tool_aug implementations
│   │   ├── run_full_baseline.py      # Main RQ1 runner
│   │   └── generate_report.py        # Generates reports/rq1_report.html
│   └── rulearena/                    # RQ2 experiments (all 3 domains)
│       ├── experiments/              # l0_python, l0f_cot, l1_ptool, l3_react
│       ├── run_single.py             # Per-domain experiment runner
│       ├── aggregate_results.py      # Aggregate benchmark_results/rulearena/
│       └── generate_report.py        # Generates reports/rq2_report.html
├── external/
│   ├── RuleArena/                    # Cloned benchmark repository
│   └── ptool_framework/              # PTP framework
├── results/
│   └── rq1/                          # RQ1 experiment outputs
├── benchmark_results/
│   └── rulearena/                    # RQ2 per-run result JSONs
└── reports/
    ├── rq1_report.html               # Generated RQ1 HTML report
    └── rq2_report.html               # Generated RQ2 HTML report
```

---

## Quick Start

### Setup

```bash
pip install -r requirements.txt
export TOGETHER_API_KEY="your-key-here"
```

---

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

# Aggregate all results
python -m benchmark.rulearena.aggregate_results
```

---

## Reports

Generated HTML reports are written to `reports/`:

- **`reports/rq1_report.html`** — RQ1 results (airline domain, multi-model comparison)
- **`reports/rq2_report.html`** — RQ2 results (full spectrum, all three domains)

To regenerate:

```bash
python -m benchmark.rq1.generate_report
python -m benchmark.rulearena.generate_report
```

---

## Results Summary

### RQ1: Airline Domain (DeepSeek-V3, Complexity Level 0)

| Experiment | Accuracy | Cost/Problem |
|------------|----------|-------------|
| L1 PTool (Pure) | ~80% | ~$0.0005 |
| L1 PTool (Transparent) | ~85% | ~$0.0008 |
| CoT Baseline | ~5% | ~$0.0010 |
| Tool-Augmented | ~40% | ~$0.0008 |

### RQ2: All Domains (DeepSeek-V3)

See `reports/rq2_report.html` for the full breakdown.

---

## Key Concepts

### Program Trace Prompting (PTP)

Python controls the workflow; LLMs handle extraction and reasoning. This separates concerns:
- **Extraction (LLM)**: Convert natural language to structured parameters
- **Calculation (Python)**: Apply rules deterministically

### Behavior Distillation

By analyzing execution traces from L3 agents, we systematically distill their behavior into L1 workflows that are more reliable, cheaper, and debuggable.

---

## References

- [RuleArena Benchmark](https://github.com/SkyRiver-2000/RuleArena) — Zhou et al., ACL 2025
- [Together.ai Documentation](https://docs.together.ai/) — LLM API reference

---

## License

Research project for academic purposes. MIT License.
