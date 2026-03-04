# Claude Playbook: RuleArena RQ2 Experiments — Step-by-Step

> **How to use this:** Work through each phase in order. At each CHECKPOINT, YOU run the command yourself and verify the output before telling Claude to proceed to the next phase. Do not skip phases.

> **Setup:** Add `rq2_experiment_prompt.md` to your Claude project context (see instructions at bottom). Then use the phase prompts below one at a time.

---

## Phase 1: Infrastructure Skeleton

### Prompt to Claude:
```
Read the file rq2_experiment_prompt.md carefully — it is the master specification for this entire project.

Implement Phase 1 (Infrastructure) only:
1. Create the directory structure under benchmark/rulearena/ as specified
2. Implement config.py with ExperimentLevel enum, MODEL_CONFIG, and ABLATION_CONFIGS dict
3. Implement dataset/loader.py with RuleArenaDataset and RuleArenaInstance dataclass
   - Load from external/RuleArena/ (assume it's already cloned)
   - Implement stratified_sample(n, seed) 
4. Implement experiments/base.py with BaseExperiment ABC and ExperimentResult dataclass
5. Implement metrics/aggregator.py with MetricsAggregator and AggregatedMetrics
   - Field names MUST match MedCalc's schema (see rq2_experiment_prompt.md Step 4.2 and 4.3)

Do NOT implement any experiments yet. Do NOT implement the report generator yet.
Stop after these 5 files are working.
```

### CHECKPOINT 1 — You run:
```bash
# First, make sure RuleArena data exists
ls external/RuleArena/
# If not: git clone https://github.com/SkyRiver-2000/RuleArena external/RuleArena

# Test the dataset loader
python -c "
from benchmark.rulearena.dataset.loader import RuleArenaDataset
ds = RuleArenaDataset()
print(f'Total instances: {len(ds)}')
print(f'Domains: {set(inst.domain for inst in ds.instances)}')
print(f'Complexity levels: {set(inst.complexity_level for inst in ds.instances)}')
sample = ds.stratified_sample(9, seed=42)
print(f'Stratified sample (9): {len(sample)} instances')
for inst in sample[:3]:
    print(f'  {inst.instance_id} | {inst.domain} | level={inst.complexity_level}')
"
```

### Success criteria:
- [ ] Prints total instances (should be ~816)
- [ ] Shows 3 domains: airline, nba, tax
- [ ] Shows 3 complexity levels: 0, 1, 2
- [ ] Stratified sample returns 9 instances spread across domains/levels
- [ ] No import errors

---

## Phase 2: L0F Chain-of-Thought Experiment (End-to-End Pipeline Test)

### Prompt to Claude:
```
Reference rq2_experiment_prompt.md, section on L0F and Step 5.

Implement the L0F Chain-of-Thought experiment:
1. experiments/l0f_cot.py — single LLM call with rules + CoT prompting
2. A minimal runner that can execute just this one experiment on N instances
3. Answer parsing: extract "ANSWER: <number>" from LLM response
4. Cost tracking using Together.ai token counts and pricing ($0.30/M input, $0.90/M output)
5. Save results as JSON with the ExperimentResult schema from base.py

Use model "deepseek-ai/DeepSeek-V3" via Together.ai API. Temperature=0, seed=42.
Include robust error handling — never crash on a single instance failure.

Make it runnable with: python -m benchmark.rulearena.run_single --experiment l0f_cot --n 3
(create run_single.py as a simple script, not the full CLI yet)
```

### CHECKPOINT 2 — You run:
```bash
# Run on 3 instances (should cost < $0.01)
python -m benchmark.rulearena.run_single --experiment l0f_cot --n 3

# Expected output (something like):
# Instance airline_000: predicted=850.0, expected=1245.0, correct=False, cost=$0.0014, time=2.3s
# Instance nba_000: predicted=..., expected=..., correct=..., cost=$0.0012, time=1.8s
# Instance tax_000: predicted=..., expected=..., correct=..., cost=$0.0018, time=2.1s
# ---
# Accuracy: 0/3 (0.0%)
# Total cost: $0.0044
# Results saved to: benchmark_results/rulearena/l0f_cot_debug.json
```

### Success criteria:
- [ ] API calls succeed (no auth or model errors)
- [ ] Each instance prints predicted vs expected with correct/incorrect
- [ ] Cost is tracked per instance (should be ~$0.001-0.003 each)
- [ ] Results JSON file is written with correct ExperimentResult fields
- [ ] Failures are logged, not crashed

---

## Phase 3: L1 PTool Experiment (Core Hypothesis)

### Prompt to Claude:
```
Reference rq2_experiment_prompt.md, section on L1 and Step 5.

Implement the L1 PTool extraction experiment:
1. experiments/l1_ptool.py — two-phase: LLM extracts JSON params, Python calculates
2. calculators/airline.py — deterministic fee calculator from extracted params
   - Use RuleArena's reference implementation as guide (check external/RuleArena/ for their eval code)
   - Start with airline domain only. NBA and tax calculators can be stubs for now.
3. Define the extraction JSON schema for airline domain (see rq2_experiment_prompt.md Step 5, L1 section)
4. Log extracted JSON in metadata["extraction_json"]
5. Make it runnable with the same run_single.py: --experiment l1_ptool --n 3

Important: The extraction prompt should include the rules text AND the target JSON schema.
The Python calculator must be deterministic — same params always produce same answer.
```

### CHECKPOINT 3 — You run:
```bash
# Run L1 on 3 airline instances
python -m benchmark.rulearena.run_single --experiment l1_ptool --n 3 --domain airline

# Expected output:
# Instance airline_000: predicted=1245.0, expected=1245.0, correct=True, cost=$0.0012, time=1.9s
#   Extracted: {"num_bags": 3, "bags": [...], "route_type": "domestic", ...}
# Instance airline_001: predicted=200.0, expected=200.0, correct=True, cost=$0.0011, time=1.7s
# ...
# Accuracy: 2/3 (66.7%)  [or higher — L1 should beat L0F]
```

### Success criteria:
- [ ] Extraction prompt returns valid JSON (not free-text)
- [ ] Python calculator produces numeric answers from extracted params
- [ ] L1 accuracy is visibly higher than L0F on same instances
- [ ] `metadata["extraction_json"]` is populated in results
- [ ] Extraction failures are caught and logged (not crashed)

---

## Phase 4: Report Generator + metrics.json

### Prompt to Claude:
```
Reference rq2_experiment_prompt.md, Step 7 (Output Format) and MedCalc's reports/generator.py.

Implement:
1. metrics/aggregator.py updates — aggregate ExperimentResults into AggregatedMetrics
   with by_category (domain) and by_calculator (domain_level_N) breakdowns
2. reports/generator.py — generate self-contained report.html with:
   - Executive summary cards (experiments run, best accuracy, total cost)
   - Comparison table (all experiments side by side)
   - Plotly bar chart: accuracy by experiment
   - Plotly scatter: cost vs accuracy
   - Per-domain breakdown table
   - Per-complexity heatmap
3. Save metrics.json alongside report.html
4. reports/regenerate_report.py — rebuild report.html from metrics.json

Update run_single.py to accept --report flag that generates the report after running.
```

### CHECKPOINT 4 — You run:
```bash
# Run both L0F and L1 on 10 instances, generate report
python -m benchmark.rulearena.run_single --experiment l0f_cot l1_ptool --n 10 --report

# Then open the report
open benchmark_results/rulearena/report.html  # macOS
# or: start benchmark_results/rulearena/report.html  # Windows

# Verify metrics.json
python -c "
import json
with open('benchmark_results/rulearena/metrics.json') as f:
    data = json.load(f)
for exp_name, m in data.items():
    print(f'{exp_name}: acc={m[\"accuracy_tolerance\"]:.1%}, cost=\${m[\"total_cost_usd\"]:.4f}')
    for cat, cm in m.get('by_category', {}).items():
        print(f'  {cat}: acc={cm[\"accuracy_tolerance\"]:.1%}, n={cm[\"total_instances\"]}')
"
```

### Success criteria:
- [ ] report.html opens in browser with working Plotly charts
- [ ] Comparison table shows L0F and L1 side by side
- [ ] L1 accuracy > L0F accuracy (confirms core hypothesis even on 10 instances)
- [ ] metrics.json has correct schema (check field names match MedCalc's)
- [ ] by_category and by_calculator are populated
- [ ] regenerate_report.py rebuilds identical report from metrics.json

---

## Phase 5: L1-TA + L3 Experiments

### Prompt to Claude:
```
Reference rq2_experiment_prompt.md, sections on L1-TA and L3.

Implement two more experiments:

1. experiments/l1ta_tool_augmented.py
   - LLM generates Python code to solve the problem
   - Execute code in sandboxed environment (subprocess or restricted exec)
   - Capture printed output as answer
   - Log generated code in metadata["code_generated"]

2. experiments/l3_react.py
   - Think/Act/Observe loop, max 10 steps
   - Define 3 tools per domain (extract, lookup, calculate) as Python functions
   - Start with airline domain tools only (stubs for NBA/tax)
   - Log full trace in metadata["trace"]
   - Track total LLM calls in metadata["num_steps"]

Both must use the same ExperimentResult schema. Add them to run_single.py.
```

### CHECKPOINT 5 — You run:
```bash
# Test L1-TA on 3 instances
python -m benchmark.rulearena.run_single --experiment l1ta_tool_augmented --n 3 --domain airline

# Test L3 on 3 instances (this will be slower and more expensive)
python -m benchmark.rulearena.run_single --experiment l3_react --n 3 --domain airline

# Verify L3 traces are logged
python -c "
import json
with open('benchmark_results/rulearena/l3_react_debug.json') as f:
    results = json.load(f)
r = results[0]
print(f'Steps taken: {r[\"metadata\"][\"num_steps\"]}')
print(f'Trace preview: {r[\"metadata\"][\"trace\"][:500]}')
print(f'Cost: \${r[\"cost_usd\"]:.4f}')
"
```

### Success criteria:
- [ ] L1-TA generates runnable code and captures output
- [ ] L3 agent completes multi-step traces (2-8 steps typical)
- [ ] L3 cost per instance is noticeably higher than L1 (~5-10x)
- [ ] Both write valid ExperimentResult JSON
- [ ] Code execution failures (L1-TA) are caught, not crashed

---

## Phase 6: Full Run + Final Report

### Prompt to Claude:
```
Reference rq2_experiment_prompt.md, Step 6 (Running Experiments).

Build the full benchmark runner:
1. runner.py — BenchmarkRunner class that runs multiple experiments sequentially
2. cli.py — Click-based CLI with run, list, report subcommands
3. Add caching (cachier or manual file-based) to avoid re-running API calls during dev
4. Add progress bars (tqdm) for long runs

The full CLI should support:
  python -m benchmark.rulearena run --experiments all --debug      # 30 instances
  python -m benchmark.rulearena run --experiments all --seed 42    # full 816 instances
  python -m benchmark.rulearena run --experiments l0f_cot l1_ptool # specific experiments
  python -m benchmark.rulearena list                               # show available experiments
  python -m benchmark.rulearena report --results-dir ./benchmark_results/rulearena
```

### CHECKPOINT 6 — You run:
```bash
# Debug run: all experiments on 30 instances (~$0.50-1.00)
python -m benchmark.rulearena run --experiments all --debug --seed 42

# Open final report
open benchmark_results/rulearena/report.html

# Verify all experiments present
python -c "
import json
with open('benchmark_results/rulearena/metrics.json') as f:
    data = json.load(f)
print('Experiments:', list(data.keys()))
for name, m in sorted(data.items(), key=lambda x: x[1]['accuracy_tolerance'], reverse=True):
    print(f'  {name:25s} acc={m[\"accuracy_tolerance\"]:6.1%}  cost=\${m[\"total_cost_usd\"]:.4f}')
"
```

### Success criteria:
- [ ] All experiments (L0F, L1, L1-TA, L3) complete without errors
- [ ] Report shows all experiments with comparison charts
- [ ] Ranking is roughly: L1 > L1-TA > L3 > L0F (validates hypothesis)
- [ ] Total debug cost < $1.00
- [ ] Ready for full run

---

## Phase 7: Production Run

**This phase is YOU, not Claude.** Run the full experiment suite.

```bash
# Full run: all 816 instances, all experiments
# Estimated cost: $20-32, estimated time: 2-4 hours
python -m benchmark.rulearena run --experiments l0f_cot l1_ptool l1ta_tool_augmented l3_react \
    --seed 42 \
    --output-dir benchmark_results/rulearena_full

# Generate final report
python -m benchmark.rulearena report --results-dir benchmark_results/rulearena_full

# Archive results
cp benchmark_results/rulearena_full/metrics.json results/rq2_final_metrics.json
cp benchmark_results/rulearena_full/report.html results/rq2_final_report.html
git add results/ && git commit -m "RQ2: Full L0F/L1/L1-TA/L3 results on DeepSeek-V3"
```

### Final success criteria:
- [ ] 816 instances x 4 experiments = 3264 total evaluations completed
- [ ] metrics.json schema matches MedCalc's (ready for cross-benchmark analysis)
- [ ] Report tells a clear story: structured extraction (L1) beats all alternatives
- [ ] Total cost within $30-40 budget
- [ ] Results committed to git with reproducibility info (seed, model, timestamp)

---

## Quick Reference: Phase Summary

| Phase | What | You Verify | Est. Cost |
|-------|------|-----------|-----------|
| 1 | Infrastructure + data loader | Dataset loads, imports work | $0 |
| 2 | L0F CoT experiment | API calls succeed, 3 results print | <$0.01 |
| 3 | L1 PTool experiment | Extraction works, L1 > L0F | <$0.01 |
| 4 | Report generator | report.html opens with charts | $0 |
| 5 | L1-TA + L3 experiments | Both produce results, L3 traces logged | <$0.10 |
| 6 | Full runner + CLI | Debug run (30 instances) works end-to-end | <$1.00 |
| 7 | Production run (YOU) | Full 816-instance results | $20-32 |
