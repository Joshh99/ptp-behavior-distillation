# PTP Behavior Distillation Research

Research project exploring behavior distillation for agentic AI systems using Program Trace Prompting (PTP).

**Collaborator**: Professor William Cohen (CMU ML / Google)

## Project Structure

- `experiments/` - Research experiments organized by phase
  - `01-ptp-replication/` - Core PTP implementation and validation
  - `02-workflow-spectrum/` - L0-L3 complexity analysis
  - `03-distillation-methods/` - Trace-based workflow distillation
  - `04-comparative-analysis/` - Systematic benchmarking
- `docs/` - Research log, meeting notes, decisions
- `results/` - Experimental outputs, comparisons, figures
- `data/` - BBH dataset and evaluation data
- `src/` - Reusable code (ptools, workflows, utilities)

## Progress

- [ ] Setup: API access, secretagent, BBH dataset
- [ ] Phase 1: PTP core replication
- [ ] Phase 2: Workflow spectrum exploration  
- [ ] Phase 3: Distillation prototypes
- [ ] Phase 4: Comparative evaluation

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Set up GitHub Models API
export GITHUB_TOKEN="your_token_here"

# Run experiments
cd experiments/01-ptp-replication
python sports_understanding.py
```

See `docs/research-log.md` for detailed progress notes.