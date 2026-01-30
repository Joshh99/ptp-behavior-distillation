# RuleArena Airline Dataset Inspection Report

**Date**: January 30, 2026  
**Repository**: https://github.com/SkyRiver-2000/RuleArena  
**Local Path**: `external/RuleArena/airline/`

---

## Executive Summary

The RuleArena airline dataset contains **300 synthesized baggage fee problems** across 3 complexity levels, supported by structured fee tables (CSV) and comprehensive reference rules (markdown/text).

---

## 1. Directory Structure

```
external/RuleArena/airline/
├── fee_tables/                    # Structured lookup tables
│   ├── bag_1/                     # First checked bag fees
│   │   ├── 0.csv                  # Standard fees by region/class
│   │   └── 1.csv                  # Alternative fee schedule
│   ├── bag_2/                     # Second checked bag fees
│   ├── bag_3/                     # Third checked bag fees
│   └── bag_4/                     # Fourth+ checked bag fees
├── synthesized_problems/          # Test instances
│   ├── comp_0.jsonl               # Complexity 0 (5 bags)
│   ├── comp_1.jsonl               # Complexity 1 (8 bags)
│   └── comp_2.jsonl               # Complexity 2 (11 bags)
├── reference_rules.txt            # Full rules in markdown format
├── reference_rules_textual.txt    # Plain text version
├── auto_test.py                   # Automated testing script
├── compute_answer.py              # Answer computation logic
├── gen_questions.py               # Problem generation script
├── micro_evaluation.py            # Evaluation utilities
└── structure.py                   # Data structure definitions
```

---

## 2. Problem Counts by Complexity Level

| Complexity | Problems | Items per Problem | Description |
|------------|----------|-------------------|-------------|
| `comp_0`   | 100      | 5 bags            | Simple scenarios |
| `comp_1`   | 100      | 8 bags            | Medium complexity |
| `comp_2`   | 100      | 11 bags           | Complex scenarios |
| **Total**  | **300**  | —                 | — |

---

## 3. CSV Fee Tables Structure

### Location
```
fee_tables/bag_{1,2,3,4}/*.csv
```

### File Count
- **bag_1/**: 2 CSV files
- **bag_2/**: 2 CSV files  
- **bag_3/**: 2 CSV files
- **bag_4/**: 2 CSV files
- **Total**: 8 CSV files

### Column Names (All Files)
```
(unnamed), Basic Economy, Main Cabin, Main Plus, Premium Economy, Business, First
```

The first column (unnamed) contains **region/destination** names.

### Sample Data: First Checked Bag (`bag_1/0.csv`)

| Region       | Basic Economy | Main Cabin | Main Plus | Premium Economy | Business | First |
|--------------|---------------|------------|-----------|-----------------|----------|-------|
| U.S.         | $40           | $40        | $0        | $0              | $0       | $0    |
| Puerto Rico  | $40           | $40        | $0        | $0              | $0       | $0    |
| Canada       | $35           | $35        | $0        | $0              | $0       | $0    |
| Mexico       | $35           | $35        | $0        | $0              | $0       | $0    |
| Cuba         | $35           | $0         | $0        | $0              | $0       | $0    |
| Europe       | $75           | $0         | $0        | $0              | $0       | $0    |
| China        | $75           | $0         | $0        | $0              | $0       | $0    |

### Fee Progression by Bag Number
| Bag # | Domestic (Economy) | International (Economy) |
|-------|-------------------|-------------------------|
| 1st   | $40               | $35-$75                |
| 2nd   | $45               | $45-$150               |
| 3rd   | $150              | $150-$200              |
| 4th+  | $200              | $200                   |

---

## 4. Reference Rules Format

### File: `reference_rules.txt` (196 lines)

The reference rules are written in **Markdown format** with hierarchical sections:

```markdown
# Bag fees

## Carry-on bags
### Personal items
### Carry-on items

## Checked bags
### First Bag
| Regions | Basic Economy | Main Cabin | ... |

### Second Bag
### Third Bag
### Fourth Bag +

### Complimentary Bags
### Weight and Size
#### Overweight Bags
#### Oversize Bags
```

### Rule Categories Covered

1. **Carry-on Rules**
   - Personal item dimensions: 18 × 14 × 8 inches
   - Carry-on dimensions: 22 × 14 × 9 inches
   - Exemptions: diaper bags, breast pumps, strollers, medical devices

2. **Checked Bag Fees** (by region and cabin class)
   - Domestic U.S./Canada
   - Mexico/Caribbean/Central America
   - South America (multiple sub-regions)
   - Europe/Israel/Qatar
   - Asia-Pacific (India, China, Japan, Korea, Hong Kong, Australia, NZ)

3. **Complimentary Bag Eligibility**
   - Credit cardmember benefits
   - AAdvantage status tiers (Gold, Platinum, Executive Platinum)
   - oneworld alliance status
   - Military personnel
   - Premium cabin classes

4. **Weight/Size Limits and Overages**
   - Standard: 62 in / 50 lbs
   - First/Business: 70 lbs for complimentary bags
   - Overweight fees: $30 (50-53 lbs), $100-$200 (53-70 lbs), $200-$450 (70-100 lbs)
   - Oversize fees: $30 (62-65 in), $150-$200 (65-115 in)

---

## 5. Problem Instance Format (JSONL)

Each line in `comp_X.jsonl` is a JSON object with:

### Schema
```json
{
  "prompt": "Natural language problem description",
  "info": {
    "base_price": 180,
    "customer_class": "Main Cabin",
    "routine": "U.S.",
    "direction": 0,
    "bag_list": [
      {
        "id": 1,
        "name": "backpack",
        "size": [22, 13, 6],
        "weight": 10
      }
    ]
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | string | Natural language scenario describing passenger and items |
| `info.base_price` | int | Ticket price in USD |
| `info.customer_class` | string | One of: "Basic Economy", "Main Cabin", "Main Plus", "Premium Economy", "Business", "First" |
| `info.routine` | string | Region/route (e.g., "U.S.", "Canada", "Europe", "China") |
| `info.direction` | int | 0 = outbound, 1 = return |
| `info.bag_list` | array | List of items with id, name, size [L,W,H], weight |

### Example Problem (comp_0)

**Prompt:**
> Sarah is a Main Cabin Class passenger flying from Orlando to Philadelphia with the following items:
> 1. A backpack: 22 x 13 x 6 inches, 10 lbs;
> 2. A luggage box: 44 x 22 x 20 inches, 69 lbs;
> 3. A luggage box: 34 x 18 x 12 inches, 51 lbs;
> 4. A backpack: 38 x 22 x 16 inches, 84 lbs;
> 5. A backpack: 38 x 14 x 11 inches, 90 lbs;
> 
> Sarah's flight ticket is $180.

**Structured Info:**
- Class: Main Cabin
- Route: U.S. (domestic)
- 5 items with varying sizes and weights

---

## 6. Key Observations

### Strengths for L1 vs L3 Research

1. **Ground Truth Available**: The `compute_answer.py` script can calculate exact correct answers, enabling objective evaluation.

2. **Complexity Gradient**: Three complexity levels allow systematic testing of where L3 agents begin to outperform L1.

3. **Rule Ambiguity**: Natural language rules contain edge cases (Cuba exceptions, complimentary bag eligibility, overweight vs oversize precedence) that test reasoning capabilities.

4. **Structured + Unstructured**: 
   - CSV tables = machine-readable lookup
   - Markdown rules = requires NLP to parse conditions

### Challenges

1. **No Pre-computed Answers**: The JSONL files contain only inputs; answers must be computed using `compute_answer.py`.

2. **Implicit Rules**: Some rules (e.g., "higher fee between oversize and overweight applies") require multi-step reasoning.

3. **Exception Handling**: Cuba has special bidirectional rules that vary from other regions.

### Recommended Next Steps

1. **Parse `compute_answer.py`** to understand the ground truth calculation logic.
2. **Generate answer labels** for all 300 problems.
3. **Build L1 PTool** that:
   - Extracts parameters from natural language prompt
   - Uses CSV lookup for base fees
   - Applies overweight/oversize rules
4. **Build L3 ReAct agent** with access to same fee tables and rules.
5. **Compare** accuracy, cost, and latency across complexity levels.

---

## 7. File Checksums

| File | Line Count |
|------|------------|
| `reference_rules.txt` | 196 |
| `comp_0.jsonl` | 100 |
| `comp_1.jsonl` | 100 |
| `comp_2.jsonl` | 100 |
| `bag_1/0.csv` | 22 |
| `bag_2/0.csv` | 22 |
| `bag_3/0.csv` | 21 |
| `bag_4/0.csv` | 21 |

---

*Report generated by automated inspection script.*
