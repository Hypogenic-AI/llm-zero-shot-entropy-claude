# Can LLMs Zero-Shot Generate Max-Entropy Questions?

Can a language model, given a set of items (e.g., fruits), generate a yes/no question that splits the set exactly in half? This project systematically tests this capability across modern LLMs, prompt strategies, and set types.

## Key Findings

- **Yes, LLMs can do this** — but only with explicit prompting. Chain-of-thought prompting achieves binary entropy > 0.94 on all 8-item datasets tested.
- **Without guidance, GPT-4.1 is surprisingly bad** — generating questions with near-zero information gain on homogeneous sets (e.g., 0.027 entropy on all-fruit sets).
- **Set composition matters most**: Mixed-category sets (animals + vehicles) → 100% perfect splits. Homogeneous sets (all fruits) → 65% perfect splits.
- **Chain-of-thought >> explicit instruction >> basic prompt** — the reasoning step lets the model count and verify the split before committing.
- **Smaller model, better defaults**: GPT-4o-mini outperforms GPT-4.1 on basic prompts, suggesting the stronger model defaults to "interesting" rather than "discriminative" questions.

## Quick Stats

| Condition | Mean Binary Entropy | Perfect Split Rate |
|-----------|-------------------:|------------------:|
| GPT-4.1 + basic prompt | 0.44 | 20% |
| GPT-4.1 + explicit split | 0.95 | 66% |
| GPT-4.1 + chain-of-thought | 0.98 | 75% |
| GPT-4o-mini + chain-of-thought | 0.97 | 70% |

## Reproduction

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai numpy scipy matplotlib seaborn pandas

# Run experiment (requires OPENAI_API_KEY)
python src/experiment.py

# Run analysis
python src/analyze.py
```

## Project Structure

```
├── REPORT.md              # Full research report with results
├── planning.md            # Experimental design and methodology
├── src/
│   ├── experiment.py      # Main experiment: generate questions, judge items, compute EIG
│   └── analyze.py         # Statistical analysis and visualization
├── datasets/game_sets/    # 8 item-set datasets (fruits, animals, mcrae, etc.)
├── results/
│   ├── raw_results.json   # All 1,049 trial results
│   ├── summary_table.csv  # Aggregated metrics
│   ├── statistical_tests.json
│   └── plots/             # Visualizations
├── literature_review.md   # Survey of related work
└── resources.md           # Catalog of datasets and papers
```

## Methodology

1. **Generate**: Prompt LLM to produce a yes/no question for an item set
2. **Annotate**: GPT-4.1 (temperature=0) labels each item yes/no
3. **Measure**: Binary entropy of the split (1.0 = perfect half, 0.0 = all same)

Tested across 8 datasets x 3 prompt strategies x 2 models = 1,049 trials.

See [REPORT.md](REPORT.md) for full results and analysis.
