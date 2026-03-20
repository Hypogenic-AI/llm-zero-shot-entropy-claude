# Research Plan: Can LLMs Zero-Shot Max Entropy Questions?

## Motivation & Novelty Assessment

### Why This Research Matters
Binary search is the optimal strategy for narrowing down possibilities — asking a yes/no question that splits a set exactly in half eliminates the maximum information per question. This is fundamental to efficient information seeking. If LLMs can do this zero-shot, they can serve as efficient question generators in interactive systems (diagnosis, 20 Questions, troubleshooting). If they can't, we need to understand why and when they fail.

### Gap in Existing Work
Prior work (Mazzaccara et al. 2024, Bertolazzi et al. 2023) evaluates LLMs in multi-turn 20Q games, finding zero-shot EIG ~0.3 (far from optimal 1.0). However:
1. **No single-question focus**: Prior work embeds question quality in game performance. We isolate the core capability: can a model generate ONE question that splits a set in half?
2. **No modern model evaluation**: Prior work tests GPT-3.5 and LLaMA 2. GPT-4.1 and other 2025 models are untested.
3. **No systematic difficulty analysis**: What properties of a set make splitting hard? Prior work notes abstract > concrete difficulty but doesn't categorize systematically.
4. **No prompt strategy comparison**: How much does prompt design matter for this task?

### Our Novel Contribution
We provide the first systematic evaluation of modern LLMs' ability to generate single yes/no questions that achieve maximum entropy (equal split) over item sets, with:
- Multiple models (GPT-4.1, GPT-4o-mini)
- Multiple prompt strategies (basic, explicit split instruction, chain-of-thought)
- Multiple set types varying in difficulty (homogeneous categories, mixed categories, abstract items)
- Quantitative difficulty analysis identifying what makes sets hard to split

### Experiment Justification
- **Exp 1 (Prompt strategies)**: Tests whether explicit instruction to "split in half" helps vs. generic "ask a yes/no question." Establishes best prompt approach.
- **Exp 2 (Across datasets)**: Tests the core hypothesis across set types of varying difficulty. Identifies when LLMs succeed and fail.
- **Exp 3 (Set difficulty analysis)**: Systematically categorizes what properties make sets hard to split (homogeneity, abstractness, size).

## Research Question
Can modern LLMs, given a set of N items, generate a single yes/no question that splits the set into two groups of equal or near-equal size (maximizing information entropy)? What set properties make this task difficult?

## Hypothesis Decomposition
1. **H1**: LLMs can generate questions that split mixed-category sets (e.g., animals + vehicles) near-perfectly, since category membership is an obvious binary feature.
2. **H2**: LLMs struggle with homogeneous sets (e.g., all fruits), where no single obvious categorical split exists.
3. **H3**: Explicit prompting ("split into exactly half") improves split quality over naive prompting.
4. **H4**: Chain-of-thought prompting further improves performance by encouraging the model to reason about item properties.
5. **H5**: Larger sets (16 items) are harder to split evenly than smaller sets (8 items).

## Proposed Methodology

### Approach
For each item set, prompt an LLM to generate a yes/no question. Then use a judge model (GPT-4.1) to annotate each item as "yes" or "no" for that question. Compute EIG from the resulting split.

### Experimental Steps
1. **Prompt design**: Create 3 prompt strategies (basic, explicit-split, chain-of-thought)
2. **Question generation**: For each (dataset, prompt strategy, model) combination, generate questions
3. **Answer annotation**: Use GPT-4.1 as oracle to label each item yes/no for the generated question
4. **EIG computation**: Calculate information gain from the split
5. **Analysis**: Compare across conditions, identify difficulty factors

### Datasets (pre-downloaded)
| Dataset | Sets | Items/set | Expected difficulty |
|---------|------|-----------|-------------------|
| mixed_categories_8 | 20 | 8 | Easy (cross-category) |
| fruits_8 | 20 | 8 | Hard (homogeneous) |
| animals_8 | 20 | 8 | Hard (homogeneous) |
| mcrae_8 | 90 | 8 | Medium (2 categories) |
| things_8 | 90 | 8 | Medium-Hard (diverse) |
| mcrae_16 | 90 | 16 | Hard (larger sets) |

### Models
- **GPT-4.1** (primary): State-of-the-art, via OpenAI API
- **GPT-4o-mini** (secondary): Smaller/cheaper model for comparison

### Baselines
- **Optimal**: EIG = 1.0 (perfect 4:4 split for 8 items)
- **Random**: Expected EIG for random binary assignment ~ 0.81 for 8 items

### Evaluation Metrics
1. **EIG**: Primary metric. H(prior) - E[H(posterior)]. For 8 items with uniform prior, optimal = 1.0 bit.
2. **Perfect split rate**: % of questions achieving exact N/2:N/2 split
3. **Near-perfect rate**: % achieving splits within 1 item of half (e.g., 3:5 or 5:3 for 8 items)
4. **Mean absolute deviation from half**: |yes_count - N/2|

### Statistical Analysis Plan
- Compare EIG across prompt strategies using paired t-tests (same sets)
- Compare across datasets using ANOVA
- Report 95% confidence intervals
- Effect sizes (Cohen's d) for pairwise comparisons
- Significance level: α = 0.05

## Expected Outcomes
- Mixed-category sets: EIG > 0.9 (easy to find cross-category question)
- Homogeneous sets (fruits, animals): EIG 0.5-0.8 (harder, requires finding subtle distinguishing features)
- Chain-of-thought should improve over basic prompting
- GPT-4.1 should outperform GPT-4o-mini

## Timeline
- Planning: 15 min ✓
- Setup: 10 min
- Implementation: 45 min
- Experiments: 60 min
- Analysis: 30 min
- Documentation: 20 min

## Potential Challenges
1. **Annotation ambiguity**: Some items may be ambiguous for a given question. Mitigation: use GPT-4.1 as consistent oracle, force binary yes/no.
2. **API rate limits**: Many calls needed. Mitigation: batch requests, add retries.
3. **Question quality**: Model may generate trivial or nonsensical questions. Mitigation: log all questions for manual inspection.

## Success Criteria
- Complete evaluation across all datasets and prompt strategies
- Clear evidence for or against the hypothesis
- Quantitative difficulty ranking of set types
- Statistical significance of key comparisons
