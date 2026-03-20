# Literature Review: Can LLMs Zero-Shot Max Entropy Questions?

## Research Area Overview

This research investigates whether large language models can, in a zero-shot setting, generate yes/no questions that split a given set of items into two groups of equal or near-equal size — i.e., questions that maximize information entropy. This connects to the 20 Questions game paradigm from cognitive science, information-theoretic question asking, and LLM reasoning under uncertainty.

The core metric is **Expected Information Gain (EIG)**: for a yes/no question over a set of N items, EIG is maximized when the question splits items into two equal halves (N/2 : N/2), achieving entropy of 1.0 bit. This is equivalent to a single step of binary search.

## Key Papers

### Paper 1: Learning to Ask Informative Questions (Mazzaccara et al., 2024)
- **Authors**: Davide Mazzaccara, Alberto Testoni, Raffaella Bernardi
- **Venue**: arXiv:2406.17453 (2024)
- **Key Contribution**: Uses DPO to train LLMs to ask higher-EIG questions in 20Q games
- **Methodology**:
  - Samples multiple questions from LLaMA 2-CHAT 7B for each game
  - Computes EIG for each question by having the model annotate yes/no for all items
  - Creates preference pairs: optimal (EIG=1) vs suboptimal (EIG<0.8)
  - Trains with DPO to prefer high-EIG questions
- **Datasets**: McRae concepts (66 concepts, 6 categories), 8-item candidate sets from 2 categories of 4 each. Test sets: INLG, Things, Celebrities, INLG-16, BigBench.
- **Key Results**:
  - Zero-shot LLaMA 2-CHAT 7B: EIG = 0.29-0.35 across test sets
  - DPO-trained: EIG = 0.40-0.47 (significant improvement)
  - DPO improves task success (S@1) by +10-14% and reduces questions needed by 2-3
  - Abstract concepts much harder than concrete ones (7.7% vs 25.0% zero-shot success)
  - DPO generalizes across domains but struggles with BigBench (large, diverse sets)
- **Code**: https://github.com/dmazzaccara/LearningToAsk
- **Relevance**: **Most directly relevant paper.** Explicitly defines optimal questions as those splitting items into equal halves. Shows zero-shot LLMs have low EIG (~0.3), far from optimal (1.0). Demonstrates that training can improve but not fully solve this.

### Paper 2: ChatGPT's Information Seeking Strategy (Bertolazzi et al., 2023)
- **Authors**: Leonardo Bertolazzi, Davide Mazzaccara, Filippo Merlo, Raffaella Bernardi
- **Venue**: INLG 2023
- **Key Contribution**: First evaluation of ChatGPT's information-seeking strategy in 20Q using hierarchical hypothesis spaces
- **Methodology**:
  - Builds hierarchical hypothesis spaces using McRae norms and GPT norms
  - 8-item sets (2-level hierarchy) and 16-item sets (3-level hierarchy)
  - Tests ChatGPT (gpt-3.5-turbo) as both Questioner and Answerer
  - Measures EIG and compares to optimal half-split agent
- **Key Results**:
  - ChatGPT far from optimal when updating space internally
  - Performance improves when prompted to explicitly track remaining candidates
  - Questions categorized as Constraint-Seeking (CS) vs Hypothesis-Scanning (HS)
- **Code**: https://github.com/leobertolazzi/20q-chatgpt
- **Relevance**: **Foundational work** for our research. Establishes the experimental paradigm, datasets, and metrics we build on. Shows ChatGPT's gap from optimal question generation.

### Paper 3: Probing Multi-turn Planning via 20 Questions (Zhang et al., 2024)
- **Authors**: Yizhe Zhang, Jiarui Lu, Navdeep Jaitly (Apple)
- **Venue**: ACL 2024 (arXiv:2310.01468)
- **Key Contribution**: Systematic evaluation of multiple LLMs as guessers in 20Q, plus RL/BC training approaches
- **Methodology**:
  - Entity-Deduction Arena (EDA) with Things (500 entities) and Celebrities (500) datasets
  - GPT-3.5-turbo as judge, various LLMs as guessers
  - Evaluates: GPT-4, GPT-3.5, Claude-1/2, Vicuna 7B/13B, Mistral 7B
  - Uses Behavior Cloning and PPO to improve Vicuna models
- **Key Results**:
  - GPT-4 best performer (~31% success on Things, ~50% on Celebrities)
  - Effective strategy: binary-tree-like narrowing, then enumeration
  - Failure modes: early enumeration, redundancy, inconsistency
  - Weaker models fall into repetitive patterns
- **Code**: https://github.com/apple/ml-entity-deduction-arena
- **Relevance**: Confirms that strong LLMs use binary-partitioning strategies intuitively. Provides datasets and evaluation framework. Shows significant performance gaps between models.

### Paper 4: Uncertainty of Thoughts (Hu et al., 2024)
- **Authors**: Zhiyuan Hu et al.
- **Venue**: NeurIPS 2024 (arXiv:2402.03271)
- **Key Contribution**: Uncertainty-aware planning for LLM information seeking using EIG-based rewards
- **Relevance**: Proposes algorithmic approach (simulation + reward propagation) to select optimal questions. Improves LLM success rate by 38.1%. Shows that explicit uncertainty tracking helps.

### Paper 5: QuestBench (Li et al., 2025)
- **Authors**: Belinda Z. Li, Been Kim, Zi Wang (Google DeepMind)
- **Venue**: arXiv:2503.22674
- **Key Contribution**: Benchmark for evaluating if LLMs can ask the right question to acquire missing information
- **Relevance**: Shows models struggle to identify the right question even when they can solve the fully-specified version. Relevant to understanding zero-shot question generation limitations.

### Paper 6: GuessingGame (2025)
- **Venue**: EMNLP 2025 (arXiv:2509.19593)
- **Key Contribution**: Directly measures information gain of LLM questions using Bayesian and entropy-based metrics
- **Relevance**: Finds higher IG reduces expected game length by 43%. Provides IG measurement methodology.

### Paper 7: BED-LLM (2025)
- **Venue**: arXiv:2508.21184
- **Key Contribution**: Sequential Bayesian experimental design with EIG for optimal question selection
- **Relevance**: Shows principled EIG estimation doubles success rate vs. naive QA. Demonstrates that marginal entropy alone is insufficient.

### Paper 8: From Passive to Active Reasoning (Zhou et al., 2025)
- **Venue**: ICML 2025 (arXiv:2506.08295)
- **Key Contribution**: AR-Bench evaluating active reasoning including "guessing numbers" tasks
- **Relevance**: Shows LLMs struggle at active reasoning — frequently fail to maximize expected information gain.

## Common Methodologies

1. **20 Questions Game paradigm**: Player asks yes/no questions to identify target from candidate set. Used in Papers 1-3. Provides controlled evaluation of question informativeness.
2. **Expected Information Gain (EIG)**: `EIG(Q) = H(prior) - E[H(posterior|answer)]`. For uniform prior over N items, optimal question achieves EIG = 1.0 bit. Used in Papers 1, 2, 4, 6, 7.
3. **Feature norms**: McRae norms (human-annotated) and GPT norms (LLM-generated) used to build structured hypothesis spaces. Used in Papers 1, 2.
4. **DPO / RL training**: Preference optimization using EIG-ranked question pairs. Used in Papers 1, 3.

## Standard Baselines

1. **Optimal agent**: Always asks the half-split question (EIG = 1.0). Needs log2(N) questions for N items.
2. **Random question**: Expected EIG depends on question distribution, typically much lower than optimal.
3. **Zero-shot LLM**: LLM generates questions without training on EIG signal. Typical EIG: 0.29-0.35 (Paper 1).

## Evaluation Metrics

1. **Expected Information Gain (EIG)**: Primary metric. Measures how well a question splits the item space. Range [0, 1] for binary questions.
2. **Task success (S@1)**: Percentage of games where target is identified within one dialogue.
3. **Average questions (AQ)**: Number of questions needed in successful games.
4. **Split ratio**: |yes_group|:|no_group| — perfect is N/2:N/2.

## Datasets in the Literature

1. **McRae concept sets** (8 items, 16 items): Used in Papers 1, 2. 6 categories (mammal, bird, clothing, weapon, fruit, vegetable). Well-structured taxonomically.
2. **Things dataset** (500 common entities): Used in Paper 3. Diverse categories.
3. **Celebrities dataset** (500 celebrity names): Used in Paper 3. Requires factual knowledge.
4. **BigBench** (29 items per set): Used in Paper 1. Mix of concrete and abstract concepts — hardest setting.

## Gaps and Opportunities

1. **Zero-shot evaluation across modern models**: Prior work mostly tests older models (GPT-3.5, LLaMA 2). No systematic zero-shot evaluation of GPT-4o, Claude 3.5/4, Gemini, etc.
2. **Single-question focus**: Prior work evaluates multi-turn games. Our focus on single-question splitting is simpler and more directly tests the core capability.
3. **Difficulty analysis**: What makes certain sets harder to split? Prior work notes abstract > concrete difficulty but doesn't systematically categorize set difficulty.
4. **Set composition effects**: How does category homogeneity affect splitting ability? No prior work tests this systematically.

## Recommendations for Our Experiment

### Recommended datasets
1. **McRae 8-item sets** (90 games): Primary benchmark, well-established, clear optimal split exists
2. **Things 8-item sets** (90 games): Out-of-domain test
3. **BigBench sets** (29 games): Hard test with abstract concepts
4. **Custom sets** (fruits, animals, mixed): Test specific hypotheses about difficulty

### Recommended baselines
1. **Optimal agent**: Always achieves EIG = 1.0 (4:4 split for 8 items)
2. **Random split**: Expected EIG for random binary assignment
3. **Zero-shot LLM (no context)**: Just ask "generate a yes/no question that splits these items in half"
4. **Zero-shot LLM (with reasoning)**: Chain-of-thought prompting

### Recommended metrics
1. **EIG** of the generated question (primary)
2. **Split ratio** (|yes|:|no|)
3. **Perfect split rate**: % of questions achieving exact half split
4. **Category alignment**: For structured sets, does the question align with the underlying category boundary?

### Methodological considerations
- Need an oracle/annotator to determine yes/no for each item given a question
- Can use the same LLM or a separate (stronger) model as annotator
- Must handle ambiguous questions (some items may be "maybe")
- Should test multiple prompt formulations
- Should vary set size (8, 16, larger) to test scaling
