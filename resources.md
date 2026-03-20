# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project "Can LLMs zero-shot max entropy questions?" The project investigates whether LLMs can generate yes/no questions that split a given set into two equal halves in a zero-shot setting.

## Papers

Total papers downloaded: 14

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Learning to Ask Informative Questions | Mazzaccara et al. | 2024 | papers/2406.17453_learning_informative_questions_eig.pdf | DPO training for high-EIG questions; zero-shot EIG ~0.3 |
| ChatGPT's Information Seeking Strategy | Bertolazzi et al. | 2023 | papers/bertolazzi2023_chatgpt_20q.pdf | Hierarchical 20Q evaluation of ChatGPT |
| Probing Multi-turn Planning via 20Q | Zhang et al. | 2024 | papers/2310.01468_20q_llm_planning.pdf | Entity-Deduction Arena; GPT-4 best at 20Q |
| Uncertainty of Thoughts | Hu et al. | 2024 | papers/2402.03271_uncertainty_of_thoughts.pdf | EIG-based planning for LLMs |
| QuestBench | Li et al. | 2025 | papers/2503.22674_questbench.pdf | Benchmark for LLM question asking |
| Can LLMs Ask Good Questions? | Zhang et al. | 2025 | papers/2501.03491_can_llms_ask_good_questions.pdf | LLM question quality evaluation |
| GameArena | - | 2025 | papers/2412.06394_gamearena.pdf | LLM evaluation through games including Akinator |
| Max Info Gain Coding | Li & Fan | 2024 | papers/2405.16753_max_info_gain_coding.pdf | Information-theoretic foundation for 20Q |
| 20Q to Distinguish LLMs | - | 2024 | papers/2409.10338_20q_distinguish_llms.pdf | Using binary questions to discriminate LLMs |
| Semantic Feature Verification | - | 2023 | papers/2304.05591_semantic_feature_verification.pdf | Probing LLMs on concept-feature pairs |
| Human-Machine Feature Listing | - | 2023 | papers/2304.05012_human_machine_feature_listing.pdf | LLMs generating semantic features |
| Curiosity-Driven Questioning | Javaji & Zhu | 2024 | papers/2409.17172_curiosity_questioning.pdf | LLM question generation quality |
| From Passive to Active Reasoning | Zhou et al. | 2025 | papers/2506.08295_active_reasoning_llm.pdf | LLMs struggle at active reasoning / EIG |
| GuessingGame | - | 2025 | papers/2509.19593_guessinggame_info_gain.pdf | Measuring IG of LLM questions |
| BED-LLM | - | 2025 | papers/2508.21184_bed_llm.pdf | Bayesian experimental design for LLM questions |

See papers/README.md for detailed descriptions.

## Datasets

Total datasets: 8 game set files

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| McRae 8-item | Bertolazzi 2023 | 90 sets | Binary splitting | datasets/game_sets/mcrae_8.json | Taxonomic; 4+4 from 2 categories |
| GPT-norms 8-item | Bertolazzi 2023 | 90 sets | Binary splitting | datasets/game_sets/gpt_8.json | GPT-generated feature norms |
| McRae 16-item | Bertolazzi 2023 | 90 sets | Binary splitting | datasets/game_sets/mcrae_16.json | 3-level hierarchy |
| Things 8-item | Zhang 2024 | 90 sets | Binary splitting | datasets/game_sets/things_8.json | Common objects, diverse |
| BigBench | Srivastava 2023 | 29 sets | Binary splitting | datasets/game_sets/bigbench.json | 29 items each, abstract+concrete |
| Fruits 8-item | Custom | 20 sets | Binary splitting | datasets/game_sets/fruits_8.json | Homogeneous fruit category |
| Mixed categories | Custom | 20 sets | Binary splitting | datasets/game_sets/mixed_categories_8.json | Easy: animals+vehicles, colors+countries |
| Animals 8-item | Custom | 20 sets | Binary splitting | datasets/game_sets/animals_8.json | Hard: all animals, must find sub-categories |

See datasets/README.md for detailed descriptions.

## Code Repositories

Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| LearningToAsk | github.com/dmazzaccara/LearningToAsk | DPO training for EIG questions | code/LearningToAsk/ | Contains game sets, training scripts, EIG computation |
| ml-entity-deduction-arena | github.com/apple/ml-entity-deduction-arena | 20Q game evaluation framework | code/ml-entity-deduction-arena/ | Things/Celebrities datasets, game playing code |
| 20q-chatgpt | github.com/leobertolazzi/20q-chatgpt | ChatGPT 20Q evaluation | code/20q-chatgpt/ | Hierarchical hypothesis spaces, McRae/GPT game sets |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with multiple queries (twenty questions LLM, information gain question generation, binary search questions LM)
2. Supplemented with web search for broader coverage of the niche topic
3. Focused on papers from 2023-2025 in the 20Q / information-seeking / LLM question generation space

### Selection Criteria
- Papers directly studying LLM question generation informativeness (highest priority)
- Papers on 20Q game with LLMs
- Papers on information-theoretic question asking
- Papers on LLM semantic/property knowledge (relevant for answering/annotating)

### Challenges Encountered
- This is a relatively niche research area; paper-finder's initial queries returned few results
- Broader keyword searches needed to find the core papers
- Some papers at intersection of cognitive science and NLP, requiring cross-domain search

### Gaps and Workarounds
- No paper directly tests single-question entropy maximization (all use multi-turn games) — this is the research gap we aim to fill
- Limited work on modern models (GPT-4o, Claude 3.5+) in this setting

## Recommendations for Experiment Design

1. **Primary dataset(s)**: McRae 8-item sets (90 games) — well-established benchmark with known optimal splits. Supplement with Things, BigBench, and custom sets for difficulty analysis.

2. **Baseline methods**:
   - Optimal agent (always achieves EIG=1.0)
   - Zero-shot LLM with simple prompt
   - Zero-shot LLM with chain-of-thought
   - Zero-shot LLM with explicit instruction to split evenly

3. **Evaluation metrics**:
   - EIG of generated question (primary)
   - Perfect split rate (% achieving exact half)
   - Split ratio distribution
   - Per-category analysis (easy vs hard sets)

4. **Code to adapt/reuse**:
   - `code/LearningToAsk/` — EIG computation logic, game set format
   - `code/20q-chatgpt/` — Hypothesis space construction, game evaluation
   - `code/ml-entity-deduction-arena/` — LLM prompting for judge/guesser roles
