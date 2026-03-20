# Downloaded Papers

## Core Papers (Deep Read)

1. **Learning to Ask Informative Questions** (2406.17453_learning_informative_questions_eig.pdf)
   - Authors: Mazzaccara, Testoni, Bernardi
   - Year: 2024
   - arXiv: 2406.17453
   - Why relevant: Most directly relevant — trains LLMs to maximize EIG in 20Q. Shows zero-shot EIG ~0.3, optimal is 1.0. Uses DPO with EIG-ranked preference pairs.

2. **ChatGPT's Information Seeking Strategy** (bertolazzi2023_chatgpt_20q.pdf)
   - Authors: Bertolazzi, Mazzaccara, Merlo, Bernardi
   - Year: 2023 (INLG)
   - Why relevant: Establishes the experimental paradigm — hierarchical 20Q with McRae/GPT feature norms. Shows ChatGPT far from optimal. Provides our primary datasets.

3. **Probing Multi-turn Planning via 20 Questions** (2310.01468_20q_llm_planning.pdf)
   - Authors: Zhang, Lu, Jaitly (Apple)
   - Year: 2024 (ACL)
   - arXiv: 2310.01468
   - Why relevant: Systematic LLM benchmark on 20Q. GPT-4 uses binary-partitioning strategy. Provides Things/Celebrities datasets and failure mode taxonomy.

## Supporting Papers

4. **Uncertainty of Thoughts** (2402.03271_uncertainty_of_thoughts.pdf)
   - Year: 2024 (NeurIPS)
   - arXiv: 2402.03271
   - Why relevant: EIG-based planning algorithm for LLM question asking.

5. **QuestBench** (2503.22674_questbench.pdf)
   - Year: 2025
   - arXiv: 2503.22674
   - Why relevant: Benchmark for LLM question asking under missing information.

6. **Can LLMs Ask Good Questions?** (2501.03491_can_llms_ask_good_questions.pdf)
   - Year: 2025
   - arXiv: 2501.03491
   - Why relevant: Evaluates LLM question quality across multiple dimensions.

7. **GameArena** (2412.06394_gamearena.pdf)
   - Year: 2025 (ICLR)
   - arXiv: 2412.06394
   - Why relevant: Includes Akinator game testing LLM binary question strategies.

8. **Max Information Gain Coding** (2405.16753_max_info_gain_coding.pdf)
   - Year: 2024
   - arXiv: 2405.16753
   - Why relevant: Information-theoretic foundation for constrained querying.

9. **20Q to Distinguish LLMs** (2409.10338_20q_distinguish_llms.pdf)
   - Year: 2024
   - arXiv: 2409.10338
   - Why relevant: Uses binary questions to discriminate between LLMs.

10. **Semantic Feature Verification in FLAN-T5** (2304.05591_semantic_feature_verification.pdf)
    - Year: 2023 (ICLR Tiny Paper)
    - arXiv: 2304.05591
    - Why relevant: Tests LLM yes/no feature-concept verification accuracy.

11. **Human-Machine Cooperation for Semantic Features** (2304.05012_human_machine_feature_listing.pdf)
    - Year: 2023
    - arXiv: 2304.05012
    - Why relevant: LLMs generating semantic feature norms for concepts.

12. **Curiosity-Driven Questioning** (2409.17172_curiosity_questioning.pdf)
    - Year: 2024
    - arXiv: 2409.17172
    - Why relevant: LLM question generation quality evaluation.

13. **From Passive to Active Reasoning** (2506.08295_active_reasoning_llm.pdf)
    - Year: 2025 (ICML)
    - arXiv: 2506.08295
    - Why relevant: AR-Bench shows LLMs struggle at active reasoning and EIG optimization.

14. **GuessingGame** (2509.19593_guessinggame_info_gain.pdf)
    - Year: 2025 (EMNLP)
    - arXiv: 2509.19593
    - Why relevant: Directly measures information gain of LLM questions.

15. **BED-LLM** (2508.21184_bed_llm.pdf)
    - Year: 2025
    - arXiv: 2508.21184
    - Why relevant: Bayesian experimental design with EIG for optimal LLM questions.
