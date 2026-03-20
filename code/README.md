# Cloned Repositories

## Repo 1: LearningToAsk
- **URL**: https://github.com/dmazzaccara/LearningToAsk
- **Purpose**: DPO training to improve LLM question informativeness (EIG) in 20Q games
- **Location**: code/LearningToAsk/
- **Key files**:
  - `scripts/` — Training and evaluation scripts
  - `data/game_sets/` — Game set definitions (McRae, Things, Celebrities, BigBench)
  - `data/game_sets/test/contrast_sets_8.json` — 90 test games with 8 items each
- **Dependencies**: See requirements.txt and environment.yml
- **Relevance**: Provides EIG computation methodology, game set format, and evaluation pipeline. Our experiment can reuse the game sets and adapt the EIG computation.

## Repo 2: ml-entity-deduction-arena
- **URL**: https://github.com/apple/ml-entity-deduction-arena
- **Purpose**: Entity-Deduction Arena (EDA) for evaluating LLMs on 20Q game
- **Location**: code/ml-entity-deduction-arena/
- **Key files**:
  - `GPT_Q20.py` — Main game playing script (Things)
  - `GPT_Q20_celebrity.py` — Celebrity variant
  - `game.py` — Game logic
  - `data/things/` — 500 Things entities (train/dev/test splits)
  - `data/celebrities/` — 500 Celebrity names
- **Dependencies**: See requirements.txt (openai, etc.)
- **Relevance**: Provides LLM-as-judge prompting patterns, game evaluation logic, and entity lists. The judge prompting approach is useful for our annotator (determining yes/no for each item given a question).

## Repo 3: 20q-chatgpt
- **URL**: https://github.com/leobertolazzi/20q-chatgpt
- **Purpose**: Evaluating ChatGPT's information-seeking strategy in hierarchical 20Q
- **Location**: code/20q-chatgpt/
- **Key files**:
  - `data/game_sets/8_mcrae/contrast_sets.json` — 90 McRae-norm 8-item games
  - `data/game_sets/8_gpt/contrast_sets.json` — 90 GPT-norm 8-item games
  - `data/game_sets/16_mcrae/contrast_sets.json` — 90 McRae-norm 16-item games
  - `data/game_sets/8_mcrae/contrast_sets_with_features.json` — Sets with feature annotations
  - `scripts/` — Game playing and analysis scripts
  - `results/` — Pre-computed results
- **Dependencies**: See requirements.txt
- **Relevance**: Primary source of our benchmark datasets. The `contrast_sets_with_features.json` files include feature annotations that define the "correct" splits, useful for evaluation.
