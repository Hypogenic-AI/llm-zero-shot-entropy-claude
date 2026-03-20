# Datasets

This directory contains datasets for the research project: "Can LLMs zero-shot max entropy questions?"

## Dataset Overview

All datasets are in `game_sets/` as JSON files. Each file contains sets of items that an LLM should split into two equal groups using a single yes/no question.

### Format

Each JSON file maps game IDs to game objects:
```json
{
  "0": {
    "items": ["elk", "chicken", "robin", "starling", "fox", "partridge", "hamster", "buffalo"]
  }
}
```

The task: given `items`, generate a yes/no question that splits them into two groups of equal (or near-equal) size.

## Datasets

### 1. McRae 8-item sets (`mcrae_8.json`)
- **Source**: Bertolazzi et al. (2023), based on McRae feature norms (McRae et al., 2005)
- **Size**: 90 sets, 8 items each
- **Structure**: Taxonomically organized — 4 items from category A + 4 from category B (e.g., 4 birds + 4 mammals)
- **Difficulty**: Easy — clear category boundary exists
- **Categories**: mammal, bird, clothing, weapon, fruit, vegetable

### 2. GPT-norms 8-item sets (`gpt_8.json`)
- **Source**: Bertolazzi et al. (2023), based on GPT-3.5 feature norms (Hansen & Hebart, 2022)
- **Size**: 90 sets, 8 items each
- **Structure**: Same taxonomic structure as McRae but using GPT-generated feature norms
- **Difficulty**: Easy-Medium

### 3. McRae 16-item sets (`mcrae_16.json`)
- **Source**: Bertolazzi et al. (2023)
- **Size**: 90 sets, 16 items each
- **Structure**: 3-level hierarchy (8:8, then 4:4, then 2:2)
- **Difficulty**: Medium — more items, deeper hierarchy

### 4. Things 8-item sets (`things_8.json`)
- **Source**: Zhang et al. (2024) / Mazzaccara et al. (2024)
- **Size**: 90 sets, 8 items each
- **Structure**: Common-life objects from unseen categories
- **Difficulty**: Medium — less clear taxonomic boundaries

### 5. BigBench sets (`bigbench.json`)
- **Source**: Srivastava et al. (2023) / Mazzaccara et al. (2024)
- **Size**: 29 sets, 29 items each
- **Structure**: Mix of concrete and abstract concepts
- **Difficulty**: Hard — abstract concepts, large sets, no clear category structure

### 6. Fruits 8-item sets (`fruits_8.json`)
- **Source**: Custom-created for this project
- **Size**: 20 sets, 8 fruits each
- **Structure**: All items are fruits (homogeneous category)
- **Difficulty**: Medium — must find sub-category distinctions within fruits

### 7. Mixed categories 8-item sets (`mixed_categories_8.json`)
- **Source**: Custom-created for this project
- **Size**: 20 sets, 8 items each
- **Structure**: 4 items from each of 2 categories (animals+vehicles or colors+countries)
- **Difficulty**: Easy — obvious category boundary

### 8. Animals 8-item sets (`animals_8.json`)
- **Source**: Custom-created for this project
- **Size**: 20 sets, 8 animals each
- **Structure**: All items are animals (homogeneous category)
- **Difficulty**: Hard — must find sub-category distinctions within animals

## Loading

```python
import json

with open("datasets/game_sets/mcrae_8.json") as f:
    games = json.load(f)

for game_id, game in games.items():
    items = game["items"]
    # Pass items to LLM and ask for a splitting question
```

## Evaluation

For each generated question:
1. Apply the question to each item in the set (LLM answers yes/no)
2. Count items in yes-group and no-group
3. Compute entropy: `H = -p*log2(p) - (1-p)*log2(1-p)` where `p = |yes_group| / |total|`
4. Maximum entropy = 1.0 (perfect 4:4 split for 8 items)

## Original Sources

- McRae norms: McRae et al. (2005). "Semantic feature production norms for a large set of living and nonliving things." Behavior Research Methods.
- GPT norms: Hansen & Hebart (2022). Semantic features from GPT-3. GitHub: ViCCo-Group/semantic_features_gpt_3
- Things/Celebrities: Zhang et al. (2024). "Probing the Multi-turn Planning Capabilities of LLMs via 20 Question Games." ACL 2024.
- BigBench: Srivastava et al. (2023). "Beyond the Imitation Game." TMLR.
