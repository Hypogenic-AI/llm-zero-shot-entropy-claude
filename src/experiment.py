"""
Experiment: Can LLMs zero-shot generate max-entropy yes/no questions?

Given a set of items, we ask an LLM to generate a yes/no question that splits the set
into two equal halves. We then use a judge model to annotate each item, compute the
resulting split, and measure Expected Information Gain (EIG).
"""

import json
import os
import sys
import time
import random
import asyncio
import logging
from pathlib import Path
from math import log2
from datetime import datetime

import numpy as np
from openai import AsyncOpenAI

# Configuration
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets" / "game_sets"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "logs" / "experiment.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

client = AsyncOpenAI()

# ── Prompt strategies ──────────────────────────────────────────────────────

PROMPTS = {
    "basic": (
        "Here is a set of items:\n{items_str}\n\n"
        "Generate a single yes/no question about these items."
    ),
    "explicit_split": (
        "Here is a set of {n} items:\n{items_str}\n\n"
        "Generate a single yes/no question such that exactly {half} of the items "
        "would be answered 'yes' and exactly {half} would be answered 'no'. "
        "The goal is to split the set into two equal halves. "
        "Output ONLY the question, nothing else."
    ),
    "chain_of_thought": (
        "Here is a set of {n} items:\n{items_str}\n\n"
        "Your task: generate a yes/no question that splits these items into two "
        "groups of equal size ({half} yes, {half} no).\n\n"
        "Think step by step:\n"
        "1. Look at the items and identify properties that some have and others don't.\n"
        "2. Find a property that exactly {half} items share.\n"
        "3. Formulate a yes/no question based on that property.\n\n"
        "First show your reasoning, then on the last line write QUESTION: <your question>"
    ),
}

JUDGE_PROMPT = (
    "For the item \"{item}\", answer the following yes/no question with ONLY "
    "\"yes\" or \"no\" (one word, lowercase).\n\n"
    "Question: {question}"
)

# Batch judge prompt - annotate all items at once for efficiency
BATCH_JUDGE_PROMPT = (
    "For each item below, answer the following yes/no question with ONLY "
    "\"yes\" or \"no\". Return a JSON object mapping each item to \"yes\" or \"no\".\n\n"
    "Question: {question}\n\n"
    "Items:\n{items_str}\n\n"
    "Return ONLY valid JSON, no explanation. Example format:\n"
    '{{"{example1}": "yes", "{example2}": "no"}}'
)


# ── EIG computation ────────────────────────────────────────────────────────

def compute_eig(yes_count: int, total: int) -> float:
    """Compute Expected Information Gain for a binary split.

    EIG = H(prior) - E[H(posterior)]
    For uniform prior over `total` items:
      H(prior) = log2(total)
      After observing answer, posterior is uniform over yes_count or no_count items.
      E[H(posterior)] = (yes_count/total)*log2(yes_count) + (no_count/total)*log2(no_count)

    Normalized to [0,1] by dividing by H(prior).
    Actually, let's use the standard binary entropy approach:
    EIG = 1 bit when split is 50/50 (for the binary question).
    """
    if total == 0:
        return 0.0
    no_count = total - yes_count
    if yes_count == 0 or no_count == 0:
        return 0.0  # No information gained

    p_yes = yes_count / total
    p_no = no_count / total

    # Prior entropy (uniform): log2(total)
    h_prior = log2(total)

    # Expected posterior entropy:
    # If answer is "yes" (prob p_yes): H(posterior) = log2(yes_count)
    # If answer is "no" (prob p_no): H(posterior) = log2(no_count)
    h_posterior = p_yes * log2(yes_count) + p_no * log2(no_count)

    eig = h_prior - h_posterior
    return eig


def compute_normalized_eig(yes_count: int, total: int) -> float:
    """EIG normalized to [0, 1] where 1 = perfect half split."""
    eig = compute_eig(yes_count, total)
    max_eig = compute_eig(total // 2, total)  # Best possible
    if max_eig == 0:
        return 0.0
    return eig / max_eig


def compute_binary_entropy(yes_count: int, total: int) -> float:
    """Binary entropy of the split: H(p) = -p*log2(p) - (1-p)*log2(1-p).
    This equals 1.0 for a perfect 50/50 split."""
    if total == 0 or yes_count == 0 or yes_count == total:
        return 0.0
    p = yes_count / total
    return -p * log2(p) - (1 - p) * log2(1 - p)


# ── API calls ──────────────────────────────────────────────────────────────

async def generate_question(
    items: list[str],
    prompt_strategy: str,
    model: str = "gpt-4.1",
    temperature: float = 0.3,
) -> str:
    """Generate a yes/no question for splitting the item set."""
    n = len(items)
    half = n // 2
    items_str = "\n".join(f"- {item}" for item in items)

    prompt_template = PROMPTS[prompt_strategy]
    prompt = prompt_template.format(
        items_str=items_str, n=n, half=half,
        example1=items[0], example2=items[1],
    )

    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=300,
            )
            text = resp.choices[0].message.content.strip()

            # For CoT, extract the question after "QUESTION:"
            if prompt_strategy == "chain_of_thought" and "QUESTION:" in text:
                text_parts = text.split("QUESTION:")
                question = text_parts[-1].strip()
                return question

            return text
        except Exception as e:
            log.warning(f"API error (attempt {attempt+1}): {e}")
            await asyncio.sleep(2 ** attempt)

    return ""


async def judge_items(
    items: list[str],
    question: str,
    model: str = "gpt-4.1",
) -> dict[str, str]:
    """Use judge model to annotate each item as yes/no for the question."""
    items_str = "\n".join(f"- {item}" for item in items)

    prompt = BATCH_JUDGE_PROMPT.format(
        question=question,
        items_str=items_str,
        example1=items[0],
        example2=items[-1],
    )

    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            text = resp.choices[0].message.content.strip()

            # Parse JSON response
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            annotations = json.loads(text)

            # Normalize: ensure all items are present and values are yes/no
            result = {}
            for item in items:
                val = annotations.get(item, "").lower().strip()
                if val not in ("yes", "no"):
                    # Try case-insensitive key matching
                    for k, v in annotations.items():
                        if k.lower() == item.lower():
                            val = v.lower().strip()
                            break
                if val not in ("yes", "no"):
                    val = "unknown"
                result[item] = val

            return result

        except (json.JSONDecodeError, Exception) as e:
            log.warning(f"Judge error (attempt {attempt+1}): {e}")
            await asyncio.sleep(2 ** attempt)

    # Fallback: return empty
    return {item: "unknown" for item in items}


# ── Experiment runner ──────────────────────────────────────────────────────

async def run_single_trial(
    items: list[str],
    prompt_strategy: str,
    gen_model: str,
    judge_model: str = "gpt-4.1",
) -> dict:
    """Run one trial: generate question, judge items, compute metrics."""
    question = await generate_question(items, prompt_strategy, model=gen_model)
    if not question:
        return {"error": "Failed to generate question"}

    annotations = await judge_items(items, question, model=judge_model)

    yes_items = [it for it, ans in annotations.items() if ans == "yes"]
    no_items = [it for it, ans in annotations.items() if ans == "no"]
    unknown_items = [it for it, ans in annotations.items() if ans == "unknown"]

    total = len(items)
    yes_count = len(yes_items)
    no_count = len(no_items)

    eig = compute_eig(yes_count, total)
    binary_entropy = compute_binary_entropy(yes_count, total)
    deviation = abs(yes_count - total / 2)
    perfect_split = (yes_count == total // 2)

    return {
        "items": items,
        "question": question,
        "annotations": annotations,
        "yes_items": yes_items,
        "no_items": no_items,
        "unknown_items": unknown_items,
        "yes_count": yes_count,
        "no_count": no_count,
        "total": total,
        "eig": eig,
        "binary_entropy": binary_entropy,
        "deviation_from_half": deviation,
        "perfect_split": perfect_split,
        "split_ratio": f"{yes_count}:{no_count}",
        "prompt_strategy": prompt_strategy,
        "gen_model": gen_model,
        "judge_model": judge_model,
    }


async def run_dataset_experiment(
    dataset_name: str,
    dataset: dict,
    prompt_strategies: list[str],
    gen_models: list[str],
    judge_model: str = "gpt-4.1",
    max_sets: int | None = None,
) -> list[dict]:
    """Run experiment on a full dataset across strategies and models."""
    results = []
    set_ids = list(dataset.keys())
    if max_sets:
        set_ids = set_ids[:max_sets]

    total_trials = len(set_ids) * len(prompt_strategies) * len(gen_models)
    log.info(f"Running {total_trials} trials for dataset '{dataset_name}'")

    # Process in batches to manage concurrency
    BATCH_SIZE = 10
    trials = []
    for set_id in set_ids:
        items = dataset[set_id]["items"]
        for strategy in prompt_strategies:
            for model in gen_models:
                trials.append((set_id, items, strategy, model))

    for batch_start in range(0, len(trials), BATCH_SIZE):
        batch = trials[batch_start:batch_start + BATCH_SIZE]
        tasks = []
        for set_id, items, strategy, model in batch:
            tasks.append(run_single_trial(items, strategy, model, judge_model))

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for (set_id, items, strategy, model), result in zip(batch, batch_results):
            if isinstance(result, Exception):
                log.error(f"Trial failed: {set_id}/{strategy}/{model}: {result}")
                result = {"error": str(result)}
            result["dataset"] = dataset_name
            result["set_id"] = set_id
            results.append(result)

        done = min(batch_start + BATCH_SIZE, len(trials))
        log.info(f"  Progress: {done}/{len(trials)} trials complete")

    return results


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    log.info("=" * 60)
    log.info("Starting experiment: Can LLMs zero-shot max entropy questions?")
    log.info(f"Timestamp: {datetime.now().isoformat()}")
    log.info(f"Seed: {SEED}")
    log.info("=" * 60)

    # Load datasets
    datasets = {}
    for fname in sorted(DATASETS_DIR.glob("*.json")):
        name = fname.stem
        with open(fname) as f:
            datasets[name] = json.load(f)
        log.info(f"Loaded dataset '{name}': {len(datasets[name])} sets")

    # Experiment configuration
    prompt_strategies = ["basic", "explicit_split", "chain_of_thought"]
    gen_models = ["gpt-4.1", "gpt-4o-mini"]
    judge_model = "gpt-4.1"

    # Dataset limits - use all for small datasets, sample for large ones
    dataset_limits = {
        "fruits_8": None,         # 20 sets - use all
        "animals_8": None,        # 20 sets - use all
        "mixed_categories_8": None,  # 20 sets - use all
        "mcrae_8": 30,            # 90 sets - sample 30
        "things_8": 30,           # 90 sets - sample 30
        "mcrae_16": 20,           # 90 sets - sample 20 (larger sets)
        "gpt_8": 20,              # 90 sets - sample 20
        "bigbench": 15,           # 29 sets - sample 15
    }

    all_results = []

    for dataset_name, dataset in datasets.items():
        limit = dataset_limits.get(dataset_name, 20)
        results = await run_dataset_experiment(
            dataset_name=dataset_name,
            dataset=dataset,
            prompt_strategies=prompt_strategies,
            gen_models=gen_models,
            judge_model=judge_model,
            max_sets=limit,
        )
        all_results.extend(results)
        log.info(f"Completed dataset '{dataset_name}': {len(results)} trials")

    # Save raw results
    output_path = RESULTS_DIR / "raw_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Saved {len(all_results)} results to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    # Group by dataset, strategy, model
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        if "error" not in r or "eig" in r:
            key = (r["dataset"], r["prompt_strategy"], r["gen_model"])
            grouped[key].append(r)

    print(f"\n{'Dataset':<25} {'Strategy':<18} {'Model':<15} {'Mean EIG':>10} {'Binary H':>10} {'Perfect%':>10} {'N':>5}")
    print("-" * 100)

    for (ds, strat, model), trials in sorted(grouped.items()):
        eigs = [t["eig"] for t in trials if "eig" in t]
        entropies = [t["binary_entropy"] for t in trials if "binary_entropy" in t]
        perfect = [t["perfect_split"] for t in trials if "perfect_split" in t]
        if eigs:
            print(f"{ds:<25} {strat:<18} {model:<15} {np.mean(eigs):>10.3f} {np.mean(entropies):>10.3f} {100*np.mean(perfect):>9.1f}% {len(eigs):>5}")

    # Save config
    config = {
        "seed": SEED,
        "prompt_strategies": prompt_strategies,
        "gen_models": gen_models,
        "judge_model": judge_model,
        "dataset_limits": dataset_limits,
        "timestamp": datetime.now().isoformat(),
        "total_trials": len(all_results),
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("Experiment complete!")
    return all_results


if __name__ == "__main__":
    results = asyncio.run(main())
