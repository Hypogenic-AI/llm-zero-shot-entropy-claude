"""
Analysis script: Generate statistics, visualizations, and tables from experiment results.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path
from scipy import stats
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)


def load_results():
    with open(RESULTS_DIR / "raw_results.json") as f:
        raw = json.load(f)
    # Filter out errors
    valid = [r for r in raw if "eig" in r and "error" not in r]
    df = pd.DataFrame(valid)
    return df


def summary_table(df):
    """Create summary table grouped by dataset, strategy, model."""
    grouped = df.groupby(["dataset", "prompt_strategy", "gen_model"]).agg(
        mean_eig=("eig", "mean"),
        std_eig=("eig", "std"),
        mean_entropy=("binary_entropy", "mean"),
        perfect_split_rate=("perfect_split", "mean"),
        mean_deviation=("deviation_from_half", "mean"),
        count=("eig", "count"),
    ).round(3)
    grouped["perfect_split_rate"] = (grouped["perfect_split_rate"] * 100).round(1)
    return grouped


def plot_eig_by_dataset_and_strategy(df):
    """Bar plot: mean EIG by dataset, grouped by prompt strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, model in enumerate(df["gen_model"].unique()):
        ax = axes[idx]
        sub = df[df["gen_model"] == model]
        pivot = sub.groupby(["dataset", "prompt_strategy"])["binary_entropy"].mean().unstack()

        # Order datasets by mean entropy (ascending = hardest first)
        order = pivot.mean(axis=1).sort_values().index
        pivot = pivot.loc[order]

        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(f"Binary Entropy by Dataset & Strategy ({model})")
        ax.set_ylabel("Binary Entropy (1.0 = perfect split)")
        ax.set_xlabel("")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Optimal")
        ax.legend(fontsize=9, loc="lower right")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eig_by_dataset_strategy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'eig_by_dataset_strategy.png'}")


def plot_split_distributions(df):
    """Histogram of split ratios across all trials."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    strategies = df["prompt_strategy"].unique()
    models = df["gen_model"].unique()

    for i, model in enumerate(models):
        for j, strategy in enumerate(strategies):
            ax = axes[i][j]
            sub = df[(df["gen_model"] == model) & (df["prompt_strategy"] == strategy)]
            ax.hist(sub["yes_count"], bins=range(0, sub["total"].max() + 2),
                    edgecolor="black", alpha=0.7, color=sns.color_palette()[j])
            ax.set_title(f"{model} / {strategy}")
            ax.set_xlabel("Number of 'yes' items")
            ax.set_ylabel("Count")
            # Add vertical line at half
            if len(sub) > 0:
                half = sub["total"].iloc[0] / 2
                ax.axvline(x=half, color="red", linestyle="--", alpha=0.7, label=f"Half={half:.0f}")
                ax.legend()

    plt.suptitle("Distribution of Split Sizes (yes-count)", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "split_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'split_distributions.png'}")


def plot_perfect_split_rate(df):
    """Bar chart: perfect split rate by dataset."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Aggregate across models, show by dataset and strategy
    pivot = df.groupby(["dataset", "prompt_strategy"])["perfect_split"].mean().unstack() * 100

    order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]

    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Perfect Split Rate (%) by Dataset and Prompt Strategy")
    ax.set_ylabel("Perfect Split Rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Strategy")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "perfect_split_rate.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'perfect_split_rate.png'}")


def plot_model_comparison(df):
    """Compare models across datasets."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Best strategy per model-dataset
    best = df.groupby(["dataset", "gen_model"])["binary_entropy"].mean().unstack()
    order = best.mean(axis=1).sort_values(ascending=False).index
    best = best.loc[order]

    best.plot(kind="bar", ax=ax, width=0.7)
    ax.set_title("Model Comparison: Mean Binary Entropy by Dataset")
    ax.set_ylabel("Binary Entropy (1.0 = optimal)")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Optimal")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'model_comparison.png'}")


def plot_dataset_difficulty(df):
    """Scatter: dataset difficulty (mean entropy) vs perfect split rate."""
    fig, ax = plt.subplots(figsize=(10, 7))

    agg = df.groupby("dataset").agg(
        mean_entropy=("binary_entropy", "mean"),
        perfect_rate=("perfect_split", "mean"),
        n_items=("total", "first"),
    )
    agg["perfect_rate"] *= 100

    colors = sns.color_palette("husl", len(agg))
    for idx, (ds, row) in enumerate(agg.iterrows()):
        ax.scatter(row["mean_entropy"], row["perfect_rate"], s=150, c=[colors[idx]],
                   edgecolors="black", zorder=5)
        ax.annotate(ds, (row["mean_entropy"], row["perfect_rate"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)

    ax.set_xlabel("Mean Binary Entropy")
    ax.set_ylabel("Perfect Split Rate (%)")
    ax.set_title("Dataset Difficulty: Entropy vs Perfect Split Rate")
    ax.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dataset_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'dataset_difficulty.png'}")


def statistical_tests(df):
    """Run statistical comparisons."""
    results = {}

    # 1. Compare prompt strategies (paired by set_id within each dataset)
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    strategies = sorted(df["prompt_strategy"].unique())
    models = sorted(df["gen_model"].unique())

    # Strategy comparison for primary model
    primary_model = "gpt-4.1"
    sub = df[df["gen_model"] == primary_model]

    print(f"\n--- Strategy Comparison (model={primary_model}) ---")
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            vals1 = sub[sub["prompt_strategy"] == s1]["binary_entropy"].values
            vals2 = sub[sub["prompt_strategy"] == s2]["binary_entropy"].values
            min_n = min(len(vals1), len(vals2))
            if min_n < 5:
                continue
            t_stat, p_val = stats.ttest_ind(vals1[:min_n], vals2[:min_n])
            d = (np.mean(vals1) - np.mean(vals2)) / np.sqrt((np.std(vals1)**2 + np.std(vals2)**2) / 2)
            print(f"  {s1} vs {s2}: t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d:.3f}")
            results[f"{s1}_vs_{s2}"] = {"t": t_stat, "p": p_val, "d": d}

    # 2. Model comparison
    print(f"\n--- Model Comparison (strategy=explicit_split) ---")
    sub = df[df["prompt_strategy"] == "explicit_split"]
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            vals1 = sub[sub["gen_model"] == m1]["binary_entropy"].values
            vals2 = sub[sub["gen_model"] == m2]["binary_entropy"].values
            if len(vals1) < 5 or len(vals2) < 5:
                continue
            t_stat, p_val = stats.ttest_ind(vals1, vals2)
            d = (np.mean(vals1) - np.mean(vals2)) / np.sqrt((np.std(vals1)**2 + np.std(vals2)**2) / 2)
            print(f"  {m1} vs {m2}: t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d:.3f}")
            results[f"{m1}_vs_{m2}"] = {"t": t_stat, "p": p_val, "d": d}

    # 3. ANOVA across datasets
    print(f"\n--- ANOVA: EIG across datasets (strategy=explicit_split, model=gpt-4.1) ---")
    sub = df[(df["prompt_strategy"] == "explicit_split") & (df["gen_model"] == "gpt-4.1")]
    groups = [g["binary_entropy"].values for _, g in sub.groupby("dataset")]
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"  F={f_stat:.3f}, p={p_val:.6f}")
        results["anova_datasets"] = {"F": f_stat, "p": p_val}

    # Save
    with open(RESULTS_DIR / "statistical_tests.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    return results


def example_questions(df, n=5):
    """Print example questions from each dataset."""
    print("\n" + "=" * 60)
    print("EXAMPLE QUESTIONS")
    print("=" * 60)

    examples = []
    for dataset in df["dataset"].unique():
        sub = df[(df["dataset"] == dataset) & (df["prompt_strategy"] == "explicit_split") &
                 (df["gen_model"] == "gpt-4.1")]
        if len(sub) == 0:
            continue

        # Show best and worst
        best = sub.loc[sub["binary_entropy"].idxmax()]
        worst = sub.loc[sub["binary_entropy"].idxmin()]

        print(f"\n--- {dataset} ---")
        print(f"  BEST (entropy={best['binary_entropy']:.3f}, split={best['split_ratio']}):")
        print(f"    Items: {', '.join(best['items'])}")
        print(f"    Q: {best['question']}")
        print(f"    Yes: {best['yes_items']}")
        print(f"    No: {best['no_items']}")

        print(f"  WORST (entropy={worst['binary_entropy']:.3f}, split={worst['split_ratio']}):")
        print(f"    Items: {', '.join(worst['items'])}")
        print(f"    Q: {worst['question']}")
        print(f"    Yes: {worst['yes_items']}")
        print(f"    No: {worst['no_items']}")

        examples.append({
            "dataset": dataset,
            "best": {"entropy": best["binary_entropy"], "split": best["split_ratio"],
                      "items": best["items"], "question": best["question"],
                      "yes": best["yes_items"], "no": best["no_items"]},
            "worst": {"entropy": worst["binary_entropy"], "split": worst["split_ratio"],
                       "items": worst["items"], "question": worst["question"],
                       "yes": worst["yes_items"], "no": worst["no_items"]},
        })

    with open(RESULTS_DIR / "example_questions.json", "w") as f:
        json.dump(examples, f, indent=2)

    return examples


def main():
    print("Loading results...")
    df = load_results()
    print(f"Loaded {len(df)} valid trials")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    summary = summary_table(df)
    print(summary.to_string())
    summary.to_csv(RESULTS_DIR / "summary_table.csv")

    # Visualizations
    print("\nGenerating plots...")
    plot_eig_by_dataset_and_strategy(df)
    plot_split_distributions(df)
    plot_perfect_split_rate(df)
    plot_model_comparison(df)
    plot_dataset_difficulty(df)

    # Statistical tests
    stat_results = statistical_tests(df)

    # Example questions
    examples = example_questions(df)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
