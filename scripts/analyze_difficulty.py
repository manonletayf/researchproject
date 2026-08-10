"""Teste l'effet de structural_complexity sur les 3 metriques RQ2,
globalement et a l'interieur de chaque topologie."""
import sys
import pandas as pd
from scipy import stats
from pathlib import Path

BASE = Path(__file__).parent.parent

def kw_by_complexity(df, metric_col, label, subset_label=""):
    tag = f" ({subset_label})" if subset_label else ""
    print(f"\n--- Kruskal-Wallis: {label} par complexite{tag} ---")
    groups = [df[df["structural_complexity"] == c][metric_col].values
              for c in ["easy", "medium", "hard"]]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        print("Donnees insuffisantes pour ce sous-groupe.")
        return
    h_stat, p = stats.kruskal(*groups)
    sig = "SIGNIFICATIF" if p < 0.05 else "non significatif"
    print(f"H={h_stat:.3f}, p={p:.4f} -> {sig}")
    print(df.groupby("structural_complexity")[metric_col].agg(["mean", "median", "count"]).round(4))

def main():
    df = pd.read_csv(BASE / "full_run_results" / "consolidated_dataset.csv")

    print("=" * 70)
    print("EFFET DE LA COMPLEXITE STRUCTURELLE")
    print("=" * 70)

    for metric_col, label in [
        ("rtd_score", "RTD"),
        ("total_llm_latency_seconds", "Latence"),
        ("substantive_reasoning_score", "Couverture substantielle"),
    ]:
        kw_by_complexity(df, metric_col, label)  # global
        for condition in ["hierarchical", "decentralized", "shared_pool"]:
            sub = df[df["condition"] == condition]
            kw_by_complexity(sub, metric_col, label, subset_label=condition)

    # crosstab RTD categorise x complexite x topologie (inspire de RQ5 du papier source)
    print("\n--- Table croisee: moyenne RTD par complexite x topologie ---")
    pivot = df.pivot_table(values="rtd_score", index="condition", columns="structural_complexity", aggfunc="mean")
    print(pivot.round(4))

if __name__ == "__main__":
    main()