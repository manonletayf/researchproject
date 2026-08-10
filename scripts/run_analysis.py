"""Analyse statistique complete RQ2 (Jour 6) : Kruskal-Wallis, Mann-Whitney,
et ANOVA factorielle pour les 3 metriques (RTD, latence, couverture
substantielle du raisonnement), avec test d'interaction topologie x schema.
Affiche les resultats dans le terminal et les sauvegarde dans un rapport texte."""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from io import StringIO
import statsmodels.api as sm
from statsmodels.formula.api import ols

BASE = Path(__file__).parent.parent

class Tee:
    """Ecrit simultanement dans le terminal et dans un buffer pour sauvegarde."""
    def __init__(self):
        self.buffer = StringIO()

    def write(self, text):
        sys.__stdout__.write(text)
        self.buffer.write(text)

    def flush(self):
        sys.__stdout__.flush()


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def kruskal_wallis_by_topology(df, metric_col, label):
    print(f"\n--- Kruskal-Wallis: {label} par topologie ---")
    groups = [df[df["condition"] == c][metric_col].values
              for c in ["hierarchical", "decentralized", "shared_pool"]]
    h_stat, p = stats.kruskal(*groups)
    n_total = len(df)
    eta_sq = (h_stat - 3 + 1) / (n_total - 3)  # approximation eta^2 pour Kruskal-Wallis
    print(f"H={h_stat:.3f}, p={p:.6f}, eta^2 approx={eta_sq:.3f}")
    sig = "SIGNIFICATIF" if p < 0.05 else "non significatif"
    print(f"-> {sig} (seuil 0.05)")

    print("\nMoyennes par condition:")
    print(df.groupby("condition")[metric_col].agg(["mean", "median", "std"]).round(4))
    return h_stat, p


def mann_whitney_by_schema(df, metric_col, label):
    print(f"\n--- Mann-Whitney: {label} par schema (toutes conditions confondues) ---")
    free = df[df["schema_mode"] == "free_text"][metric_col].values
    typed = df[df["schema_mode"] == "typed"][metric_col].values
    u_stat, p = stats.mannwhitneyu(free, typed, alternative="two-sided")
    print(f"U={u_stat:.1f}, p={p:.6f}")
    sig = "SIGNIFICATIF" if p < 0.05 else "non significatif"
    print(f"-> {sig} (seuil 0.05)")
    print(f"Moyenne free_text: {free.mean():.4f} | Moyenne typed: {typed.mean():.4f}")
    return u_stat, p


def mann_whitney_by_schema_within_condition(df, metric_col, label):
    print(f"\n--- Mann-Whitney: {label} par schema, DANS CHAQUE condition (verification interaction) ---")
    results = {}
    for condition in ["hierarchical", "decentralized", "shared_pool"]:
        sub = df[df["condition"] == condition]
        free = sub[sub["schema_mode"] == "free_text"][metric_col].values
        typed = sub[sub["schema_mode"] == "typed"][metric_col].values
        u_stat, p = stats.mannwhitneyu(free, typed, alternative="two-sided")
        diff = typed.mean() - free.mean()
        sig = "SIGNIFICATIF" if p < 0.05 else "n.s."
        print(f"  {condition:<15} typed-free_text={diff:+.4f}  p={p:.4f}  ({sig})")
        results[condition] = {"p": p, "diff": diff}
    return results


def factorial_anova(df, metric_col, label, log_transform=False):
    print(f"\n--- ANOVA factorielle 2 facteurs: {label} ---")
    data = df.copy()
    if log_transform:
        # log(x + 1) pour eviter log(0) si jamais une valeur est nulle
        data[metric_col] = np.log1p(data[metric_col])
        print("(variable log-transformee: log(x+1))")

    model = ols(f"{metric_col} ~ C(condition) * C(schema_mode)", data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table.round(5))

    interaction_p = anova_table.loc["C(condition):C(schema_mode)", "PR(>F)"]
    sig = "SIGNIFICATIVE" if interaction_p < 0.05 else "non significative"
    print(f"\n-> Interaction topologie x schema: p={interaction_p:.4f} ({sig})")
    return anova_table


def main():
    df = pd.read_csv(BASE / "full_run_results" / "consolidated_dataset.csv")

    sys.stdout = Tee()

    section("JOUR 6 - ANALYSE STATISTIQUE RQ2")
    print(f"Dataset: {len(df)} runs, {df['condition'].nunique()} conditions x {df['schema_mode'].nunique()} modes")

    # ============ RTD ============
    section("METRIQUE 1: PERTE D'INFORMATION (RTD)")
    kruskal_wallis_by_topology(df, "rtd_score", "RTD")
    mann_whitney_by_schema(df, "rtd_score", "RTD")
    mann_whitney_by_schema_within_condition(df, "rtd_score", "RTD")
    factorial_anova(df, "rtd_score", "RTD (ANOVA classique malgre non-normalite, a titre indicatif)")

    # ============ LATENCE ============
    section("METRIQUE 2: LATENCE DE COORDINATION")
    kruskal_wallis_by_topology(df, "total_llm_latency_seconds", "Latence")
    mann_whitney_by_schema(df, "total_llm_latency_seconds", "Latence")
    mann_whitney_by_schema_within_condition(df, "total_llm_latency_seconds", "Latence")
    factorial_anova(df, "total_llm_latency_seconds", "Latence", log_transform=True)

    # ============ COUVERTURE SUBSTANTIELLE ============
    section("METRIQUE 3: COUVERTURE SUBSTANTIELLE DU RAISONNEMENT")
    kruskal_wallis_by_topology(df, "substantive_reasoning_score", "Couverture substantielle")
    mann_whitney_by_schema(df, "substantive_reasoning_score", "Couverture substantielle")
    mann_whitney_by_schema_within_condition(df, "substantive_reasoning_score", "Couverture substantielle")
    factorial_anova(df, "substantive_reasoning_score", "Couverture substantielle (ANOVA classique malgre non-normalite, a titre indicatif)")

    # ============ CORRECTION MULTIPLE ============
    section("RAPPEL - CORRECTION POUR COMPARAISONS MULTIPLES")
    print("3 metriques testees independamment (Kruskal-Wallis topologie).")
    print("Seuil Bonferroni ajuste: 0.05 / 3 =", round(0.05/3, 4))
    print("A appliquer lors de l'interpretation finale des p-values ci-dessus.")

    output_path = BASE / "full_run_results" / "day6_analysis_report.txt"
    with open(output_path, "w") as f:
        f.write(sys.stdout.buffer.getvalue())

    sys.stdout = sys.__stdout__
    print(f"\n\nRapport complet sauvegarde dans: {output_path}")


if __name__ == "__main__":
    main()