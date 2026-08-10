"""Categorise chaque run selon ou la perte du tracer se produit
(jamais reproduit / perdu en transit / preserve), et teste l'association
avec topologie x schema via chi-carre, plus un test cible sur l'anomalie
shared_pool + typed."""
import json
import pandas as pd
from scipy import stats
from pathlib import Path

BASE = Path(__file__).parent.parent

def categorize(row):
    if row["root_output_score"] == 0:
        return "never_reproduced"
    elif row["final_output_score"] == 0:
        return "lost_in_transit"
    else:
        return "preserved"

def main():
    records = [json.loads(l) for l in open(BASE / "full_run_results" / "audit_scores.jsonl")]
    df = pd.DataFrame(records)
    df["loss_category"] = df.apply(categorize, axis=1)

    print("=" * 70)
    print("CATEGORISATION DE LA PERTE DE TRACEUR")
    print("=" * 70)

    print("\n--- Distribution globale ---")
    print(df["loss_category"].value_counts())

    print("\n--- Table croisee: categorie x condition ---")
    ct_condition = pd.crosstab(df["condition"], df["loss_category"])
    print(ct_condition)
    chi2, p, dof, expected = stats.chi2_contingency(ct_condition)
    print(f"Chi2={chi2:.3f}, p={p:.4f}, dof={dof} -> {'SIGNIFICATIF' if p < 0.05 else 'non significatif'}")

    print("\n--- Table croisee: categorie x schema_mode ---")
    ct_schema = pd.crosstab(df["schema_mode"], df["loss_category"])
    print(ct_schema)
    chi2, p, dof, expected = stats.chi2_contingency(ct_schema)
    print(f"Chi2={chi2:.3f}, p={p:.4f}, dof={dof} -> {'SIGNIFICATIF' if p < 0.05 else 'non significatif'}")

    print("\n--- Table croisee complete: condition x schema x categorie ---")
    df["cell"] = df["condition"] + "_" + df["schema_mode"]
    ct_full = pd.crosstab(df["cell"], df["loss_category"])
    print(ct_full)

    # Test cible: shared_pool+typed (racine=0) vs tout le reste
    print("\n--- Test cible: anomalie shared_pool + typed (racine jamais reproduite) ---")
    df["is_shared_pool_typed"] = (df["condition"] == "shared_pool") & (df["schema_mode"] == "typed")
    df["root_zero"] = df["root_output_score"] == 0
    table_2x2 = pd.crosstab(df["is_shared_pool_typed"], df["root_zero"])
    print(table_2x2)
    odds_ratio, p_fisher = stats.fisher_exact(table_2x2)
    print(f"Fisher exact: OR={odds_ratio:.3f}, p={p_fisher:.4f} -> {'SIGNIFICATIF' if p_fisher < 0.05 else 'non significatif'}")

if __name__ == "__main__":
    main()