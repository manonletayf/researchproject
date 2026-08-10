"""Analyse statistique de la latence normalisee : par tour, par token,
et pente de croissance intra-run. Fusionne token_latency_metrics.jsonl
avec le dataset principal pour recuperer condition/schema_mode/complexity."""
import json
import pandas as pd
from scipy import stats
from pathlib import Path

BASE = Path(__file__).parent.parent

def kw_and_mw(df, metric_col, label):
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    valid = df.dropna(subset=[metric_col])
    print(f"n valides: {len(valid)} / {len(df)}")

    print(f"\n--- Kruskal-Wallis: {label} par topologie ---")
    groups = [valid[valid["condition"] == c][metric_col].values
              for c in ["hierarchical", "decentralized", "shared_pool"]]
    h, p = stats.kruskal(*groups)
    print(f"H={h:.3f}, p={p:.4f} -> {'SIGNIFICATIF' if p < 0.05 else 'non significatif'}")
    print(valid.groupby("condition")[metric_col].agg(["mean", "median", "std"]).round(6))

    print(f"\n--- Mann-Whitney: {label} par schema (global) ---")
    free = valid[valid["schema_mode"] == "free_text"][metric_col].values
    typed = valid[valid["schema_mode"] == "typed"][metric_col].values
    u, p = stats.mannwhitneyu(free, typed)
    print(f"U={u:.1f}, p={p:.4f} -> {'SIGNIFICATIF' if p < 0.05 else 'non significatif'}")
    print(f"free_text mean={free.mean():.6f} | typed mean={typed.mean():.6f}")

    print(f"\n--- Mann-Whitney: {label} par schema, DANS CHAQUE condition ---")
    for condition in ["hierarchical", "decentralized", "shared_pool"]:
        sub = valid[valid["condition"] == condition]
        f = sub[sub["schema_mode"] == "free_text"][metric_col].values
        t = sub[sub["schema_mode"] == "typed"][metric_col].values
        if len(f) == 0 or len(t) == 0:
            continue
        u, p = stats.mannwhitneyu(f, t)
        diff = t.mean() - f.mean()
        sig = "SIGNIFICATIF" if p < 0.05 else "n.s."
        print(f"  {condition:<15} typed-free_text={diff:+.6f}  p={p:.4f}  ({sig})")

def main():
    token_records = [json.loads(l) for l in open(BASE / "full_run_results" / "token_latency_metrics.jsonl")]
    df_tokens = pd.DataFrame(token_records)

    df_main = pd.read_csv(BASE / "full_run_results" / "consolidated_dataset.csv")
    df_main = df_main[["task_id", "condition", "schema_mode", "structural_complexity"]]

    df = df_tokens.merge(df_main, on=["task_id", "condition", "schema_mode"], how="left")

    kw_and_mw(df, "latency_per_turn", "LATENCE PAR TOUR")
    kw_and_mw(df, "latency_per_token", "LATENCE PAR TOKEN")
    kw_and_mw(df, "latency_growth_slope", "PENTE DE CROISSANCE INTRA-RUN")

    output_path = BASE / "full_run_results" / "normalized_latency_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\n\nDataset fusionne sauvegarde: {output_path}")

if __name__ == "__main__":
    main()