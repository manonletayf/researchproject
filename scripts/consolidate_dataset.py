"""Consolide les 3 sources de resultats RQ2 (RTD+latence, audit, couverture
substantielle) en un seul tableau long format, une ligne par run, pret pour
l'analyse statistique du Jour 6."""
import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent.parent

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def main():
    # Source 1: RTD + latence (3 fichiers, un par condition)
    main_records = []
    for condition in ["hierarchical", "decentralized", "shared_pool"]:
        main_records.extend(load_jsonl(BASE / "full_run_results" / f"{condition}.jsonl"))

    df_main = pd.DataFrame(main_records)
    df_main = df_main[df_main["error"].isna()]  # exclut les runs en erreur (aucun normalement)

    # Source 2: couverture substantielle du raisonnement
    reasoning_records = load_jsonl(BASE / "full_run_results" / "reasoning_coverage_scores.jsonl")
    df_reasoning = pd.DataFrame(reasoning_records)[["task_id", "condition", "schema_mode", "substantive_reasoning_score"]]

    # Source 3: audit root/final (garde comme diagnostic, pas comme DV principale)
    audit_records = load_jsonl(BASE / "full_run_results" / "audit_scores.jsonl")
    df_audit = pd.DataFrame(audit_records)[["task_id", "condition", "schema_mode", "root_output_score", "final_output_score"]]

    # Fusion sur (task_id, condition, schema_mode) - cle commune aux 3 sources
    df = df_main.merge(df_reasoning, on=["task_id", "condition", "schema_mode"], how="left")
    df = df.merge(df_audit, on=["task_id", "condition", "schema_mode"], how="left")

    # Colonnes finales, dans un ordre lisible
    cols = [
        "task_id", "condition", "schema_mode", "domain", "structural_complexity",
        "rtd_score", "total_llm_latency_seconds", "wall_clock_seconds",
        "substantive_reasoning_score", "root_output_score", "final_output_score",
        "num_turns",
    ]
    df = df[cols]

    output_path = BASE / "full_run_results" / "consolidated_dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"Consolide: {len(df)} lignes, {df.shape[1]} colonnes")
    print(f"Ecrit dans: {output_path}")
    print()
    print("Verification des valeurs manquantes par colonne:")
    print(df.isna().sum())

if __name__ == "__main__":
    main()