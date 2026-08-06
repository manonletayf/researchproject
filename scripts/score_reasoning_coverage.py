"""Calcule la couverture substantielle du raisonnement, avec un critere
comparable entre modes typed et free_text : nombre total de mots produits
par tour (somme des 3 champs en typed, longueur du bloc entier en free_text),
compare a un seuil fixe a la mediane propre a chaque mode."""
import json
from pathlib import Path

BASE = Path(__file__).parent.parent

TYPED_MIN_WORDS = 241
FREE_TEXT_MIN_WORDS = 318

def is_substantial_typed(content_str):
    try:
        parsed = json.loads(content_str)
    except (json.JSONDecodeError, TypeError):
        return False
    total_words = (
        len(str(parsed.get("reasoning", "")).split())
        + len(str(parsed.get("findings", "")).split())
        + len(str(parsed.get("decision", "")).split())
    )
    return total_words >= TYPED_MIN_WORDS

def is_substantial_free_text(content_str):
    return len(content_str.split()) >= FREE_TEXT_MIN_WORDS

def score_trace(trace_data, schema_mode):
    turns = trace_data["trace"]["turns"]
    if not turns:
        return None
    check_fn = is_substantial_typed if schema_mode == "typed" else is_substantial_free_text
    substantial_count = sum(1 for t in turns if check_fn(t["content"]))
    return substantial_count / len(turns)

def main():
    all_records = []

    for condition in ["hierarchical", "decentralized", "shared_pool"]:
        jsonl_path = BASE / "full_run_results" / f"{condition}.jsonl"
        with open(jsonl_path) as f:
            run_records = [json.loads(l) for l in f]

        for run in run_records:
            if run["error"] is not None:
                continue

            trace_path = BASE / run["trace_file"]
            with open(trace_path) as f:
                trace_data = json.load(f)

            score = score_trace(trace_data, run["schema_mode"])

            all_records.append({
                "task_id": run["task_id"],
                "condition": condition,
                "schema_mode": run["schema_mode"],
                "rtd_score": run["rtd_score"],
                "substantive_reasoning_score": score,
            })

    output_path = BASE / "full_run_results" / "reasoning_coverage_scores.jsonl"
    with open(output_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    print(f"Termine: {len(all_records)} runs scores, ecrit dans {output_path}")

if __name__ == "__main__":
    main()