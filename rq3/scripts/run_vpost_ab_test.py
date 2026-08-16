"""
Test A/B pour V_post: compare le judge avec thinking active vs desactive,
sur le meme echantillon de traces RQ1 Phase 3, en mesurant precision/rappel
contre la verite terrain (error_injected connu par construction).

C'est le test qui manquait a V_live (qui n'avait pas de verite terrain et
s'est appuye sur une calibration qualitative). Ici, RQ1 Phase 3 sait
exactement si/ou une erreur a ete injectee dans chaque trace, donc on peut
calculer une vraie precision/rappel plutot que de relire des exemples a la
main.

PAS ENCORE EXECUTABLE: depend des donnees RQ1 Phase 3, qui n'existent pas
encore. Le --dry-check verifie que le fichier attendu et son schema sont
corrects, sans appel API, des que le fichier existe.

Usage:
    python run_vpost_ab_test.py --dry-check      # verifie juste le format d'entree
    python run_vpost_ab_test.py --n-sample 30    # lance le vrai test A/B (necessite RQ1 Phase 3)
"""

import argparse
import json

from vpost_config import RQ1_PHASE3_RESULTS_PATH, RQ1_PHASE3_EXPECTED_FIELDS, VPOST_AB_TEST_RESULTS_JSONL
from reasoning_judge import evaluate_reasoning_trace


def load_labeled_traces():
    if not RQ1_PHASE3_RESULTS_PATH.exists():
        return None, f"Fichier introuvable: {RQ1_PHASE3_RESULTS_PATH} (RQ1 Phase 3 pas encore termine)"

    records = []
    with open(RQ1_PHASE3_RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        return None, "Fichier trouve mais vide."

    missing_fields = RQ1_PHASE3_EXPECTED_FIELDS - set(records[0].keys())
    if missing_fields:
        return None, (
            f"Champs attendus manquants dans le premier enregistrement: {missing_fields}. "
            f"Champs presents: {sorted(records[0].keys())}. "
            "Le schema RQ1_PHASE3_EXPECTED_FIELDS dans vpost_config.py doit probablement etre "
            "ajuste au format reel produit par RQ1 Phase 3 - a faire avant de lancer le test."
        )

    return records, None


def compute_metrics(predictions: list[bool], ground_truth: list[bool]) -> dict:
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    # F1: NaN si precision ou rappel est lui-meme NaN (aucun positif predit
    # ou aucun positif reel) ; 0.0 (pas NaN) si les deux valent legitimement
    # zero ; sinon la formule standard.
    if precision != precision or recall != recall:  # test NaN (NaN != NaN)
        f1 = float("nan")
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    accuracy = (tp + tn) / len(predictions) if predictions else float("nan")

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-check", action="store_true")
    parser.add_argument("--n-sample", type=int, default=30,
                         help="Nombre de traces a evaluer dans chaque condition (thinking on / off)")
    args = parser.parse_args()

    records, error = load_labeled_traces()
    if error:
        print(f"STATUT: {error}")
        print(f"\nChemin attendu: {RQ1_PHASE3_RESULTS_PATH}")
        print("Ce script sera pret a tourner des que RQ1 Phase 3 produit ce fichier.")
        return

    print(f"{len(records)} traces etiquetees trouvees.")

    if args.dry_check:
        n_with_error = sum(1 for r in records if r.get("error_injected"))
        print(f"Dont {n_with_error} avec erreur injectee, {len(records) - n_with_error} sans.")
        print("Schema OK. Pret pour --n-sample.")
        return

    sample = records[:args.n_sample]
    ground_truth = [bool(r.get("error_injected", False)) for r in sample]

    for condition_name, use_thinking in (("thinking_off", False), ("thinking_on", True)):
        print(f"\n=== Condition: {condition_name} ===")
        predictions = []
        for i, record in enumerate(sample):
            result = evaluate_reasoning_trace(
                trace_text=record["full_trace"],
                final_answer=record.get("model_answer", ""),
                use_thinking=use_thinking,
            )
            predictions.append(result["triggered"])
            event = {
                "instance_id": record.get("instance_id"),
                "condition": condition_name,
                "ground_truth_error_injected": ground_truth[i],
                **result,
            }
            with open(VPOST_AB_TEST_RESULTS_JSONL, "a") as out:
                out.write(json.dumps(event) + "\n")
            print(f"  [{i+1}/{len(sample)}] triggered={result['triggered']} (verite terrain: {ground_truth[i]})")

        metrics = compute_metrics(predictions, ground_truth)
        print(f"  Precision={metrics['precision']:.3f}  Rappel={metrics['recall']:.3f}  "
              f"F1={metrics['f1']:.3f}  Exactitude={metrics['accuracy']:.3f}")

    print(f"\nResultats detailles: {VPOST_AB_TEST_RESULTS_JSONL}")
    print("Comparer precision/rappel/F1 entre thinking_on et thinking_off pour trancher.")


if __name__ == "__main__":
    main()