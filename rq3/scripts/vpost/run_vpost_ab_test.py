"""
Test A/B pour V_post: compare le judge avec thinking active vs desactive,
sur un echantillon equilibre de traces RQ1 Phase 3, en mesurant precision/
rappel contre la verite terrain.

Point important sur la source des donnees (confirme aupres de RQ1, aout
2026): les fichiers de perturbation (softcot_perturbation_results.jsonl,
s1_perturbation_results.jsonl) ne contiennent QUE des instances corrompues
- il n'y a pas de champ booleen "error_injected" explicite dedans, chaque
ligne EST un positif par construction. Pour calculer une precision/rappel
qui ait un sens (rien a "rater" sans negatifs, precision triviale sans
negatifs a rejeter), ce script construit lui-meme l'echantillon equilibre:
  - POSITIFS: les lignes des fichiers de perturbation (error_injected=True),
    trace complete = full_perturbed_trace
  - NEGATIFS: les instances ORIGINALES non corrompues correspondantes,
    retrouvees via original_instance_id dans les runs principaux
    (softcot_gsm8k_seed41.jsonl / s1_math_batch*.jsonl), trace complete =
    full_trace

PAS ENCORE EXECUTABLE: depend des donnees RQ1 Phase 3, rapatriees depuis
EC2. Le --dry-check verifie que les fichiers attendus et leur schema sont
corrects, sans appel API, des qu'ils existent.

Usage:
    python run_vpost_ab_test.py --dry-check      # verifie le format d'entree, sans API
    python run_vpost_ab_test.py --n-sample 30    # lance le vrai test A/B, N positifs + N negatifs PAR METHODE
"""

import argparse
import glob
import json

from vpost_config import (
    SOFTCOT_PERTURBATION_PATH,
    S1_PERTURBATION_PATH,
    SOFTCOT_MAIN_RUN_PATH,
    S1_MAIN_RUN_GLOB,
    PERTURBATION_REQUIRED_FIELDS,
    MAIN_RUN_REQUIRED_FIELDS,
    VPOST_AB_TEST_RESULTS_JSONL,
)
from reasoning_judge import evaluate_reasoning_trace

_RQ1_RESULTS_DIR = SOFTCOT_PERTURBATION_PATH.parent  # rq1-results/, pour le glob S1


def _read_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _check_files_exist():
    missing = []
    for path in (SOFTCOT_PERTURBATION_PATH, S1_PERTURBATION_PATH, SOFTCOT_MAIN_RUN_PATH):
        if not path.exists():
            missing.append(str(path))
    s1_main_files = glob.glob(str(_RQ1_RESULTS_DIR / S1_MAIN_RUN_GLOB))
    if not s1_main_files:
        missing.append(f"{_RQ1_RESULTS_DIR / S1_MAIN_RUN_GLOB} (aucun fichier ne correspond au glob)")
    return missing, s1_main_files


def load_labeled_traces():
    """
    Construit l'echantillon equilibre positifs/negatifs pour les deux
    methodes (SoftCoT, S1). Retourne (records, error_message).

    Chaque record unifie: {instance_id, method, backbone, full_trace,
    final_answer, error_injected: bool}
    """
    missing, s1_main_files = _check_files_exist()
    if missing:
        return None, "Fichiers manquants:\n  " + "\n  ".join(missing) + "\n(RQ1 Phase 3 pas encore rapatrie depuis EC2)"

    # --- Positifs (perturbation) ---
    softcot_positives = _read_jsonl(SOFTCOT_PERTURBATION_PATH)
    s1_positives = _read_jsonl(S1_PERTURBATION_PATH)

    for label, records in (("SoftCoT", softcot_positives), ("S1", s1_positives)):
        if not records:
            continue
        missing_fields = PERTURBATION_REQUIRED_FIELDS - set(records[0].keys())
        if missing_fields:
            return None, (
                f"[{label}] Champs attendus manquants: {missing_fields}. "
                f"Champs presents: {sorted(records[0].keys())}. "
                "PERTURBATION_REQUIRED_FIELDS dans vpost_config.py doit probablement etre "
                "ajuste au format reel - a faire avant de lancer le test."
            )

    if not softcot_positives and not s1_positives:
        return None, "Fichiers de perturbation trouves mais vides (0 instance des deux cotes)."

    # --- Negatifs (runs principaux, apparies par original_instance_id) ---
    softcot_main = {r["instance_id"]: r for r in _read_jsonl(SOFTCOT_MAIN_RUN_PATH)}
    s1_main = {}
    for path_str in s1_main_files:
        for r in _read_jsonl(path_str):
            s1_main[r["instance_id"]] = r

    for label, main_dict in (("SoftCoT (run principal)", softcot_main), ("S1 (run principal)", s1_main)):
        if not main_dict:
            continue
        sample_record = next(iter(main_dict.values()))
        missing_fields = MAIN_RUN_REQUIRED_FIELDS - set(sample_record.keys())
        if missing_fields:
            return None, (
                f"[{label}] Champs attendus manquants: {missing_fields}. "
                f"Champs presents: {sorted(sample_record.keys())}. "
                "MAIN_RUN_REQUIRED_FIELDS dans vpost_config.py doit probablement etre ajuste."
            )

    unified = []
    unmatched_negatives = 0

    for method, positives, main_dict in (
        ("softcot", softcot_positives, softcot_main),
        ("s1", s1_positives, s1_main),
    ):
        for pos in positives:
            unified.append({
                "instance_id": pos["instance_id"],
                "method": method,
                "backbone": pos["backbone"],
                "full_trace": pos["full_perturbed_trace"],
                "final_answer": pos.get("perturbed_answer", ""),
                "error_injected": True,
            })
            # negatif apparie: l'instance originale non corrompue
            orig_id = pos.get("original_instance_id")
            orig = main_dict.get(orig_id)
            if orig is None:
                unmatched_negatives += 1
                continue
            unified.append({
                "instance_id": orig["instance_id"],
                "method": method,
                "backbone": orig["backbone"],
                "full_trace": orig["full_trace"],
                "final_answer": orig.get("model_answer", ""),
                "error_injected": False,
            })

    if unmatched_negatives:
        print(f"ATTENTION: {unmatched_negatives} positif(s) sans negatif apparie "
              f"(original_instance_id introuvable dans le run principal) - ignores pour la paire, "
              f"mais le positif reste dans l'echantillon.")

    return unified, None


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
    parser.add_argument("--n-sample", type=int, default=20,
                         help="Nombre de PAIRES positif/negatif par methode a evaluer dans chaque "
                              "condition thinking on/off (donc jusqu'a 4*n_sample appels par condition)")
    args = parser.parse_args()

    records, error = load_labeled_traces()
    if error:
        print(f"STATUT: {error}")
        print("\nCe script sera pret a tourner des que RQ1 Phase 3 est rapatrie depuis EC2.")
        return

    n_pos = sum(1 for r in records if r["error_injected"])
    n_neg = sum(1 for r in records if not r["error_injected"])
    print(f"{len(records)} instances construites: {n_pos} positifs (erreur injectee), {n_neg} negatifs (propres).")
    for method in ("softcot", "s1"):
        n_method = sum(1 for r in records if r["method"] == method)
        print(f"  {method}: {n_method} instances")

    if args.dry_check:
        print("Schema OK. Pret pour --n-sample.")
        return

    # Echantillon equilibre: jusqu'a n_sample positifs + n_sample negatifs
    # PAR METHODE, pour ne pas laisser une methode dominer le test.
    sample = []
    for method in ("softcot", "s1"):
        pos = [r for r in records if r["method"] == method and r["error_injected"]][:args.n_sample]
        neg = [r for r in records if r["method"] == method and not r["error_injected"]][:args.n_sample]
        sample.extend(pos)
        sample.extend(neg)

    ground_truth = [r["error_injected"] for r in sample]
    print(f"\nEchantillon final: {len(sample)} instances ({sum(ground_truth)} positifs, {len(sample) - sum(ground_truth)} negatifs).")

    for condition_name, use_thinking in (("thinking_off", False), ("thinking_on", True)):
        print(f"\n=== Condition: {condition_name} ===")
        predictions = []
        for i, record in enumerate(sample):
            result = evaluate_reasoning_trace(
                trace_text=record["full_trace"],
                final_answer=record["final_answer"],
                use_thinking=use_thinking,
            )
            predictions.append(result["triggered"])
            event = {
                "instance_id": record["instance_id"],
                "method": record["method"],
                "backbone": record["backbone"],
                "condition": condition_name,
                "ground_truth_error_injected": ground_truth[i],
                **result,
            }
            with open(VPOST_AB_TEST_RESULTS_JSONL, "a") as out:
                out.write(json.dumps(event) + "\n")
            print(f"  [{i+1}/{len(sample)}] {record['method']}/{record['instance_id']}: "
                  f"triggered={result['triggered']} (verite terrain: {ground_truth[i]})")

        metrics = compute_metrics(predictions, ground_truth)
        print(f"  Precision={metrics['precision']:.3f}  Rappel={metrics['recall']:.3f}  "
              f"F1={metrics['f1']:.3f}  Exactitude={metrics['accuracy']:.3f}")

    print(f"\nResultats detailles: {VPOST_AB_TEST_RESULTS_JSONL}")
    print("Comparer precision/rappel/F1 entre thinking_on et thinking_off pour trancher.")


if __name__ == "__main__":
    main()