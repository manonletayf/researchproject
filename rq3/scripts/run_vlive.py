"""
Etape 6 du plan RQ3 - pipeline de rejeu complet V_live.

Rejoue les 204 runs RQ2 (hierarchical + decentralized + shared_pool),
message par message, dans l'ordre chronologique de trace["turns"].
Applique a chaque tour:
  - le trigger schema (etage 1, programmatique, schema_validator.py)
  - le trigger contradiction (etage 2, judge Sonnet 5, contradiction_judge.py)
  - le trigger softcot_divergence (etage 2, INACTIF sur RQ2 - voir
    contradiction_judge.check_softcot_divergence)

Ecriture progressive des trigger_event en JSONL + manifest resumable par
run_id, sur le meme principe que run_full_rq2.py (Jour 4).

A lancer depuis rq2-experiment/rq3/scripts/ (ou avec ce dossier dans le
PYTHONPATH) - les sorties atterrissent dans rq2-experiment/rq3/outputs/.

Usage:
    python run_vlive.py                # lance / reprend
    python run_vlive.py --dry-estimate # compte messages + estime le cout, sans appeler l'API
"""

import argparse
import json
from collections import deque
from datetime import datetime, timezone

from rq3_config import (
    RQ2_RESULTS_FILES,
    RQ2_EXPERIMENT_ROOT,
    RQ3_WINDOW_SIZE,
    RQ3_TRIGGER_EVENTS_JSONL,
    RQ3_MANIFEST_JSONL,
    RQ3_FAILURES_LOG,
)
from schema_validator import validate_typed_turn
from contradiction_judge import check_contradiction, check_softcot_divergence

# Tarifs Sonnet 5 (aout 2026), pour --dry-estimate uniquement.
SONNET5_INPUT_PER_MTOK = 2.00
SONNET5_OUTPUT_PER_MTOK = 10.00
EST_INPUT_TOKENS_PER_CALL = 800   # fenetre k=5 + message courant, ordre de grandeur
EST_OUTPUT_TOKENS_PER_CALL = 150  # bool + rationale courte


# ---------------------------------------------------------------------------
# Manifest / reprise (resumable par run_id, meme pattern que Jour 4)
# ---------------------------------------------------------------------------

def load_completed_run_ids() -> set[str]:
    done = set()
    if not RQ3_MANIFEST_JSONL.exists():
        return done
    with open(RQ3_MANIFEST_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("status") == "done":
                done.add(entry["run_id"])
    return done


def append_manifest(run_id: str, status: str):
    with open(RQ3_MANIFEST_JSONL, "a") as f:
        f.write(json.dumps({"run_id": run_id, "status": status,
                             "ts": datetime.now(timezone.utc).isoformat()}) + "\n")
        f.flush()


def append_trigger_event(event: dict):
    with open(RQ3_TRIGGER_EVENTS_JSONL, "a") as f:
        f.write(json.dumps(event) + "\n")
        f.flush()


def append_failure(run_id: str, error: str):
    with open(RQ3_FAILURES_LOG, "a") as f:
        f.write(json.dumps({"run_id": run_id, "error": error,
                             "ts": datetime.now(timezone.utc).isoformat()}) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Iteration sur les runs RQ2
# ---------------------------------------------------------------------------

def iter_rq2_runs():
    """Genere un dict par run, a partir des 3 fichiers de resultats."""
    for condition, results_path in RQ2_RESULTS_FILES.items():
        if not results_path.exists():
            print(f"ATTENTION: {results_path} introuvable, condition {condition} ignoree")
            continue
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("error"):
                    continue  # run rate en RQ2, pas de trace exploitable
                row["condition"] = condition
                yield row


def load_trace(row: dict) -> dict | None:
    # trace_file est deja relatif a la racine rq2-experiment/ (il inclut
    # lui-meme le prefixe "full_run_results/"), pas a full_run_results/.
    trace_path = RQ2_EXPERIMENT_ROOT / row["trace_file"]
    if not trace_path.exists():
        return None
    with open(trace_path) as f:
        return json.load(f)


def make_run_id(row: dict) -> str:
    return f"{row['condition']}__{row['task_id']}__{row['schema_mode']}"


# ---------------------------------------------------------------------------
# Traitement d'un run
# ---------------------------------------------------------------------------

def process_run(row: dict, run_id: str):
    trace = load_trace(row)
    if trace is None:
        append_failure(run_id, f"trace_file introuvable: {row.get('trace_file')}")
        return

    turns = trace.get("trace", {}).get("turns", [])
    schema_mode = row["schema_mode"]
    window = deque(maxlen=RQ3_WINDOW_SIZE)

    for idx, turn in enumerate(turns):
        message_id = f"{run_id}__turn{idx}"
        content = turn.get("content", "")

        # Etage 1 - schema (programmatique)
        schema_result = validate_typed_turn(content, schema_mode)
        append_trigger_event({
            "run_id": run_id,
            "message_id": message_id,
            "trigger_type": "schema",
            "triggered": schema_result["triggered"],
            "judge_rationale": schema_result["judge_rationale"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Etage 2 - contradiction (judge Sonnet 5)
        contradiction_result = check_contradiction(turn, list(window))
        append_trigger_event({
            "run_id": run_id,
            "message_id": message_id,
            "trigger_type": "contradiction",
            "triggered": contradiction_result["triggered"],
            "judge_rationale": contradiction_result["judge_rationale"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Etage 2 - softcot_divergence (inactif sur RQ2, voir docstring)
        softcot_result = check_softcot_divergence(turn, planner_strategy=None)
        append_trigger_event({
            "run_id": run_id,
            "message_id": message_id,
            "trigger_type": "softcot_divergence",
            "triggered": softcot_result["triggered"],
            "judge_rationale": softcot_result["judge_rationale"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        window.append(turn)


# ---------------------------------------------------------------------------
# --dry-estimate : compte messages reels, estime le cout, aucun appel API
# ---------------------------------------------------------------------------

def dry_estimate():
    total_runs = 0
    total_messages = 0
    missing_traces = 0

    for row in iter_rq2_runs():
        total_runs += 1
        trace = load_trace(row)
        if trace is None:
            missing_traces += 1
            continue
        total_messages += len(trace.get("trace", {}).get("turns", []))

    n_judge_calls = total_messages  # un appel contradiction par message
    input_cost = n_judge_calls * EST_INPUT_TOKENS_PER_CALL / 1_000_000 * SONNET5_INPUT_PER_MTOK
    output_cost = n_judge_calls * EST_OUTPUT_TOKENS_PER_CALL / 1_000_000 * SONNET5_OUTPUT_PER_MTOK

    print(f"Runs trouves            : {total_runs}")
    print(f"Traces manquantes       : {missing_traces}")
    print(f"Messages totaux (turns) : {total_messages}")
    print(f"Moyenne messages/run    : {total_messages / total_runs:.1f}" if total_runs else "n/a")
    print(f"Appels judge estimes    : {n_judge_calls}")
    print(f"Cout estime (input)     : ${input_cost:.2f}")
    print(f"Cout estime (output)    : ${output_cost:.2f}")
    print(f"Cout estime (total)     : ${input_cost + output_cost:.2f}")
    print("(estimation basee sur 800 tok in / 150 tok out par appel - ordre de grandeur, pas exact)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-estimate", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                         help="Ne traite que les N premiers runs (calibration). Appelle l'API pour de vrai.")
    args = parser.parse_args()

    if args.dry_estimate:
        dry_estimate()
        return

    done = load_completed_run_ids()
    print(f"{len(done)} runs deja traites (reprise sur manifest)")

    total = 0
    processed = 0
    for row in iter_rq2_runs():
        if args.limit is not None and processed >= args.limit:
            print(f"--limit={args.limit} atteint, arret.")
            break
        run_id = make_run_id(row)
        total += 1
        if run_id in done:
            continue
        processed += 1
        try:
            process_run(row, run_id)
            append_manifest(run_id, "done")
            print(f"[{processed}] {run_id} OK")
        except Exception as e:
            append_failure(run_id, f"{type(e).__name__}: {e}")
            append_manifest(run_id, "failed")
            print(f"[{processed}] ECHEC {run_id}: {e}")

    print(f"Termine. {total} runs vus, {processed} traites cette session.")
    print(f"Trigger events: {RQ3_TRIGGER_EVENTS_JSONL}")


if __name__ == "__main__":
    main()