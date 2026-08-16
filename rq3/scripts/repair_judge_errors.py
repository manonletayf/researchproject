"""
Repair pass for V_live: re-runs ONLY the contradiction checks that failed
with a JUDGE_ERROR in the first pass, and appends corrected trigger_events
to a separate file. Does not touch schema/softcot_divergence entries (those
never call the API and had 0 errors), and does not modify the original
vlive_trigger_events.jsonl - safer to keep the original untouched and merge
at analysis time.

Usage:
    python repair_judge_errors.py                 # repair all JUDGE_ERROR entries
    python repair_judge_errors.py --dry-count      # just count how many need repair, no API calls
"""

import argparse
import json
from collections import deque, defaultdict
from datetime import datetime, timezone

from rq3_config import RQ2_RESULTS_FILES, RQ2_EXPERIMENT_ROOT, RQ3_WINDOW_SIZE, RQ3_TRIGGER_EVENTS_JSONL, RQ3_OUTPUTS_DIR
from contradiction_judge import check_contradiction

REPAIRED_JSONL = RQ3_OUTPUTS_DIR / "vlive_trigger_events_repaired.jsonl"


def find_error_message_ids() -> set[str]:
    """Scan the original trigger_events file for contradiction entries whose
    judge_rationale indicates a JUDGE_ERROR. Returns the set of message_ids
    (run_id__turnN) needing repair, EXCLUDING any already successfully
    repaired in a prior pass (tracked via REPAIRED_JSONL)."""
    errored = set()
    with open(RQ3_TRIGGER_EVENTS_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["trigger_type"] == "contradiction" and "JUDGE_ERROR" in entry.get("judge_rationale", ""):
                errored.add(entry["message_id"])

    if REPAIRED_JSONL.exists():
        already_fixed = set()
        with open(REPAIRED_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if "JUDGE_ERROR" not in entry.get("judge_rationale", ""):
                    already_fixed.add(entry["message_id"])
        errored -= already_fixed

    return errored


def iter_rq2_runs():
    for condition, results_path in RQ2_RESULTS_FILES.items():
        if not results_path.exists():
            continue
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("error"):
                    continue
                row["condition"] = condition
                yield row


def make_run_id(row: dict) -> str:
    return f"{row['condition']}__{row['task_id']}__{row['schema_mode']}"


def load_trace(row: dict):
    trace_path = RQ2_EXPERIMENT_ROOT / row["trace_file"]
    if not trace_path.exists():
        return None
    with open(trace_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-count", action="store_true")
    args = parser.parse_args()

    error_message_ids = find_error_message_ids()
    print(f"{len(error_message_ids)} contradiction checks need repair")

    if args.dry_count:
        return

    if not error_message_ids:
        print("Nothing to repair.")
        return

    # group error message_ids by run_id, extract the turn index needed
    by_run = defaultdict(set)
    for mid in error_message_ids:
        # message_id format: {run_id}__turn{N}
        run_id, turn_part = mid.rsplit("__turn", 1)
        by_run[run_id].add(int(turn_part))

    repaired_count = 0
    still_failing = 0

    with open(REPAIRED_JSONL, "a") as out:
        for row in iter_rq2_runs():
            run_id = make_run_id(row)
            if run_id not in by_run:
                continue

            trace = load_trace(row)
            if trace is None:
                continue
            turns = trace.get("trace", {}).get("turns", [])
            needed_indices = by_run[run_id]

            window = deque(maxlen=RQ3_WINDOW_SIZE)
            for idx, turn in enumerate(turns):
                if idx in needed_indices:
                    result = check_contradiction(turn, list(window))
                    event = {
                        "run_id": run_id,
                        "message_id": f"{run_id}__turn{idx}",
                        "trigger_type": "contradiction",
                        "triggered": result["triggered"],
                        "judge_rationale": result["judge_rationale"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "repaired": True,
                    }
                    out.write(json.dumps(event) + "\n")
                    out.flush()
                    if "JUDGE_ERROR" in result["judge_rationale"]:
                        still_failing += 1
                    else:
                        repaired_count += 1
                    print(f"  {run_id}__turn{idx}: {'still failing' if 'JUDGE_ERROR' in result['judge_rationale'] else 'repaired'}")
                window.append(turn)

    print(f"\nDone. {repaired_count} repaired, {still_failing} still failing after retry.")
    print(f"Output: {REPAIRED_JSONL}")
    if still_failing:
        print("Re-run this script again to retry the remaining failures.")


if __name__ == "__main__":
    main()