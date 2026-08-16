"""
Breaks down the V_live contradiction trigger rate by topology (condition)
and schema_mode, mirroring RQ2's factorial design (topology x schema_mode).

Reads the final deduplicated file (vlive_trigger_events_final.jsonl) and
reports, per (condition, schema_mode) cell:
  - number of runs
  - number of messages evaluated
  - number of contradiction triggers
  - trigger rate per message (normalized - the number that's actually
    comparable across cells, since runs/messages per cell differ)
  - average triggers per run

Usage:
    python analyze_vlive_by_condition.py
"""

import json
from collections import defaultdict

from rq3_config import RQ3_OUTPUTS_DIR

FINAL_JSONL = RQ3_OUTPUTS_DIR / "vlive_trigger_events_final.jsonl"


def parse_run_id(run_id: str):
    """run_id = {condition}__{task_id}__{schema_mode}"""
    parts = run_id.split("__")
    condition = parts[0]
    schema_mode = parts[-1]
    return condition, schema_mode


def main():
    # cell key = (condition, schema_mode)
    messages_per_run = defaultdict(lambda: defaultdict(int))  # cell -> run_id -> message count
    triggers_per_cell = defaultdict(int)
    runs_seen = defaultdict(set)  # cell -> set of run_ids

    with open(FINAL_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["trigger_type"] != "contradiction":
                continue
            condition, schema_mode = parse_run_id(entry["run_id"])
            cell = (condition, schema_mode)
            runs_seen[cell].add(entry["run_id"])
            messages_per_run[cell][entry["run_id"]] += 1
            if entry["triggered"]:
                triggers_per_cell[cell] += 1

    print(f"{'condition':<15} {'schema_mode':<12} {'runs':>5} {'messages':>9} {'triggers':>9} {'rate/msg':>9} {'avg/run':>8}")
    print("-" * 75)

    all_cells = sorted(runs_seen.keys())
    for cell in all_cells:
        condition, schema_mode = cell
        n_runs = len(runs_seen[cell])
        n_messages = sum(messages_per_run[cell].values())
        n_triggers = triggers_per_cell[cell]
        rate = n_triggers / n_messages if n_messages else 0.0
        avg_per_run = n_triggers / n_runs if n_runs else 0.0
        print(f"{condition:<15} {schema_mode:<12} {n_runs:>5} {n_messages:>9} {n_triggers:>9} {rate:>9.3%} {avg_per_run:>8.2f}")

    # Marginals by condition (collapsing schema_mode)
    print("\n--- By topology (collapsed across schema_mode) ---")
    by_condition_runs = defaultdict(set)
    by_condition_messages = defaultdict(int)
    by_condition_triggers = defaultdict(int)
    for (condition, schema_mode), runs in runs_seen.items():
        by_condition_runs[condition] |= runs
        by_condition_messages[condition] += sum(messages_per_run[(condition, schema_mode)].values())
        by_condition_triggers[condition] += triggers_per_cell[(condition, schema_mode)]
    for condition in sorted(by_condition_runs.keys()):
        n_runs = len(by_condition_runs[condition])
        n_messages = by_condition_messages[condition]
        n_triggers = by_condition_triggers[condition]
        rate = n_triggers / n_messages if n_messages else 0.0
        print(f"{condition:<15} runs={n_runs:>4}  messages={n_messages:>5}  triggers={n_triggers:>4}  rate/msg={rate:.3%}")

    # Marginals by schema_mode (collapsing condition)
    print("\n--- By schema_mode (collapsed across topology) ---")
    by_schema_runs = defaultdict(set)
    by_schema_messages = defaultdict(int)
    by_schema_triggers = defaultdict(int)
    for (condition, schema_mode), runs in runs_seen.items():
        by_schema_runs[schema_mode] |= runs
        by_schema_messages[schema_mode] += sum(messages_per_run[(condition, schema_mode)].values())
        by_schema_triggers[schema_mode] += triggers_per_cell[(condition, schema_mode)]
    for schema_mode in sorted(by_schema_runs.keys()):
        n_runs = len(by_schema_runs[schema_mode])
        n_messages = by_schema_messages[schema_mode]
        n_triggers = by_schema_triggers[schema_mode]
        rate = n_triggers / n_messages if n_messages else 0.0
        print(f"{schema_mode:<15} runs={n_runs:>4}  messages={n_messages:>5}  triggers={n_triggers:>4}  rate/msg={rate:.3%}")


if __name__ == "__main__":
    main()