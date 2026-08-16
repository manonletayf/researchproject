"""
Cross-analysis: does the V_live contradiction rate (process-level verifier)
correlate with RQ2's outcome-level scores (constraint_anchor_addressed,
correct_outcome_reflected)?

Joins on run_id. Since audit_scores.jsonl's exact schema wasn't available
at write time, this script prints a diagnostic of the first record and the
match rate before computing statistics, so mismatches are visible rather
than silently producing wrong numbers.

Usage:
    python analyze_vlive_vs_audit.py
"""

import json
from collections import defaultdict
from pathlib import Path

from rq3_config import RQ3_OUTPUTS_DIR, RQ2_EXPERIMENT_ROOT

FINAL_JSONL = RQ3_OUTPUTS_DIR / "vlive_trigger_events_final.jsonl"
AUDIT_SCORES_PATH = RQ2_EXPERIMENT_ROOT / "full_run_results" / "audit_scores.jsonl"


def parse_run_id(run_id: str):
    parts = run_id.split("__")
    condition = parts[0]
    schema_mode = parts[-1]
    return condition, schema_mode


def compute_vlive_rates():
    """Per run_id: n_messages, n_triggers, rate."""
    msgs = defaultdict(int)
    triggers = defaultdict(int)
    with open(FINAL_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e["trigger_type"] != "contradiction":
                continue
            msgs[e["run_id"]] += 1
            if e["triggered"]:
                triggers[e["run_id"]] += 1
    rates = {rid: triggers[rid] / msgs[rid] for rid in msgs}
    return rates, msgs, triggers


def load_audit_scores():
    if not AUDIT_SCORES_PATH.exists():
        print(f"NOT FOUND: {AUDIT_SCORES_PATH}")
        return []
    records = []
    with open(AUDIT_SCORES_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    rates, msgs, triggers = compute_vlive_rates()
    print(f"V_live: {len(rates)} runs with computed contradiction rate")

    audit_records = load_audit_scores()
    print(f"audit_scores.jsonl: {len(audit_records)} records")

    if not audit_records:
        return

    print("\n--- First audit_scores record (diagnostic) ---")
    print(json.dumps(audit_records[0], indent=2))
    print("--- End diagnostic ---\n")

    # try to find a run_id-like key, or reconstruct it from
    # condition + task_id + schema_mode (actual audit_scores.jsonl schema)
    first = audit_records[0]
    id_key = None
    for candidate in ("run_id", "id"):
        if candidate in first:
            id_key = candidate
            break

    reconstruct = False
    if id_key is None and all(k in first for k in ("condition", "task_id", "schema_mode")):
        reconstruct = True
        print("No run_id field found - reconstructing as {condition}__{task_id}__{schema_mode} to match V_live format.")

    if id_key is None and not reconstruct:
        print("Could not find an obvious run_id key in audit_scores.jsonl.")
        print(f"Available keys: {list(first.keys())}")
        print("Inspect the diagnostic above and tell me which field(s) identify a run"
              " (or how to reconstruct condition/task_id/schema_mode) so I can fix the join.")
        return

    matched = 0
    unmatched = 0
    joined = []
    for rec in audit_records:
        rid = f"{rec['condition']}__{rec['task_id']}__{rec['schema_mode']}" if reconstruct else rec.get(id_key)
        if rid in rates:
            joined.append({
                "run_id": rid,
                "contradiction_rate": rates[rid],
                "n_messages": msgs[rid],
                "n_triggers": triggers[rid],
                **rec,
            })
            matched += 1
        else:
            unmatched += 1

    print(f"Matched: {matched} / {len(audit_records)}  (unmatched: {unmatched})")
    if matched == 0:
        print("No matches - run_id format likely differs between the two files.")
        print(f"Example V_live run_id: {list(rates.keys())[0]}")
        example_rid = f"{first['condition']}__{first['task_id']}__{first['schema_mode']}" if reconstruct else first.get(id_key)
        print(f"Example audit_scores reconstructed id: {example_rid}")
        return

    # group comparison: mean contradiction rate by outcome score
    # (actual fields in audit_scores.jsonl: root_output_score, final_output_score,
    # rtd_score - not constraint_anchor_addressed/correct_outcome_reflected as
    # initially referenced; using what's actually present)
    for outcome_field in ("final_output_score", "root_output_score", "rtd_score",
                           "correct_outcome_reflected", "constraint_anchor_addressed"):
        if outcome_field not in first:
            continue
        groups = defaultdict(list)
        for row in joined:
            groups[row.get(outcome_field)].append(row["contradiction_rate"])
        print(f"\n--- Mean contradiction rate by {outcome_field} ---")
        for val, rate_list in sorted(groups.items(), key=lambda x: str(x[0])):
            mean_rate = sum(rate_list) / len(rate_list)
            print(f"  {outcome_field}={val}: n={len(rate_list):>4}  mean contradiction rate={mean_rate:.3%}")


if __name__ == "__main__":
    main()