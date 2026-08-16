"""
Checks whether outcome scores (final_output_score, root_output_score,
rtd_score) themselves vary by condition (topology) and schema_mode.

This is a confound check: if success rate already differs by topology or
schema_mode, the raw contradiction-rate-vs-outcome correlation computed
earlier conflates topology/schema_mode effects with any real link between
contradiction and outcome. If confirmed, the fix is to stratify (compare
contradiction rate for success vs failure WITHIN each topology x
schema_mode cell) rather than pooling across cells.

Usage:
    python check_outcome_confound.py
"""

import json
from collections import defaultdict

from rq3_config import RQ2_EXPERIMENT_ROOT, RQ3_OUTPUTS_DIR

AUDIT_SCORES_PATH = RQ2_EXPERIMENT_ROOT / "full_run_results" / "audit_scores.jsonl"
FINAL_JSONL = RQ3_OUTPUTS_DIR / "vlive_trigger_events_final.jsonl"


def load_audit_scores():
    records = []
    with open(AUDIT_SCORES_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_vlive_rates():
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
    return {rid: triggers[rid] / msgs[rid] for rid in msgs}, msgs, triggers


def main():
    records = load_audit_scores()
    rates, msgs, triggers = compute_vlive_rates()

    # Part 1: does mean final_output_score vary by (condition, schema_mode)?
    cell_scores = defaultdict(list)
    for r in records:
        cell = (r["condition"], r["schema_mode"])
        cell_scores[cell].append(r["final_output_score"])

    print("=== Part 1: mean final_output_score by (topology, schema_mode) ===")
    print(f"{'condition':<15} {'schema_mode':<12} {'n':>4} {'mean_score':>11}")
    print("-" * 45)
    for cell in sorted(cell_scores.keys()):
        scores = cell_scores[cell]
        mean = sum(scores) / len(scores)
        print(f"{cell[0]:<15} {cell[1]:<12} {len(scores):>4} {mean:>11.3f}")

    # Part 2: WITHIN each cell, compare contradiction rate for
    # success (score==1.0) vs failure (score==0.0), ignoring partial scores
    print("\n=== Part 2: contradiction rate, success vs failure, WITHIN each cell ===")
    print(f"{'condition':<15} {'schema_mode':<12} {'fail_n':>7} {'fail_rate':>10} {'success_n':>10} {'success_rate':>13}")
    print("-" * 75)

    cell_groups = defaultdict(lambda: {"fail": [], "success": []})
    for r in records:
        cell = (r["condition"], r["schema_mode"])
        run_id = f"{r['condition']}__{r['task_id']}__{r['schema_mode']}"
        if run_id not in rates:
            continue
        rate = rates[run_id]
        if r["final_output_score"] == 0.0:
            cell_groups[cell]["fail"].append(rate)
        elif r["final_output_score"] == 1.0:
            cell_groups[cell]["success"].append(rate)
        # partial scores (0.5, 0.66...) excluded from this binary comparison

    for cell in sorted(cell_groups.keys()):
        fail = cell_groups[cell]["fail"]
        success = cell_groups[cell]["success"]
        fail_rate = sum(fail) / len(fail) if fail else float("nan")
        success_rate = sum(success) / len(success) if success else float("nan")
        print(f"{cell[0]:<15} {cell[1]:<12} {len(fail):>7} {fail_rate:>10.3%} {len(success):>10} {success_rate:>13.3%}")


if __name__ == "__main__":
    main()