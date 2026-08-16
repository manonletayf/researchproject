"""
Draws a stratified random sample of detected contradictions (triggered=true)
across the three topologies, with full context (the evaluated message +
its preceding window), for manual qualitative review at scale.

No API calls - pure reading of already-produced trigger_events and traces.

Usage:
    python sample_contradictions_for_review.py [--n-per-condition 6] [--seed 42]
"""

import argparse
import json
import random
from collections import defaultdict, deque

from rq3_config import RQ2_RESULTS_FILES, RQ2_EXPERIMENT_ROOT, RQ3_WINDOW_SIZE, RQ3_OUTPUTS_DIR

FINAL_JSONL = RQ3_OUTPUTS_DIR / "vlive_trigger_events_final.jsonl"


def load_triggered_contradictions():
    by_condition = defaultdict(list)
    with open(FINAL_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e["trigger_type"] == "contradiction" and e["triggered"]:
                condition = e["run_id"].split("__")[0]
                by_condition[condition].append(e)
    return by_condition


def load_trace_for_run(run_id: str):
    condition = run_id.split("__")[0]
    results_path = RQ2_RESULTS_FILES[condition]
    with open(results_path) as f:
        for line in f:
            row = json.loads(line)
            if f"{condition}__{row['task_id']}__{row['schema_mode']}" == run_id:
                trace_path = RQ2_EXPERIMENT_ROOT / row["trace_file"]
                with open(trace_path) as tf:
                    return json.load(tf)
    return None


def get_context(run_id: str, turn_idx: int):
    """Returns (window_turns, current_turn) for a given run_id + turn index."""
    trace = load_trace_for_run(run_id)
    if trace is None:
        return None, None
    turns = trace.get("trace", {}).get("turns", [])
    if turn_idx >= len(turns):
        return None, None
    window = deque(maxlen=RQ3_WINDOW_SIZE)
    for idx, turn in enumerate(turns):
        if idx == turn_idx:
            return list(window), turn
        window.append(turn)
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-condition", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)  # fixed seed for reproducibility (cite in thesis if used)

    by_condition = load_triggered_contradictions()

    for condition in sorted(by_condition.keys()):
        pool = by_condition[condition]
        sample = random.sample(pool, min(args.n_per_condition, len(pool)))
        print(f"\n{'='*80}\nCONDITION: {condition} ({len(pool)} total triggers, sampling {len(sample)})\n{'='*80}")

        for entry in sample:
            run_id = entry["run_id"]
            turn_idx = int(entry["message_id"].rsplit("__turn", 1)[1])
            window, current = get_context(run_id, turn_idx)

            print(f"\n--- {entry['message_id']} ---")
            print(f"JUDGE RATIONALE: {entry['judge_rationale']}")
            if current:
                print(f"\nEVALUATED MESSAGE [{current.get('agent_id', '?')}]:")
                print(f"  {current.get('content', '')[:600]}")
            if window:
                print(f"\nPRECEDING WINDOW ({len(window)} turns):")
                for w in window:
                    print(f"  [{w.get('agent_id', '?')}]: {w.get('content', '')[:200]}")
            print()


if __name__ == "__main__":
    main()