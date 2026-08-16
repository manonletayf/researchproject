"""
Merges vlive_trigger_events.jsonl (original) with
vlive_trigger_events_repaired.jsonl (repair passes) into a single clean
file, keeping only the LATEST entry per (message_id, trigger_type).

Since repaired entries are contradiction-only and appended across possibly
multiple repair passes, the last occurrence in file order is the correct
final value (later repair attempts supersede earlier ones).

Usage:
    python merge_vlive_final.py
"""

import json

from rq3_config import RQ3_TRIGGER_EVENTS_JSONL, RQ3_OUTPUTS_DIR

REPAIRED_JSONL = RQ3_OUTPUTS_DIR / "vlive_trigger_events_repaired.jsonl"
FINAL_JSONL = RQ3_OUTPUTS_DIR / "vlive_trigger_events_final.jsonl"


def main():
    latest = {}  # (message_id, trigger_type) -> event dict, last one wins

    for path in (RQ3_TRIGGER_EVENTS_JSONL, REPAIRED_JSONL):
        if not path.exists():
            print(f"skip (not found): {path}")
            continue
        n = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key = (entry["message_id"], entry["trigger_type"])
                latest[key] = entry
                n += 1
        print(f"read {n} lines from {path.name}")

    with open(FINAL_JSONL, "w") as out:
        for entry in latest.values():
            out.write(json.dumps(entry) + "\n")

    total = len(latest)
    still_error = sum(1 for e in latest.values() if "JUDGE_ERROR" in e.get("judge_rationale", ""))
    n_contradiction_true = sum(1 for e in latest.values() if e["trigger_type"] == "contradiction" and e["triggered"])
    n_schema_true = sum(1 for e in latest.values() if e["trigger_type"] == "schema" and e["triggered"])

    print(f"\nFinal merged file: {FINAL_JSONL}")
    print(f"total events (deduplicated): {total}")
    print(f"remaining JUDGE_ERROR: {still_error}")
    print(f"contradiction triggered=true: {n_contradiction_true}")
    print(f"schema triggered=true: {n_schema_true}")


if __name__ == "__main__":
    main()