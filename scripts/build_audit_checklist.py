import json, glob, os

def extract_tracers(task):
    """Handle both single-tracer and multi_constraint RTD injection formats."""
    rtd = task.get("injections", {}).get("rtd", {})
    multi = rtd.get("multi_constraint", {})
    if multi and "constraints" in multi:
        return [
            {"tracer_id": c.get("tracer_id", ""), "anchor": c.get("anchor", "")}
            for c in multi["constraints"]
        ]
    if rtd.get("tracer_id"):
        return [{"tracer_id": rtd.get("tracer_id", ""), "anchor": rtd.get("anchor", "")}]
    return []

def extract_checklist(task):
    tracers = extract_tracers(task)
    points = []
    for t in tracers:
        points.append({"point": "tracer_present_in_root_output", "detail": t["tracer_id"]})
        points.append({"point": "tracer_present_in_final_output", "detail": t["tracer_id"]})
        points.append({"point": "constraint_anchor_addressed", "detail": t["anchor"]})
    points.append({
        "point": "correct_outcome_reflected",
        "detail": task.get("ground_truth", {}).get("correct_outcome", "")[:200]
    })
    return {
        "task_id": task["task_id"],
        "topology": task["topology"]["type"],
        "num_tracers": len(tracers),
        "expected_points": points,
    }

all_checklists = []
for group in ["hierarchical", "decentralized", "shared_pool"]:
    for path in glob.glob(f"rq2_tasks/{group}/*.json"):
        with open(path) as f:
            task = json.load(f)
        all_checklists.append(extract_checklist(task))

with open("rq2_tasks/audit_checklist.json", "w") as f:
    json.dump(all_checklists, f, indent=2)

empty = sum(1 for c in all_checklists if c["num_tracers"] == 0)
print(f"Built checklist for {len(all_checklists)} tasks")
print(f"Tasks with zero tracers detected: {empty}")