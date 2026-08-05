"""Calcule les points d'audit tracer_present_in_root_output et tracer_present_in_final_output
pour chaque run, en croisant les traces du Jour 4 avec la checklist du Jour 1.

Utilise DAGTopology pour identifier correctement la couche la plus profonde du graphe,
plutot qu'une approximation par position chronologique (le speaking_order boucle sur
le cycle d'agents, donc "les derniers tours" ne correspondent pas forcement a la
couche la plus profonde)."""
import sys
import json
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "AgentCollabBench"))

from agentcollabbench.metrics.rtd.topology import DAGTopology


def load_checklist():
    with open(BASE / "rq2_tasks" / "audit_checklist.json") as f:
        checklist = json.load(f)
    return {c["task_id"]: c for c in checklist}


def load_task_json(condition, task_id):
    """Charge le fichier de tache original pour reconstruire la topologie."""
    path = BASE / "rq2_tasks" / condition / f"{task_id}.json"
    with open(path) as f:
        return json.load(f)


def score_trace(trace_data, expected_points, task_json):
    """Verifie tracer_present_in_root_output et tracer_present_in_final_output
    en repliquant fidelement la logique de _build_messages() dans rtd.py :
    la couche la plus profonde inclut le contenu des agents de cette couche
    ET output["result"], qui peut porter le tracer independamment de qui
    l'a produit."""
    turns = trace_data["trace"]["turns"]
    if not turns:
        return []

    topo = DAGTopology(task_json["topology"])
    depth = topo.dag_depth

    root_content = turns[0]["content"]

    final_layer_parts = []
    for t in turns:
        try:
            if topo.agent_layer(t["agent_id"]) == depth:
                final_layer_parts.append(t["content"])
        except KeyError:
            continue

    # Replique _build_messages(): output["result"] est toujours inclus
    # a la couche la plus profonde, peu importe quel agent l'a produit
    result = trace_data["output"].get("result", "")
    if result:
        final_layer_parts.append(result)

    final_layer_content = " ".join(final_layer_parts)

    results = []
    for point in expected_points:
        if point["point"] == "tracer_present_in_root_output":
            found = point["detail"] in root_content
            results.append({"point": point["point"], "detail": point["detail"], "found": found})
        elif point["point"] == "tracer_present_in_final_output":
            found = point["detail"] in final_layer_content
            results.append({"point": point["point"], "detail": point["detail"], "found": found})

    return results


def main():
    checklist_by_id = load_checklist()
    all_records = []

    for condition in ["hierarchical", "decentralized", "shared_pool"]:
        jsonl_path = BASE / "full_run_results" / f"{condition}.jsonl"
        with open(jsonl_path) as f:
            run_records = [json.loads(l) for l in f]

        for run in run_records:
            if run["error"] is not None:
                continue

            task_id = run["task_id"]
            schema_mode = run["schema_mode"]
            checklist_entry = checklist_by_id.get(task_id)
            if checklist_entry is None:
                print(f"ATTENTION: pas de checklist pour {task_id}")
                continue

            task_json = load_task_json(condition, task_id)

            trace_path = BASE / run["trace_file"]
            with open(trace_path) as f:
                trace_data = json.load(f)

            audit_points = score_trace(trace_data, checklist_entry["expected_points"], task_json)
            root_points = [p for p in audit_points if p["point"] == "tracer_present_in_root_output"]
            final_points = [p for p in audit_points if p["point"] == "tracer_present_in_final_output"]

            root_score = sum(p["found"] for p in root_points) / len(root_points) if root_points else None
            final_score = sum(p["found"] for p in final_points) / len(final_points) if final_points else None

            all_records.append({
                "task_id": task_id,
                "condition": condition,
                "schema_mode": schema_mode,
                "rtd_score": run["rtd_score"],
                "num_tracers": checklist_entry.get("num_tracers", 1),
                "root_output_score": root_score,
                "final_output_score": final_score,
                "audit_detail": audit_points,
            })

    output_path = BASE / "full_run_results" / "audit_scores.jsonl"
    with open(output_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    print(f"Termine: {len(all_records)} runs scores, ecrit dans {output_path}")


if __name__ == "__main__":
    main()