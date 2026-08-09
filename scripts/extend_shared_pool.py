"""Etend le corpus shared_pool a 36 taches. Utilise d'abord les 9 taches
fully_connected jamais touchees, puis complete avec 9 taches deja utilisees
dans decentralized (chevauchement assume et documente comme limite -
seules 45 taches fully_connected existent au total dans le corpus source,
insuffisant pour 36+36 sans recoupement)."""
import json, glob, random, os

SRC = "AgentCollabBench/tasks"
OUT = "rq2_tasks/shared_pool"
random.seed(45)

POOL_PREFIX = (
    "You post updates to and read from a SHARED TEAM WORKSPACE visible to "
    "all agents simultaneously, rather than messaging any single teammate directly. "
    "Before responding, review everything currently posted in the workspace and "
    "build on it. "
)

def collect(topo_type, exclude_ids):
    matches = []
    for path in glob.glob(f"{SRC}/*.json"):
        with open(path) as f:
            task = json.load(f)
        if "rtd" not in task.get("metric_applicability", []):
            continue
        if task.get("topology", {}).get("type") != topo_type:
            continue
        if task["task_id"] in exclude_ids:
            continue
        matches.append((path, task))
    return matches

def build_pool_task(task, source_id_used):
    task = dict(task)
    original_id = task["task_id"]
    task["task_id"] = original_id.replace("TASK-", "TASK-POOL-")

    # evite les collisions si l'id existe deja (cas du chevauchement)
    out_path = f"{OUT}/{task['task_id']}.json"
    if os.path.exists(out_path):
        return None  # deja construit lors de l'extension initiale

    task["topology"] = dict(task["topology"])
    task["topology"]["type"] = "custom_graph"
    new_agents = []
    for agent in task["topology"]["agents"]:
        agent = dict(agent)
        agent["system_prompt"] = POOL_PREFIX + agent["system_prompt"]
        new_agents.append(agent)
    task["topology"]["agents"] = new_agents
    return task

# IDs deja presents dans shared_pool (sous leur forme TASK-POOL-...)
existing_pool_ids_raw = {
    json.load(open(p))["task_id"].replace("TASK-POOL-", "TASK-")
    for p in glob.glob(f"{OUT}/*.json")
}
print(f"shared_pool: {len(existing_pool_ids_raw)} taches existantes")

# IDs deja utilisees dans decentralized (pour le chevauchement controle)
used_decentralized = {json.load(open(p))["task_id"] for p in glob.glob("rq2_tasks/decentralized/*.json")}

# D'abord: taches fully_connected jamais touchees ni par shared_pool ni par decentralized
never_touched = collect("fully_connected", existing_pool_ids_raw | used_decentralized)
print(f"  Taches jamais touchees disponibles: {len(never_touched)}")

random.shuffle(never_touched)
n_needed = 18
from_never_touched = never_touched[:n_needed]
remaining_needed = n_needed - len(from_never_touched)

print(f"  Utilisation de {len(from_never_touched)} taches jamais touchees")

added = 0
for path, task in from_never_touched:
    new_task = build_pool_task(task, source_id_used=False)
    if new_task:
        with open(f"{OUT}/{new_task['task_id']}.json", "w") as f:
            json.dump(new_task, f, indent=2)
        added += 1

# Ensuite, si besoin: complement via chevauchement avec decentralized
if remaining_needed > 0:
    overlap_candidates = [
        (p, json.load(open(p))) for p in glob.glob("rq2_tasks/decentralized/*.json")
    ]
    random.shuffle(overlap_candidates)
    overlap_selected = overlap_candidates[:remaining_needed]
    print(f"  CHEVAUCHEMENT ASSUME: {len(overlap_selected)} taches reprises depuis decentralized")

    for path, task in overlap_selected:
        new_task = build_pool_task(task, source_id_used=True)
        if new_task:
            with open(f"{OUT}/{new_task['task_id']}.json", "w") as f:
                json.dump(new_task, f, indent=2)
            added += 1

print(f"Total ajoute: {added}")
print(f"Total final shared_pool: {len(existing_pool_ids_raw) + added}")