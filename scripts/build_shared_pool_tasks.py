import json, glob, random, os

SRC = "AgentCollabBench/tasks"
OUT = "rq2_tasks/shared_pool"
os.makedirs(OUT, exist_ok=True)

random.seed(43)  # seed différente pour ne pas reprendre les mêmes tâches que decentralized

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

# exclut les tâches déjà utilisées dans decentralized pour éviter le chevauchement
already_used = set()
for path in glob.glob("rq2_tasks/decentralized/*.json"):
    with open(path) as f:
        already_used.add(json.load(f)["task_id"])

fully = collect("fully_connected", already_used)
print(f"fully_connected RTD tasks available (excluding decentralized set): {len(fully)}")

random.shuffle(fully)
N = 18
selected = fully[:N]

for path, task in selected:
    task = dict(task)  # copie superficielle
    task["task_id"] = task["task_id"].replace("TASK-", "TASK-POOL-")
    task["topology"] = dict(task["topology"])
    task["topology"]["type"] = "custom_graph"  # accepté par le validateur, sémantique blackboard dans les prompts

    new_agents = []
    for agent in task["topology"]["agents"]:
        agent = dict(agent)
        agent["system_prompt"] = POOL_PREFIX + agent["system_prompt"]
        new_agents.append(agent)
    task["topology"]["agents"] = new_agents

    out_path = os.path.join(OUT, f"{task['task_id']}.json")
    with open(out_path, "w") as f:
        json.dump(task, f, indent=2)

print(f"Wrote {len(selected)} shared-pool tasks to {OUT}")