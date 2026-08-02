import json, glob, random, shutil, os

SRC = "AgentCollabBench/tasks"
OUT = "rq2_tasks/hierarchical"
OUT2 = "rq2_tasks/decentralized"
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT2, exist_ok=True)

random.seed(42)

def collect(topo_type):
    matches = []
    for path in glob.glob(f"{SRC}/*.json"):
        with open(path) as f:
            task = json.load(f)
        if "rtd" not in task.get("metric_applicability", []):
            continue
        if task.get("topology", {}).get("type") != topo_type:
            continue
        matches.append((path, task))
    return matches

branching = collect("branching_tree")
fully = collect("fully_connected")

print(f"branching_tree RTD tasks found: {len(branching)}")
print(f"fully_connected RTD tasks found: {len(fully)}")

N = 18

def balanced_sample(matches, n):
    by_domain = {}
    for path, task in matches:
        by_domain.setdefault(task["domain"], []).append((path, task))
    for d in by_domain:
        random.shuffle(by_domain[d])
    result = []
    domains = list(by_domain.keys())
    i = 0
    while len(result) < n and any(by_domain.values()):
        d = domains[i % len(domains)]
        if by_domain[d]:
            result.append(by_domain[d].pop())
        i += 1
    return result

for path, task in balanced_sample(branching, N):
    shutil.copy(path, os.path.join(OUT, os.path.basename(path)))

for path, task in balanced_sample(fully, N):
    shutil.copy(path, os.path.join(OUT2, os.path.basename(path)))

print(f"Copied {len(os.listdir(OUT))} hierarchical tasks to {OUT}")
print(f"Copied {len(os.listdir(OUT2))} decentralized tasks to {OUT2}")
