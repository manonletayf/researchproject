"""Etend les corpus hierarchical et decentralized a 36 taches chacun,
en ajoutant 18 nouvelles taches distinctes de celles deja utilisees."""
import json, glob, random, shutil, os

SRC = "AgentCollabBench/tasks"
random.seed(44)  # nouvelle seed, distincte des precedentes

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

def extend_condition(condition, topo_type, n_new):
    out_dir = f"rq2_tasks/{condition}"
    existing_ids = {json.load(open(p))["task_id"] for p in glob.glob(f"{out_dir}/*.json")}
    print(f"{condition}: {len(existing_ids)} taches existantes")

    available = collect(topo_type, existing_ids)
    print(f"  {len(available)} taches disponibles pour extension")

    new_tasks = balanced_sample(available, n_new)
    for path, task in new_tasks:
        shutil.copy(path, os.path.join(out_dir, os.path.basename(path)))
    print(f"  {len(new_tasks)} nouvelles taches ajoutees -> total {len(existing_ids) + len(new_tasks)}")

extend_condition("hierarchical", "branching_tree", 18)
extend_condition("decentralized", "fully_connected", 18)