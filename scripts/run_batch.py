"""Exécute un lot de runs RQ2 (topologie x schéma) et sauvegarde progressivement en JSONL."""
import sys, json, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "AgentCollabBench"))

from config import get_rq2_provider
from agentcollabbench.harness.runner import TaskRunner
from agentcollabbench.metrics.rtd.rtd import RTDMetric

def already_done(output_path, task_id, schema_mode):
    """Vérifie si ce (task_id, schema_mode) a déjà un résultat sauvegardé."""
    if not Path(output_path).exists():
        return False
    with open(output_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["task_id"] == task_id and rec["schema_mode"] == schema_mode:
                return True
    return False

def run_batch(condition, task_dir, output_path, schema_modes=("free_text", "typed")):
    task_files = sorted(Path(task_dir).glob("*.json"))
    total = len(task_files) * len(schema_modes)
    done = 0

    for task_file in task_files:
        with open(task_file) as f:
            task_json = json.load(f)
        task_id = task_json["task_id"]

        for schema_mode in schema_modes:
            done += 1
            if already_done(output_path, task_id, schema_mode):
                print(f"[{done}/{total}] SKIP (deja fait): {task_id} / {schema_mode}")
                continue

            print(f"[{done}/{total}] Running: {task_id} / {schema_mode} ...", flush=True)
            start = time.time()
            try:
                provider = get_rq2_provider()
                runner = TaskRunner(provider=provider, metric="rtd", schema_mode=schema_mode)
                result = runner.run_task_file(str(task_file))

                rtd_metric = RTDMetric()
                score = rtd_metric.compute(result.scenario, result.output, result.trace)

                record = {
                    "task_id": task_id,
                    "condition": condition,
                    "schema_mode": schema_mode,
                    "structural_complexity": task_json.get("structural_complexity"),
                    "domain": task_json.get("domain"),
                    "rtd_score": score,
                    "num_turns": len(result.trace["turns"]),
                    "total_llm_latency_seconds": result.trace["timestamps"].get("total_llm_latency_seconds"),
                    "wall_clock_seconds": round(time.time() - start, 2),
                    "error": None,
                }
            except Exception as e:
                record = {
                    "task_id": task_id,
                    "condition": condition,
                    "schema_mode": schema_mode,
                    "structural_complexity": task_json.get("structural_complexity"),
                    "domain": task_json.get("domain"),
                    "rtd_score": None,
                    "num_turns": None,
                    "total_llm_latency_seconds": None,
                    "wall_clock_seconds": round(time.time() - start, 2),
                    "error": str(e),
                }
                print(f"   ERREUR: {e}")

            with open(output_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            status = "OK" if record["error"] is None else "FAILED"
            print(f"   -> {status} | RTD={record['rtd_score']} | {record['wall_clock_seconds']}s")

    print(f"\nTerminé: {done}/{total} runs traités (voir {output_path})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["hierarchical", "decentralized", "shared_pool"])
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    task_dir = base / "rq2_tasks" / args.condition
    output_path = base / "full_run_results" / f"{args.condition}.jsonl"
    output_path.parent.mkdir(exist_ok=True)

    run_batch(args.condition, task_dir, output_path)