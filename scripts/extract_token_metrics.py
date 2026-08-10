"""Extrait les metriques de latence normalisee (par tour, par token) et la
pente de croissance de latence intra-run depuis les traces completes.
Necessaire car input_tokens/output_tokens ne sont sauvegardes que dans les
traces, pas dans le resume JSONL."""
import json
from pathlib import Path
from scipy import stats as scipy_stats

BASE = Path(__file__).parent.parent

def extract_metrics(run, trace_data):
    api_calls = trace_data["trace"].get("api_calls", [])
    if not api_calls:
        return None

    total_input = sum(c.get("input_tokens", 0) or 0 for c in api_calls)
    total_output = sum(c.get("output_tokens", 0) or 0 for c in api_calls)
    total_tokens = total_input + total_output
    num_turns = run["num_turns"]
    total_latency = run["total_llm_latency_seconds"]

    latency_per_turn = total_latency / num_turns if num_turns else None
    latency_per_token = total_latency / total_tokens if total_tokens else None

    # pente de latence par index d'appel (0, 1, 2, ...) - regression lineaire simple
    latencies = [c.get("latency_seconds", 0) or 0 for c in api_calls]
    slope = None
    if len(latencies) >= 3:
        indices = list(range(len(latencies)))
        result = scipy_stats.linregress(indices, latencies)
        slope = result.slope

    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_tokens,
        "latency_per_turn": latency_per_turn,
        "latency_per_token": latency_per_token,
        "latency_growth_slope": slope,
    }

def main():
    all_records = []
    for condition in ["hierarchical", "decentralized", "shared_pool"]:
        jsonl_path = BASE / "full_run_results" / f"{condition}.jsonl"
        with open(jsonl_path) as f:
            run_records = [json.loads(l) for l in f]

        for run in run_records:
            if run["error"] is not None:
                continue
            trace_path = BASE / run["trace_file"]
            with open(trace_path) as f:
                trace_data = json.load(f)

            metrics = extract_metrics(run, trace_data)
            if metrics is None:
                continue

            all_records.append({
                "task_id": run["task_id"],
                "condition": condition,
                "schema_mode": run["schema_mode"],
                **metrics,
            })

    output_path = BASE / "full_run_results" / "token_latency_metrics.jsonl"
    with open(output_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    print(f"Termine: {len(all_records)} runs, ecrit dans {output_path}")

if __name__ == "__main__":
    main()