"""
Formal statistical test of the topology x schema_mode effect on the
contradiction trigger rate, aggregated at the RUN level (not message
level) to respect the fact that messages within a run are not independent
observations - consistent with the RQ2 analysis approach (factorial
ANOVA on run-level outcomes).

Requires pandas, scipy, statsmodels (already used in the RQ2 analysis
pipeline per the project's established toolkit - if missing here, install
with: pip install pandas scipy statsmodels

Usage:
    python analyze_vlive_factorial.py
"""

import json
from collections import defaultdict

from rq3_config import RQ3_OUTPUTS_DIR

FINAL_JSONL = RQ3_OUTPUTS_DIR / "vlive_trigger_events_final.jsonl"


def load_per_run_rates():
    msgs = defaultdict(int)
    triggers = defaultdict(int)
    with open(FINAL_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e["trigger_type"] != "contradiction":
                continue
            msgs[e["run_id"]] += 1
            if e["triggered"]:
                triggers[e["run_id"]] += 1

    rows = []
    for run_id in msgs:
        parts = run_id.split("__")
        condition = parts[0]
        schema_mode = parts[-1]
        rows.append({
            "run_id": run_id,
            "condition": condition,
            "schema_mode": schema_mode,
            "n_messages": msgs[run_id],
            "n_triggers": triggers[run_id],
            "rate": triggers[run_id] / msgs[run_id],
        })
    return rows


def main():
    rows = load_per_run_rates()
    print(f"{len(rows)} runs loaded")

    try:
        import pandas as pd
        import numpy as np
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
    except ImportError as e:
        print(f"\nMissing package: {e}")
        print("Install with: pip install pandas scipy statsmodels")
        print("\nFalling back to descriptive stats only (no formal test).")
        _fallback_descriptive(rows)
        return

    df = pd.DataFrame(rows)

    # Arcsine-square-root transform for proportions - standard variance-
    # stabilizing transform before ANOVA on rate/proportion data, since
    # raw proportions violate the homogeneity-of-variance assumption
    # (variance depends on the mean for binomial proportions).
    df["rate_asin"] = np.arcsin(np.sqrt(df["rate"]))

    print("\n=== Two-way ANOVA on arcsine-transformed contradiction rate ===")
    print("Model: rate_asin ~ C(condition) * C(schema_mode)\n")

    model = ols("rate_asin ~ C(condition) * C(schema_mode)", data=df).fit()
    anova_table = anova_lm(model, typ=2)
    print(anova_table)

    print("\n=== Interpretation guide ===")
    print("C(condition)              -> main effect of topology")
    print("C(schema_mode)            -> main effect of schema_mode")
    print("C(condition):C(schema_mode) -> interaction (the pattern noted earlier:")
    print("                                schema_mode's protective effect differs by topology)")
    print("PR(>F) < 0.05 (or your chosen alpha) indicates a significant effect.")

    # Also report un-transformed cell means for reporting in the thesis
    print("\n=== Cell means (untransformed rate) for reporting ===")
    cell_means = df.groupby(["condition", "schema_mode"])["rate"].agg(["mean", "std", "count"])
    print(cell_means)


def _fallback_descriptive(rows):
    from collections import defaultdict
    cells = defaultdict(list)
    for r in rows:
        cells[(r["condition"], r["schema_mode"])].append(r["rate"])
    for cell, rates in sorted(cells.items()):
        mean = sum(rates) / len(rates)
        print(f"{cell}: n={len(rates)}, mean rate={mean:.3%}")


if __name__ == "__main__":
    main()