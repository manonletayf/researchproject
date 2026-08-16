"""
Etape 3 du plan RQ3 - trigger schema (V_live, etage 1, programmatique).

Valide un tour (turn) contre RQ2_TYPED_RESPONSE_SCHEMA_FIELDS quand le run
est en schema_mode="typed". Ne fait AUCUN appel LLM - c'est le point de
l'etage 1: filtrer ce qui est detectable sans judge avant d'appeler Sonnet.

En schema_mode="free_text", le check est skip (retourne triggered=False,
schema non applicable).
"""

import json

from rq3_config import RQ2_TYPED_RESPONSE_SCHEMA_FIELDS


def validate_typed_turn(content: str, schema_mode: str) -> dict:
    """
    content: the raw "content" field of a turn (string).
    schema_mode: "typed" or "free_text", as present in the RQ2 results
                 file (hierarchical.jsonl etc.)

    Returns a dict ready to be merged into a trigger_event:
        {triggered: bool, judge_rationale: str}
    """
    if schema_mode != "typed":
        return {"triggered": False, "judge_rationale": "schema_mode=free_text, check not applicable"}

    # 1. Content must be valid JSON
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {"triggered": True, "judge_rationale": "content is not valid JSON"}

    if not isinstance(parsed, dict):
        return {"triggered": True, "judge_rationale": f"valid JSON but not an object (type={type(parsed).__name__})"}

    violations = []

    # 2. Required fields present
    for field_name, spec in RQ2_TYPED_RESPONSE_SCHEMA_FIELDS.items():
        if spec["required"] and field_name not in parsed:
            violations.append(f"missing field: {field_name}")

    # 3. Correct types for present fields
    for field_name, spec in RQ2_TYPED_RESPONSE_SCHEMA_FIELDS.items():
        if field_name in parsed:
            expected_type = spec["type"]
            value = parsed[field_name]
            if not isinstance(value, expected_type):
                violations.append(
                    f"invalid type for {field_name}: expected {expected_type}, got {type(value).__name__}"
                )

    # 4. No unexpected fields (sign of the model drifting off-schema)
    extra_fields = set(parsed.keys()) - set(RQ2_TYPED_RESPONSE_SCHEMA_FIELDS.keys())
    if extra_fields:
        violations.append(f"unexpected fields: {sorted(extra_fields)}")

    if violations:
        return {"triggered": True, "judge_rationale": "; ".join(violations)}

    return {"triggered": False, "judge_rationale": "conforms to schema"}