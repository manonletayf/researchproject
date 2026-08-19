"""
Harness de logging JSONL commun pour RQ1 (S1, Coconut, SoftCoT).
Un fichier JSONL par run. Append-only, résumable : au chargement,
lit les instance_id déjà présents pour permettre de sauter les
instances déjà traitées après un crash/interruption.
"""
import json
import time
from pathlib import Path


class RunLogger:
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._completed_ids = self._load_completed_ids()

    def _load_completed_ids(self):
        ids = set()
        if self.output_path.exists():
            with open(self.output_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        ids.add(rec.get("instance_id"))
                    except json.JSONDecodeError:
                        continue
        return ids

    def is_done(self, instance_id):
        return instance_id in self._completed_ids

    def log(self, record: dict):
        assert "instance_id" in record, "record must include instance_id"
        record.setdefault("logged_at", time.time())
        with open(self.output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._completed_ids.add(record["instance_id"])
