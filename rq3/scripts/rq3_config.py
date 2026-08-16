"""
Config RQ3 - verrouille le modele judge et les chemins, sur le meme
principe que scripts/config.py en RQ2 (RQ2_BACKBONE_MODEL).

A placer dans ~/rq2-experiment/rq3/scripts/rq3_config.py, c-a-d dans le
sous-dossier "rq3" separe, a l'interieur du repo rq2-experiment existant.
Structure attendue:

    rq2-experiment/
      full_run_results/        <- logs RQ2 (source, inchange)
      rq3/
        scripts/
          rq3_config.py         <- ce fichier
          schema_validator.py
          contradiction_judge.py
          run_vlive.py
        outputs/                <- cree automatiquement, sorties RQ3

Tous les chemins ci-dessous sont calcules relativement a l'emplacement de
ce fichier (Path(__file__)) plutot qu'a Path.home(), pour ne pas dependre
du nom d'utilisateur ni d'un lancement depuis un dossier particulier.
"""

from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent          # rq2-experiment/rq3/scripts
_RQ3_DIR = _SCRIPTS_DIR.parent                            # rq2-experiment/rq3
_RQ2_EXPERIMENT_ROOT = _RQ3_DIR.parent                    # rq2-experiment

# ---------------------------------------------------------------------------
# Modele judge - famille differente du backbone evalue (gpt-4.1-mini)
# ---------------------------------------------------------------------------
RQ3_JUDGE_MODEL = "claude-sonnet-5"
RQ3_JUDGE_MAX_TOKENS = 2048  # remonte de 1024: 3 messages precis epuisaient systematiquement le budget en thinking, jamais de JSON produit

# ---------------------------------------------------------------------------
# Fenetre glissante pour la detection de contradiction
# ---------------------------------------------------------------------------
RQ3_WINDOW_SIZE = 5

# ---------------------------------------------------------------------------
# Chemins RQ2 (source des logs a rejouer) - inchanges, un niveau au-dessus
# du dossier rq3/
# ---------------------------------------------------------------------------
RQ2_EXPERIMENT_ROOT = _RQ2_EXPERIMENT_ROOT
RQ2_ROOT = _RQ2_EXPERIMENT_ROOT / "full_run_results"

RQ2_RESULTS_FILES = {
    "hierarchical": RQ2_ROOT / "hierarchical.jsonl",
    "decentralized": RQ2_ROOT / "decentralized.jsonl",
    "shared_pool": RQ2_ROOT / "shared_pool.jsonl",
}

RQ2_TRACES_ROOT = RQ2_ROOT / "traces"

# ---------------------------------------------------------------------------
# Champ de sortie RQ2_TYPED_RESPONSE_SCHEMA (mode "typed"), pour la
# validation programmatique en etape 1 de V_live.
# reconstruit depuis Jour 2 - a CONFIRMER cote implementation reelle,
# en particulier le type de "findings" (string vs liste).
# ---------------------------------------------------------------------------
RQ2_TYPED_RESPONSE_SCHEMA_FIELDS = {
    "reasoning": {"type": str, "required": True},
    "findings": {"type": str, "required": True},   # TODO confirmer: str ou list ?
    "decision": {"type": str, "required": True},
    "confidence": {"type": (int, float, type(None)), "required": True},  # nullable
}

# ---------------------------------------------------------------------------
# Sorties RQ3 - a l'interieur du sous-dossier rq3/output/ (nom confirme
# par la structure reelle du projet)
# ---------------------------------------------------------------------------
RQ3_OUTPUTS_DIR = _RQ3_DIR / "output"
RQ3_TRIGGER_EVENTS_JSONL = RQ3_OUTPUTS_DIR / "vlive_trigger_events.jsonl"
RQ3_MANIFEST_JSONL = RQ3_OUTPUTS_DIR / "vlive_manifest.jsonl"
RQ3_FAILURES_LOG = RQ3_OUTPUTS_DIR / "vlive_failures.jsonl"

RQ3_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)