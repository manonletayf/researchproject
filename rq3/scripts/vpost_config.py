"""
Config V_post - squelette prepare avant l'arrivee des donnees RQ1 Phase 3.

Structure attendue, coherente avec rq3/scripts/rq3_config.py (V_live):

    rq2-experiment/
      rq1-results/                  <- TODO: confirmer le chemin exact une
                                        fois RQ1 Phase 3 termine (pas encore
                                        fixe a ce stade)
      rq3/
        scripts/
          vpost_config.py            <- ce fichier
          reasoning_judge.py
          run_vpost_ab_test.py
        output/
          vpost_ab_test_results.jsonl

ATTENTION - point non resolu a trancher avant le premier run reel:
RQ1 s'execute sur une instance AWS EC2 (~/manon-letayf/), separee de
rq2-experiment qui vit sur le Mac local. Le fichier de resultats Phase 3
devra donc etre RAPATRIE depuis EC2 (scp/rsync) avant que ce pipeline
puisse le lire - RQ1_PHASE3_RESULTS_PATH suppose qu'il a deja ete copie
sous rq2-experiment/rq1-results/ localement, ce qui n'est pas automatique.
"""

from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_RQ3_DIR = _SCRIPTS_DIR.parent
_RQ2_EXPERIMENT_ROOT = _RQ3_DIR.parent

# ---------------------------------------------------------------------------
# Judge - meme modele que V_live par defaut, pour la coherence a travers RQ3
# (famille differente des backbones RQ1: Qwen2.5-7B/32B, GPT-2)
# ---------------------------------------------------------------------------
VPOST_JUDGE_MODEL = "claude-sonnet-5"

# IMPORTANT: max_tokens doit rester strictement superieur au budget de
# thinking (VPOST_THINKING_BUDGET_TOKENS) + une marge pour le JSON de sortie
# lui-meme. Sinon, meme bug que celui rencontre sur V_live: le thinking
# consomme tout le budget avant de produire le moindre texte de reponse.
VPOST_THINKING_BUDGET_TOKENS = 4096
VPOST_JUDGE_MAX_TOKENS = 6144  # thinking budget (4096) + marge de sortie (~2048)

_MAX_RETRIES = 4
_RETRY_BASE_DELAY_SECONDS = 2

# ---------------------------------------------------------------------------
# Donnees source RQ1 Phase 3 - TODO CONFIRMER une fois le protocole de
# perturbation finalise. Schema attendu, base sur le RunLogger deja etabli
# pour RQ1 (voir memoire de session): instance_id, run_id, method, task,
# backbone, is_baseline_no_cot, seed, gold_answer, gold_plan_steps,
# model_answer, correct, full_trace, trace_token_count, prefix_token_count,
# reasoning_units, latency_s.
#
# Champs supplementaires ATTENDUS pour Phase 3 (injection d'erreur), noms a
# CONFIRMER aupres du protocole final RQ1 - ceux-ci sont des noms provisoires
# coherents avec la terminologie deja utilisee dans les echanges RQ1:
#   - "error_injected": bool
#   - "injection_location": reference a l'endroit de la trace ou l'erreur a
#     ete injectee (index de step ou offset de token - TODO confirmer format)
#   - "injection_type": str, type d'erreur injectee
# ---------------------------------------------------------------------------
RQ1_PHASE3_RESULTS_PATH = _RQ2_EXPERIMENT_ROOT / "rq1-results" / "phase3_perturbation_results.jsonl"  # TODO CONFIRMER chemin exact

RQ1_PHASE3_EXPECTED_FIELDS = {
    "instance_id", "run_id", "method", "task", "backbone",
    "gold_answer", "model_answer", "correct", "full_trace",
    "error_injected", "injection_location", "injection_type",  # TODO confirmer noms exacts
}

# ---------------------------------------------------------------------------
# Sorties V_post
# ---------------------------------------------------------------------------
VPOST_OUTPUTS_DIR = _RQ3_DIR / "output"
VPOST_AB_TEST_RESULTS_JSONL = VPOST_OUTPUTS_DIR / "vpost_ab_test_results.jsonl"

VPOST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)