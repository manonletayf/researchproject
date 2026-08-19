"""
Config V_post - donnees source RQ1 Phase 3 (perturbation SoftCoT/S1).

Structure attendue, coherente avec rq3/scripts/rq3_config.py (V_live):

    rq2-experiment/
      rq1-results/                              <- TODO: confirmer une fois
        softcot_perturbation_results.jsonl          rapatrie depuis EC2
        s1_perturbation_results.jsonl
        softcot_gsm8k_seed41.jsonl                  <- run principal SoftCoT
        s1_math_batch*.jsonl                        <- run(s) principal(aux) S1
      rq3/
        scripts/
          vpost/                     <- sous-dossier dedie a V_post
            vpost_config.py            <- ce fichier
            reasoning_judge.py
            run_vpost_ab_test.py
            test_reasoning_judge_synthetic.py
        output/
          vpost_ab_test_results.jsonl   <- reste dans rq3/output/, deja
                                            prefixe "vpost_" (namespace
                                            suffisant, pas de sous-dossier
                                            de sortie separe pour l'instant)

ATTENTION - point non resolu a trancher avant le premier run reel:
RQ1 s'execute sur une instance AWS EC2 (~/manon-letayf/), separee de
rq2-experiment qui vit sur le Mac local. Les fichiers ci-dessus devront
donc etre RAPATRIES depuis EC2 (scp/rsync) avant que ce pipeline puisse
les lire - les chemins ci-dessous supposent qu'ils ont deja ete copies
localement sous rq2-experiment/rq1-results/, ce qui n'est pas automatique.

ATTENTION - semantique de error_injected=True, a garder en tete a
l'interpretation des resultats:
Tant que verify_corruption_changes_answer n'est pas integree cote scripts
de perturbation RQ1 (point differe, cf. document RQ1), error_injected=True
certifie uniquement qu'UNE ETAPE CORROMPUE EST PRESENTE dans la trace -
PAS que la reponse finale en a ete affectee. Une corruption peut tres bien
laisser la reponse finale correcte (le modele "absorbe" l'erreur) ou
incorrecte. La rubrique du judge V_post (reasoning_judge.py) evalue la
COHERENCE INTERNE de la trace et son soutien au resultat final annonce -
donc conceptuellement bien alignee avec "detecter une etape corrompue",
independamment de l'effet en aval sur l'exactitude numerique du resultat.
Mais precision/rappel calcules ici mesurent "le judge detecte-t-il la
presence d'une etape corrompue", pas "le judge detecte-t-il un mauvais
resultat final" - distinction a garder explicite dans le chapitre.
"""

from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent          # rq3/scripts/vpost/
_RQ3_SCRIPTS_DIR = _SCRIPTS_DIR.parent                    # rq3/scripts/
_RQ3_DIR = _RQ3_SCRIPTS_DIR.parent                        # rq3/
_RQ2_EXPERIMENT_ROOT = _RQ3_DIR.parent                    # rq2-experiment/

# ---------------------------------------------------------------------------
# Judge - meme modele que V_live par defaut, pour la coherence a travers RQ3
# (famille differente des backbones RQ1: Qwen2.5-7B/32B, GPT-2)
# ---------------------------------------------------------------------------
VPOST_JUDGE_MODEL = "claude-sonnet-5"

# IMPORTANT: sur ce modele, le thinking "classique" (budget_tokens fixe) est
# deprecie - il faut utiliser le mode "adaptive" + le parametre effort.
# max_tokens reste un plafond dur sur la somme thinking+reponse; a des
# niveaux d'effort eleves, Claude peut consommer beaucoup de tokens de
# reflexion - garder une marge confortable au-dessus de la taille de sortie
# attendue (~150-300 tokens pour un JSON triggered+rationale).
VPOST_JUDGE_MAX_TOKENS = 6144

# Niveau d'effort pour la condition "thinking_on" du test A/B. "low" est la
# recommandation usuelle pour des taches de classification simples (voir
# doc Anthropic); on part sur "medium" pour donner une vraie chance au
# thinking dans le test A/B avant de conclure qu'il n'apporte rien - a
# ajuster selon le resultat du pilote.
VPOST_THINKING_EFFORT = "medium"

_MAX_RETRIES = 4
_RETRY_BASE_DELAY_SECONDS = 2

# ---------------------------------------------------------------------------
# Donnees source RQ1 Phase 3 - noms de champs confirmes (voir echange RQ1,
# aout 2026). Chaque fichier de perturbation ne contient QUE des instances
# corrompues (positifs, error_injected implicite = True pour toute ligne):
# il n'y a pas de champ booleen "error_injected" explicite dans ces
# fichiers - les negatifs doivent etre reconstruits separement depuis les
# runs principaux non corrompus, via original_instance_id (voir
# run_vpost_ab_test.py: load_labeled_traces()).
# ---------------------------------------------------------------------------

# Chemins TODO CONFIRMER une fois les fichiers rapatries depuis EC2 -
# noms de fichiers devines a partir des scripts cites (run_softcot_
# perturbation.py, run_s1_perturbation.py), a corriger si differents.
SOFTCOT_PERTURBATION_PATH = _RQ2_EXPERIMENT_ROOT / "rq1-results" / "softcot_perturbation_results.jsonl"  # TODO CONFIRMER
S1_PERTURBATION_PATH = _RQ2_EXPERIMENT_ROOT / "rq1-results" / "s1_perturbation_results.jsonl"  # TODO CONFIRMER

# Runs principaux (non corrompus) - source des exemples negatifs, apparies
# via original_instance_id. s1_math_batch*.jsonl est un glob (plusieurs
# fichiers batch possibles, cf. RQ1 S1 splitte en 5 batches de 100).
SOFTCOT_MAIN_RUN_PATH = _RQ2_EXPERIMENT_ROOT / "rq1-results" / "softcot_gsm8k_seed41.jsonl"  # TODO CONFIRMER
S1_MAIN_RUN_GLOB = "s1_math_batch*.jsonl"  # cherche dans rq1-results/, TODO CONFIRMER

# Champs communs aux deux fichiers de perturbation (SoftCoT et S1), avec
# les noms specifiques a chaque methode entre parentheses:
#   instance_id, run_id, method, original_instance_id, gold_answer,
#   original_model_answer, full_perturbed_trace, perturbed_answer,
#   behavioral_sensitivity, propagation_visible,
#   corrupt_step_index (SoftCoT) / corrupt_paragraph_index (S1),
#   original_step_text (SoftCoT) / original_paragraph_text (S1),
#   corrupted_step_text (SoftCoT) / corrupted_continuation... (S1),
#   corruption_prompt, perturbed_response
# Champ backbone: cle JSON confirmee = "backbone" dans les deux fichiers de
# perturbation (grep direct sur le code source, aout 2026):
#   run_softcot_perturbation.py, ligne 156: 'backbone': arg.base_model_id
#   run_s1_perturbation.py, ligne 127: 'backbone': arg.model_id
# Meme nom de cle des deux cotes malgre des noms d'argument CLI differents
# (--base_model_id pour SoftCoT, --model_id pour S1) - une seule cle a lire.

PERTURBATION_REQUIRED_FIELDS = {
    "instance_id", "method", "original_instance_id", "gold_answer",
    "full_perturbed_trace", "perturbed_answer", "backbone",
}

MAIN_RUN_REQUIRED_FIELDS = {"instance_id", "full_trace", "model_answer", "backbone"}

# ---------------------------------------------------------------------------
# Sorties V_post
# ---------------------------------------------------------------------------
VPOST_OUTPUTS_DIR = _RQ3_DIR / "output"
VPOST_AB_TEST_RESULTS_JSONL = VPOST_OUTPUTS_DIR / "vpost_ab_test_results.jsonl"

VPOST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)