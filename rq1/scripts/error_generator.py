"""
Generateur d'erreur LLM pour le protocole d'injection RQ1 (S1, SoftCoT).
Reecrit une etape de raisonnement en y introduisant une erreur logique
ou arithmetique subtile qui change sa conclusion.
"""
import os
from openai import OpenAI

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def corrupt_step(step_text, model='gpt-4o-mini'):
    """
    Reecrit step_text en y introduisant une erreur subtile.
    Retourne (texte_corrompu, prompt_utilise) pour tracabilite.
    """
    prompt = (
        "Voici une etape de raisonnement mathematique :\n\n"
        f"{step_text}\n\n"
        "Reecris cette etape en y introduisant UNE SEULE erreur logique ou "
        "arithmetique subtile qui change sa conclusion. L'erreur doit "
        "paraitre plausible, pas absurde. Garde le meme style et la meme "
        "longueur approximative. Ne commente pas, ne signale pas l'erreur, "
        "renvoie uniquement le texte corrompu."
    )
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    corrupted = response.choices[0].message.content.strip()
    return corrupted, prompt


def verify_corruption_changes_answer(original_final_answer, corrupted_leads_to,
                                      gold_answer):
    """
    Verification a posteriori : l'erreur injectee doit rendre la reponse
    fausse par rapport au gold_answer (sinon l'injection est inutile).
    """
    return corrupted_leads_to != gold_answer
