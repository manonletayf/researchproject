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


class OpenAICreditError(Exception):
    """Levee quand la cle API OpenAI n'a plus de credit/quota disponible."""
    pass


def corrupt_step(step_text, model='gpt-4o-mini'):
    """
    Reecrit step_text en y introduisant une erreur subtile.
    Retourne (texte_corrompu, prompt_utilise) pour tracabilite.
    Leve OpenAICreditError si le quota/credit est epuise -- a intercepter
    par l'appelant pour arreter proprement le script plutot que de
    continuer a echouer sur chaque instance suivante.
    """
    import openai
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
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
    except openai.RateLimitError as e:
        raise OpenAICreditError(
            f'Credit/quota OpenAI epuise ou limite de taux atteinte : {e}'
        ) from e
    except openai.AuthenticationError as e:
        raise OpenAICreditError(
            f'Cle API OpenAI invalide ou expiree : {e}'
        ) from e
    corrupted = response.choices[0].message.content.strip()
    return corrupted, prompt


def verify_corruption_changes_answer(original_final_answer, corrupted_leads_to,
                                      gold_answer):
    """
    Verification a posteriori : l'erreur injectee doit rendre la reponse
    fausse par rapport au gold_answer (sinon l'injection est inutile).
    """
    return corrupted_leads_to != gold_answer
