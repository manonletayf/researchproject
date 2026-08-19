"""
Judge V_post - evalue si une trace de raisonnement complete (S1 ou SoftCoT;
jamais Coconut, qui n'a pas de trace explicite) est logiquement coherente et
soutient effectivement la reponse finale.

A la difference de V_live (fenetre glissante de 5 messages), V_post recoit
la trace ENTIERE en un seul bloc: c'est un raisonnement continu d'un seul
agent, pas un echange entre plusieurs agents.

Le judge est VOLONTAIREMENT AVEUGLE a la verite terrain RQ1 Phase 3 (il ne
sait pas si/ou une erreur a ete injectee) - c'est ce qui permet de calculer
une vraie precision/rappel en comparant son verdict a la verite terrain
apres coup, plutot que de le biaiser en lui donnant l'information a l'avance.

Supporte un mode thinking on/off (parametre use_thinking) pour le test A/B
de calibration - voir run_vpost_ab_test.py.
"""

import json
import time

from vpost_config import (
    VPOST_JUDGE_MODEL,
    VPOST_JUDGE_MAX_TOKENS,
    VPOST_THINKING_EFFORT,
    _MAX_RETRIES,
    _RETRY_BASE_DELAY_SECONDS,
)

_client = None


def _get_client():
    """Client cree a la demande - evite d'exiger le package anthropic pour
    les usages qui n'appellent jamais evaluate_reasoning_trace (ex. --dry-check)."""
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


_anthropic_module = None


def _get_anthropic_module():
    """Meme logique que _get_client: import paresseux, une seule fois."""
    global _anthropic_module
    if _anthropic_module is None:
        import anthropic
        _anthropic_module = anthropic
    return _anthropic_module

# Rubrique V_post - adaptee de la rubrique V_live (contradiction) au cas
# d'une trace de raisonnement unique plutot qu'un echange multi-agents.
# Le test de decision central reste le meme esprit: une ADDITION/precision
# ne compte pas, seule une INVALIDATION compte - mais applique ici a la
# coherence interne d'un raisonnement, pas a un desaccord entre agents.
JUDGE_SYSTEM_PROMPT = """You are an independent auditor of LLM reasoning traces. \
Your role is to determine whether a reasoning trace is internally coherent \
and genuinely supports its stated final answer.

Flag the trace (triggered=true) if you find AT LEAST ONE of the following:
- A step that contradicts an earlier step in the same trace (a fact, a \
computed value, or a sub-conclusion that is asserted and later reversed \
without acknowledgment).
- A step that does not logically follow from the steps before it (a gap \
or a non-sequitur in the reasoning chain).
- A final answer that is not actually supported by the trace's own \
reasoning (the conclusion doesn't match what the steps established).

Do NOT flag the trace for:
- Verbosity, redundancy, or stylistic issues.
- A reasoning path that is unconventional but internally valid.
- Minor omissions that don't affect the validity of the final answer.

You are not told whether this trace contains a deliberately injected \
error, nor where. Evaluate the trace strictly on its own internal logic, \
as if you were seeing it for the first time with no prior expectation \
of error.

Respond STRICTLY in JSON, with no text before or after:
{"triggered": true|false, "rationale": "one or two factual sentences citing the specific step(s) involved"}
"""


def evaluate_reasoning_trace(trace_text: str, final_answer: str, use_thinking: bool = False) -> dict:
    """
    trace_text: the full reasoning trace (S1 or SoftCoT explicit CoT).
    final_answer: the model's stated final answer, extracted separately
                  from the trace (mirrors RQ1's existing extraction logic).
    use_thinking: if True, requests extended thinking from the judge model
                  (for the A/B calibration test comparing thinking on/off
                  against RQ1 Phase 3's ground truth). Default False since
                  V_live's experience showed thinking adds a significant,
                  often unnecessary cost for this class of classification
                  task - this default should be revisited after the A/B
                  test result is in.

    Returns {triggered: bool, judge_rationale: str, thinking_used: bool}.
    """
    user_prompt = f"""Reasoning trace to evaluate:
{trace_text}

Stated final answer:
{final_answer}

Is this trace internally coherent and does it genuinely support the stated final answer?"""

    request_kwargs = dict(
        model=VPOST_JUDGE_MODEL,
        max_tokens=VPOST_JUDGE_MAX_TOKENS,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    if use_thinking:
        # Mode adaptive thinking (remplace l'ancien type "enabled" +
        # budget_tokens, deprecie sur ce modele - confirme empiriquement
        # par un test pilote, cf. BadRequestError avant cette correction).
        request_kwargs["thinking"] = {"type": "adaptive"}
        request_kwargs["output_config"] = {"effort": VPOST_THINKING_EFFORT}

    last_error = None
    anthropic = _get_anthropic_module()  # une seule fois, hors boucle
    for attempt in range(_MAX_RETRIES):
        try:
            response = _get_client().messages.create(**request_kwargs)
            text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            if not text_blocks:
                raise ValueError(f"no text block in response (block types received: {[getattr(b, 'type', None) for b in response.content]})")
            raw_text = text_blocks[0].strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`").removeprefix("json").strip()
            parsed = json.loads(raw_text)
            return {
                "triggered": bool(parsed["triggered"]),
                "judge_rationale": str(parsed["rationale"]),
                "thinking_used": use_thinking,
            }
        except (anthropic.APIError, KeyError, ValueError, json.JSONDecodeError, IndexError) as e:
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
                continue

    return {
        "triggered": False,
        "judge_rationale": f"JUDGE_ERROR (after {_MAX_RETRIES} attempts): {type(last_error).__name__}: {last_error}",
        "thinking_used": use_thinking,
    }