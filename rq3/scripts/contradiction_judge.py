"""
Etape 4 du plan RQ3 - trigger contradiction inter-agents (V_live, etage 2).

Le judge (Claude Sonnet 5, famille differente du backbone evalue
gpt-4.1-mini) compare le message courant a la fenetre glissante des k=5
derniers messages pertinents du meme run, et rend un jugement
bool + rationale.

Necessite la variable d'environnement ANTHROPIC_API_KEY.
"""

import json
import os
import time

import anthropic

from rq3_config import RQ3_JUDGE_MODEL, RQ3_JUDGE_MAX_TOKENS, RQ3_WINDOW_SIZE

_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

_MAX_RETRIES = 4
_RETRY_BASE_DELAY_SECONDS = 2  # backoff: 2s, 4s, 8s, 16s

# Judge rubric - RQ3 plan step 1, V_live / coherence component.
# Single system prompt, parameterized by the evaluated content (unified
# with the future V_post component: only the context given in the user
# message changes).
JUDGE_SYSTEM_PROMPT = """You are an independent auditor of multi-agent systems. \
Your role is to detect FACTUAL CONTRADICTIONS between an agent's most recent \
message and the preceding messages in the same run (same task, same agents).

A contradiction = a statement that REVERSES or INVALIDATES a statement made \
earlier: a different numeric value for the same quantity, a status that \
changes without the change being justified/announced as such, a process \
order that gets reversed, a decision that contradicts an earlier decision, \
an agent that claims to be unaware of information already established in \
the preceding messages (e.g. saying "awaiting X" when X was already \
provided earlier in the window).

Do NOT count as a contradiction (even if the information is new):
- a rephrasing or summary of what was already said
- a refinement or clarification that does not reverse the meaning
- the ADDITION of a piece of data, detail, or justification that simply \
had not been mentioned before, WITHOUT any earlier statement becoming \
false as a result. Information that was absent is not information that \
was contradicted.
- an explicit, acknowledged disagreement between agents (e.g. an agent \
challenging another's assumption while flagging it as such)

The key distinction: an ADDITION leaves every earlier statement true and \
adds a new one alongside it. A CONTRADICTION makes an earlier statement \
false or obsolete. Check which of the two you are looking at before \
answering.

Respond STRICTLY in JSON, with no text before or after:
{"triggered": true|false, "rationale": "one or two factual sentences, citing the conflicting element if triggered=true"}
"""


def _format_window(window: list[dict]) -> str:
    lines = []
    for turn in window:
        lines.append(f"[{turn.get('agent_id', '?')}]: {turn.get('content', '')}")
    return "\n\n".join(lines) if lines else "(no preceding message in the window)"


def check_contradiction(current_turn: dict, preceding_turns: list[dict]) -> dict:
    """
    current_turn: {"agent_id": ..., "content": ...} - the turn to evaluate
    preceding_turns: list of preceding turns in the same run, already
                      truncated to RQ3_WINDOW_SIZE (done by the caller via
                      a deque)

    Returns {triggered: bool, judge_rationale: str}. On parsing failure or
    API error, triggered=False with the failure reason explicitly recorded
    (for auditing the judge itself, not silently swallowed).
    """
    window_text = _format_window(preceding_turns)
    current_text = f"[{current_turn.get('agent_id', '?')}]: {current_turn.get('content', '')}"

    user_prompt = f"""Preceding messages (context window, chronological order):
{window_text}

Message to evaluate:
{current_text}

Does this message factually contradict anything in the preceding messages?"""

    for attempt in range(_MAX_RETRIES):
        try:
            response = _client.messages.create(
                model=RQ3_JUDGE_MODEL,
                max_tokens=RQ3_JUDGE_MAX_TOKENS,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            # Ne pas supposer que content[0] est le bloc texte: un ThinkingBlock
            # peut le preceder. On cherche explicitement le premier bloc "text".
            text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            if not text_blocks:
                raise ValueError(f"no text block in response (block types received: {[getattr(b, 'type', None) for b in response.content]})")
            raw_text = text_blocks[0].strip()
            # tolerance in case the model still wraps the answer in ```
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`").removeprefix("json").strip()
            parsed = json.loads(raw_text)
            return {
                "triggered": bool(parsed["triggered"]),
                "judge_rationale": str(parsed["rationale"]),
            }
        except (anthropic.APIError, KeyError, ValueError, json.JSONDecodeError, IndexError) as e:
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                # Transient errors (529 overloaded, rare JSON truncation due
                # to thinking): a retry usually succeeds.
                time.sleep(_RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
                continue

    return {
        "triggered": False,
        "judge_rationale": f"JUDGE_ERROR (after {_MAX_RETRIES} attempts): {type(last_error).__name__}: {last_error}",
    }


def check_softcot_divergence(current_turn: dict, planner_strategy: str | None) -> dict:
    """
    Stage 2, softcot_divergence trigger - INACTIVE for RQ2 runs.

    RQ2 runs all use the generic gpt-4.1-mini backbone; none of them embed
    an S1/Coconut/SoftCoT planning strategy. This trigger is wired here so
    the V_live pipeline is ready to activate on future runs (once RQ1 has
    run and the prefix-ablation test is available), but it never fires on
    the current RQ2 logs.
    """
    if planner_strategy != "softcot":
        return {
            "triggered": False,
            "judge_rationale": "softcot_divergence not applicable: no SoftCoT planning strategy is associated with this run (RQ2 uses a generic backbone)",
        }
    # Anchor point for the future wiring to the RQ1 ablation test.
    raise NotImplementedError(
        "softcot_divergence will be wired to the RQ1 prefix-ablation test (RQ3 plan step 5), not yet available."
    )