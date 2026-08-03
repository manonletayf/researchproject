"""Central configuration for the RQ2 experiment (topology x message-schema).

Why this file exists
---------------------
agentcollabbench.harness.provider.PROVIDERS classes each ship with their own
generic fallback `model` default (e.g. OpenAIProvider defaults to
"gpt-4o-mini"). Those defaults are NOT tied to any specific paper's protocol
-- they exist so each provider class works standalone. Relying on them is
exactly how gpt-4o-mini slipped into early RQ2 test commands instead of the
protocol's gpt-4.1-mini.

Rule going forward: every call to `create_provider(...)` in scripts/ must
pass `model=` explicitly, sourced from the constants below -- never from a
harness default. Import from this module, don't hardcode model strings in
individual scripts.

Source of truth: AgentCollabBench/README.md, paper-protocol section.
"""

from __future__ import annotations

from typing import Any

from agentcollabbench.harness.provider import LLMProvider, create_provider

# ---------------------------------------------------------------------------
# Paper-protocol backbones
# ---------------------------------------------------------------------------

#: Proprietary backbone used for the RQ2 pilot and main runs (native OpenAI API).
RQ2_BACKBONE_MODEL: str = "gpt-4.1-mini"
RQ2_BACKBONE_PROVIDER: str = "openai"

#: Secondary proprietary backbone from the paper protocol (native Gemini API),
#: kept here for when/if RQ2 adds a cross-backbone robustness check.
RQ2_SECONDARY_BACKBONE_MODEL: str = "gemini-2.5-flash-lite"
RQ2_SECONDARY_BACKBONE_PROVIDER: str = "gemini"

#: Open-weight backbones, routed only through OpenRouter per the paper protocol.
#: Not part of the RQ2 3x2 design as currently scoped -- listed for reuse if
#: an open-weight replication arm gets added later.
RQ2_OPENWEIGHT_MODELS: dict[str, str] = {
    "qwen3.5-35b": "qwen/qwen3.5-35b-a3b",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
}
RQ2_OPENWEIGHT_PROVIDER: str = "openrouter"

#: Judge model for any LLM-as-judge scoring (mirrors AGENTCOLLABBENCH_JUDGE_MODEL
#: in the README) -- centralized here too so judge calls never drift from the
#: main backbone by accident.
RQ2_JUDGE_MODEL: str = "gpt-4.1-mini"


def get_rq2_provider(**overrides: Any) -> LLMProvider:
    """Create the standard RQ2 backbone provider (gpt-4.1-mini via OpenAI).

    Always prefer this over calling `create_provider("openai", ...)` directly
    in scripts/ -- it guarantees the protocol model even if you forget to
    pass `model=`, and its default only changes here, once, if the protocol
    backbone is ever revised.

    Args:
        **overrides: Passed through to create_provider, can override
            `model`, `temperature`, `max_tokens`, `api_key`, etc. Use
            sparingly and document why in the calling script if you do.

    Returns:
        An initialized LLMProvider using the RQ2 protocol backbone.
    """
    kwargs: dict[str, Any] = {"model": RQ2_BACKBONE_MODEL}
    kwargs.update(overrides)
    return create_provider(RQ2_BACKBONE_PROVIDER, **kwargs)