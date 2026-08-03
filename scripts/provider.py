"""LLM provider abstraction for multi-agent evaluation."""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Environment variables can still be supplied directly; dotenv is a convenience.
    pass


@dataclass
class Message:
    """A single chat message."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None  # Provider-specific raw response

def _summarize_response(response: Any, max_chars: int = 1000) -> str:
    """Return a compact representation of an SDK response for error messages."""

    try:
        if hasattr(response, "model_dump_json"):
            text = response.model_dump_json()
        elif hasattr(response, "model_dump"):
            text = str(response.model_dump())
        else:
            text = repr(response)
    except Exception:
        text = repr(response)
    return text if len(text) <= max_chars else text[:max_chars] + "...<truncated>"


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Subclass this to add new providers (Anthropic, local models, etc.).
    """

    @abstractmethod
    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: List of Message objects (system, user, assistant).
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.).
                response_schema: Optional dict with keys ``name`` and ``schema``
                    (JSON Schema). When provided, providers that support native
                    structured output (OpenAI, OpenRouter) constrain generation
                    to that schema. Providers without native support silently
                    ignore this kwarg — callers relying on RQ2's schema_mode
                    should therefore pair it with a provider that honors it.

        Returns:
            LLMResponse with the assistant's reply.
        """
        ...

    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g., 'openai', 'anthropic')."""
        ...


# ---------------------------------------------------------------------------
# RQ2 typed-schema support (topology x message-schema experiment)
# ---------------------------------------------------------------------------

#: Default structured-output schema for the "typed" condition of RQ2.
#: `findings` is the designated free-text field: RTD tracer/anchor strings
#: must land here so that `str(content)`-based detection in rtd.py still
#: finds them after JSON serialization.
RQ2_TYPED_RESPONSE_SCHEMA: dict[str, Any] = {
    "name": "agent_turn_response",
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Brief reasoning for this turn's contribution.",
            },
            "findings": {
                "type": "string",
                "description": (
                    "The substantive content of this turn: analysis, evidence, "
                    "proposed edits, or answer text. Preserve any identifiers, "
                    "constraints, or references mentioned earlier in the "
                    "conversation verbatim."
                ),
            },
            "decision": {
                "type": "string",
                "description": "This agent's concrete decision or recommendation, if any.",
            },
            "confidence": {
                "type": ["number", "null"],
                "description": (
                    "Self-reported confidence in [0, 1], or null if the agent "
                    "does not have a meaningful confidence estimate for this turn."
                ),
            },
        },
        "required": ["reasoning", "findings", "decision", "confidence"],
        "additionalProperties": False,
    },
}

#: Provider identifiers known to honor `response_schema` natively. Used only
#: for a best-effort warning in TaskRunner — instantiating schema_mode="typed"
#: with a provider outside this set will NOT raise, it will silently produce
#: free-text content, contaminating the "typed" condition. Extend this set
#: as more providers gain native json_schema/strict support.
SCHEMA_MODE_SUPPORTED_PROVIDERS: set[str] = {"openai", "openrouter"}


def build_openai_compatible_response_format(response_schema: dict[str, Any]) -> dict[str, Any]:
    """Build a Chat-Completions-style `response_format` payload from a schema.

    Shared by any OpenAI-compatible provider (OpenAI itself, OpenRouter,
    and OpenAI-compatible gateways) that supports the `json_schema`
    structured-output mode.

    Args:
        response_schema: dict with `name` (str) and `schema` (JSON Schema dict).

    Returns:
        A `response_format` dict ready to pass to `chat.completions.create`.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_schema.get("name", "agent_turn_response"),
            "schema": response_schema["schema"],
            "strict": True,
        },
    }


class OpenAIProvider(LLMProvider):
    """OpenAI chat completion provider."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        import openai

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = openai.OpenAI(api_key=api_key)

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        model = kwargs.get("model", self.model)

        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response_schema = kwargs.get("response_schema")
        if response_schema:
            request_kwargs["response_format"] = build_openai_compatible_response_format(
                response_schema
            )

        response = self._client.chat.completions.create(**request_kwargs)

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            raw=response,
        )

    def provider_name(self) -> str:
        return "openai"


class AnthropicProvider(LLMProvider):
    """Anthropic chat completion provider (Claude models)."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. "
                "Install it with: pip install anthropic"
            ) from exc

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        model = kwargs.get("model", self.model)

        # Separate system prompt from the rest
        system_content = ""
        api_messages = []
        for m in messages:
            if m.role == "system":
                system_content = m.content
            else:
                api_messages.append({"role": m.role, "content": m.content})

        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if system_content:
            create_kwargs["system"] = system_content
        # Anthropic temperature range is 0–1
        create_kwargs["temperature"] = min(max(float(temperature), 0.0), 1.0)

        response = self._client.messages.create(**create_kwargs)

        content = ""
        if response.content:
            content = response.content[0].text if hasattr(response.content[0], "text") else ""

        usage = response.usage
        return LLMResponse(
            content=content,
            model=response.model,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            raw=response,
        )

    def provider_name(self) -> str:
        return "anthropic"


class GoogleProvider(LLMProvider):
    """Google Generative AI provider (Gemini models) — legacy google.generativeai SDK."""

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai package is required for GoogleProvider. "
                "Install it with: pip install google-generativeai"
            ) from exc

        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if api_key:
            genai.configure(api_key=api_key)

        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        self._genai = genai
        self._model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
        )

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        # Build Gemini-style history from messages
        system_parts = []
        history = []
        last_user_content = ""

        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "user":
                last_user_content = m.content
                if history or not system_parts:
                    history.append({"role": "user", "parts": [m.content]})
            elif m.role == "assistant":
                history.append({"role": "model", "parts": [m.content]})

        # Prepend system prompt to the first user message if present
        if system_parts and history:
            system_text = "\n\n".join(system_parts)
            first_user = history[0]
            if first_user["role"] == "user":
                first_user["parts"] = [system_text + "\n\n" + first_user["parts"][0]]
        elif system_parts:
            last_user_content = "\n\n".join(system_parts) + "\n\n" + last_user_content

        chat = self._model.start_chat(history=history[:-1] if len(history) > 1 else [])
        send_content = history[-1]["parts"][0] if history else last_user_content
        response = chat.send_message(send_content)

        content = response.text if hasattr(response, "text") else ""
        usage = getattr(response, "usage_metadata", None)

        return LLMResponse(
            content=content,
            model=self.model_name,
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            raw=response,
        )

    def provider_name(self) -> str:
        return "google"


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the new google.genai SDK (recommended)."""

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite-preview",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        try:
            import google.genai as genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise ImportError(
                "google-genai package is required for GeminiProvider. "
                "Install it with: pip install google-genai"
            ) from exc

        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._genai_types = genai_types

        import os

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(api_key=resolved_key)

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        from google.genai import types as genai_types

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Separate system instructions from the conversation
        system_parts: list[str] = []
        contents: list[Any] = []

        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "user":
                contents.append(
                    genai_types.Content(role="user", parts=[genai_types.Part(text=m.content)])
                )
            elif m.role == "assistant":
                contents.append(
                    genai_types.Content(role="model", parts=[genai_types.Part(text=m.content)])
                )

        config = genai_types.GenerateContentConfig(
            system_instruction="\n\n".join(system_parts) if system_parts else None,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        content = response.text or ""
        usage = getattr(response, "usage_metadata", None)

        return LLMResponse(
            content=content,
            model=self.model_name,
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            raw=response,
        )

    def provider_name(self) -> str:
        return "gemini"


class OllamaProvider(LLMProvider):
    """OpenAI-compatible provider for local models (Ollama, vLLM, LM Studio, etc.)."""

    def __init__(
        self,
        model: str = "llama3.1:70b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        import openai

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        model = kwargs.get("model", self.model)

        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        response = self._client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            raw=response,
        )

    def provider_name(self) -> str:
        return "ollama"


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider — OpenAI-compatible gateway to open-weight and proprietary models.

    Uses the OpenAI SDK pointed at https://openrouter.ai/api/v1.
    API key is read from the OPENROUTER_API_KEY environment variable or passed directly.

    Example models:
        qwen/qwen3.5-35b-a3b
        meta-llama/llama-3.1-70b-instruct
        mistralai/mixtral-8x7b-instruct
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str = "qwen/qwen3.5-35b-a3b",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        site_url: str = "https://github.com/agentcollabbench/agentcollabbench",
        site_name: str = "AgentCollabBench",
        provider_routing: dict[str, Any] | None = None,
    ):
        import os

        import openai

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider_routing = provider_routing

        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY in your .env "
                "or pass api_key= to OpenRouterProvider."
            )

        self._client = openai.OpenAI(
            base_url=self.OPENROUTER_BASE_URL,
            api_key=resolved_key,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            },
        )

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        model = kwargs.get("model", self.model)

        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        provider_routing = kwargs.get("provider_routing", self.provider_routing)
        if provider_routing:
            request_kwargs["extra_body"] = {"provider": provider_routing}

        response_schema = kwargs.get("response_schema")
        if response_schema:
            # Note: not every OpenRouter-routed model honors json_schema mode.
            # Unsupported models typically ignore the field or degrade to
            # plain text; check the model's OpenRouter capability page if
            # typed-mode outputs come back unstructured.
            request_kwargs["response_format"] = build_openai_compatible_response_format(
                response_schema
            )

        rate_limit_retries = int(
            kwargs.get("rate_limit_retries", os.environ.get("OPENROUTER_RATE_LIMIT_RETRIES", 60))
        )
        retry_base_seconds = float(
            kwargs.get(
                "rate_limit_retry_base_seconds",
                os.environ.get("OPENROUTER_RATE_LIMIT_RETRY_BASE_SECONDS", 2.0),
            )
        )
        retry_max_seconds = float(
            kwargs.get(
                "rate_limit_retry_max_seconds",
                os.environ.get("OPENROUTER_RATE_LIMIT_RETRY_MAX_SECONDS", 30.0),
            )
        )

        response = None
        for attempt in range(rate_limit_retries + 1):
            try:
                response = self._client.chat.completions.create(**request_kwargs)
                break
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                message = str(exc).lower()
                is_rate_limit = (
                    status_code == 429
                    or status_code == 502
                    or "429" in message
                    or "502" in message
                    or "rate limit" in message
                    or "rate-limited" in message
                    or "capacity exceeded" in message
                )
                if not is_rate_limit or attempt >= rate_limit_retries:
                    raise

                delay = min(retry_max_seconds, retry_base_seconds * (2 ** min(attempt, 4)))
                logger.warning(
                    "OpenRouter rate limit for %s; retrying same turn in %.1fs (%d/%d)",
                    model,
                    delay,
                    attempt + 1,
                    rate_limit_retries,
                )
                time.sleep(delay)

        if response is None:
            raise RuntimeError("OpenRouter request did not return a response")

        choices = getattr(response, "choices", None)
        if not choices:
            error_payload = getattr(response, "error", None)
            error_summary = _summarize_response(error_payload or response)
            retryable_empty_response = any(
                marker in error_summary.lower()
                for marker in ("429", "502", "rate limit", "rate-limited", "capacity exceeded")
            )
            if retryable_empty_response:
                for attempt in range(rate_limit_retries):
                    delay = min(retry_max_seconds, retry_base_seconds * (2 ** min(attempt, 4)))
                    logger.warning(
                        "OpenRouter empty retryable response for %s; retrying same turn in %.1fs (%d/%d): %s",
                        model,
                        delay,
                        attempt + 1,
                        rate_limit_retries,
                        error_summary,
                    )
                    time.sleep(delay)
                    try:
                        response = self._client.chat.completions.create(**request_kwargs)
                    except Exception as exc:
                        status_code = getattr(exc, "status_code", None)
                        message = str(exc).lower()
                        retryable_exc = (
                            status_code == 429
                            or status_code == 502
                            or "429" in message
                            or "502" in message
                            or "rate limit" in message
                            or "rate-limited" in message
                            or "capacity exceeded" in message
                        )
                        if retryable_exc:
                            error_summary = str(exc)
                            continue
                        raise
                    choices = getattr(response, "choices", None)
                    if choices:
                        break
                    error_payload = getattr(response, "error", None)
                    error_summary = _summarize_response(error_payload or response)
                if not choices:
                    raise RuntimeError(
                        "OpenRouter returned no chat choices after retries "
                        f"(model={getattr(response, 'model', model)!r}, "
                        f"provider={getattr(response, 'provider', None)!r}, "
                        f"response={_summarize_response(response)})"
                    )
            else:
                raise RuntimeError(
                    "OpenRouter returned no chat choices "
                    f"(model={getattr(response, 'model', model)!r}, "
                    f"provider={getattr(response, 'provider', None)!r}, "
                    f"response={_summarize_response(response)})"
                )

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise RuntimeError(
                "OpenRouter returned a choice without a message "
                f"(model={getattr(response, 'model', model)!r}, "
                f"provider={getattr(response, 'provider', None)!r}, "
                f"choice={_summarize_response(choice)})"
            )

        content = getattr(message, "content", "") or ""
        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=content if isinstance(content, str) else str(content),
            model=getattr(response, "model", model),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            raw=response,
        )

    def provider_name(self) -> str:
        return "openrouter"


class KimiProvider(LLMProvider):
    """Moonshot Kimi API (OpenAI-compatible).

    Default endpoint is the international gateway ``https://api.moonshot.ai/v1``.
    For China mainland, use ``MOONSHOT_BASE_URL=https://api.moonshot.cn/v1`` or pass
    ``base_url=`` to ``create_provider``.

    API key: ``MOONSHOT_API_KEY`` or ``KIMI_API_KEY`` (used when ``api_key`` is not
    passed explicitly).

    Recommended model for live runs is ``kimi-k2.6``. For Kimi ``kimi-k2*`` chat
    models, requests send ``extra_body={"thinking": {"type": "disabled"}}`` by
    default so completions behave like standard chat APIs; pass
    ``disable_thinking=False`` to keep native reasoning if needed.

    The Moonshot API rejects arbitrary ``temperature`` values for ``kimi-k2*``:
    only ``0.6`` is accepted; this provider forces that value in ``chat()`` for
    those models (callers' ``--temperature`` is ignored for them).
    """

    DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
    # Moonshot returns 400 if temperature != 0.6 for kimi-k2* chat models.
    KIMI_K2_FIXED_TEMPERATURE: float = 0.6

    def __init__(
        self,
        model: str = "kimi-k2.6",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.6,
        max_tokens: int = 8192,
        disable_thinking: bool = True,
    ) -> None:
        import openai

        resolved_key = (
            api_key
            or os.environ.get("MOONSHOT_API_KEY")
            or os.environ.get("KIMI_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "Kimi/Moonshot API key not found. Set MOONSHOT_API_KEY or KIMI_API_KEY "
                "in your environment, or pass api_key= to KimiProvider."
            )

        resolved_url = (
            (base_url or os.environ.get("MOONSHOT_BASE_URL") or self.DEFAULT_BASE_URL).rstrip(
                "/"
            )
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        self._client = openai.OpenAI(base_url=resolved_url, api_key=resolved_key)

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        temperature = float(kwargs.get("temperature", self.temperature))
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        model = kwargs.get("model", self.model)
        disable_thinking = kwargs.get("disable_thinking", self.disable_thinking)

        if isinstance(model, str) and model.startswith("kimi-k2"):
            temperature = self.KIMI_K2_FIXED_TEMPERATURE

        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        want_disable_thinking = (
            disable_thinking and isinstance(model, str) and model.startswith("kimi-k2")
        )
        if want_disable_thinking:
            request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        response = self._client.chat.completions.create(**request_kwargs)

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            raw=response,
        )

    def provider_name(self) -> str:
        return "kimi"


class DeepSeekProvider(LLMProvider):
    """DeepSeek API (OpenAI-compatible chat completions).

    Base URL: ``https://api.deepseek.com``. API key: ``DEEPSEEK_API_KEY``.
    Override with ``DEEPSEEK_BASE_URL`` if needed.

    V4 models enable "thinking" mode by default; for deterministic judge-style
    outputs this provider sends ``thinking: {type: disabled}`` when
    ``disable_thinking`` is True (default) and the model id starts with
    ``deepseek-v4``.
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        disable_thinking: bool = True,
    ) -> None:
        import openai

        resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not resolved_key:
            raise ValueError(
                "DeepSeek API key not found. Set DEEPSEEK_API_KEY in your environment "
                "or pass api_key= to DeepSeekProvider."
            )
        resolved_url = (
            base_url or os.environ.get("DEEPSEEK_BASE_URL") or self.DEFAULT_BASE_URL
        ).rstrip("/")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        self._client = openai.OpenAI(base_url=resolved_url, api_key=resolved_key)

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        temperature = float(kwargs.get("temperature", self.temperature))
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        model_arg = kwargs.get("model", self.model)
        model = model_arg
        disable_thinking = kwargs.get("disable_thinking", self.disable_thinking)

        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if disable_thinking and isinstance(model, str) and model.startswith("deepseek-v4"):
            request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        response = self._client.chat.completions.create(**request_kwargs)

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model or model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            raw=response,
        )

    def provider_name(self) -> str:
        return "deepseek"


class GroqProvider(LLMProvider):
    """GroqCloud chat API (OpenAI-compatible).

    Endpoint: ``https://api.groq.com/openai/v1``. API key: ``GROQ_API_KEY``.

    Groq lists Llama 3.1 8B as ``llama-3.1-8b-instant``. The HuggingFace-style id
    ``meta-llama/llama-3.1-8b-instruct`` is accepted and mapped to that model id.
    """

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    _MODEL_ALIASES: dict[str, str] = {
        "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b-instant",
        "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b-instant",
    }

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        import openai

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY in your environment "
                "or pass api_key= to GroqProvider."
            )
        resolved_url = (
            base_url or os.environ.get("GROQ_BASE_URL") or self.DEFAULT_BASE_URL
        ).rstrip("/")

        self.model = self._MODEL_ALIASES.get(model, model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = openai.OpenAI(base_url=resolved_url, api_key=resolved_key)

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        model_arg = kwargs.get("model", self.model)
        model = self._MODEL_ALIASES.get(model_arg, model_arg)

        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        response = self._client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model or model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            raw=response,
        )

    def provider_name(self) -> str:
        return "groq"


# Registry of available providers for convenience
PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
    "openrouter": OpenRouterProvider,
    "kimi": KimiProvider,
    "moonshot": KimiProvider,
    "groq": GroqProvider,
    "deepseek": DeepSeekProvider,
}


def create_provider(name: str, **kwargs: Any) -> LLMProvider:
    """Factory to create a provider by name.

    Args:
        name: Provider name — one of:
            'openai'      → OpenAI API (gpt-4.1-mini, gpt-4o, …)
            'anthropic'   → Anthropic API (claude-* models)
            'google'      → Legacy google-generativeai SDK (Gemini)
            'gemini'      → New google-genai SDK (gemini-2.5-flash-lite, …)
            'ollama'      → Local OpenAI-compatible endpoint
            'openrouter'  → OpenRouter gateway (qwen/qwen3.5-35b-a3b, …)
            'kimi'/'moonshot' → Moonshot Kimi API OpenAI-compat (MOONSHOT_API_KEY; kimi-k2.6)
            'groq'        → GroqCloud OpenAI-compat (GROQ_API_KEY; llama-3.1-8b-instant, …)
            'deepseek'    → DeepSeek API OpenAI-compat (DEEPSEEK_API_KEY; deepseek-v4-flash, …)
        **kwargs: Passed to the provider constructor (model, api_key, temperature, …).

    Note:
        In the paper protocol, proprietary backbones use native ``openai`` / ``gemini``;
        open-weight backbones (Qwen, Llama) use ``openrouter`` only.

    Returns:
        An initialized LLMProvider instance.
    """
    if name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    return PROVIDERS[name](**kwargs)


# LLM judge for CPR and IDR only.
CPR_IDR_JUDGE_PROVIDER = "deepseek"
CPR_IDR_JUDGE_MODEL = "deepseek-v4-flash"
# Held-out / sensitivity judge (DeepSeek API model id).
DEEPSEEK_V4_FLASH_MODEL = "deepseek-v4-flash"
_cpr_idr_judge_provider_cache: dict[tuple[str, str, str], LLMProvider] = {}


def get_cpr_idr_judge_provider(
    provider_name: str | None = None,
    model_name: str | None = None,
    provider_routing: dict[str, Any] | None = None,
) -> LLMProvider:
    """Return a shared provider for CPR/IDR ``judge_provider``.

    Defaults to the paper judge provider/model unless overridden by
    ``AGENTCOLLABBENCH_JUDGE_PROVIDER`` and ``AGENTCOLLABBENCH_JUDGE_MODEL``.
    """
    provider_name = (
        provider_name or os.environ.get("AGENTCOLLABBENCH_JUDGE_PROVIDER") or CPR_IDR_JUDGE_PROVIDER
    )
    model_name = model_name or os.environ.get("AGENTCOLLABBENCH_JUDGE_MODEL") or CPR_IDR_JUDGE_MODEL
    route_key = json.dumps(provider_routing, sort_keys=True) if provider_routing else ""
    key = (provider_name, model_name, route_key)
    if key not in _cpr_idr_judge_provider_cache:
        kwargs: dict[str, Any] = {"model": model_name}
        if provider_name == "openai":
            kwargs["api_key"] = os.environ.get("OPENAI_API_KEY")
        elif provider_name in {"google", "gemini"}:
            kwargs["api_key"] = os.environ.get("GEMINI_API_KEY")
        elif provider_name == "anthropic":
            kwargs["api_key"] = os.environ.get("ANTHROPIC_API_KEY")
        elif provider_name == "openrouter":
            kwargs["api_key"] = os.environ.get("OPENROUTER_API_KEY")
            if provider_routing:
                kwargs["provider_routing"] = provider_routing
        elif provider_name == "groq":
            kwargs["api_key"] = os.environ.get("GROQ_API_KEY")
        elif provider_name in ("kimi", "moonshot"):
            kwargs["api_key"] = (
                os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY")
            )
        elif provider_name == "deepseek":
            kwargs["api_key"] = os.environ.get("DEEPSEEK_API_KEY")

        _cpr_idr_judge_provider_cache[key] = create_provider(provider_name, **kwargs)
    return _cpr_idr_judge_provider_cache[key]
