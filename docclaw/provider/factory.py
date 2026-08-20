"""Construct DocClaw LLM providers from runtime config."""

from __future__ import annotations

from docclaw.config.schema import ProviderConfig, ProvidersConfig
from docclaw.provider.base import LLMProvider
from docclaw.provider.anthropic_provider import (
    AnthropicProvider,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_ANTHROPIC_URL,
    DEFAULT_ANTHROPIC_VERSION,
)
from docclaw.provider.gemini_provider import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_URL,
    GeminiProvider,
)
from docclaw.provider.openai_codex_provider import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_URL,
    DEFAULT_ORIGINATOR,
    OpenAICodexProvider,
)
from docclaw.provider.openai_provider import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_URL,
    OpenAIProvider,
)


def make_provider(config: ProviderConfig) -> LLMProvider:
    if config.name == "openai_codex":
        return OpenAICodexProvider(
            default_model=config.model or DEFAULT_CODEX_MODEL,
            url=config.url or DEFAULT_CODEX_URL,
            originator=config.originator or DEFAULT_ORIGINATOR,
            verify_ssl=config.verify_ssl,
            allow_insecure_ssl_fallback=config.allow_insecure_ssl_fallback,
            request_timeout=config.request_timeout,
        )
    if config.name == "openai":
        return OpenAIProvider(
            default_model=config.model or DEFAULT_OPENAI_MODEL,
            api_key=config.api_key or "",
            url=config.url or DEFAULT_OPENAI_URL,
            verify_ssl=config.verify_ssl,
            allow_insecure_ssl_fallback=config.allow_insecure_ssl_fallback,
            request_timeout=config.request_timeout,
        )
    if config.name == "anthropic":
        return AnthropicProvider(
            default_model=config.model or DEFAULT_ANTHROPIC_MODEL,
            api_key=config.api_key or "",
            url=config.url or DEFAULT_ANTHROPIC_URL,
            anthropic_version=config.api_version or DEFAULT_ANTHROPIC_VERSION,
            verify_ssl=config.verify_ssl,
            allow_insecure_ssl_fallback=config.allow_insecure_ssl_fallback,
            request_timeout=config.request_timeout,
        )
    if config.name == "gemini":
        return GeminiProvider(
            default_model=config.model or DEFAULT_GEMINI_MODEL,
            api_key=config.api_key or "",
            url_template=config.url or DEFAULT_GEMINI_URL,
            verify_ssl=config.verify_ssl,
            allow_insecure_ssl_fallback=config.allow_insecure_ssl_fallback,
            request_timeout=config.request_timeout,
        )
    raise ValueError(f"unsupported provider.name: {config.name}")


def make_named_provider(configs: ProvidersConfig, name: str) -> LLMProvider:
    return make_provider(configs.require(name))
