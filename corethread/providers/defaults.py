"""Provider wrapper that applies model-profile generation defaults."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from corethread.config import ModelProfile
from corethread.models import ChatCompletionRequest, ChatCompletionResponse
from corethread.providers.base import ChatOptions, Provider, ProviderHealth

__all__ = ["ProfileDefaultsProvider", "apply_profile_defaults"]


def _with_profile_defaults(
    profile: ModelProfile,
    request: ChatCompletionRequest,
    options: ChatOptions | None,
    timeout_s: float,
) -> tuple[ChatCompletionRequest, ChatOptions | None, float]:
    update: dict[str, Any] = {}
    for key in ("temperature", "top_p", "max_tokens", "stop", "seed"):
        profile_value = getattr(profile, key, None)
        if getattr(request, key, None) is None and profile_value is not None:
            update[key] = profile_value
    request_with_defaults = request.model_copy(update=update) if update else request

    merged_options: ChatOptions = dict(options or {})  # type: ignore[assignment]
    for key in ("temperature", "top_p", "seed", "stop"):
        value = getattr(profile, key, None)
        if value is not None and key not in merged_options:
            merged_options[key] = value  # type: ignore[literal-required]

    effective_timeout = profile.timeout_s or timeout_s
    return request_with_defaults, merged_options or None, effective_timeout


def apply_profile_defaults(provider: Provider, profile: ModelProfile) -> Provider:
    """Patch provider.chat in place so isinstance(provider, ConcreteProvider) holds."""

    original_chat = provider.chat
    original_stream_chat = getattr(provider, "stream_chat", None)

    async def chat(
        request: ChatCompletionRequest,
        *,
        options: ChatOptions | None = None,
        model_override: str | None = None,
        timeout_s: float = 120.0,
    ) -> ChatCompletionResponse:
        request_with_defaults, merged_options, effective_timeout = _with_profile_defaults(
            profile,
            request,
            options,
            timeout_s,
        )
        return await original_chat(
            request_with_defaults,
            options=merged_options,
            model_override=model_override,
            timeout_s=effective_timeout,
        )

    provider.chat = chat  # type: ignore[method-assign]
    if callable(original_stream_chat):

        async def stream_chat(
            request: ChatCompletionRequest,
            *,
            options: ChatOptions | None = None,
            model_override: str | None = None,
            timeout_s: float = 120.0,
            on_final: Callable[[ChatCompletionResponse], None] | None = None,
        ) -> AsyncIterator[dict[str, Any]]:
            request_with_defaults, merged_options, effective_timeout = (
                _with_profile_defaults(
                    profile,
                    request,
                    options,
                    timeout_s,
                )
            )
            async for chunk in original_stream_chat(
                request_with_defaults,
                options=merged_options,
                model_override=model_override,
                timeout_s=effective_timeout,
                on_final=on_final,
            ):
                yield chunk

        provider.stream_chat = stream_chat  # type: ignore[attr-defined]
    return provider


class ProfileDefaultsProvider(Provider):
    """Apply profile defaults before delegating to a concrete provider."""

    def __init__(self, provider: Provider, profile: ModelProfile) -> None:
        self._provider = provider
        self._profile = profile
        self.name = provider.name

    async def chat(
        self,
        request: ChatCompletionRequest,
        *,
        options: ChatOptions | None = None,
        model_override: str | None = None,
        timeout_s: float = 120.0,
    ) -> ChatCompletionResponse:
        request_with_defaults, merged_options, effective_timeout = _with_profile_defaults(
            self._profile,
            request,
            options,
            timeout_s,
        )
        return await self._provider.chat(
            request_with_defaults,
            options=merged_options or None,
            model_override=model_override,
            timeout_s=effective_timeout,
        )

    async def health(self, *, timeout_s: float = 2.0) -> ProviderHealth:
        return await self._provider.health(timeout_s=timeout_s)

    async def warmup(self, *, timeout_s: float = 300.0) -> None:
        await self._provider.warmup(timeout_s=timeout_s)
