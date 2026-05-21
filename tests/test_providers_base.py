"""Tests for providers/base.py — pure contract introspection (no I/O).

Locks:
- Provider is abstract: Provider() directly raises TypeError.
- Provider.chat signature: (self, request, *, options=None, timeout_s=120.0).
  options and timeout_s are KEYWORD_ONLY (structural defense against positional
  argument swaps — same discipline as to_openai_chat_completion).
- Provider.health signature: (self, *, timeout_s=2.0). timeout_s KEYWORD_ONLY.
- Provider.warmup signature: (self, *, timeout_s=300.0). timeout_s KEYWORD_ONLY.
- ChatOptions is a TypedDict(total=False) with exactly 8 keys: num_ctx,
  keep_alive, format, temperature, top_p, seed, num_predict, stop (D-05).
- ProviderHealth is a TypedDict(total=True) with exactly 3 keys: kind, state,
  last_error (D-11).
- The 'name' class attribute is annotated as str (concrete subclass sets it).
"""

from __future__ import annotations

import inspect
import typing

import pytest

from corethread.providers.base import ChatOptions, Provider, ProviderHealth


def test_provider_abc_rejects_instantiation() -> None:
    """Provider is abstract — direct instantiation must raise TypeError."""
    with pytest.raises(TypeError) as info:
        Provider()  # type: ignore[abstract]
    assert "abstract" in str(info.value).lower()


def test_provider_has_name_annotation_str() -> None:
    """D-04: subclasses set ``name: str`` as a class attribute (e.g. ``name = "ollama"``)."""
    # name is type-annotated in the ABC (not assigned a value). Subclasses assign.
    # providers/base.py uses ``from __future__ import annotations``, so
    # ``__annotations__`` values are ForwardRef strings. Resolve to runtime
    # types with typing.get_type_hints.
    hints = typing.get_type_hints(Provider)
    assert "name" in hints
    assert hints["name"] is str


def test_provider_chat_signature() -> None:
    """D-04 + D-06: chat has (self, request, *, options=None, timeout_s=120.0)
    with options + timeout_s keyword-only."""
    sig = inspect.signature(Provider.chat)
    params = sig.parameters
    # self + request + * + options + timeout_s
    assert list(params.keys()) == ["self", "request", "options", "model_override", "timeout_s"]
    assert params["request"].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    )
    assert params["options"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["timeout_s"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["options"].default is None
    assert params["timeout_s"].default == 120.0


def test_provider_health_signature() -> None:
    """D-06: health has (self, *, timeout_s=2.0) with timeout_s keyword-only."""
    sig = inspect.signature(Provider.health)
    params = sig.parameters
    assert list(params.keys()) == ["self", "timeout_s"]
    assert params["timeout_s"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["timeout_s"].default == 2.0


def test_provider_warmup_signature() -> None:
    """D-06: warmup has (self, *, timeout_s=300.0) with timeout_s keyword-only."""
    sig = inspect.signature(Provider.warmup)
    params = sig.parameters
    assert list(params.keys()) == ["self", "timeout_s"]
    assert params["timeout_s"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["timeout_s"].default == 300.0


def test_provider_methods_are_async() -> None:
    """D-04: every abstract method is async def."""
    assert inspect.iscoroutinefunction(Provider.chat)
    assert inspect.iscoroutinefunction(Provider.health)
    assert inspect.iscoroutinefunction(Provider.warmup)


def test_provider_methods_are_abstract() -> None:
    """D-04: chat/health/warmup are all @abstractmethod."""
    assert getattr(Provider.chat, "__isabstractmethod__", False) is True
    assert getattr(Provider.health, "__isabstractmethod__", False) is True
    assert getattr(Provider.warmup, "__isabstractmethod__", False) is True


def test_chat_options_is_total_false_with_eight_keys() -> None:
    """D-05: ChatOptions is TypedDict(total=False) with exactly 8 keys."""
    assert ChatOptions.__total__ is False
    expected = {
        "num_ctx",
        "keep_alive",
        "format",
        "temperature",
        "top_p",
        "seed",
        "num_predict",
        "stop",
    }
    assert set(ChatOptions.__annotations__.keys()) == expected


def test_chat_options_key_types() -> None:
    """D-05: key types match the documented contract."""
    # providers/base.py uses ``from __future__ import annotations``, so
    # ChatOptions.__annotations__ values are strings. Use typing.get_type_hints
    # to resolve to real runtime types (Union types surface as types.UnionType).
    anns = typing.get_type_hints(ChatOptions)
    # num_ctx is plain int
    assert anns["num_ctx"] is int
    # keep_alive is ``int | str`` — the annotation resolves to a types.UnionType;
    # verifying structurally via repr is the stable check (avoids 3.9/3.10/3.11 Union API drift).
    assert "int" in repr(anns["keep_alive"]) and "str" in repr(anns["keep_alive"])
    # format is ``dict | str``
    assert "dict" in repr(anns["format"]) and "str" in repr(anns["format"])
    # temperature / top_p are float
    assert anns["temperature"] is float
    assert anns["top_p"] is float
    # seed / num_predict are int
    assert anns["seed"] is int
    assert anns["num_predict"] is int


def test_provider_health_is_total_true_with_three_keys() -> None:
    """D-11: ProviderHealth is TypedDict(total=True) with kind, state, last_error."""
    assert ProviderHealth.__total__ is True
    assert set(ProviderHealth.__annotations__.keys()) == {"kind", "state", "last_error"}


def test_concrete_subclass_can_be_constructed() -> None:
    """A minimal concrete subclass (overriding all three abstract methods) is
    instantiable — proves the ABC contract is satisfiable."""

    class MockProvider(Provider):
        name = "mock"

        async def chat(self, request, *, options=None, timeout_s=120.0):  # type: ignore[override]
            raise NotImplementedError

        async def health(self, *, timeout_s=2.0):  # type: ignore[override]
            raise NotImplementedError

        async def warmup(self, *, timeout_s=300.0):  # type: ignore[override]
            raise NotImplementedError

    p = MockProvider()
    assert p.name == "mock"


def test_concrete_subclass_missing_method_still_abstract() -> None:
    """If a subclass fails to override any abstract method, instantiation
    still raises TypeError — this is what makes the ABC load-bearing."""

    class HalfProvider(Provider):
        name = "half"

        async def chat(self, request, *, options=None, timeout_s=120.0):  # type: ignore[override]
            return None  # type: ignore[return-value]

        # health and warmup intentionally NOT overridden

    with pytest.raises(TypeError):
        HalfProvider()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Phase 3 / Plan 01 (D-09 / JDG-06) — model_override kwarg contract tests
# ---------------------------------------------------------------------------


def test_provider_chat_signature_includes_model_override() -> None:
    """D-09: Provider.chat must have a KEYWORD_ONLY `model_override: str | None`
    parameter with default None, positioned between `options` and `timeout_s`.

    Closes JDG-06 at the ABC layer — the Judge (Phase 3) dispatches against
    cfg.judge.model via this kwarg, reusing the same OllamaProvider instance
    + shared httpx.AsyncClient (Pitfall #11).
    """
    sig = inspect.signature(Provider.chat)
    params = sig.parameters
    # Order is load-bearing: options -> model_override -> timeout_s.
    assert list(params.keys()) == ["self", "request", "options", "model_override", "timeout_s"], (
        f"unexpected order: {list(params.keys())}"
    )
    assert params["model_override"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["model_override"].default is None


def test_provider_chat_model_override_type_hint_is_optional_str() -> None:
    """D-09: the model_override annotation resolves to str | None at runtime
    (PEP-604 union). Future refactors that change the type (e.g. to a
    NewType wrapper) break this assertion — intentional: the ABC contract
    is a plain optional string.

    NOTE: `typing.get_type_hints` with `from __future__ import annotations`
    in providers/base.py resolves the `str | None` PEP-604 annotation into
    a `types.UnionType` at runtime (3.10+). We structurally verify the shape
    via `typing.get_args` rather than comparing to `typing.Optional[str]`
    directly — the equivalence is true but ruff UP045 flags the literal form.
    """
    hints = typing.get_type_hints(Provider.chat)
    # typing.get_args(str | None) == (str, type(None)) — stable across 3.10+.
    assert set(typing.get_args(hints["model_override"])) == {str, type(None)}
