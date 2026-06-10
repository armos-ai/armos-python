# SPDX-License-Identifier: MIT
from types import SimpleNamespace
from typing import Any, Literal
from ..guard import Armos
from .base import (
    _MaskingMixin, _AsyncMaskingMixin,
    _StreamingDemasker, _ArmosAnthropicAsyncStream,
    SYSTEM_HINT,
)


class _ArmosAnthropicStream:
    """
    Wraps an Anthropic Stream[MessageStreamEvent].
    Demaskes PII tokens in content_block_delta text events, handling tokens
    split across multiple chunks. Flushes buffer before content_block_stop.
    """

    def __init__(self, stream: Any, demask_fn):
        self._stream = stream
        self._demasker = _StreamingDemasker(demask_fn)

    def __iter__(self):
        for event in self._stream:
            event_type = getattr(event, "type", None)

            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta and getattr(delta, "type", None) == "text_delta":
                    text = getattr(delta, "text", "") or ""
                    safe = self._demasker.feed(text)
                    if safe:
                        delta.text = safe
                        yield event
                    # else: buffering — skip
                else:
                    yield event

            elif event_type == "content_block_stop":
                remaining = self._demasker.flush()
                if remaining:
                    # Emit a synthetic text_delta before the stop event
                    yield SimpleNamespace(
                        type="content_block_delta",
                        index=getattr(event, "index", 0),
                        delta=SimpleNamespace(type="text_delta", text=remaining),
                    )
                yield event

            else:
                yield event

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args):
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class ArmosMessages(_MaskingMixin):
    """Wraps anthropic.resources.Messages. Intercepts create() only."""

    def __init__(self, messages: Any, guard: Armos):
        self._messages = messages
        self._guard = guard

    def create(self, **kwargs) -> Any:
        has_pii = False
        if "messages" in kwargs:
            kwargs["messages"], has_pii = self._mask_messages(kwargs["messages"])

        if has_pii:
            existing = kwargs.get("system") or ""
            kwargs["system"] = (existing + "\n\n" + SYSTEM_HINT).strip() if existing else SYSTEM_HINT

        response = self._messages.create(**kwargs)

        if kwargs.get("stream"):
            return _ArmosAnthropicStream(response, self._guard.demask)
        return self._demask_response(response)

    def _demask_response(self, response: Any) -> Any:
        if not hasattr(response, "content"):
            return response
        for block in response.content:
            if hasattr(block, "text") and block.text:
                block.text = self._guard.demask(block.text)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class ArmosAnthropic:
    """
    Drop-in replacement for anthropic.Anthropic.
    Masks PII in prompts before sending to Anthropic.
    Restores real values in responses.

    Usage:
        from anthropic import Anthropic
        from armos import ArmosAnthropic
        client = ArmosAnthropic(Anthropic())
    """

    def __init__(
        self,
        client: Any,
        store: Literal["armos"] | None = None,
        armos_api_key: str | None = None,
        armos_base_url: str = "https://proxy.armos.dev",
    ):
        self._client = client
        self._guard = Armos(store=store, armos_api_key=armos_api_key, armos_base_url=armos_base_url)
        self.messages = ArmosMessages(client.messages, self._guard)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ── Async wrappers ────────────────────────────────────────────────────────────

class ArmosAsyncMessages(_AsyncMaskingMixin):
    """Wraps anthropic.resources.AsyncMessages."""

    def __init__(self, messages: Any, guard: Armos):
        self._messages = messages
        self._guard = guard

    async def create(self, **kwargs) -> Any:
        has_pii = False
        if "messages" in kwargs:
            kwargs["messages"], has_pii = await self._mask_messages_async(kwargs["messages"])

        if has_pii:
            existing = kwargs.get("system") or ""
            kwargs["system"] = (existing + "\n\n" + SYSTEM_HINT).strip() if existing else SYSTEM_HINT

        response = await self._messages.create(**kwargs)

        if kwargs.get("stream"):
            return _ArmosAnthropicAsyncStream(response, self._guard.demask)
        return self._demask_response(response)

    def _demask_response(self, response: Any) -> Any:
        if not hasattr(response, "content"):
            return response
        for block in response.content:
            if hasattr(block, "text") and block.text:
                block.text = self._guard.demask(block.text)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class ArmosAsyncAnthropic:
    """
    Drop-in replacement for anthropic.AsyncAnthropic.
    Masks PII in prompts before sending to Anthropic.
    Restores real values in responses.

    Usage:
        from anthropic import AsyncAnthropic
        from armos import ArmosAsyncAnthropic
        client = ArmosAsyncAnthropic(AsyncAnthropic())
    """

    def __init__(
        self,
        client: Any,
        store: Literal["armos"] | None = None,
        armos_api_key: str | None = None,
        armos_base_url: str = "https://proxy.armos.dev",
    ):
        self._client = client
        self._guard = Armos(store=store, armos_api_key=armos_api_key, armos_base_url=armos_base_url)
        self.messages = ArmosAsyncMessages(client.messages, self._guard)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
