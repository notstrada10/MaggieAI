"""Retry-on-transient-failure for the DeepSeek inference client.

Motivation: an end-to-end run of the RespondeoQA eval (174 items)
hit ~18% failure with `httpx.ConnectError` clustered at the tail —
Docker's embedded DNS flapping under sustained load. The Anthropic
SDK retries internally; the DeepSeek client (raw httpx) did not.
"""

from __future__ import annotations

import httpx
import pytest

from maggieai.inference.client import GenerationRequest, Message
from maggieai.inference.deepseek_api import DeepSeekApiClient


def _make_client(transport: httpx.MockTransport, *, max_retries: int = 3) -> DeepSeekApiClient:
    """Build a client whose underlying httpx.AsyncClient uses the mock transport.

    The constructor reaches into settings to read the API key, so we pass
    `api_key="x"` to bypass that. We then swap in a fresh AsyncClient bound
    to the mock transport — replacing the real one to avoid network I/O.
    """
    client = DeepSeekApiClient(
        api_key="x",
        base_url="https://api.deepseek.test/v1",
        model="deepseek-v4-pro",
        max_retries=max_retries,
        retry_initial_delay=0.0,  # no real sleeping in tests
    )
    client._client = httpx.AsyncClient(  # type: ignore[attr-defined]
        transport=transport,
        headers={"Authorization": "Bearer x"},
    )
    return client


def _ok_payload() -> dict[str, object]:
    return {
        "choices": [{"message": {"content": "salve"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _request() -> GenerationRequest:
    return GenerationRequest(messages=[Message(role="user", content="hi")], max_tokens=8)


async def test_succeeds_after_one_dns_flake() -> None:
    """Mirrors the production failure: first attempt raises ConnectError, second succeeds."""
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("[Errno -2] Name or service not known")
        return httpx.Response(200, json=_ok_payload())

    client = _make_client(httpx.MockTransport(handler))
    resp = await client.generate(_request())
    assert resp.text == "salve"
    assert calls["n"] == 2
    await client.aclose()


async def test_retries_on_5xx_and_429() -> None:
    statuses = [500, 429, 200]
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        status = statuses[calls["n"]]
        calls["n"] += 1
        if status == 200:
            return httpx.Response(200, json=_ok_payload())
        return httpx.Response(status, json={"error": "transient"})

    client = _make_client(httpx.MockTransport(handler))
    await client.generate(_request())
    assert calls["n"] == 3
    await client.aclose()


async def test_does_not_retry_on_4xx() -> None:
    """A 401 (bad key) or 422 (bad request) won't recover on retry — fail fast."""
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "invalid api key"})

    client = _make_client(httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await client.generate(_request())
    assert calls["n"] == 1
    await client.aclose()


async def test_gives_up_after_max_retries() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("dns down")

    client = _make_client(httpx.MockTransport(handler), max_retries=2)
    with pytest.raises(httpx.ConnectError):
        await client.generate(_request())
    # max_retries=2 means: initial attempt + 2 retries = 3 calls
    assert calls["n"] == 3
    await client.aclose()
