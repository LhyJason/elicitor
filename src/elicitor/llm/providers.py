"""LLM providers: where a request is sent (SPEC §14).

A provider turns a `Request` into a `Completion` (the raw reply text plus token counts). It does
not parse or retry; the client does that. Transport problems raise `TransportError`.

- `ollama`: Ollama's native chat API, with the role's JSON schema as `format`.
- `openai_compatible`: any server with an OpenAI-style `/v1/chat/completions` (vLLM, llama.cpp,
  Ollama's own compatibility endpoint).
- `mock`: no server; calls the role's mock function.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import httpx

from elicitor.rng import derive_rng

if TYPE_CHECKING:
    from elicitor.llm.roles import Role


class TransportError(RuntimeError):
    """The server could not be reached or answered with an HTTP error."""


@dataclass(frozen=True)
class Request:
    role: Role
    context: dict[str, Any]
    model: str
    messages: list[dict[str, str]]
    schema: dict[str, Any]
    temperature: float
    max_tokens: int
    seed: int


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    name: str

    def complete(self, request: Request) -> Completion: ...


def _http_client(timeout: float, transport: httpx.BaseTransport | None) -> httpx.Client:
    # trust_env=False: the servers are local (or on the same cluster node), and a proxy set in
    # the environment would otherwise intercept requests to 127.0.0.1.
    return httpx.Client(timeout=timeout, trust_env=False, transport=transport)


def _post(client: httpx.Client, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        raise TransportError(f"{url}: {exc}") from exc


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        num_ctx: int = 4096,
        think: bool | None = False,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.think = think
        self._http = _http_client(timeout, transport)

    def payload(self, request: Request) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "stream": False,
            "format": request.schema,
            "options": {
                "temperature": request.temperature,
                "seed": request.seed,
                "num_predict": request.max_tokens,
                # A different num_ctx makes Ollama reload the model, so it is pinned.
                "num_ctx": self.num_ctx,
            },
        }
        if self.think is not None:
            payload["think"] = self.think
        return payload

    def complete(self, request: Request) -> Completion:
        data = _post(self._http, f"{self.base_url}/api/chat", self.payload(request))
        return Completion(
            text=data.get("message", {}).get("content", ""),
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            extra={"done_reason": data.get("done_reason")},
        )


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 120.0,
        api_key: str = "none",
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._http = _http_client(timeout, transport)

    def payload(self, request: Request) -> dict[str, Any]:
        return {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "seed": request.seed,
            "max_tokens": request.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": request.role.name, "schema": request.schema},
            },
        }

    def complete(self, request: Request) -> Completion:
        self._http.headers["Authorization"] = f"Bearer {self.api_key}"
        data = _post(self._http, f"{self.base_url}/v1/chat/completions", self.payload(request))
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise TransportError(f"unexpected reply shape: {exc}") from exc
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )


class MockProvider:
    """Deterministic: the role's mock function gets a generator seeded by the request seed."""

    name = "mock"

    def complete(self, request: Request) -> Completion:
        rng = derive_rng("mock", request.role.name, request.seed)
        reply = request.role.mock(request.context, rng)
        return Completion(text=reply.model_dump_json(by_alias=True))


def make_provider(name: str, **params: Any) -> Provider:
    registry: dict[str, type] = {
        "ollama": OllamaProvider,
        "openai_compatible": OpenAICompatibleProvider,
        "mock": MockProvider,
    }
    if name not in registry:
        raise ValueError(f"unknown LLM provider {name!r}; choose from {sorted(registry)}")
    return registry[name](**params)
