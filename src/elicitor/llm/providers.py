"""LLM providers: where a request is sent (SPEC §14).

A provider turns a `Request` into a `Completion` (the raw reply text plus token counts). It does
not parse or retry; the client does that. Transport problems raise `TransportError`.

- `ollama`: Ollama's native chat API, with the role's JSON schema as `format`.
- `openai_compatible`: any server with an OpenAI-style `/v1/chat/completions` (vLLM, llama.cpp,
  Ollama's own compatibility endpoint).
- `mock`: no server; calls the role's mock function.

`RoutedProvider` spreads requests over several servers of one provider kind: each request goes
to a server of its role's model group, the one with the fewest requests in flight (SPEC §14, M5).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
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
        think: bool | Mapping[str, bool] | None = False,
        transport: httpx.BaseTransport | None = None,
    ):
        """`think` is sent as given, or looked up by the role's model group when it is a table
        (a group not in the table gets no `think` field: models without a thinking mode)."""
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
        think = self.think
        if isinstance(think, Mapping):
            think = think.get(request.role.group)
        if think is not None:
            payload["think"] = think
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


class RoutedProvider:
    """Several servers per model group; `name` is the servers' provider name, so cache keys do
    not depend on which server answered."""

    def __init__(self, pools: Mapping[str, Sequence[Provider]]):
        if not pools or any(not servers for servers in pools.values()):
            raise ValueError("every model group needs at least one server")
        names = {server.name for servers in pools.values() for server in servers}
        if len(names) != 1:
            raise ValueError(f"all servers must be of one provider kind, got {sorted(names)}")
        self.name = names.pop()
        self.pools = {group: list(servers) for group, servers in pools.items()}
        self._lock = threading.Lock()
        self.in_flight = {group: [0] * len(servers) for group, servers in self.pools.items()}
        self.sent = {group: [0] * len(servers) for group, servers in self.pools.items()}

    def complete(self, request: Request) -> Completion:
        group = request.role.group
        if group not in self.pools:
            raise ValueError(f"no server for model group {group!r}")
        with self._lock:
            loads = self.in_flight[group]
            index = loads.index(min(loads))  # ties go to list order
            loads[index] += 1
            self.sent[group][index] += 1
        try:
            return self.pools[group][index].complete(request)
        finally:
            with self._lock:
                loads[index] -= 1


def make_provider(name: str, **params: Any) -> Provider:
    registry: dict[str, type] = {
        "ollama": OllamaProvider,
        "openai_compatible": OpenAICompatibleProvider,
        "mock": MockProvider,
    }
    if name not in registry:
        raise ValueError(f"unknown LLM provider {name!r}; choose from {sorted(registry)}")
    return registry[name](**params)
