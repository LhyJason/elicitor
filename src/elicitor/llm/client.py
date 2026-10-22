"""The LLM client: render a role's prompt, check the cache, call the provider, validate, retry.

`call` tries up to `max_attempts` times; attempt i sends seed + i, so a retry at temperature 0 is
not a copy of the failed reply. Only validated replies are cached. Transport errors are retried
with a short back-off and then raised, which stops the run (SPEC §14, M2 details).

The client is thread-safe (SPEC §14, M5 details). Identical requests are single-flight: while one
is in flight, another with the same cache key waits for its reply. Statistics go to the calling
thread's current tally, which `track()` replaces for the length of a task.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from elicitor.llm.cache import ReplyCache, cache_key
from elicitor.llm.providers import (
    Provider,
    Request,
    RoutedProvider,
    TransportError,
    make_provider,
)
from elicitor.llm.roles import ROLE_NAMES, Role, build_roles


class LLMParseError(RuntimeError):
    """Every attempt returned a reply that did not validate against the role's schema."""


class LLMTransportError(RuntimeError):
    """The server stayed unreachable after the retries."""


@dataclass
class RoleStats:
    calls: int = 0
    cache_hits: int = 0
    attempts: int = 0  # requests actually sent to the provider
    parse_failures: int = 0  # attempts whose reply did not validate
    failed_calls: int = 0  # calls where every attempt failed
    transport_errors: int = 0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def parse_reply(model: type[BaseModel], text: str) -> BaseModel | None:
    """Validate a reply; if that fails, retry on the outermost {...} (drops code fences etc.)."""
    try:
        return model.model_validate_json(text)
    except ValidationError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return model.model_validate_json(text[start : end + 1])
        except ValidationError:
            pass
    return None


class LLMClient:
    def __init__(
        self,
        roles: Mapping[str, Role],
        provider: Provider,
        models: Mapping[str, str],
        cache: ReplyCache | None = None,
        max_attempts: int = 3,
        transport_retries: int = 2,
        retry_wait_s: float = 1.0,
        role_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        missing = {role.group for role in roles.values()} - set(models)
        if missing:
            raise ValueError(f"no model configured for group(s) {sorted(missing)}")
        self.roles = dict(roles)
        self.provider = provider
        self.models = dict(models)
        self.cache = cache
        self.max_attempts = max_attempts
        self.transport_retries = transport_retries
        self.retry_wait_s = retry_wait_s
        self.role_overrides = {k: dict(v) for k, v in (role_overrides or {}).items()}
        self._default_stats = self.new_stats()
        self._local = threading.local()
        self._flight_lock = threading.Lock()
        self._in_flight: dict[str, threading.Event] = {}

    # ------------------------------------------------------------ statistics

    def new_stats(self) -> dict[str, RoleStats]:
        return {name: RoleStats() for name in self.roles}

    @property
    def stats(self) -> dict[str, RoleStats]:
        """The calling thread's tally (the client-wide one outside `track`)."""
        return getattr(self._local, "stats", None) or self._default_stats

    @contextlib.contextmanager
    def track(self) -> Iterator[dict[str, RoleStats]]:
        """Count this thread's calls in a fresh tally while the block runs."""
        previous = getattr(self._local, "stats", None)
        self._local.stats = tally = self.new_stats()
        try:
            yield tally
        finally:
            self._local.stats = previous

    # ------------------------------------------------------------ calls

    def call(self, role_name: str, context: Mapping[str, Any], seed: int = 0) -> BaseModel:
        role = self.roles[role_name]
        overrides = self.role_overrides.get(role_name, {})
        temperature = float(overrides.get("temperature", role.temperature))
        max_tokens = int(overrides.get("max_tokens", role.max_tokens))
        model_name = self.models[role.group]
        output = role.output(context)
        messages = role.prompt.render(context)
        schema = output.model_json_schema(by_alias=True)
        stats = self.stats[role_name]
        stats.calls += 1

        for attempt in range(self.max_attempts):
            request = Request(
                role=role,
                context=dict(context),
                model=model_name,
                messages=messages,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed + attempt,
            )
            key = cache_key(
                provider=self.provider.name,
                model=model_name,
                role=role_name,
                messages=messages,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed + attempt,
            )
            if self.cache is None:
                reply = self._attempt(request, output, None, stats)
            else:
                reply = self._attempt_cached(request, output, key, stats)
            if reply is not None:
                return reply

        stats.failed_calls += 1
        raise LLMParseError(f"role {role_name}: no valid reply in {self.max_attempts} attempts")

    def _attempt_cached(
        self, request: Request, output: type[BaseModel], key: str, stats: RoleStats
    ) -> BaseModel | None:
        """One attempt through the cache; waits instead of sending when the same key is in
        flight, and then reads the reply it left (or none, if it did not validate)."""
        assert self.cache is not None
        while True:
            cached = self.cache.get(key)
            if cached is not None:
                reply = parse_reply(output, cached)
                if reply is not None:
                    stats.cache_hits += 1
                    return reply
            with self._flight_lock:
                event = self._in_flight.get(key)
                leader = event is None
                if leader:
                    event = self._in_flight[key] = threading.Event()
            if leader:
                break
            event.wait()
            cached = self.cache.get(key)
            if cached is None:
                # The leader's reply did not validate (or it hit a transport error): this
                # attempt failed the same way, without a request of its own.
                return None
        try:
            return self._attempt(request, output, key, stats)
        finally:
            with self._flight_lock:
                del self._in_flight[key]
            event.set()

    def _attempt(
        self, request: Request, output: type[BaseModel], key: str | None, stats: RoleStats
    ) -> BaseModel | None:
        completion = self._send(request, stats)
        reply = parse_reply(output, completion.text)
        if reply is None:
            stats.parse_failures += 1
            return None
        if self.cache is not None and key is not None:
            self.cache.put(
                key,
                role=request.role.name,
                model=request.model,
                text=completion.text,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            )
        return reply

    def _send(self, request: Request, stats: RoleStats):
        for retry in range(self.transport_retries + 1):
            stats.attempts += 1
            start = time.perf_counter()
            try:
                completion = self.provider.complete(request)
            except TransportError as exc:
                stats.transport_errors += 1
                if retry == self.transport_retries:
                    raise LLMTransportError(str(exc)) from exc
                time.sleep(self.retry_wait_s * 2**retry)
                continue
            finally:
                stats.latency_s += time.perf_counter() - start
            stats.prompt_tokens += completion.prompt_tokens
            stats.completion_tokens += completion.completion_tokens
            return completion
        raise AssertionError("unreachable")

    # ------------------------------------------------------------ reporting

    def reset_stats(self) -> None:
        """Start the calling thread's tally from zero."""
        tally = self.stats
        for name in tally:
            tally[name] = RoleStats()

    def stats_dict(self, tally: Mapping[str, RoleStats] | None = None) -> dict[str, Any]:
        tally = self.stats if tally is None else tally
        return {
            "provider": self.provider.name,
            "models": self.models,
            "roles": {
                name: {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(s).items()}
                for name, s in tally.items()
                if s.calls
            },
        }

    def write_stats(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.stats_dict(), indent=2) + "\n", encoding="utf-8")

    def stats_table(self) -> str:
        fields = ("calls", "cache_hits", "attempts", "parse_failures", "failed_calls",
                  "transport_errors", "latency_s")  # fmt: skip
        rows = [(name, asdict(s)) for name, s in self.stats.items() if s.calls]
        if not rows:
            return "no LLM calls"
        lines = ["role".ljust(10) + "".join(f.rjust(17) for f in fields)]
        for name, s in rows:
            cells = (f"{s[f]:.1f}" if isinstance(s[f], float) else str(s[f]) for f in fields)
            lines.append(name.ljust(10) + "".join(c.rjust(17) for c in cells))
        return "\n".join(lines)

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()


BASE_URL_ENV = "ELICITOR_BASE_URL"
SERVERS_ENV = {"agent": "ELICITOR_SERVERS_AGENT", "user": "ELICITOR_SERVERS_USER"}
_CONFIG_KEYS = {
    "provider", "models", "base_url", "timeout", "num_ctx", "think", "api_key", "cache",
    "max_attempts", "roles",
}  # fmt: skip
_PROVIDER_KEYS = {
    "ollama": ("base_url", "timeout", "num_ctx", "think"),
    "openai_compatible": ("base_url", "timeout", "api_key"),
    "mock": (),
}


def server_urls(base_url: Any) -> dict[str, list[str]] | None:
    """Per-group server lists from `base_url` and the environment; None for a single server.

    `base_url` is one URL or a table {group: URL or [URLs]}; ELICITOR_SERVERS_<GROUP> (comma
    separated) replaces a group's list.
    """
    table = dict(base_url) if isinstance(base_url, Mapping) else {}
    for group, var in SERVERS_ENV.items():
        if os.environ.get(var):
            table[group] = os.environ[var]
    if not table:
        return None
    if set(table) != set(SERVERS_ENV):
        raise ValueError(f"[llm] base_url table needs exactly the groups {sorted(SERVERS_ENV)}")
    urls = {}
    for group, value in table.items():
        items = value.split(",") if isinstance(value, str) else list(value)
        urls[group] = [u.strip() for u in items if u.strip()]
        if not urls[group]:
            raise ValueError(f"no server URL for model group {group!r}")
    return urls


def build_client(
    config: Mapping[str, Any], prompts_dir: str | Path, default_cache: str | Path | None
) -> LLMClient:
    """Build a client from an experiment file's `[llm]` table.

    `cache = false` turns the cache off; a path string overrides `default_cache`. The environment
    variable ELICITOR_BASE_URL, when set, overrides a single `base_url` (a cluster job sets it to
    the port its server picked); ELICITOR_SERVERS_AGENT / _USER give several servers per model
    group (SPEC §14, M5 details).
    """
    unknown = set(config) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"[llm]: unknown keys {sorted(unknown)}")
    provider_name = config.get("provider", "mock")
    if provider_name not in _PROVIDER_KEYS:
        raise ValueError(f"unknown LLM provider {provider_name!r}")
    params = {k: config[k] for k in _PROVIDER_KEYS[provider_name] if k in config}
    provider: Provider
    if "base_url" in _PROVIDER_KEYS[provider_name]:
        pools = server_urls(params.get("base_url"))
        if pools is not None:
            provider = RoutedProvider(
                {
                    group: [make_provider(provider_name, **{**params, "base_url": url})
                            for url in urls]
                    for group, urls in pools.items()
                }
            )  # fmt: skip
        else:
            if os.environ.get(BASE_URL_ENV):
                params["base_url"] = os.environ[BASE_URL_ENV]
            provider = make_provider(provider_name, **params)
    else:
        provider = make_provider(provider_name, **params)

    default_model = "mock" if provider_name == "mock" else None
    models = dict(config.get("models", {}))
    for group in ("agent", "user"):
        models.setdefault(group, default_model)
    if None in models.values():
        raise ValueError("[llm.models] needs a model for 'agent' and 'user'")

    overrides = dict(config.get("roles", {}))
    bad = set(overrides) - set(ROLE_NAMES)
    if bad:
        raise ValueError(f"[llm.roles]: unknown roles {sorted(bad)}")

    cache_setting = config.get("cache", True)
    cache = None
    if cache_setting is not False:
        path = default_cache if cache_setting is True else cache_setting
        if path is not None:
            cache = ReplyCache(path)

    return LLMClient(
        roles=build_roles(prompts_dir),
        provider=provider,
        models=models,
        cache=cache,
        max_attempts=int(config.get("max_attempts", 3)),
        role_overrides=overrides,
    )
