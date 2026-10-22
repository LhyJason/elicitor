"""The LLM client: render a role's prompt, check the cache, call the provider, validate, retry.

`call` tries up to `max_attempts` times; attempt i sends seed + i, so a retry at temperature 0 is
not a copy of the failed reply. Only validated replies are cached. Transport errors are retried
with a short back-off and then raised, which stops the run (SPEC §14, M2 details).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from elicitor.llm.cache import ReplyCache, cache_key
from elicitor.llm.providers import Provider, Request, TransportError, make_provider
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
        self.stats = {name: RoleStats() for name in self.roles}

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
            cached = self.cache.get(key) if self.cache is not None else None
            if cached is not None:
                reply = parse_reply(output, cached)
                if reply is not None:
                    stats.cache_hits += 1
                    return reply

            completion = self._send(request, stats)
            reply = parse_reply(output, completion.text)
            if reply is None:
                stats.parse_failures += 1
                continue
            if self.cache is not None:
                self.cache.put(
                    key,
                    role=role_name,
                    model=model_name,
                    text=completion.text,
                    prompt_tokens=completion.prompt_tokens,
                    completion_tokens=completion.completion_tokens,
                )
            return reply

        stats.failed_calls += 1
        raise LLMParseError(f"role {role_name}: no valid reply in {self.max_attempts} attempts")

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

    def stats_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider.name,
            "models": self.models,
            "roles": {
                name: {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(s).items()}
                for name, s in self.stats.items()
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
_CONFIG_KEYS = {
    "provider", "models", "base_url", "timeout", "num_ctx", "think", "api_key", "cache",
    "max_attempts", "roles",
}  # fmt: skip
_PROVIDER_KEYS = {
    "ollama": ("base_url", "timeout", "num_ctx", "think"),
    "openai_compatible": ("base_url", "timeout", "api_key"),
    "mock": (),
}


def build_client(
    config: Mapping[str, Any], prompts_dir: str | Path, default_cache: str | Path | None
) -> LLMClient:
    """Build a client from an experiment file's `[llm]` table.

    `cache = false` turns the cache off; a path string overrides `default_cache`. The environment
    variable ELICITOR_BASE_URL, when set, overrides `base_url` (a cluster job sets it to the port
    its server picked).
    """
    unknown = set(config) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"[llm]: unknown keys {sorted(unknown)}")
    provider_name = config.get("provider", "mock")
    if provider_name not in _PROVIDER_KEYS:
        raise ValueError(f"unknown LLM provider {provider_name!r}")
    params = {k: config[k] for k in _PROVIDER_KEYS[provider_name] if k in config}
    if "base_url" in _PROVIDER_KEYS[provider_name] and os.environ.get(BASE_URL_ENV):
        params["base_url"] = os.environ[BASE_URL_ENV]
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
