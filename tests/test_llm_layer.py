"""LLM layer: cache, roles and prompts, client retries, providers (no server needed)."""

import json
import random
import re

import httpx
import pytest

from elicitor.llm import LLMParseError, LLMTransportError, build_client
from elicitor.llm.cache import ReplyCache, cache_key
from elicitor.llm.client import LLMClient, parse_reply
from elicitor.llm.providers import (
    Completion,
    MockProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    Request,
    TransportError,
)
from elicitor.llm.roles import ROLE_NAMES, PhraseReply, build_roles, update_model, value_lines
from elicitor.personas import load_personas
from tests.conftest import TRAVEL

PROMPTS = TRAVEL / "prompts"


@pytest.fixture(scope="module")
def roles():
    return build_roles(PROMPTS)


@pytest.fixture
def persona():
    return load_personas(TRAVEL / "personas.jsonl", split="dev")[0]


def update_context(schema, answer="Somewhere by the sea, please."):
    slot = schema.slot("setting")
    return {
        "slot": slot,
        "slot_description": slot.description,
        "values": value_lines(slot),
        "question": slot.question,
        "answer": answer,
        "no_preference_note": "",
    }


class Scripted:
    """Provider that replays a list of reply texts (or exceptions) and records the requests."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[Request] = []

    def complete(self, request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Completion(text=reply, prompt_tokens=10, completion_tokens=5)


def client_with(roles, provider, cache=None):
    return LLMClient(roles, provider, {"agent": "m", "user": "m"}, cache=cache, retry_wait_s=0.0)


GOOD_UPDATE = json.dumps(
    {
        "evidence": "by the sea",
        "scores": {"beach": 10, "city": 2, "mountains": 1, "countryside": 3, "no_preference": 1},
    }
)


# ------------------------------------------------------------------ cache


def test_cache_key_is_stable_and_sensitive():
    a = cache_key(model="m", messages=[{"role": "user", "content": "hi"}], seed=1)
    b = cache_key(seed=1, messages=[{"content": "hi", "role": "user"}], model="m")
    assert a == b
    assert a != cache_key(model="m", messages=[{"role": "user", "content": "hi"}], seed=2)


def test_cache_round_trip_and_persistence(tmp_path):
    path = tmp_path / "c.sqlite"
    cache = ReplyCache(path)
    assert len(cache) == 0 and cache.get("k") is None
    cache.put("k", role="update", model="m", text="{}")
    assert cache.get("k") == "{}"
    cache.close()
    reopened = ReplyCache(path)
    assert reopened.get("k") == "{}" and len(reopened) == 1
    reopened.close()


def test_empty_cache_is_still_used(tmp_path, roles, schema):
    """Regression: an empty cache has len 0 and must not be treated as 'no cache'."""
    cache = ReplyCache(tmp_path / "c.sqlite")
    provider = Scripted([GOOD_UPDATE])
    client = client_with(roles, provider, cache)
    first = client.call("update", update_context(schema))
    second = client.call("update", update_context(schema))
    assert first == second
    assert len(provider.requests) == 1
    assert client.stats["update"].cache_hits == 1
    client.close()


# ------------------------------------------------------------------ roles and prompts


def test_every_role_has_a_prompt_and_a_mock(roles):
    assert set(roles) == set(ROLE_NAMES)
    for role in roles.values():
        assert role.group in ("agent", "user")
        assert role.max_tokens > 0 and 0 <= role.temperature <= 1


def test_update_schema_requires_a_bounded_score_for_every_value(schema):
    slot = schema.slot("budget")
    js = update_model(slot).model_json_schema(by_alias=True)
    scores = js["$defs"][f"Scores_{slot.name}"]
    assert scores["required"] == list(slot.value_names)
    assert scores["additionalProperties"] is False
    assert all(p["minimum"] == 0 and p["maximum"] == 10 for p in scores["properties"].values())


def test_prompts_render_without_leftover_placeholders(roles, schema, persona):
    from elicitor.simulator.llm import load_styles, profile_lines

    slot = schema.slot("pace")
    contexts = {
        "update": update_context(schema),
        "phrase": {"slot": slot, "slot_description": slot.description, "template": slot.question},
        "simulate": {
            "slot": slot,
            "persona": persona,
            "profile": profile_lines(schema, persona),
            "backstory": "",
            "style_guide": load_styles(PROMPTS / "styles.toml")[persona.style],
            "question": slot.question,
            "topic": slot.description,
        },
    }
    for name, role in roles.items():
        messages = role.prompt.render(contexts[name])
        assert [m["role"] for m in messages] == ["system", "user"]
        # Labels may contain "$100"; an unfilled placeholder looks like "$name".
        assert not any(re.search(r"\$[a-z_]", m["content"]) for m in messages), name
        reply = role.mock(contexts[name], random.Random(0))
        assert isinstance(reply, role.output(contexts[name]))


def test_styles_cover_every_persona_style():
    from elicitor.personas import STYLES
    from elicitor.simulator.llm import load_styles

    assert set(load_styles(PROMPTS / "styles.toml")) == set(STYLES)


def test_phrase_reply_must_be_a_question():
    assert PhraseReply(question="  Where to?  ").question == "Where to?"
    with pytest.raises(ValueError):
        PhraseReply(question="Tell me where you want to go.")


def test_parse_reply_tolerates_code_fences(schema):
    model = update_model(schema.slot("setting"))
    assert parse_reply(model, f"```json\n{GOOD_UPDATE}\n```") is not None
    assert parse_reply(model, '{"evidence": "", "scores": {"beach": 11}}') is None
    assert parse_reply(model, "not json") is None


# ------------------------------------------------------------------ client


def test_retry_after_parse_failure_uses_a_new_seed(roles, schema):
    provider = Scripted(["oops", GOOD_UPDATE])
    client = client_with(roles, provider)
    reply = client.call("update", update_context(schema), seed=5)
    assert reply.scores.v0 == 10  # beach
    assert [r.seed for r in provider.requests] == [5, 6]
    s = client.stats["update"]
    assert (s.calls, s.attempts, s.parse_failures, s.failed_calls) == (1, 2, 1, 0)


def test_call_fails_after_max_attempts(roles, schema):
    client = client_with(roles, Scripted(["bad"] * 3))
    with pytest.raises(LLMParseError):
        client.call("update", update_context(schema))
    s = client.stats["update"]
    assert (s.attempts, s.parse_failures, s.failed_calls) == (3, 3, 1)


def test_transport_errors_are_retried_then_raised(roles, schema):
    client = client_with(roles, Scripted([TransportError("down"), GOOD_UPDATE]))
    client.call("update", update_context(schema))
    assert client.stats["update"].transport_errors == 1

    client = client_with(roles, Scripted([TransportError("down")] * 3))
    with pytest.raises(LLMTransportError):
        client.call("update", update_context(schema))
    assert client.stats["update"].transport_errors == 3


def test_role_overrides_and_model_groups(roles, schema):
    provider = Scripted([GOOD_UPDATE])
    client = LLMClient(
        roles,
        provider,
        {"agent": "small", "user": "big"},
        role_overrides={"update": {"temperature": 0.3}},
    )
    client.call("update", update_context(schema))
    request = provider.requests[0]
    assert request.model == "small" and request.temperature == 0.3
    assert request.schema["properties"]["scores"]


def test_missing_model_group_is_an_error(roles):
    with pytest.raises(ValueError, match="no model"):
        LLMClient(roles, MockProvider(), {"agent": "m"})


def test_stats_are_written(tmp_path, roles, schema):
    client = client_with(roles, Scripted([GOOD_UPDATE]))
    client.call("update", update_context(schema))
    client.write_stats(tmp_path / "s.json")
    stats = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert stats["roles"]["update"]["calls"] == 1
    assert "phrase" not in stats["roles"]  # roles that were never called are left out
    assert "update" in client.stats_table()


def test_build_client_checks_its_table(tmp_path):
    client = build_client({}, PROMPTS, None)
    assert client.provider.name == "mock" and client.cache is None
    with pytest.raises(ValueError, match="unknown keys"):
        build_client({"modle": "x"}, PROMPTS, None)
    with pytest.raises(ValueError, match="needs a model"):
        build_client({"provider": "ollama"}, PROMPTS, None)
    with pytest.raises(ValueError, match="unknown roles"):
        build_client({"roles": {"judge": {}}}, PROMPTS, None)
    client = build_client({"cache": False}, PROMPTS, tmp_path / "x.sqlite")
    assert client.cache is None


# ------------------------------------------------------------------ providers


def request_for(roles, schema):
    ctx = update_context(schema)
    role = roles["update"]
    return Request(
        role=role,
        context=ctx,
        model="qwen3:4b",
        messages=role.prompt.render(ctx),
        schema=update_model(ctx["slot"]).model_json_schema(by_alias=True),
        temperature=0.0,
        max_tokens=50,
        seed=7,
    )


def test_ollama_request_and_reply(roles, schema):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": GOOD_UPDATE}, "prompt_eval_count": 90, "eval_count": 30},
        )

    provider = OllamaProvider(transport=httpx.MockTransport(handler), num_ctx=2048)
    completion = provider.complete(request_for(roles, schema))
    body = seen["body"]
    assert seen["url"].endswith("/api/chat")
    assert body["think"] is False and body["stream"] is False
    assert body["format"]["properties"]["scores"]
    assert body["options"] == {"temperature": 0.0, "seed": 7, "num_predict": 50, "num_ctx": 2048}
    assert completion.text == GOOD_UPDATE
    assert (completion.prompt_tokens, completion.completion_tokens) == (90, 30)


def test_ollama_can_omit_think(roles, schema):
    provider = OllamaProvider(think=None)
    assert "think" not in provider.payload(request_for(roles, schema))


def test_openai_compatible_request_and_reply(roles, schema):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": GOOD_UPDATE}}],
                "usage": {"prompt_tokens": 80, "completion_tokens": 20},
            },
        )

    provider = OpenAICompatibleProvider(transport=httpx.MockTransport(handler))
    completion = provider.complete(request_for(roles, schema))
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert completion.text == GOOD_UPDATE and completion.completion_tokens == 20


def test_http_errors_become_transport_errors(roles, schema):
    provider = OllamaProvider(
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    )
    with pytest.raises(TransportError):
        provider.complete(request_for(roles, schema))


def test_mock_provider_is_deterministic(roles, schema, persona):
    from elicitor.simulator.llm import load_styles, profile_lines

    slot = schema.slot("climate")
    ctx = {
        "slot": slot,
        "persona": persona,
        "profile": profile_lines(schema, persona),
        "backstory": "",
        "style_guide": load_styles(PROMPTS / "styles.toml")[persona.style],
        "question": slot.question,
        "topic": slot.description,
    }
    client = client_with(roles, MockProvider())
    answers = {client.call("simulate", ctx, seed=s).answer for s in range(30)}
    again = {client.call("simulate", ctx, seed=s).answer for s in range(30)}
    assert answers == again
    assert len(answers) > 1  # noise: some seeds name a different value


def test_base_url_can_come_from_the_environment(monkeypatch):
    config = {"provider": "ollama", "models": {"agent": "a", "user": "u"}, "cache": False}
    assert build_client(config, PROMPTS, None).provider.base_url == "http://127.0.0.1:11434"
    monkeypatch.setenv("ELICITOR_BASE_URL", "http://127.0.0.1:40123/")
    assert build_client(config, PROMPTS, None).provider.base_url == "http://127.0.0.1:40123"
