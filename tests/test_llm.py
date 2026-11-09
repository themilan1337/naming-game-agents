"""LLM adapter tests. No network and no API keys: only the stub and httpx mock transports."""

import json
import random
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from namegame.llm import (
    SYSTEM_PROMPT,
    AnthropicClient,
    CallStats,
    LLMAgent,
    MissingKeyError,
    OpenAIClient,
    StubClient,
    history_prompt,
    llm_factory,
    make_client,
    memory_prompt,
    parse_name,
    parse_prompt_names,
    probe_memory,
    run_probe,
)
from namegame.networks import complete_network
from namegame.sim import run_emergence, run_tipping

POOL = ["emma", "maria", "david", "john", "anna", "lisa", "mark", "paul", "jane", "sam"]


class Scripted:
    """Client that replays fixed replies; used to test parsing, retries and fallback."""

    name = "scripted"
    is_stub = False

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        return self.replies.pop(0)


def test_parse_name():
    assert parse_name("Emma") == "emma"
    assert parse_name('  "maria".\n') == "maria"
    assert parse_name("Sure, emma") is None
    assert parse_name("") is None
    assert parse_name("1234") is None


def test_memory_prompt_roundtrip_through_parser():
    p = memory_prompt(["b", "a", "a"], 12)
    assert parse_prompt_names(p) == ["b", "a", "a"]
    assert "at most your last 12" in p


def test_history_prompt_roundtrip_through_parser():
    p = history_prompt([("x", "y"), ("x", "x")], 12)
    assert "1. you: x / partner: y" in p
    assert parse_prompt_names(p) == ["x", "y", "x", "x"]


def test_stub_is_labelled_and_picks_most_frequent_newest_on_tie():
    stub = StubClient()
    assert stub.is_stub and "STUB" in stub.name and "not a language model" in stub.name
    assert stub.complete(SYSTEM_PROMPT, memory_prompt(["a", "b", "b", "a"], 12)) == "a"
    assert stub.complete(SYSTEM_PROMPT, memory_prompt(["a", "b", "b"], 12)) == "b"


def test_agent_retries_once_then_falls_back_to_newest_memory_name():
    client = Scripted(["I think emma is nice", "still not one word"])
    stats = CallStats()
    agent = LLMAgent(12, client, "memory", stats)
    agent.memory[:] = ["zed", "amy"]
    assert agent.choose(random.Random(0)) == "zed"
    assert (stats.calls, stats.retries, stats.fallbacks) == (2, 2, 1)
    assert client.prompts[1].endswith("Reply with exactly one name.")


def test_agent_accepts_retry_reply():
    client = Scripted(["blah blah", "Emma"])
    stats = CallStats()
    agent = LLMAgent(12, client, "memory", stats)
    agent.memory[:] = ["zed"]
    assert agent.choose(random.Random(0)) == "emma"
    assert (stats.calls, stats.retries, stats.fallbacks) == (2, 1, 0)


def test_agent_memory_follows_modal_update_and_history_keeps_rounds():
    agent = LLMAgent(3, StubClient(), "history", CallStats())
    agent.memory[:] = ["a"]
    agent.update("a", "b")
    agent.update("b", "b")
    assert agent.memory == ["b", "b"]  # first update removed own "a", pushed "b"; then match
    assert agent.history == [("b", "b"), ("a", "b")]
    assert "1. you: b / partner: b" in agent.build_prompt()


def test_invalid_mode():
    with pytest.raises(ValueError):
        LLMAgent(12, StubClient(), "chat")


def test_stub_agents_reach_convention_in_a_network_and_are_threadsafe():
    stats = CallStats()
    net = complete_network(12)
    run = run_emergence(
        llm_factory(StubClient(), 12, "memory", stats),
        net,
        12,
        random.Random(3),
        max_rounds=60,
        executor=ThreadPoolExecutor(4),
        name_pool=POOL,
    )
    assert run.converged_round is not None
    assert stats.fallbacks == 0
    assert stats.calls >= 12


def test_probe_memory_counts_and_assignment_randomised():
    rng = random.Random(0)
    seen = set()
    for _ in range(20):
        mem, target = probe_memory(5, 12, rng)
        assert mem.count(target) == 5 and len(mem) == 12 and len(set(mem)) == 2
        seen.add(target)
    assert len(seen) == 2


def test_run_probe_with_stub_is_majority_rule_and_flagged():
    rows, stats = run_probe(StubClient(), m=12, reps=5, seed=1)
    assert len(rows) == 13 * 5
    assert all(r["is_stub"] == 1 for r in rows)
    for r in rows:
        k = r["k_target"]
        if k > 6:
            assert r["chose_target"] == 1
        elif k < 6:
            assert r["chose_target"] == 0
    assert stats.fallbacks == 0


def test_make_client_requires_keys_and_known_provider(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert make_client("stub").is_stub
    with pytest.raises(MissingKeyError):
        make_client("anthropic:some-model")
    with pytest.raises(MissingKeyError):
        make_client("openai:some-model")
    with pytest.raises(ValueError):
        make_client("anthropic")
    with pytest.raises(ValueError):
        make_client("gemini:x")


def test_anthropic_request_shape_and_parsing():
    c = AnthropicClient("model-x", temperature=0.7, api_key="k")
    url, headers, payload = c.request("sys", "usr")
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "k" and headers["anthropic-version"] == "2023-06-01"
    assert payload["system"] == "sys" and payload["messages"] == [
        {"role": "user", "content": "usr"}
    ]
    assert payload["temperature"] == 0.7
    no_temp = AnthropicClient("model-x", api_key="k").request("s", "u")[2]
    assert "temperature" not in no_temp

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["model"] == "model-x"
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Emma"}]})

    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    assert c.complete("sys", "usr") == "Emma"


def test_openai_request_shape_parsing_and_retry():
    c = OpenAIClient("model-y", api_key="k", base_url="http://local/v1/")
    url, headers, payload = c.request("sys", "usr")
    assert url == "http://local/v1/chat/completions"
    assert headers["Authorization"] == "Bearer k"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={})
        return httpx.Response(200, json={"choices": [{"message": {"content": "david"}}]})

    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    assert c.complete("sys", "usr") == "david"
    assert calls["n"] == 2


def test_stub_tipping_with_named_conventions():
    stats = CallStats()
    run = run_tipping(
        llm_factory(StubClient(), 12, "memory", stats),
        complete_network(24),
        12,
        10,
        random.Random(1),
        max_rounds=40,
        old_name="emma",
        new_name="david",
    )
    assert run.rounds[-1].new_share > 0.9  # 10 of 24 committed: a majority-rule stub flips
    assert stats.fallbacks == 0


# ---- transport: retries, timeouts, thread-safe counters, token limit (PL-31 T7) ----


def _client_with(handler, cls=OpenAIClient):
    c = cls("m", api_key="k")
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_callstats_add_is_threadsafe():
    stats = CallStats()

    def work(_):
        for _ in range(1000):
            stats.add(calls=1)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(work, range(8)))
    assert stats.calls == 8000


def test_retry_after_header_is_honoured(monkeypatch):
    delays = []
    monkeypatch.setattr("time.sleep", delays.append)
    seen = {"n": 0}

    def handler(request):
        seen["n"] += 1
        if seen["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"}, json={})
        if seen["n"] == 2:
            return httpx.Response(429, headers={"Retry-After": "900"}, json={})
        return httpx.Response(200, json={"choices": [{"message": {"content": "emma"}}]})

    assert _client_with(handler).complete("s", "u") == "emma"
    assert delays[0] == 3
    assert max(delays) <= 60


def test_timeout_is_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    seen = {"n": 0}

    def handler(request):
        seen["n"] += 1
        if seen["n"] == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "anna"}]})

    assert _client_with(handler, AnthropicClient).complete("s", "u") == "anna"
    assert seen["n"] == 2


def test_gives_up_after_four_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    seen = {"n": 0}

    def handler(request):
        seen["n"] += 1
        return httpx.Response(503, json={})

    with pytest.raises(httpx.HTTPStatusError):
        _client_with(handler).complete("s", "u")
    assert seen["n"] == 4


def test_max_tokens_is_settable():
    assert AnthropicClient("m", api_key="k", max_tokens=64).request("s", "u")[2]["max_tokens"] == 64
    assert OpenAIClient("m", api_key="k", max_tokens=64).request("s", "u")[2]["max_tokens"] == 64
    assert AnthropicClient("m", api_key="k").request("s", "u")[2]["max_tokens"] == 16
    assert OpenAIClient("m", api_key="k").request("s", "u")[2]["max_tokens"] == 16


def test_openai_token_parameter_name_can_be_switched():
    c = OpenAIClient("m", api_key="k", max_tokens=32, token_param="max_completion_tokens")
    payload = c.request("s", "u")[2]
    assert payload["max_completion_tokens"] == 32 and "max_tokens" not in payload
    with pytest.raises(ValueError):
        OpenAIClient("m", api_key="k", token_param="nonsense")


# ---- neutral probe prompt (PL-31 T8) ----


class Recorder:
    name = "recorder"
    is_stub = False

    def __init__(self):
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        return "maria"


def test_probe_prompt_does_not_claim_an_order():
    rec = Recorder()
    run_probe(rec, m=12, reps=1)
    assert rec.prompts
    for p in rec.prompts:
        assert "Names in your memory:" in p
        assert "newest first" not in p
        assert len(parse_prompt_names(p)) == 12


def test_game_prompt_keeps_the_order_claim():
    p = memory_prompt(["b", "a"], 12)
    assert "newest first" in p
    assert parse_prompt_names(p) == ["b", "a"]
    unordered = memory_prompt(["b", "a"], 12, ordered=False)
    assert "newest first" not in unordered
    assert parse_prompt_names(unordered) == ["b", "a"]


def test_llm_agent_orders_by_default_and_probe_agent_does_not():
    agent = LLMAgent(12, Recorder())
    agent.memory[:] = ["x", "y"]
    assert "newest first" in agent.build_prompt()
    agent.ordered = False
    assert "newest first" not in agent.build_prompt()
