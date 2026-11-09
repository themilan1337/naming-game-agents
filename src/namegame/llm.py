"""LLM agents for the naming game behind a provider-agnostic client.

* ``LLMClient`` is the only thing the game needs: ``complete(system, user) -> text``.
* ``AnthropicClient`` and ``OpenAIClient`` read their API key from the environment
  (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``). They were written from the public API
  descriptions and have not been tested against a live endpoint. The runs reported in the
  note used ``ClaudeCLIClient`` (the ``claude-cli:<model>`` client), which needs no key.
* ``CallBudget`` and ``BudgetedClient`` put a hard cap (calls, and cost from user-supplied
  prices) around any client. No price is built in: none was verified.
* ``StubClient`` is a deterministic, rule-based stand-in used by the tests and the smoke run.
  It is not a language model. Everything it produces is labelled ``STUB`` and must never be
  reported as an LLM result.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx

from .agents import Agent

SYSTEM_PROMPT = (
    "You are one player in a naming game. In every round you are paired with one other player. "
    "Both of you write a name for the same unfamiliar face at the same time, without talking. "
    "If the two names are identical you both earn a small reward; if they differ you both lose "
    "a little. No name is better than any other, only matching counts. "
    "Answer with a single first name in lowercase letters and nothing else."
)

MEMORY_LINE = "Names in your memory, newest first: "
MEMORY_LINE_UNORDERED = "Names in your memory: "
HISTORY_HEADER = "Your most recent rounds, newest first (you / your partner):"
ASK = "Which name do you write this round?"


class LLMClient(Protocol):
    name: str
    is_stub: bool

    def complete(self, system: str, user: str) -> str: ...


class MissingKeyError(RuntimeError):
    pass


class StubClient:
    """Deterministic stand-in: answers with the most frequent remembered name (newest wins ties).

    It parses the prompt the same way a reader would, so tests exercise the real prompt format.
    """

    name = "STUB (rule-based stand-in, not a language model)"
    is_stub = True

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        names = parse_prompt_names(user)
        if not names:
            return "emma"
        counts = Counter(names)
        best = max(counts.values())
        return next(n for n in names if counts[n] == best)


def parse_prompt_names(user: str) -> list[str]:
    """Names shown to the model, newest first (works for both prompt modes)."""
    for line in user.splitlines():
        for prefix in (MEMORY_LINE, MEMORY_LINE_UNORDERED):
            if line.startswith(prefix):
                return [x.strip() for x in line[len(prefix) :].split(",") if x.strip()]
    names = []
    for line in user.splitlines():
        m = re.match(r"\s*\d+\.\s+you: (\S+)\s*/\s*partner: (\S+)", line)
        if m:
            names.extend([m.group(1), m.group(2)])
    return names


class _HTTPClient:
    is_stub = False
    max_attempts = 4
    max_wait = 60.0
    # HTTP statuses worth another try: rate limit, server errors, Anthropic "overloaded" (529)
    retry_statuses = (429, 500, 502, 503, 529)

    def __init__(
        self,
        model: str,
        temperature: float | None,
        timeout: float = 60.0,
        max_tokens: int = 16,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._http = httpx.Client(timeout=timeout)

    def _wait(self, resp: httpx.Response | None, attempt: int) -> float:
        """Seconds to wait: the Retry-After header (seconds) if present, else 2**attempt; max 60."""
        if resp is not None:
            try:
                return min(max(float(resp.headers["Retry-After"]), 0.0), self.max_wait)
            except (KeyError, ValueError):
                pass
        return min(2.0**attempt, self.max_wait)

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        for attempt in range(self.max_attempts):
            last = attempt == self.max_attempts - 1
            try:
                resp = self._http.post(url, headers=headers, json=payload)
            except (httpx.TimeoutException, httpx.TransportError):
                if last:
                    raise
                time.sleep(self._wait(None, attempt))
                continue
            if resp.status_code in self.retry_statuses and not last:
                time.sleep(self._wait(resp, attempt))
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("unreachable")


class AnthropicClient(_HTTPClient):
    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 16,
    ) -> None:
        super().__init__(model, temperature, max_tokens=max_tokens)
        self.name = f"anthropic:{model}"
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise MissingKeyError("set ANTHROPIC_API_KEY to use an anthropic:<model> client")
        self.base_url = base_url.rstrip("/")

    def request(self, system: str, user: str) -> tuple[str, dict, dict]:
        payload: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        return f"{self.base_url}/v1/messages", headers, payload

    def complete_with_usage(self, system: str, user: str) -> tuple[str, dict]:
        """Reply text and ``{"input_tokens", "output_tokens"}`` as the API reports them."""
        url, headers, payload = self.request(system, user)
        data = self._post(url, headers, payload)
        text = "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )
        u = data.get("usage") or {}
        usage = {}
        if "input_tokens" in u and "output_tokens" in u:
            usage = {"input_tokens": u["input_tokens"], "output_tokens": u["output_tokens"]}
        return text, usage

    def complete(self, system: str, user: str) -> str:
        return self.complete_with_usage(system, user)[0]


class OpenAIClient(_HTTPClient):
    """Chat-completions client; also works with OpenAI-compatible servers via ``base_url``."""

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 16,
        token_param: str = "max_tokens",
    ) -> None:
        super().__init__(model, temperature, max_tokens=max_tokens)
        if token_param not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("token_param must be 'max_tokens' or 'max_completion_tokens'")
        self.token_param = token_param
        self.name = f"openai:{model}"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise MissingKeyError("set OPENAI_API_KEY to use an openai:<model> client")
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")

    def request(self, system: str, user: str) -> tuple[str, dict, dict]:
        payload: dict = {
            "model": self.model,
            self.token_param: self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        headers = {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        return f"{self.base_url}/chat/completions", headers, payload

    def complete_with_usage(self, system: str, user: str) -> tuple[str, dict]:
        """Reply text and ``{"input_tokens", "output_tokens"}`` (OpenAI: prompt/completion)."""
        url, headers, payload = self.request(system, user)
        data = self._post(url, headers, payload)
        text = data["choices"][0]["message"]["content"] or ""
        u = data.get("usage") or {}
        usage = {}
        if "prompt_tokens" in u and "completion_tokens" in u:
            usage = {"input_tokens": u["prompt_tokens"], "output_tokens": u["completion_tokens"]}
        return text, usage

    def complete(self, system: str, user: str) -> str:
        return self.complete_with_usage(system, user)[0]


class ClaudeCLIClient:
    """Claude models through the local Claude command-line client (headless, no API key, no HTTP).

    Each call is ``claude -p --model <model> --tools "" --system-prompt <system>`` run in an empty
    temporary directory with no settings sources, so no project file or memory reaches the model.
    The CLI has no temperature or token-limit flag: sampling is the CLI default and the reply
    length is not capped (a reply that is not one word counts as a parse failure as usual).
    Every call is appended to ``raw_log`` (JSON lines: time, model, prompt, raw reply, token usage,
    the CLI's list-price cost, duration). The cost is notional on a subscription plan.
    """

    is_stub = False

    def __init__(self, model: str, timeout: float = 120.0, raw_log: Path | None = None) -> None:
        self.model = model
        self.name = f"claude-cli:{model}"
        self.timeout = timeout
        self.raw_log = raw_log
        self._cwd = tempfile.mkdtemp(prefix="namegame_cli_")
        self._lock = threading.Lock()

    def _command(self, system: str) -> list[str]:
        return [
            "claude", "-p", "--model", self.model, "--tools", "",
            "--system-prompt", system, "--output-format", "json",
            "--no-session-persistence", "--setting-sources", "", "--disable-slash-commands",
        ]  # fmt: skip

    def complete_with_usage(self, system: str, user: str) -> tuple[str, dict]:
        t0 = time.time()
        for attempt in range(3):
            try:
                proc = subprocess.run(
                    self._command(system), input=user, capture_output=True, text=True,
                    timeout=self.timeout, cwd=self._cwd, check=False,
                )  # fmt: skip
                data = json.loads(proc.stdout)
                if not data.get("is_error"):
                    break
                err = str(data.get("result"))[:200]
            except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
                err = type(e).__name__
            if attempt == 2:
                raise RuntimeError(f"claude CLI failed 3 times: {err}")
            time.sleep(2.0 * (attempt + 1))
        u = data.get("usage") or {}
        usage = {
            "input_tokens": u.get("input_tokens", 0)
            + u.get("cache_creation_input_tokens", 0)
            + u.get("cache_read_input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0),
        }
        text = data.get("result") or ""
        if self.raw_log is not None:
            row = {
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                "model": self.model,
                "user": user,
                "reply": text,
                "usage": u,
                "cost_usd_list_price": data.get("total_cost_usd"),
                "duration_ms": data.get("duration_ms"),
                "attempts": attempt + 1,
            }
            with self._lock:
                self.raw_log.parent.mkdir(parents=True, exist_ok=True)
                with self.raw_log.open("a") as f:
                    f.write(json.dumps(row) + "\n")
        return text, usage

    def complete(self, system: str, user: str) -> str:
        return self.complete_with_usage(system, user)[0]


def make_client(
    spec: str,
    temperature: float | None = None,
    max_tokens: int = 16,
    token_param: str = "max_tokens",
) -> LLMClient:
    """``stub``, ``anthropic:<model>`` or ``openai:<model>``. No default model is hard-coded.
    ``token_param`` only applies to OpenAI (``max_tokens`` or ``max_completion_tokens``)."""
    if spec == "stub":
        return StubClient()
    provider, _, model = spec.partition(":")
    if provider == "anthropic" and model:
        return AnthropicClient(model, temperature, max_tokens=max_tokens)
    if provider == "openai" and model:
        return OpenAIClient(model, temperature, max_tokens=max_tokens, token_param=token_param)
    if provider == "claude-cli" and model:
        if temperature is not None:
            raise ValueError("the claude CLI has no temperature flag; omit --temperature")
        return ClaudeCLIClient(model)
    raise ValueError(
        "client must be 'stub', 'anthropic:<model>', 'openai:<model>' or 'claude-cli:<model>'"
    )


class BudgetExceeded(RuntimeError):
    """The call or cost cap was reached; no further call is made."""


class CallBudget:
    """Thread-safe cap on the number of calls and on the cost computed from user-given prices.

    ``check()`` runs before every call and reserves it; ``charge(usage)`` runs after it with the
    token counts the API reported. Prices are USD per million tokens and are supplied by the
    user: none is built in. With several worker threads, calls already in flight when the cost
    cap is crossed still finish, so the cost can exceed the cap by at most ``workers`` calls.
    Retries inside one HTTP request (429, timeouts; up to 4 attempts) are not counted as calls,
    and a timed-out request may be billed without a usage report, so the caps are estimates, not
    hard limits.
    """

    def __init__(
        self,
        max_calls: int | None = None,
        max_cost_usd: float | None = None,
        price_in: float | None = None,
        price_out: float | None = None,
    ) -> None:
        if max_cost_usd is not None and (price_in is None or price_out is None):
            raise ValueError("a cost cap needs both prices (USD per million tokens)")
        self.max_calls = max_calls
        self.max_cost_usd = max_cost_usd
        self.price_in = price_in
        self.price_out = price_out
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self._lock = threading.Lock()

    @property
    def cost_usd(self) -> float | None:
        if self.price_in is None or self.price_out is None:
            return None
        return (self.tokens_in * self.price_in + self.tokens_out * self.price_out) / 1e6

    def preload(self, calls: int, tokens_in: int, tokens_out: int) -> None:
        """Count spend of an earlier invocation (a resumed run), so the cap covers the whole run."""
        with self._lock:
            self.calls += calls
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out

    def check(self) -> None:
        with self._lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise BudgetExceeded(f"call cap reached ({self.max_calls} calls)")
            cost = self.cost_usd
            if self.max_cost_usd is not None and cost is not None and cost >= self.max_cost_usd:
                raise BudgetExceeded(f"cost cap reached ({cost:.4f} of {self.max_cost_usd} USD)")
            self.calls += 1

    def charge(self, usage: dict, strict: bool = True) -> None:
        with self._lock:
            if self.max_cost_usd is not None and strict and not usage:
                raise BudgetExceeded(
                    "the reply reported no token usage, so the cost cap cannot hold"
                )
            self.tokens_in += int(usage.get("input_tokens", 0))
            self.tokens_out += int(usage.get("output_tokens", 0))


class BudgetedClient:
    """Wraps any client with a :class:`CallBudget`; forwards ``name`` and ``is_stub``."""

    def __init__(self, client: LLMClient, budget: CallBudget) -> None:
        self.client = client
        self.budget = budget
        self.name = client.name
        self.is_stub = client.is_stub

    def complete(self, system: str, user: str) -> str:
        self.budget.check()
        inner = getattr(self.client, "complete_with_usage", None)
        if inner is None:
            text, usage = self.client.complete(system, user), {}
        else:
            text, usage = inner(system, user)
        self.budget.charge(usage, strict=not self.is_stub)
        return text


NAME_RE = re.compile(r"[a-z]+")


def parse_name(text: str) -> str | None:
    """The single lowercase name in a reply, or None if the reply is not exactly one word."""
    words = NAME_RE.findall(text.lower())
    return words[0] if len(words) == 1 else None


def memory_prompt(memory: Sequence[str], m: int, ordered: bool = True) -> str:
    """The game prompt (``ordered=True``: the list is newest first, as in the game) or the probe
    prompt (``ordered=False``: the probe shuffles the memory, so no order is claimed)."""
    line = MEMORY_LINE if ordered else MEMORY_LINE_UNORDERED
    return f"{line}{', '.join(memory)}\n(You remember at most your last {m} encounters.)\n{ASK}"


def history_prompt(history: Sequence[tuple[str, str]], m: int) -> str:
    lines = [HISTORY_HEADER]
    for i, (own, heard) in enumerate(history[:m], start=1):
        lines.append(f"{i}. you: {own} / partner: {heard}")
    return "\n".join(lines) + f"\n{ASK}"


@dataclass
class CallStats:
    calls: int = 0
    retries: int = 0
    fallbacks: int = 0
    log: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, calls: int = 0, retries: int = 0, fallbacks: int = 0) -> None:
        """Thread-safe increment (the simulation may call the client from several threads)."""
        with self._lock:
            self.calls += calls
            self.retries += retries
            self.fallbacks += fallbacks


class LLMAgent(Agent):
    """A player whose choice is made by a language model.

    ``mode="memory"``: the model sees exactly the memory the rule-based agents use (the modal
    FIFO queue), so any difference in behaviour comes from the choice rule alone.
    ``mode="history"``: the model sees its last rounds as "you / partner" pairs and does its own
    bookkeeping; the modal memory is still kept for analysis.
    If a reply is not a single word the agent asks once more, then falls back to its newest
    remembered name; the counts are kept in ``stats`` and must be reported with any result.
    """

    kind = "llm"

    def __init__(
        self,
        m: int,
        client: LLMClient,
        mode: str = "memory",
        stats: CallStats | None = None,
        keep_log: bool = False,
    ) -> None:
        super().__init__(m)
        if mode not in ("memory", "history"):
            raise ValueError("mode must be 'memory' or 'history'")
        self.client = client
        self.mode = mode
        self.stats = stats if stats is not None else CallStats()
        self.keep_log = keep_log
        self.history: list[tuple[str, str]] = []
        self.ordered = True  # False in the probe, whose memories are shuffled

    def build_prompt(self) -> str:
        if self.mode == "history" and self.history:
            return history_prompt(self.history, self.m)
        return memory_prompt(self.memory, self.m, self.ordered)

    def choose(self, rng: random.Random) -> str:  # noqa: ARG002
        prompt = self.build_prompt()
        name = None
        reply = ""
        for attempt in range(2):
            self.stats.add(calls=1)
            text = prompt if attempt == 0 else prompt + "\nReply with exactly one name."
            try:
                reply = self.client.complete(SYSTEM_PROMPT, text)
            except BudgetExceeded:
                self.stats.add(calls=-1)  # the budget refused this call: it was never made
                raise
            name = parse_name(reply)
            if name is not None:
                break
            self.stats.add(retries=1)
        fell_back = name is None
        if fell_back:
            self.stats.add(fallbacks=1)
            name = self.memory[0]
        if self.keep_log:
            self.stats.log.append(
                {"prompt": prompt, "reply": reply, "name": name, "fallback": fell_back}
            )
        return name

    def update(self, own: str, heard: str) -> None:
        super().update(own, heard)
        self.history.insert(0, (own, heard))
        del self.history[self.m :]


def llm_factory(client: LLMClient, m: int, mode: str, stats: CallStats, keep_log: bool = False):
    """Agent factory for :func:`namegame.sim.run_emergence` (all agents share ``stats``)."""

    def make(_index: int, _rng: random.Random) -> Agent:
        return LLMAgent(m, client, mode, stats, keep_log)

    return make


# --------------------------------------------------------------------------------------------
# Memory probe: show the model memories with k slots of one name and m-k of another.
# --------------------------------------------------------------------------------------------

PROBE_NAMES = ("maria", "david")


def probe_memory(k: int, m: int, rng: random.Random) -> tuple[list[str], str]:
    """A random-order memory with ``k`` slots of the target name and ``m-k`` of the other.
    The target/other assignment is randomised so that name priors cancel out."""
    a, b = rng.sample(PROBE_NAMES, 2)
    mem = [a] * k + [b] * (m - k)
    rng.shuffle(mem)
    return mem, a


def run_probe(
    client: LLMClient,
    m: int = 12,
    reps: int = 20,
    seed: int = 0,
    mode: str = "memory",
    stats: CallStats | None = None,
    skip: set[tuple[int, int]] | None = None,
    on_row: Callable[[dict], None] | None = None,
) -> tuple[list[dict], CallStats]:
    """Ask ``reps`` times for each k = 0..m how often the model writes the name that fills k
    slots. Returns one row per call made. In ``history`` mode the memory is shown as rounds in
    which the player met that name in the matching slots (partner column), own column repeated.

    ``skip`` holds (k, rep) units that an earlier run finished (resume): their memories are still
    drawn so the random stream, and therefore every later memory, is the same as in an
    uninterrupted run, but the model is not asked. ``on_row`` is called after every finished row
    (the CLI writes its checkpoint there)."""
    rng = random.Random(f"probe|{seed}|{m}")
    stats = stats if stats is not None else CallStats()
    skip = skip or set()
    rows = []
    for k in range(m + 1):
        for rep in range(reps):
            mem, target = probe_memory(k, m, rng)
            if (k, rep) in skip:
                continue
            agent = LLMAgent(m, client, mode, stats)
            agent.ordered = False
            agent.memory[:] = mem
            if mode == "history":
                agent.history = [(x, x) for x in mem]
            name = agent.choose(rng)
            row = {
                "k_target": k,
                "m": m,
                "rep": rep,
                "target": target,
                "answer": name,
                "chose_target": int(name == target),
                "client": client.name,
                "is_stub": int(client.is_stub),
                "mode": mode,
            }
            rows.append(row)
            if on_row is not None:
                on_row(row)
    return rows, stats
