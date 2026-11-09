"""Call and cost caps, dry run, refusal to run a real client uncapped. No network, no keys."""

import httpx
import pytest

from namegame import cli
from namegame.llm import BudgetedClient, BudgetExceeded, CallBudget, StubClient


class Usage:
    """Client that answers 'emma' and reports token usage like the real clients do."""

    name = "scripted-usage"
    is_stub = False

    def __init__(self, tokens_in=1000, tokens_out=10):
        self.tokens = (tokens_in, tokens_out)
        self.calls = 0

    def complete_with_usage(self, system, user):
        self.calls += 1
        return "emma", {"input_tokens": self.tokens[0], "output_tokens": self.tokens[1]}

    def complete(self, system, user):
        return self.complete_with_usage(system, user)[0]


def test_max_calls_stops_at_the_cap():
    inner = Usage()
    client = BudgetedClient(inner, CallBudget(max_calls=5))
    for _ in range(5):
        assert client.complete("s", "u") == "emma"
    with pytest.raises(BudgetExceeded, match="call cap"):
        client.complete("s", "u")
    assert inner.calls == 5


def test_cost_cap_uses_given_prices():
    # 1,000 input tokens at 3.0 and 10 output tokens at 15.0 USD per million = 0.00315 USD
    inner = Usage(1000, 10)
    budget = CallBudget(max_cost_usd=0.003, price_in=3.0, price_out=15.0)
    client = BudgetedClient(inner, budget)
    client.complete("s", "u")
    assert budget.cost_usd == pytest.approx(0.00315)
    with pytest.raises(BudgetExceeded, match="cost cap"):
        client.complete("s", "u")
    assert inner.calls == 1


def test_cost_cap_refuses_a_reply_without_usage():
    class NoUsage(Usage):
        def complete_with_usage(self, system, user):
            return "emma", {}

    budget = CallBudget(max_cost_usd=1.0, price_in=1.0, price_out=1.0)
    with pytest.raises(BudgetExceeded, match="no token usage"):
        BudgetedClient(NoUsage(), budget).complete("s", "u")


def test_cost_cap_requires_prices(tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(
            ["llm-probe", "--client", "stub", "--max-cost-usd", "1", "--results-dir", str(tmp_path)]
        )
    msg = str(e.value.code)
    assert "--price-in" in msg and "--price-out" in msg and "no price is built in" in msg
    assert not list(tmp_path.rglob("*"))
    with pytest.raises(SystemExit):
        cli.main(
            ["llm-probe", "--client", "stub", "--max-cost-usd", "1", "--price-in", "3",
             "--results-dir", str(tmp_path)]
        )  # fmt: skip


def test_real_client_without_cap_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    called = []
    with monkeypatch.context() as m:
        m.setattr(cli, "make_client", lambda *a, **k: called.append(1))
        with pytest.raises(SystemExit) as e:
            cli.main(["llm-probe", "--client", "anthropic:m", "--results-dir", str(tmp_path)])
    assert "--max-calls" in str(e.value.code)
    assert called == []
    # the stub is never refused
    cli.main(["llm-probe", "--client", "stub", "--reps", "1", "--results-dir", str(tmp_path)])
    assert (tmp_path / "llm" / "STUB_stub_probe.csv").exists()


@pytest.mark.parametrize(
    ("argv", "planned"),
    [
        (["llm-probe", "--reps", "20"], 260),
        (["llm-game"], 8640),
        (["llm-tipping"], 29925),
    ],
)
def test_dry_run_prints_planned_calls(argv, planned, monkeypatch, tmp_path, capsys):
    def no_client(*a, **k):
        raise AssertionError("a client was created during a dry run")

    def no_http(self, *a, **k):
        raise AssertionError("HTTP during a dry run")

    monkeypatch.setattr(cli, "make_client", no_client)
    monkeypatch.setattr(httpx.Client, "post", no_http)
    cli.main(
        [*argv, "--client", "anthropic:example-model", "--dry-run", "--max-calls", "300",
         "--results-dir", str(tmp_path)]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert f"{planned:,} planned calls" in out.splitlines()[0]
    assert f"{2 * planned:,}" in out  # worst case with one re-ask per call
    assert not list(tmp_path.rglob("*")), "a dry run writes nothing"


def test_dry_run_worst_case_cost_needs_prices_and_warns_above_cap(tmp_path, capsys):
    cli.main(
        ["llm-probe", "--client", "openai:m", "--reps", "20", "--dry-run", "--max-calls", "100",
         "--max-cost-usd", "1", "--price-in", "1", "--price-out", "2",
         "--results-dir", str(tmp_path)]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert "worst-case cost" in out and "USD" in out
    assert "WARNING" in out and "--max-calls" in out


def test_stub_is_not_budget_blocked_by_missing_usage(tmp_path):
    inner = StubClient()
    client = BudgetedClient(inner, CallBudget(max_cost_usd=1.0, price_in=1.0, price_out=1.0))
    assert client.is_stub and client.complete("s", "Names in your memory: a, a\nq") == "a"
