"""Checkpoints, resume, fallback-rate validity and budget stops, through the CLI.

No network and no keys: scripted clients and httpx mock transports only. Results go to tmp_path.
"""

import json
import threading

import httpx
import pytest

from namegame import charts, cli
from namegame.llm import AnthropicClient

CAP = ["--max-calls", "100000"]


class Scripted:
    """Answers with a fixed reply, counts calls, and raises on the call numbers in ``fail_on``."""

    name = "scripted"
    is_stub = False

    def __init__(self, reply="maria", fail_on=()):
        self.reply = reply
        self.fail_on = set(fail_on)
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, system, user):
        with self._lock:
            self.calls += 1
            n = self.calls
        if n in self.fail_on:
            raise RuntimeError(f"simulated failure on call {n}")
        return self.reply


def use(monkeypatch, client):
    monkeypatch.setattr(cli, "make_client", lambda *a, **k: client)
    return client


def run(tmp_path, *argv):
    cli.main([*argv, "--client", "openai:test", "--results-dir", str(tmp_path)])


def llm_dir(tmp_path):
    return tmp_path / "llm"


def read_csv(path):
    import csv

    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_probe_resumes_after_interruption(monkeypatch, tmp_path):
    use(monkeypatch, Scripted(fail_on={30}))
    with pytest.raises(RuntimeError, match="call 30"):
        run(tmp_path, "llm-probe", "--reps", "3", *CAP)
    ck = llm_dir(tmp_path) / "openai_test_probe.checkpoint.jsonl"
    assert len(ck.read_text().splitlines()) == 1 + 29  # header + 29 finished units
    assert not (llm_dir(tmp_path) / "openai_test_probe.csv").exists()

    second = use(monkeypatch, Scripted())
    run(tmp_path, "llm-probe", "--reps", "3", "--resume", *CAP)
    rows = read_csv(llm_dir(tmp_path) / "openai_test_probe.csv")
    assert len(rows) == 13 * 3
    assert len({(r["k_target"], r["rep"]) for r in rows}) == 13 * 3
    assert second.calls == 13 * 3 - 29  # the 29 finished units are not asked again
    meta = json.loads((llm_dir(tmp_path) / "openai_test_probe.json").read_text())
    assert meta["resumed_units"] == 29 and meta["calls"] == 39 and meta["valid"] is True


def test_game_resumes_at_the_unfinished_game(monkeypatch, tmp_path):
    args = ["llm-game", "--n", "6", "--games", "3", "--rounds", "3", *CAP]
    # a game is 3 rounds x 6 players = 18 calls; call 20 is in the second game
    use(monkeypatch, Scripted(reply="emma", fail_on={20}))
    with pytest.raises(RuntimeError, match="call 20"):
        run(tmp_path, *args)
    second = use(monkeypatch, Scripted(reply="emma"))
    run(tmp_path, *args, "--resume")
    assert second.calls == 2 * 18, "game 1 must not be played again"
    rows = read_csv(llm_dir(tmp_path) / "openai_test_game_complete_N6.csv")
    assert sorted({int(r["game"]) for r in rows}) == [0, 1, 2] and len(rows) == 3 * 3
    meta = json.loads((llm_dir(tmp_path) / "openai_test_game_complete_N6.json").read_text())
    assert meta["resumed_units"] == 1 and meta["calls"] == 54


def test_tipping_resumes_per_confed_and_game(monkeypatch, tmp_path):
    args = ["llm-tipping", "--n", "6", "--confed", "1", "2", "--games", "2", "--rounds", "2", *CAP]
    # learners: 5 then 4; calls per game 2 x 5 = 10 and 2 x 4 = 8; call 12 is in unit (1, game 1)
    use(monkeypatch, Scripted(reply="emma", fail_on={12}))
    with pytest.raises(RuntimeError):
        run(tmp_path, *args)
    second = use(monkeypatch, Scripted(reply="emma"))
    run(tmp_path, *args, "--resume")
    assert second.calls == 10 + 8 + 8  # units (1, g1), (2, g0), (2, g1) only
    rows = read_csv(llm_dir(tmp_path) / "openai_test_tipping_complete_N6.csv")
    assert [r["n_confed"] for r in rows] == ["1", "2"]


def test_third_call_timeout_resumes_from_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr("namegame.llm.time.sleep", lambda s: None)
    seen = []

    def handler(request):
        seen.append(1)
        if 3 <= len(seen) <= 6:  # the 3rd call times out on all 4 attempts
            raise httpx.ReadTimeout("simulated", request=request)
        body = {
            "content": [{"type": "text", "text": "maria"}],
            "usage": {"input_tokens": 50, "output_tokens": 2},
        }
        return httpx.Response(200, json=body)

    def make(*a, **k):
        client = AnthropicClient("m", api_key="k")
        client._http = httpx.Client(transport=httpx.MockTransport(handler))
        return client

    monkeypatch.setattr(cli, "make_client", make)
    argv = ["llm-probe", "--client", "anthropic:m", "--reps", "1", "--max-calls", "100"]
    with pytest.raises(httpx.ReadTimeout):
        cli.main([*argv, "--results-dir", str(tmp_path)])
    ck = llm_dir(tmp_path) / "anthropic_m_probe.checkpoint.jsonl"
    assert len(ck.read_text().splitlines()) == 1 + 2  # header and the first two units
    cli.main([*argv, "--resume", "--results-dir", str(tmp_path)])
    assert len(read_csv(llm_dir(tmp_path) / "anthropic_m_probe.csv")) == 13
    meta = json.loads((llm_dir(tmp_path) / "anthropic_m_probe.json").read_text())
    assert meta["resumed_units"] == 2 and meta["valid"] is True
    assert meta["budget"]["tokens_in"] == 13 * 50  # token usage survives the resume


@pytest.mark.parametrize(
    ("argv", "stem", "suffix"),
    [
        (["llm-probe", "--reps", "1"], "openai_test_probe", "probe.csv"),
        (
            ["llm-game", "--n", "6", "--games", "1", "--rounds", "2"],
            "openai_test_game_complete_N6",
            "game_complete_N6.csv",
        ),
        (
            ["llm-tipping", "--n", "6", "--confed", "1", "--games", "1", "--rounds", "2"],
            "openai_test_tipping_complete_N6",
            "tipping_complete_N6.csv",
        ),
    ],
)
def test_high_fallback_rate_run_is_marked_invalid(
    monkeypatch, tmp_path, capsys, argv, stem, suffix
):
    use(monkeypatch, Scripted(reply="I think it is emma"))  # never a single word
    with pytest.raises(SystemExit) as e:
        run(tmp_path, *argv, *CAP)
    assert e.value.code == 3
    out = capsys.readouterr().out
    assert "INVALID" in out and "fallback" in out
    d = llm_dir(tmp_path)
    assert (d / f"{stem}.INVALID.csv").exists() and (d / f"{stem}.INVALID.json").exists()
    assert not (d / f"{stem}.csv").exists()
    meta = json.loads((d / f"{stem}.INVALID.json").read_text())
    assert meta["valid"] is False and meta["fallback_rate"] == 1.0
    assert charts.llm_files(tmp_path, suffix) == [], "charts.py must not pick up an invalid run"


def test_a_few_fallbacks_below_the_limit_are_valid(monkeypatch, tmp_path):
    class Rare(Scripted):
        def complete(self, system, user):
            with self._lock:
                self.calls += 1
                bad = self.calls in (1, 2)  # one choice: first ask and re-ask both fail
            return "no idea at all" if bad else "maria"

    use(monkeypatch, Rare())
    run(tmp_path, "llm-probe", "--reps", "20", *CAP)  # 260 choices, 1 fallback = 0.4 percent
    meta = json.loads((llm_dir(tmp_path) / "openai_test_probe.json").read_text())
    assert meta["fallbacks"] == 1 and meta["valid"] is True
    assert len(charts.llm_files(tmp_path, "probe.csv")) == 1  # a valid run is drawn


def test_budget_stop_writes_partial_and_resume_finishes(monkeypatch, tmp_path, capsys):
    use(monkeypatch, Scripted())
    with pytest.raises(SystemExit) as e:
        run(tmp_path, "llm-probe", "--reps", "2", "--max-calls", "10")
    assert e.value.code == 3
    assert "call cap" in capsys.readouterr().out
    d = llm_dir(tmp_path)
    assert len(read_csv(d / "openai_test_probe.PARTIAL.csv")) == 10
    meta = json.loads((d / "openai_test_probe.PARTIAL.json").read_text())
    assert meta["complete"] is False and meta["valid"] is False
    assert not (d / "openai_test_probe.csv").exists()
    assert len((d / "openai_test_probe.checkpoint.jsonl").read_text().splitlines()) == 1 + 10

    # the cap covers the whole run: resuming with the same cap stops at once, a larger cap finishes
    second = use(monkeypatch, Scripted())
    with pytest.raises(SystemExit):
        run(tmp_path, "llm-probe", "--reps", "2", "--max-calls", "10", "--resume")
    assert second.calls == 0
    run(tmp_path, "llm-probe", "--reps", "2", "--max-calls", "100", "--resume")
    assert len(read_csv(d / "openai_test_probe.csv")) == 26
    assert not (d / "openai_test_probe.PARTIAL.csv").exists(), "stale partial file is removed"


def test_existing_checkpoint_needs_resume_and_the_same_settings(monkeypatch, tmp_path):
    use(monkeypatch, Scripted(fail_on={5}))
    with pytest.raises(RuntimeError):
        run(tmp_path, "llm-probe", "--reps", "2", *CAP)
    use(monkeypatch, Scripted())
    with pytest.raises(SystemExit, match="--resume"):
        run(tmp_path, "llm-probe", "--reps", "2", *CAP)
    with pytest.raises(SystemExit, match="other settings"):
        run(tmp_path, "llm-probe", "--reps", "3", "--resume", *CAP)
    with pytest.raises(SystemExit, match="other settings"):
        run(tmp_path, "llm-probe", "--reps", "2", "--seed", "7", "--resume", *CAP)


def test_a_torn_last_checkpoint_line_is_ignored(monkeypatch, tmp_path):
    use(monkeypatch, Scripted(fail_on={5}))
    with pytest.raises(RuntimeError):
        run(tmp_path, "llm-probe", "--reps", "2", *CAP)
    ck = llm_dir(tmp_path) / "openai_test_probe.checkpoint.jsonl"
    ck.write_text(ck.read_text() + '{"unit": "k9|r0", "ro')  # a kill in the middle of a write
    second = use(monkeypatch, Scripted())
    run(tmp_path, "llm-probe", "--reps", "2", "--resume", *CAP)
    assert second.calls == 26 - 4


@pytest.mark.parametrize(
    "argv",
    [
        ["llm-game", "--n", "6", "--games", "3", "--rounds", "4", "--workers", "1"],
        ["llm-probe", "--reps", "3"],
        ["llm-tipping", "--n", "6", "--confed", "1", "2", "--games", "2", "--rounds", "4"]
        + ["--workers", "1"],
    ],
)
def test_cap_holds_across_stop_and_resume(monkeypatch, tmp_path, argv):
    """Two runs with the same cap must not make more real calls than the cap in total."""
    cap = "30"
    first = use(monkeypatch, Scripted(reply="emma"))
    with pytest.raises(SystemExit):
        run(tmp_path, *argv, "--max-calls", cap)
    second = use(monkeypatch, Scripted(reply="emma"))
    with pytest.raises(SystemExit):
        run(tmp_path, *argv, "--resume", "--max-calls", cap)
    assert first.calls == 30
    assert first.calls + second.calls <= 30
    for meta_file in llm_dir(tmp_path).glob("*.json"):
        assert json.loads(meta_file.read_text())["calls"] <= 30
