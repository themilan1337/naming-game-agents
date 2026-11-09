"""ClaudeCLIClient with the subprocess mocked: no model is called."""

import json
import subprocess

import pytest

from namegame import llm


def fake_run(stdout, record=None):
    def run(cmd, **kw):
        if record is not None:
            record.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    return run


OK = json.dumps(
    {
        "result": "maria",
        "is_error": False,
        "total_cost_usd": 0.0025,
        "duration_ms": 1700,
        "usage": {
            "input_tokens": 2,
            "cache_creation_input_tokens": 600,
            "cache_read_input_tokens": 20,
            "output_tokens": 4,
        },
    }
)


def test_reply_usage_command_and_raw_log(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(subprocess, "run", fake_run(OK, calls))
    c = llm.make_client("claude-cli:claude-sonnet-5-5")
    c.raw_log = tmp_path / "raw" / "x.jsonl"
    text, usage = c.complete_with_usage("sys", "Names in your memory: maria")
    assert text == "maria"
    assert usage == {"input_tokens": 622, "output_tokens": 4}
    cmd, kw = calls[0]
    assert cmd[:3] == ["claude", "-p", "--model"] and "claude-sonnet-5-5" in cmd
    assert cmd[cmd.index("--tools") + 1] == "" and cmd[cmd.index("--setting-sources") + 1] == ""
    assert kw["input"] == "Names in your memory: maria"
    row = json.loads(c.raw_log.read_text())
    assert row["reply"] == "maria" and row["model"] == "claude-sonnet-5-5"
    assert row["cost_usd_list_price"] == 0.0025


def test_error_is_retried_then_raised(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_run(json.dumps({"is_error": True, "result": "x"})))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="3 times"):
        llm.make_client("claude-cli:m").complete("s", "u")


def test_temperature_refused():
    with pytest.raises(ValueError):
        llm.make_client("claude-cli:m", temperature=0.5)
