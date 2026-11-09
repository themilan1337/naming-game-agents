"""Stub run of every LLM command through the CLI. The stub is not a language model: these tests
check the pipeline (files, labels, validity flags), never a model result."""

import json

import pytest

from namegame import cli

CASES = {
    "llm-probe": ["--reps", "1"],
    "llm-game": ["--n", "6", "--games", "1", "--rounds", "3"],
    "llm-tipping": ["--n", "6", "--confed", "1", "--games", "1", "--rounds", "3"],
}


@pytest.mark.parametrize("cmd", CASES)
def test_stub_probe_game_tipping_run_and_are_labelled_stub(cmd, tmp_path):
    cli.main([cmd, "--client", "stub", *CASES[cmd], "--results-dir", str(tmp_path)])
    files = sorted((tmp_path / "llm").iterdir())
    assert files and all(f.name.startswith("STUB_") for f in files), [f.name for f in files]
    metas = [f for f in files if f.suffix == ".json"]
    assert len(metas) == 1
    meta = json.loads(metas[0].read_text())
    assert meta["is_stub"] is True
    assert meta["fallbacks"] == 0 and meta["valid"] is True and meta["complete"] is True
    assert "STUB" in meta["client"]
    assert meta["budget"]["max_calls"] is None and meta["calls"] > 0


def test_stub_never_gets_a_real_name(tmp_path):
    cli.main(
        ["llm-probe", "--client", "stub", "--label", "foo", "--reps", "1"]
        + ["--results-dir", str(tmp_path)]
    )
    names = sorted(p.name for p in (tmp_path / "llm").iterdir())
    assert names and all(n.startswith("STUB_foo_probe") for n in names), names


def test_stub_rerun_starts_over_instead_of_refusing(tmp_path):
    argv = ["llm-probe", "--client", "stub", "--reps", "1", "--results-dir", str(tmp_path)]
    cli.main(argv)
    cli.main(argv)  # an existing stub checkpoint is replaced, not an error
    assert (tmp_path / "llm" / "STUB_stub_probe.csv").exists()


def test_stub_runs_do_not_touch_the_committed_results(tmp_path):
    before = sorted(p.name for p in (cli.ROOT / "results" / "llm").iterdir())
    cli.main(["llm-probe", "--client", "stub", "--reps", "1", "--results-dir", str(tmp_path)])
    after = sorted(p.name for p in (cli.ROOT / "results" / "llm").iterdir())
    assert before == after
