"""The tipping command covers both optimizing tie rules by default."""

import csv

from namegame import cli


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_tipping_includes_recent_tie_agent_by_default(tmp_path):
    cli.main(
        ["tipping", "--sims", "2", "--sizes", "24", "--processes", "1"]
        + ["--results-dir", str(tmp_path)]
    )
    points = _read(tmp_path / "repro_tipping_points.csv")
    agents = [r["agent"] for r in points]
    assert agents == ["threshold", "imitate", "optimize", "luce", "optimize_recent_tie"]
    complete = _read(tmp_path / "repro_tipping_complete.csv")
    assert {r["agent"] for r in complete} == set(agents)


def test_tipping_agents_option_restricts_the_agents(tmp_path):
    cli.main(
        ["tipping", "--sims", "2", "--sizes", "24", "--processes", "1"]
        + ["--agents", "threshold", "--results-dir", str(tmp_path)]
    )
    assert [r["agent"] for r in _read(tmp_path / "repro_tipping_points.csv")] == ["threshold"]
