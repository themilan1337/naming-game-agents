import csv
import subprocess
from pathlib import Path

from namegame import cli

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_data.sh"


def test_write_csv_uses_lf_only(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    out = tmp_path / "results" / "r.csv"
    cli.write_csv(out, [{"a": 1, "b": "x"}])
    assert b"\r" not in out.read_bytes()
    with open(out, newline="") as fh:
        assert list(csv.DictReader(fh)) == [{"a": "1", "b": "x"}]


def test_write_csv_outside_root_does_not_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path / "elsewhere")
    cli.write_csv(tmp_path / "r.csv", [{"a": 1}])
    assert (tmp_path / "r.csv").exists()


def test_fetch_script_is_valid_bash_and_has_checksum_fallback():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    assert "shasum -a 256" in SCRIPT.read_text()
