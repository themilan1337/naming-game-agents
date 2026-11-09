"""The table-s8 gate (scripts/check_table_s8.py) on small fixture CSVs."""

import csv
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_table_s8.py"
spec = importlib.util.spec_from_file_location("check_table_s8", SCRIPT)
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)

GOOD = {
    ("threshold", 24): -0.06,
    ("threshold", 48): 0.38,
    ("imitate", 24): 0.50,
    ("luce", 96): 0.40,
    ("optimize_recent_tie", 24): 0.17,
    ("optimize", 24): -1.64,
    ("optimize", 48): -1.74,
    ("optimize", 96): -1.36,
}


def write(tmp_path, overrides=None):
    rows = dict(GOOD)
    rows.update(overrides or {})
    path = tmp_path / "t.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["agent", "N", "diff_mean"])
        w.writeheader()
        for (agent, n), d in rows.items():
            w.writerow({"agent": agent, "N": n, "diff_mean": d})
    return path


def test_passes_when_rows_inside_tolerance(tmp_path, capsys):
    assert chk.main([str(write(tmp_path))]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == "table-s8 check: pass"
    assert lines[-2] == chk.NOTE_OPTIMIZE
    assert "1.2 to 1.9 rounds faster than published and is not a match" in lines[-2]


def test_fails_when_a_matched_row_is_off(tmp_path, capsys):
    assert chk.main([str(write(tmp_path, {("imitate", 24): 0.9}))]) == 1
    out = capsys.readouterr().out
    assert "imitate" in out and "FAIL" in out
    assert out.strip().splitlines()[-1] == "table-s8 check: FAIL"


def test_fails_when_random_tie_rows_leave_the_expected_band(tmp_path, capsys):
    assert chk.main([str(write(tmp_path, {("optimize", 24): -0.2}))]) == 1
    assert "optimize" in capsys.readouterr().out


def test_fails_on_missing_file_rows(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("agent,N,diff_mean\n")
    assert chk.main([str(p)]) == 1
