"""Chart smoke tests on tiny fixtures: files are written, and STUB series never show up as LLM."""

import csv
import json

from namegame import charts
from namegame.cli import build_parser


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _probe_rows(client, is_stub):
    return [
        {
            "k_target": k,
            "m": 12,
            "rep": 0,
            "target": "a",
            "answer": "a",
            "chose_target": int(k > 6),
            "client": client,
            "is_stub": is_stub,
            "mode": "memory",
        }
        for k in range(13)
    ]


def test_rule_curves_match_the_rules():
    tp = dict(charts.rule_curve("threshold"))
    assert tp[8] == 1.0 and tp[4] == 0.0 and abs(tp[7] - 7 / 12) < 1e-9
    op = dict(charts.rule_curve("optimize"))
    assert op[7] == 1.0 and op[6] == 0.5 and op[5] == 0.0
    assert dict(charts.rule_curve("luce"))[3] == 0.25


def test_probe_figure_hides_stub_unless_requested(tmp_path):
    res = tmp_path / "results"
    _write(res / "llm" / "STUB_x_probe.csv", _probe_rows("STUB", 1))
    (res / "llm" / "STUB_x_probe.json").write_text(json.dumps({"is_stub": True, "client": "STUB"}))
    _write(res / "llm" / "real_probe.csv", _probe_rows("anthropic:m", 0))
    (res / "llm" / "real_probe.json").write_text(
        json.dumps({"is_stub": False, "client": "anthropic:m"})
    )
    charts.INCLUDE_STUB = False
    assert [m.get("client") for _, m in charts.llm_files(res, "probe.csv")] == ["anthropic:m"]
    out = charts.fig_probe(res, tmp_path / "fig")
    assert out is not None and out.exists()
    charts.INCLUDE_STUB = True
    try:
        assert len(charts.llm_files(res, "probe.csv")) == 2
    finally:
        charts.INCLUDE_STUB = False


def test_make_all_with_no_results_writes_only_probe_note(tmp_path):
    made = charts.make_all(tmp_path / "empty", tmp_path / "out")
    assert [p.name for p in made] == ["fig6_llm_probe.png"]


def test_convergence_and_tipping_figures_from_small_files(tmp_path):
    res = tmp_path / "r"
    conv = []
    for agent in ("threshold", "imitate", "optimize", "optimize_recent_tie", "luce"):
        for n in (24, 48, 96):
            conv.append(
                {
                    "agent": agent,
                    "N": n,
                    "mean_rounds": 20.0,
                    "sd_rounds": 5.0,
                    "published_mean": 21.0,
                    "published_sd": 5.0,
                }
            )
    _write(res / "repro_tableS8_convergence.csv", conv)
    tip = [
        {"agent": a, "N": 24, "fraction": f, "mean_final_new_share": min(1.0, f * 3)}
        for a in charts.RULES
        for f in (0.1, 0.2, 0.3)
    ]
    _write(res / "repro_tipping_complete.csv", tip)
    assert charts.fig_convergence(res, tmp_path / "o").exists()
    assert charts.fig_tipping(res, tmp_path / "o").exists()


def test_cli_parser_has_all_commands():
    p = build_parser()
    for cmd in (
        "table-s8",
        "tipping",
        "empirical",
        "topologies",
        "llm-probe",
        "llm-game",
        "llm-tipping",
        "charts",
    ):
        args = p.parse_args([cmd] + (["--client", "stub"] if cmd.startswith("llm") else []))
        assert args.cmd == cmd
