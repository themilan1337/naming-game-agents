"""Chart 2 of the note: the LLM probe and the LLM game next to the rule agents, plus the numbers.

Reads results/llm/<client>_probe.csv and <client>_game_complete_N<N>.csv (real runs only; STUB
files are ignored), plays the rule agents on the same network (same names, rounds, seed scheme)
and writes results/figures/fig7_llm_probe_and_game.png, results/llm/rule_games_N<N>.csv and
results/llm/llm_summary.json. Usage: uv run python scripts/llm_figure.py <client> [N] [rounds]
"""

import csv
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from namegame.charts import STYLE, rule_curve  # noqa: E402
from namegame.cli import NAME_POOL  # noqa: E402
from namegame.networks import make_network  # noqa: E402
from namegame.sim import run_emergence, run_tipping  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"
RULES = ("threshold", "imitate", "optimize", "luce")
SIMS = 300


def rows(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def main(client, n=24, rounds=40):
    probe = rows(R / "llm" / f"{client}_probe.csv")
    game = rows(R / "llm" / f"{client}_game_complete_N{n}.csv")
    assert not any(int(r["is_stub"]) for r in probe + game)
    by_k = defaultdict(list)
    for r in probe:
        by_k[int(r["k_target"])].append(int(r["chose_target"]))
    p_k = {k: sum(v) / len(v) for k, v in sorted(by_k.items())}
    # rule agents on the same network, names and round count (stop_at_convergence=False)
    net = make_network("complete", n, seed=2026)
    ref, conv = {}, {}
    out = []
    for kind in RULES:
        curves, cr = [], []
        for s in range(SIMS):
            rng = random.Random(f"rule|{kind}|{n}|{s}")
            run = run_emergence(kind, net, 12, rng, max_rounds=rounds, stop_at_convergence=False,
                                name_pool=NAME_POOL)  # fmt: skip
            curves.append(run.top_share())
            cr.append(run.converged_round)
        ref[kind] = [st.mean(c[t] for c in curves) for t in range(rounds)]
        got = [c for c in cr if c]
        conv[kind] = {"converged": len(got), "of": SIMS, "mean_round": round(st.mean(got), 1)}
        out += [{"agent": kind, "round": t + 1, "mean_top_share": round(ref[kind][t], 4)}
                for t in range(rounds)]  # fmt: skip
    with open(R / "llm" / f"rule_games_N{n}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["agent", "round", "mean_top_share"])
        w.writeheader()
        w.writerows(out)
    games = defaultdict(list)
    for r in game:
        games[int(r["game"])].append(r)
    gsum = []
    for g, rs in sorted(games.items()):
        rs.sort(key=lambda r: int(r["round"]))
        gsum.append({"game": g, "converged_round": rs[0]["converged_round"] or None,
                     "final_top_share": float(rs[-1]["top_share"]),
                     "final_success_rate": float(rs[-1]["success_rate"]),
                     "final_n_names": int(rs[-1]["n_names"])})  # fmt: skip
    summary = {"client": client, "N": n, "rounds": rounds, "probe_p_by_k": p_k,
               "probe_reps_per_k": {k: len(v) for k, v in sorted(by_k.items())},
               "llm_games": gsum, "rule_agents_same_setting": conv,
               "rule_final_top_share": {k: round(v[-1], 3) for k, v in ref.items()}}  # fmt: skip
    # committed-minority runs: rule agents with the same setup (names, N, rounds), LLM from its file
    tip_rule = {}
    for c in (3, 6):
        tip_rule[str(c)] = {}
        for kind in RULES:
            fin = []
            for s in range(SIMS):
                rng = random.Random(f"ruletip|{kind}|{n}|{c}|{s}")
                run = run_tipping(kind, net, 12, c, rng, max_rounds=rounds,
                                  old_name="emma", new_name="david")  # fmt: skip
                fin.append(run.rounds[-1].new_share)
            tip_rule[str(c)][kind] = {"mean_final_new_share": round(st.mean(fin), 3),
                                      "share_of_runs_above_half": round(
                                          sum(f > 0.5 for f in fin) / SIMS, 3)}  # fmt: skip
    summary["tipping_rule_agents_same_setting"] = tip_rule
    tp = R / "llm" / f"{client}_tipping_complete_N{n}.csv"
    if tp.exists():
        summary["tipping_llm"] = [
            {k: r[k] for k in ("n_confed", "fraction", "n_sims", "rounds", "mean_final_new_share")}
            for r in rows(tp)
        ]
    (R / "llm" / "llm_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for k in ("threshold", "optimize", "luce"):
        s = STYLE[k]
        a.plot(*zip(*rule_curve(k), strict=True), color=s["color"], ls=s["ls"], marker=s["marker"],
               ms=3, label=s["label"])  # fmt: skip
    a.plot(*zip(*p_k.items(), strict=True), color="black", marker="X", lw=2, ms=6,
           label=f"Claude Sonnet 5.5 ({len(probe)} calls)")  # fmt: skip
    tie_text = "6 of 12 is a tie: it wrote the\nname listed first, 6 of 6 times"
    a.annotate(tie_text, (6, 0), xytext=(1.0, 0.18), fontsize=7,
               arrowprops={"arrowstyle": "->", "lw": 0.7})  # fmt: skip
    a.set_xticks(range(13))
    a.set_xlabel("memory slots (of 12) holding the name")
    a.set_ylabel("probability the name is written")
    a.set_title("A. What the player writes, by how often a name is in its memory", fontsize=9)
    a.axvline(8, color="#0072B2", lw=0.8, ls=":")
    a.text(8.1, 0.45, "threshold:\n8 of 12", fontsize=7, color="#0072B2")
    a.legend(fontsize=7, loc="upper left")
    for k in RULES:
        s = STYLE[k]
        b.plot(range(1, rounds + 1), ref[k], color=s["color"], ls=s["ls"], label=s["label"])
    for g, rs in sorted(games.items()):
        b.plot([int(r["round"]) for r in rs], [float(r["top_share"]) for r in rs],
               color="black", lw=2 if g == 0 else 1.3, ls="-" if g == 0 else "--",
               label="Claude Sonnet 5.5, one game")  # fmt: skip
    b.set_xlabel("round")
    b.set_ylabel("share using the most common name")
    b.set_title(f"B. Share of {n} players using the most common name", fontsize=9)
    b.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(R / "figures" / "fig7_llm_probe_and_game.png", dpi=160)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 24, int(a[3]) if len(a) > 3 else 40)
