"""Charts for the naming-game results.

This module only reads the CSV files in ``results/`` and draws figures, so it can be reworked
without touching the simulator. Labels, colours and captions are written for a reader who knows
the paper but not this code. Each ``fig_*`` function takes the results folder and an output
folder, returns the path it wrote, or None when its input is missing.

Rules the figures follow so that no reader mistakes one kind of result for another:
* the paper's reproduction, our own extensions and LLM results are different figures or panels;
* anything produced by the rule-based stub client is drawn grey and labelled STUB, and is never
  mixed into the LLM series;
* missing LLM files give a visible "not run yet" note instead of an empty plot.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colour-blind-safe palette plus line styles and markers, so that the lines remain
# distinguishable in grey scale.
STYLE = {
    "threshold": {"label": "Threshold (TP)", "color": "#0072B2", "marker": "o", "ls": "-"},
    "imitate": {"label": "Imitating (IM)", "color": "#E69F00", "marker": "s", "ls": "--"},
    "optimize": {"label": "Optimizing (OP)", "color": "#D55E00", "marker": "^", "ls": "-."},
    "luce": {"label": "Probability matching (Luce)", "color": "#009E73", "marker": "D", "ls": ":"},
    "llm": {"label": "LLM agents", "color": "#000000", "marker": "X", "ls": "-"},
    "stub": {"label": "STUB (not an LLM)", "color": "#888888", "marker": "P", "ls": "--"},
}
RULES = ("threshold", "imitate", "optimize", "luce")
TOPO_LABEL = {
    "complete": "complete (as in the paper)",
    "ring": "ring lattice, k=4",
    "small_world": "small world, k=6, p=0.1",
    "random_regular": "random 6-regular",
    "scale_free": "scale free, m=3",
}

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    }
)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(x: str) -> float:
    return float(x)


INCLUDE_STUB = False  # set by make_all; stub files are only drawn on request, always in grey


def llm_files(results: Path, suffix: str) -> list[tuple[Path, dict]]:
    """(csv path, meta) for every LLM result file with this suffix; meta says if it is a stub."""
    out = []
    for p in sorted((results / "llm").glob(f"*_{suffix}")):
        meta_p = p.with_suffix(".json")
        meta = (
            json.loads(meta_p.read_text())
            if meta_p.exists()
            else {"is_stub": True, "client": "unknown"}
        )
        if meta.get("is_stub") and not INCLUDE_STUB:
            continue
        out.append((p, meta))
    return out


def _legend_below(fig, ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 0.0))


def _note(ax, text: str, x: float = 0.5, y: float = 0.5) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
        bbox={"boxstyle": "round", "fc": "white", "ec": "#bbbbbb"},
    )


# ------------------------------------------------------------------------------------------
# Reproduction of the paper
# ------------------------------------------------------------------------------------------
def fig_convergence(results: Path, out: Path) -> Path | None:
    rows = read_csv(results / "repro_tableS8_convergence.csv")
    if not rows:
        return None
    kinds = [
        ("threshold", "Threshold"),
        ("imitate", "Imitating"),
        ("optimize_recent_tie", "Optimizing\n(newest-\nname ties)"),
        ("optimize", "Optimizing\n(random\nties)"),
        ("luce", "Probability\nmatching\n(Luce)"),
    ]
    sizes = (24, 48, 96)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, n in zip(axes, sizes, strict=True):
        ax.axhspan(20, 30, color="#cccccc", alpha=0.5, lw=0, label="human networks: 20-30 rounds")
        for i, (k, _lab) in enumerate(kinds):
            r = next((x for x in rows if x["agent"] == k and int(x["N"]) == n), None)
            if r is None:
                continue
            ax.errorbar(
                i,
                _f(r["mean_rounds"]),
                yerr=_f(r["sd_rounds"]),
                fmt="o",
                color="#0072B2",
                capsize=3,
                label="our simulation (mean, bar = 1 SD)" if i == 0 else None,
            )
            if k != "optimize":
                ax.plot(
                    i,
                    _f(r["published_mean"]),
                    marker="D",
                    mfc="none",
                    color="#D55E00",
                    ls="",
                    label="published (SI Table S8)" if i == 0 else None,
                )
        ax.set_xticks(range(len(kinds)), [lab for _, lab in kinds], fontsize=7.5)
        ax.set_title(f"{n} players")
    axes[0].set_ylabel("rounds until convergence (mean)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Rounds until all players share one name (fully connected network, memory of 12 names)",
        fontsize=11,
    )
    fig.tight_layout()
    return _save(fig, out / "fig1_reproduction_convergence.png")


def fig_empirical(results: Path, out: Path) -> Path | None:
    rows = read_csv(results / "repro_empirical_accuracy.csv")
    if not rows:
        return None
    pick = [
        ("threshold", "rounds > 12", "Threshold"),
        ("imitate", "rounds > 12", "Imitating"),
        ("optimize_recent_tie", "rounds > 12", "Optimizing\n(newest-name\nties)"),
        ("optimize", "rounds > 12", "Optimizing\n(random\nties)"),
        ("luce", "rounds > 12", "Luce"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, (m, s, _lab) in enumerate(pick):
        r = next((x for x in rows if x["model"] == m and x["subset"] == s), None)
        if r is None:
            continue
        ax.bar(
            i - 0.2,
            _f(r["accuracy"]),
            0.38,
            color="#0072B2",
            label="our simulation" if i == 0 else None,
        )
        if r["published_accuracy"]:
            ax.bar(
                i + 0.2,
                _f(r["published_accuracy"]),
                0.38,
                color="#D55E00",
                label="published (PNAS Fig. 3C)" if i == 0 else None,
            )
    ax.set_xticks(range(len(pick)), [p[2] for p in pick], fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("share of human choices predicted")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Reproduction: next-choice accuracy on the human name-game data", fontsize=10)
    fig.tight_layout()
    return _save(fig, out / "fig2_reproduction_next_choice.png")


def fig_tipping(
    results: Path,
    out: Path,
    file: str = "repro_tipping_complete.csv",
    name: str = "fig3_reproduction_tipping.png",
    title: str | None = None,
    with_llm: bool = True,
) -> Path | None:
    rows = read_csv(results / file)
    if not rows:
        return None
    sizes = sorted({int(r["N"]) for r in rows})
    fig, axes = plt.subplots(
        1, len(sizes), figsize=(3.6 * len(sizes), 3.6), sharey=True, squeeze=False
    )
    for ax, n in zip(axes[0], sizes, strict=True):
        ax.axvspan(0.20, 0.25, color="#cccccc", alpha=0.6, lw=0, label="human tipping range 20-25%")
        for k in RULES:
            pts = sorted(
                (_f(r["fraction"]), _f(r["mean_final_new_share"]))
                for r in rows
                if r["agent"] == k and int(r["N"]) == n
            )
            if pts:
                s = STYLE[k]
                ax.plot(
                    *zip(*pts, strict=True),
                    color=s["color"],
                    marker=s["marker"],
                    ls=s["ls"],
                    ms=4,
                    label=s["label"],
                )
        if with_llm:
            _overlay_llm_tipping(ax, results, n)
        ax.set_title(f"{n} players")
        ax.set_xlabel("committed minority (share of players)")
    axes[0][0].set_ylabel("learners using the new name\nafter 75 rounds")
    _legend_below(fig, axes[0][0])
    fig.suptitle(
        title or "Reproduction: tipping by a committed minority, complete networks", fontsize=11
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    return _save(fig, out / name)


def _overlay_llm_tipping(ax, results: Path, n: int) -> None:
    found = False
    for p, meta in llm_files(results, f"tipping_complete_N{n}.csv"):
        rows = read_csv(p)
        pts = sorted((_f(r["fraction"]), _f(r["mean_final_new_share"])) for r in rows)
        if pts:
            s = STYLE["stub" if meta.get("is_stub") else "llm"]
            ax.plot(
                *zip(*pts, strict=True),
                color=s["color"],
                marker=s["marker"],
                ls=s["ls"],
                ms=5,
                label=f"{s['label']}: {meta.get('client', '')}"[:60],
            )
            found = True
    if not found:
        ax.text(
            0.98,
            0.04,
            "LLM series: not run yet",
            transform=ax.transAxes,
            ha="right",
            fontsize=7,
            color="#555555",
        )


# ------------------------------------------------------------------------------------------
# Extension: other topologies
# ------------------------------------------------------------------------------------------
def fig_topology_curves(results: Path, out: Path) -> Path | None:
    rows = read_csv(results / "ext_topology_curves.csv")
    if not rows:
        return None
    topos = [t for t in TOPO_LABEL if any(r["topology"] == t for r in rows)]
    n = int(rows[0]["N"])
    fig, axes = plt.subplots(
        1, len(topos), figsize=(3.2 * len(topos), 3.6), sharey=True, squeeze=False
    )
    for ax, t in zip(axes[0], topos, strict=True):
        for k in RULES:
            pts = sorted(
                (int(r["round"]), _f(r["mean_top_share"]))
                for r in rows
                if r["topology"] == t and r["agent"] == k
            )
            if pts:
                s = STYLE[k]
                ax.plot(*zip(*pts, strict=True), color=s["color"], ls=s["ls"], label=s["label"])
        _overlay_llm_game(ax, results, t, n)
        ax.set_title(TOPO_LABEL[t], fontsize=8)
        ax.set_xlabel("round")
    axes[0][0].set_ylabel("share using the most common name")
    _legend_below(fig, axes[0][0])
    fig.suptitle(
        f"Extension (not in the paper): emergence of a convention on other networks, N = {n}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    return _save(fig, out / "fig4_extension_topology_emergence.png")


def _overlay_llm_game(ax, results: Path, topology: str, n: int) -> None:
    for p, meta in llm_files(results, f"game_{topology}_N{n}.csv"):
        rows = read_csv(p)
        by_round: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            by_round[int(r["round"])].append(_f(r["top_share"]))
        pts = sorted((t, sum(v) / len(v)) for t, v in by_round.items())
        s = STYLE["stub" if meta.get("is_stub") else "llm"]
        ax.plot(
            *zip(*pts, strict=True), color=s["color"], ls=s["ls"], lw=2, label=f"{s['label']}"[:40]
        )


def fig_topology_tipping(results: Path, out: Path) -> Path | None:
    rows = read_csv(results / "ext_topology_tipping.csv")
    if not rows:
        return None
    topos = [t for t in TOPO_LABEL if any(r["topology"] == t for r in rows)]
    n = int(rows[0]["N"])
    fig, axes = plt.subplots(
        1, len(topos), figsize=(3.2 * len(topos), 3.6), sharey=True, squeeze=False
    )
    for ax, t in zip(axes[0], topos, strict=True):
        ax.axvspan(0.20, 0.25, color="#cccccc", alpha=0.6, lw=0)
        for k in RULES:
            pts = sorted(
                (_f(r["fraction"]), _f(r["mean_final_new_share"]))
                for r in rows
                if r["topology"] == t and r["agent"] == k
            )
            if pts:
                s = STYLE[k]
                ax.plot(
                    *zip(*pts, strict=True),
                    color=s["color"],
                    marker=s["marker"],
                    ls=s["ls"],
                    ms=3,
                    label=s["label"],
                )
        ax.set_title(TOPO_LABEL[t], fontsize=8)
        ax.set_xlabel("committed minority (share)")
    axes[0][0].set_ylabel("learners using the new name\nafter 75 rounds")
    _legend_below(fig, axes[0][0])
    fig.suptitle(f"Extension (not in the paper): tipping on other networks, N = {n}", fontsize=10)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    return _save(fig, out / "fig5_extension_topology_tipping.png")


# ------------------------------------------------------------------------------------------
# LLM probe: choice probability against memory composition
# ------------------------------------------------------------------------------------------
def rule_curve(kind: str, m: int = 12) -> list[tuple[int, float]]:
    """P(write the name that fills k of m slots) for a two-name memory, from the agent rules."""
    from .agents import tp_threshold

    out = []
    for k in range(m + 1):
        if kind == "threshold":
            thr = tp_threshold(m)
            p = 1.0 if k >= thr else (0.0 if m - k >= thr else k / m)
        elif kind == "optimize":
            p = 1.0 if k > m / 2 else (0.0 if k < m / 2 else 0.5)
        elif kind == "luce":
            p = k / m
        else:
            raise ValueError(kind)
        out.append((k, p))
    return out


def fig_probe(results: Path, out: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for k in ("threshold", "optimize", "luce"):
        s = STYLE[k]
        ax.plot(
            *zip(*rule_curve(k), strict=True),
            color=s["color"],
            ls=s["ls"],
            marker=s["marker"],
            ms=3,
            label=s["label"],
        )
    found = False
    for p, meta in llm_files(results, "probe.csv"):
        rows = read_csv(p)
        by_k: dict[int, list[int]] = defaultdict(list)
        for r in rows:
            by_k[int(r["k_target"])].append(int(r["chose_target"]))
        pts = sorted((k, sum(v) / len(v)) for k, v in by_k.items())
        s = STYLE["stub" if meta.get("is_stub") else "llm"]
        ax.plot(
            *zip(*pts, strict=True),
            color=s["color"],
            marker=s["marker"],
            ls=s["ls"],
            lw=2,
            ms=6,
            label=f"{s['label']}: {meta.get('client', '')}"[:70],
        )
        found = True
    if not found:
        _note(ax, "LLM series: not run yet", x=0.75, y=0.25)
    ax.set_xlabel("memory slots (out of 12) holding the name")
    ax.set_ylabel("probability the name is written")
    ax.set_xticks(range(0, 13))
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Does the choice jump at a threshold? Two-name memories", fontsize=10)
    fig.tight_layout()
    return _save(fig, out / "fig6_llm_probe.png")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def make_all(results: Path, out: Path, include_stub: bool = False) -> list[Path]:
    global INCLUDE_STUB
    INCLUDE_STUB = include_stub
    made = [
        fig_convergence(results, out),
        fig_empirical(results, out),
        fig_tipping(results, out),
        fig_topology_curves(results, out),
        fig_topology_tipping(results, out),
        fig_probe(results, out),
    ]
    return [p for p in made if p is not None]
