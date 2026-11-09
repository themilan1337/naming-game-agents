"""Batch experiments: reproduction of published numbers and the topology extension.

Every function takes an explicit base seed, derives one ``random.Random`` per task from it
(string seeds are hashed deterministically by Python), and returns plain rows (dicts) so that
results do not depend on the number of worker processes.
"""

from __future__ import annotations

import random
import statistics as st
from collections.abc import Iterable
from multiprocessing import Pool

from .networks import make_network
from .sim import n_confederates, run_emergence, run_tipping

# Published values, SI Appendix Table S8 (3000 runs each, M = 12, Zipf over ten first names).
TABLE_S8 = {
    ("threshold", 24): (21.83, 10.62),
    ("threshold", 48): (23.81, 8.01),
    ("threshold", 96): (24.13, 5.31),
    ("imitate", 24): (15.09, 6.61),
    ("imitate", 48): (17.00, 6.85),
    ("imitate", 96): (16.99, 4.72),
    ("optimize", 24): (14.44, 5.50),
    ("optimize", 48): (15.20, 4.60),
    ("optimize", 96): (14.89, 3.12),
    ("luce", 24): (24.36, 14.39),
    ("luce", 48): (31.39, 13.13),
    ("luce", 96): (40.17, 10.38),
}


def _rng(*parts: object) -> random.Random:
    return random.Random("|".join(str(p) for p in parts))


def convergence_task(args: tuple[str, int, int, int, int]) -> dict:
    """Table S8 style: rounds until convergence on a complete network, ``sims`` runs."""
    kind, n, m, sims, seed = args
    net = make_network("complete", n)
    rng = _rng("conv", kind, n, m, seed)
    rounds = []
    for _ in range(sims):
        r = run_emergence(kind, net, m, rng, max_rounds=75).converged_round
        if r is not None:
            rounds.append(r)
    return {
        "agent": kind,
        "N": n,
        "M": m,
        "n_sims": sims,
        "n_converged": len(rounds),
        "mean_rounds": st.mean(rounds),
        "sd_rounds": st.pstdev(rounds),
    }


def tipping_task(args: tuple[str, str, int, int, int, int, int, int]) -> dict:
    """Final share of learners using the committed minority's name after ``rounds`` rounds."""
    kind, topology, n, n_conf, m, sims, rounds, seed = args
    net = make_network(topology, n, seed=seed)
    rng = _rng("tip", kind, topology, n, n_conf, m, seed)
    finals, flip_rounds = [], []
    for _ in range(sims):
        run = run_tipping(kind, net, m, n_conf, rng, max_rounds=rounds)
        finals.append(run.rounds[-1].new_share)
        if run.converged_round is not None:
            flip_rounds.append(run.converged_round)
    return {
        "agent": kind,
        "topology": topology,
        "N": n,
        "M": m,
        "n_confed": n_conf,
        "fraction": n_conf / n,
        "n_sims": sims,
        "rounds": rounds,
        "mean_final_new_share": st.mean(finals),
        "p_flipped": len(flip_rounds) / sims,
        "mean_flip_round": st.mean(flip_rounds) if flip_rounds else float("nan"),
    }


def emergence_curve_task(args: tuple[str, str, int, int, int, int, int]) -> list[dict]:
    """Per-round mean top-name share and coordination success, plus convergence time, for one
    (agent, topology, N). Runs continue after convergence so curves have equal length."""
    kind, topology, n, m, sims, rounds, seed = args
    rng = _rng("curve", kind, topology, n, m, seed)
    top = [0.0] * rounds
    succ = [0.0] * rounds
    conv: list[int] = []
    for _ in range(sims):
        net = make_network(topology, n, seed=rng.randrange(10**9))
        run = run_emergence(kind, net, m, rng, max_rounds=rounds, stop_at_convergence=False)
        for t, r in enumerate(run.rounds):
            top[t] += r.top_share
            succ[t] += r.success_rate
        if run.converged_round is not None:
            conv.append(run.converged_round)
    out = []
    for t in range(rounds):
        out.append(
            {
                "agent": kind,
                "topology": topology,
                "N": n,
                "M": m,
                "n_sims": sims,
                "round": t + 1,
                "mean_top_share": top[t] / sims,
                "mean_success_rate": succ[t] / sims,
                "n_converged_by_end": len(conv),
                "mean_converged_round": st.mean(conv) if conv else float("nan"),
            }
        )
    return out


def tipping_grid(n: int, lo: float = 0.04, hi: float = 0.40, step: int = 1) -> list[int]:
    """Integer numbers of committed players between ``lo`` and ``hi`` of the network."""
    ks = sorted({k for k in range(1, n) if lo <= k / n <= hi})
    return ks[::step]


def pool_map(fn, tasks: Iterable, processes: int = 2) -> list:
    tasks = list(tasks)
    if processes <= 1:
        return [fn(t) for t in tasks]
    with Pool(processes) as pool:
        return pool.map(fn, tasks, chunksize=1)


def tipping_point(rows: list[dict], threshold: float = 0.5) -> float | None:
    """Smallest committed fraction at which the mean final adoption reaches ``threshold``,
    linearly interpolated between the two grid points around the crossing (None if never)."""
    pts = sorted((r["fraction"], r["mean_final_new_share"]) for r in rows)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        if y0 < threshold <= y1:
            return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    if pts and pts[0][1] >= threshold:
        return pts[0][0]
    return None


__all__ = [
    "TABLE_S8",
    "convergence_task",
    "emergence_curve_task",
    "n_confederates",
    "pool_map",
    "tipping_grid",
    "tipping_point",
    "tipping_task",
]
