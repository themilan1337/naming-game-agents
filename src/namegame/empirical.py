"""Replay the human name-game data through the agent rules (paper Figs. 2-3 style checks).

Data: Centola and Baronchelli (2015) and Centola et al. (2018), as released by Guilbeault,
Caplan and Yang in https://github.com/scaplan/name-game (``data/NameGame-2015-2018-all.tsv``,
fetched by ``scripts/fetch_data.sh``; not stored in this repository).

For every human player we rebuild the memory each model would hold from the interactions that
player actually had, and ask what the model predicts the player will say next.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from .agents import Agent, make_agent, update_modal_memory


@dataclass(frozen=True)
class Interaction:
    exp_id: str
    source: str
    a_id: str
    a_round: int
    a_name: str
    a_confed: bool
    b_id: str
    b_round: int
    b_name: str
    b_confed: bool


def load_interactions(path: str | Path) -> dict[str, list[Interaction]]:
    """Read the tsv; returns experiment id -> interactions in file (= chronological) order."""
    out: dict[str, list[Interaction]] = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[r["ExpID"]].append(
                Interaction(
                    exp_id=r["ExpID"],
                    source=r["SourcePaper"],
                    a_id=r["Agent_A_id"],
                    a_round=int(r["Agent_A_roundnum"]),
                    a_name=r["Agent_A_output"],
                    a_confed=r["Agent_A_isconfed"] == "TRUE",
                    b_id=r["Agent_B_id"],
                    b_round=int(r["Agent_B_roundnum"]),
                    b_name=r["Agent_B_output"],
                    b_confed=r["Agent_B_isconfed"] == "TRUE",
                )
            )
    return dict(out)


@dataclass
class Observation:
    """One human choice together with each model's prediction made from the replayed memory."""

    exp_id: str
    source: str
    round: int
    name: str
    memory_full: bool
    # kind -> deterministic prediction (None if the model would sample) ...
    predicted: dict[str, str | None]
    # ... and the probability the model gives to the observed name (for sampling models)
    p_observed: dict[str, float]


def replay(
    experiments: dict[str, list[Interaction]],
    m: int = 12,
    kinds: Iterable[str] = ("threshold", "imitate", "optimize", "luce", "two_thirds"),
    skip_confederates: bool = True,
) -> list[Observation]:
    """Replay every experiment; one Observation per (human player, round)."""
    kinds = tuple(kinds)
    observations: list[Observation] = []
    for exp_id, rows in experiments.items():
        agents: dict[str, dict[str, Agent]] = {}
        played: Counter[str] = Counter()
        for row in rows:
            for me, my_round, my_name, my_confed, other_name in (
                (row.a_id, row.a_round, row.a_name, row.a_confed, row.b_name),
                (row.b_id, row.b_round, row.b_name, row.b_confed, row.a_name),
            ):
                if me not in agents:
                    agents[me] = {k: make_agent(k, m) for k in kinds}
                if my_round != played[me] + 1:
                    raise ValueError(f"{exp_id}: player {me} round {my_round} out of order")
                mem = agents[me]
                if (not my_confed or not skip_confederates) and played[me] > 0:
                    observations.append(
                        Observation(
                            exp_id=exp_id,
                            source=row.source,
                            round=my_round,
                            name=my_name,
                            memory_full=len(next(iter(mem.values())).memory) >= m,
                            predicted={k: mem[k].predicted_name() for k in kinds},
                            p_observed={
                                k: mem[k].memory.count(my_name) / len(mem[k].memory) for k in kinds
                            },
                        )
                    )
                for agent in mem.values():
                    agent.update(my_name, other_name)
                played[me] += 1
    return observations


def accuracy(observations: Iterable[Observation], kind: str, min_round: int = 0) -> tuple[int, int]:
    """(hits, n) of the deterministic predictions of ``kind`` for rounds > ``min_round``."""
    hits = n = 0
    for o in observations:
        if o.round <= min_round:
            continue
        p = o.predicted[kind]
        if p is None:
            continue
        n += 1
        hits += p == o.name
    return hits, n


def expected_accuracy(
    observations: Iterable[Observation], kind: str, min_round: int = 0
) -> tuple[float, int]:
    """Mean probability the model assigns to the observed name, over all observations
    (used for the sampling-only Luce model, whose "prediction" is a distribution)."""
    total = 0.0
    n = 0
    for o in observations:
        if o.round <= min_round:
            continue
        total += o.p_observed[kind]
        n += 1
    return (total / n if n else float("nan")), n


def infer_conventions(rows: list[Interaction]) -> tuple[str, str]:
    """(original, alternative) names of a tipping-point experiment (Centola et al. 2018).

    Committed players ("confederates") are ordinary participants until they are activated.
    The alternative is the name they use most often; the original is the most used other name
    before the first time a committed player says the alternative. "na" is not a name.
    """
    cc: Counter[str] = Counter()
    for r in rows:
        if r.a_confed:
            cc[r.a_name] += 1
        if r.b_confed:
            cc[r.b_name] += 1
    alts = [n for n, _ in cc.most_common() if n != "na"]
    if not alts:
        raise ValueError("no committed players in this experiment")
    alt = alts[0]
    first = next(
        i
        for i, r in enumerate(rows)
        if (r.a_confed and r.a_name == alt) or (r.b_confed and r.b_name == alt)
    )
    pre: Counter[str] = Counter()
    for r in rows[:first]:
        pre[r.a_name] += 1
        pre[r.b_name] += 1
    originals = [n for n, _ in pre.most_common() if n not in (alt, "na")]
    if not originals:
        raise ValueError("no original convention before the committed players switch")
    return originals[0], alt


class SevenFive(NamedTuple):
    """Counts of the 7:5 check: trials, trials where the top name was produced, the rest."""

    trials: int
    top_hits: int
    other: int


def seven_five_check(
    experiments: dict[str, list[Interaction]],
    m: int = 12,
    high: int = 7,
    source: str = "CBBB2018",
    from_round: int = 22,
) -> SevenFive:
    """Paper, "Predicting Convention Change": when a learner's full memory has its most frequent
    name in exactly ``high`` of ``m`` slots (7 of 12), how often is that name produced next?

    Definition: over experiments whose ``source`` is ``source``, for every non-committed player's
    round ``>= from_round``, replay the modal FIFO memory (``update_modal_memory``, the same
    update for every rule); if the memory is full and one name holds exactly ``high`` slots and
    no name holds more, count a trial; it is a top hit if the player produced that name, else
    "other". The paper reports 949 observations and 29% other (TP predicts 42%, OP 0%).

    Origin: the count definition was taken from the column names and numbers of the authors'
    released output file (7:12 row: CBBB2018, from round 22, 1343 trials, 949 top name, 394 other)
    and from the paper text, and implemented independently here. Caveat: the paper says "across
    all ten networks" while the released file names CBBB2018 only; ``source`` selects which.
    """
    trials = top_hits = 0
    for rows in experiments.values():
        if not rows or rows[0].source != source:
            continue
        memories: dict[str, list[str]] = {}
        for r in rows:
            for me, my_round, name, confed, other in (
                (r.a_id, r.a_round, r.a_name, r.a_confed, r.b_name),
                (r.b_id, r.b_round, r.b_name, r.b_confed, r.a_name),
            ):
                mem = memories.setdefault(me, [])
                if not confed and my_round >= from_round and len(mem) == m:
                    counts = Counter(mem)
                    top, top_n = counts.most_common(1)[0]
                    if top_n == high and sum(c == high for c in counts.values()) == 1:
                        trials += 1
                        top_hits += name == top
                update_modal_memory(mem, m, name, other)
    return SevenFive(trials, top_hits, trials - top_hits)
