"""Agent-based naming game: emergence of a convention and tipping by a committed minority."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field

from .agents import Agent, ConfederateAgent, make_agent
from .networks import Network, pair_round

OLD = "old"
NEW = "new"

AgentFactory = Callable[[int, random.Random], Agent]


def zipf_names(w: int = 10) -> tuple[list[str], list[float]]:
    """Names "1".."w" with Zipf weights 1/i, as the first-round name distribution of the paper's
    pure simulations (ten types, after the "dropdown menu" design of Centola and Baronchelli)."""
    names = [str(i + 1) for i in range(w)]
    weights = [1.0 / (i + 1) for i in range(w)]
    return names, weights


def draw_zipf(rng: random.Random, w: int = 10, pool: Sequence[str] | None = None) -> str:
    """One first name: Zipf-weighted over ``pool`` (if given) or over "1".."w"."""
    names, weights = zipf_names(len(pool) if pool is not None else w)
    if pool is not None:
        names = list(pool)
    return rng.choices(names, weights=weights, k=1)[0]


def make_learner(kind: str | AgentFactory, index: int, m: int, rng: random.Random, memory) -> Agent:
    """A learner from an agent kind or from a factory, with its memory set to ``memory``."""
    if callable(kind):
        agent = kind(index, rng)
        agent.memory[:] = list(memory)
        return agent
    return make_agent(kind, m, memory)


@dataclass
class RoundStats:
    """Per-round summary over the non-confederate players that produced a name."""

    top_share: float
    n_names: int
    success_rate: float  # fraction of pairs whose two names matched
    new_share: float = 0.0  # share of producers using NEW (tipping runs)


@dataclass
class Run:
    rounds: list[RoundStats] = field(default_factory=list)
    converged_round: int | None = None  # first round (1-based) with <= tol deviating producers

    def top_share(self) -> list[float]:
        return [r.top_share for r in self.rounds]


def _play_round(
    agents: Sequence[Agent | ConfederateAgent],
    net: Network,
    rng: random.Random,
    executor: Executor | None = None,
) -> tuple[list[str], int, int]:
    """Play one round. Returns (names of learners, n_pairs, n_matches).

    All names of a round are chosen before any memory is updated; this is equivalent to the
    sequential order because pairs are disjoint. With an ``executor`` the choices are made
    concurrently (used for LLM agents, whose calls are slow and independent).
    """
    pairs = pair_round(net, rng)
    order = [i for pair in pairs for i in pair]
    if executor is None:
        chosen = [agents[i].choose(rng) for i in order]
    else:
        chosen = list(executor.map(lambda i: agents[i].choose(rng), order))
    learner_names: list[str] = []
    matches = 0
    for p, (i, j) in enumerate(pairs):
        na, nb = chosen[2 * p], chosen[2 * p + 1]
        a, b = agents[i], agents[j]
        if na == nb:
            matches += 1
        if not isinstance(a, ConfederateAgent):
            learner_names.append(na)
        if not isinstance(b, ConfederateAgent):
            learner_names.append(nb)
        a.update(na, nb)
        b.update(nb, na)
    return learner_names, len(pairs), matches


def run_emergence(
    kind: str | AgentFactory,
    net: Network,
    m: int,
    rng: random.Random,
    max_rounds: int = 75,
    tol: int = 2,
    w: int = 10,
    stop_at_convergence: bool = True,
    initial_names: Sequence[str] | None = None,
    executor: Executor | None = None,
    name_pool: Sequence[str] | None = None,
) -> Run:
    """Start every player with one Zipf-drawn name and play until the network converges.

    Convergence (Centola and Baronchelli 2015, followed by the paper's SI A.10): in one round at
    most ``tol`` players produce a name other than the most used one.
    ``initial_names`` fixes every player's first name (for tests). ``kind`` is an agent kind
    from :data:`agents.KINDS` or a factory ``(index, rng) -> Agent`` (mixed populations, LLM
    agents).
    """
    agents: list[Agent | ConfederateAgent] = []
    for i in range(net.n):
        seed_name = initial_names[i] if initial_names is not None else draw_zipf(rng, w, name_pool)
        agents.append(make_learner(kind, i, m, rng, [seed_name]))
    run = Run()
    for t in range(1, max_rounds + 1):
        names, n_pairs, matches = _play_round(agents, net, rng, executor)
        counts = Counter(names)
        top = counts.most_common(1)[0][1] if counts else 0
        run.rounds.append(
            RoundStats(
                top_share=top / len(names) if names else 0.0,
                n_names=len(counts),
                success_rate=matches / n_pairs if n_pairs else 0.0,
            )
        )
        if run.converged_round is None and names and len(names) - top <= tol:
            run.converged_round = t
            if stop_at_convergence:
                break
    return run


def run_tipping(
    kind: str | AgentFactory,
    net: Network,
    m: int,
    n_confed: int,
    rng: random.Random,
    max_rounds: int = 75,
    tol: int = 2,
    executor: Executor | None = None,
    old_name: str = OLD,
    new_name: str = NEW,
) -> Run:
    """Established convention OLD in every learner's memory; ``n_confed`` committed players
    (placed at random nodes) always say ``new_name``. The run records how fast learners adopt it."""
    if not 0 <= n_confed < net.n:
        raise ValueError("n_confed must be between 0 and n-1")
    confeds = set(rng.sample(range(net.n), n_confed))
    agents: list[Agent | ConfederateAgent] = []
    for i in range(net.n):
        if i in confeds:
            agents.append(ConfederateAgent(new_name))
        else:
            agents.append(make_learner(kind, i, m, rng, [old_name] * m))
    run = Run()
    for t in range(1, max_rounds + 1):
        names, n_pairs, matches = _play_round(agents, net, rng, executor)
        counts = Counter(names)
        n_new = counts.get(new_name, 0)
        run.rounds.append(
            RoundStats(
                top_share=max(counts.values()) / len(names) if names else 0.0,
                n_names=len(counts),
                success_rate=matches / n_pairs if n_pairs else 0.0,
                new_share=n_new / len(names) if names else 0.0,
            )
        )
        if run.converged_round is None and names and len(names) - n_new <= tol:
            run.converged_round = t
    return run


def n_confederates(n: int, fraction: float) -> int:
    """Number of committed players for a critical-mass fraction (rounded as in the paper's code)."""
    return int(round(fraction * n))
