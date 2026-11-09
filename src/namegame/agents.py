"""Agent rules for the naming game.

The rules follow the description in Guilbeault, Caplan and Yang, "A simple threshold captures
the social learning of conventions", PNAS 123(17):e2508061123 (2026), Models section and SI
Appendix A.1. This is an independent implementation written from that description.

Memory model ("modal" FIFO queue of size M, newest entry first):

* both players produce the same name: the name is pushed to the front; if the queue is full the
  oldest entry is dropped;
* the names differ: the agent first removes the oldest entry that holds the name it just produced
  (if there is one), then pushes the name it heard to the front (dropping the oldest entry if the
  queue is still full).

Choice rules differ only in how a name is drawn from memory:

* ``threshold`` (TP): if one name fills at least ``M - M/ln(M)`` slots the agent always produces
  it, otherwise it samples a slot uniformly (Luce choice axiom);
* ``optimize`` (OP): always the most frequent name, ties broken at random;
* ``imitate`` (IM): inventory memory, a name is drawn uniformly; if the heard name is already in
  the inventory the whole inventory is replaced by that name (Baronchelli et al. 2006, Centola
  and Baronchelli 2015);
* ``luce``: sample a slot uniformly (probability matching);
* ``two_thirds``: like TP but with the cut-off 2M/3.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence

from .released_code_rules import newest_two_names_rule

KINDS = ("threshold", "imitate", "optimize", "luce", "two_thirds", "optimize_recent_tie")


def tp_threshold(m: int) -> int:
    """Smallest number of slots (out of m) that makes a name "productive" under the TP.

    The Tolerance Principle tolerates at most m / ln(m) exceptions, so a name needs at least
    m - m / ln(m) slots. For m = 12 that is 12 - 4.83 = 7.17, hence 8 slots.
    """
    if m < 2:
        raise ValueError("memory size must be at least 2")
    return math.ceil(m - m / math.log(m) - 1e-9)


def two_thirds_threshold(m: int) -> int:
    """Smallest number of slots that is at least two thirds of m (the paper's 2/3 rule)."""
    return math.ceil(2 * m / 3 - 1e-9)


def update_modal_memory(memory: list[str], m: int, own: str, heard: str) -> None:
    """Update ``memory`` in place with the modal FIFO model; index 0 is the newest entry."""
    if own != heard:
        for i in range(len(memory) - 1, -1, -1):  # oldest occurrence of the name just produced
            if memory[i] == own:
                del memory[i]
                break
    while len(memory) >= m:
        memory.pop()
    memory.insert(0, heard)


class Agent:
    """Base class: holds the memory and the shared update rule."""

    kind = "base"

    def __init__(self, m: int, memory: Sequence[str] = ()) -> None:
        if m < 2:
            raise ValueError("memory size must be at least 2")
        self.m = m
        self.memory: list[str] = list(memory)

    def choose(self, rng: random.Random) -> str:
        raise NotImplementedError

    def update(self, own: str, heard: str) -> None:
        update_modal_memory(self.memory, self.m, own, heard)

    def predicted_name(self) -> str | None:
        """The name this agent will certainly produce next, or None if it would sample."""
        return None

    def _sample(self, rng: random.Random) -> str:
        return self.memory[rng.randrange(len(self.memory))]


class LuceAgent(Agent):
    kind = "luce"

    def choose(self, rng: random.Random) -> str:
        return self._sample(rng)


class OptimizingAgent(Agent):
    """Always produce the most frequent name in memory.

    ``tie_break="random"`` follows the paper's text ("randomly when faced with ties", Models).
    ``tie_break="recent"`` picks, among tied names, the one that entered memory most recently.
    Newest-name ties are justified by the fit to SI Table S8 (published convergence rounds), see
    README, Results 1.
    """

    kind = "optimize"

    def __init__(self, m: int, memory: Sequence[str] = (), tie_break: str = "random") -> None:
        super().__init__(m, memory)
        if tie_break not in ("random", "recent"):
            raise ValueError("tie_break must be 'random' or 'recent'")
        self.tie_break = tie_break

    def predicted_name(self) -> str | None:
        """The name the agent certainly produces next, or None when it would break a tie at random.

        With ``tie_break="random"`` this is the unique plurality (the paper: "excluding ties").
        With ``tie_break="recent"`` it is the looser test of ``released_code_rules`` (GPL-origin
        rule, kept apart), needed to reproduce the published Fig. 3C numbers. It only matters
        when replaying human data; simulations never call it.
        """
        if self.tie_break == "recent":
            return newest_two_names_rule(self.memory)
        top = _top_names(self.memory)
        return top[0] if len(top) == 1 else None

    def choose(self, rng: random.Random) -> str:
        top = _top_names(self.memory)
        if len(top) == 1 or self.tie_break == "recent":
            return top[0]
        return rng.choice(top)


class OptimizingRecentTieAgent(OptimizingAgent):
    kind = "optimize_recent_tie"

    def __init__(self, m: int, memory: Sequence[str] = ()) -> None:
        super().__init__(m, memory, tie_break="recent")


class ThresholdAgent(Agent):
    kind = "threshold"
    _cutoff = staticmethod(tp_threshold)

    def __init__(self, m: int, memory: Sequence[str] = ()) -> None:
        super().__init__(m, memory)
        self.cutoff = self._cutoff(m)

    def productive_name(self) -> str | None:
        counts = Counter(self.memory)
        if not counts:
            return None
        best = [n for n, c in counts.items() if c == max(counts.values())]
        # Two names can both reach the cut-off only for tiny m; the ordering is then arbitrary
        # but deterministic (first in memory order).
        name = best[0]
        return name if counts[name] >= self.cutoff else None

    def predicted_name(self) -> str | None:
        return self.productive_name()

    def choose(self, rng: random.Random) -> str:
        name = self.productive_name()
        return name if name is not None else self._sample(rng)


class TwoThirdsAgent(ThresholdAgent):
    kind = "two_thirds"
    _cutoff = staticmethod(two_thirds_threshold)


class ImitatingAgent(Agent):
    kind = "imitate"

    def predicted_name(self) -> str | None:
        return self.memory[0] if len(self.memory) == 1 else None

    def choose(self, rng: random.Random) -> str:
        return self._sample(rng)

    def update(self, own: str, heard: str) -> None:
        if heard in self.memory:
            self.memory[:] = [heard]
            return
        while len(self.memory) >= self.m:
            self.memory.pop()
        self.memory.insert(0, heard)


class ConfederateAgent:
    """A committed player who always produces one name and never learns."""

    kind = "confederate"

    def __init__(self, name: str) -> None:
        self.name = name
        self.memory: list[str] = []

    def choose(self, rng: random.Random) -> str:  # noqa: ARG002
        return self.name

    def update(self, own: str, heard: str) -> None:
        return None

    def predicted_name(self) -> str:
        return self.name


_CLASSES: dict[str, type[Agent]] = {
    "threshold": ThresholdAgent,
    "imitate": ImitatingAgent,
    "optimize": OptimizingAgent,
    "optimize_recent_tie": OptimizingRecentTieAgent,
    "luce": LuceAgent,
    "two_thirds": TwoThirdsAgent,
}


def make_agent(kind: str, m: int, memory: Sequence[str] = ()) -> Agent:
    try:
        return _CLASSES[kind](m, memory)
    except KeyError:
        raise ValueError(f"unknown agent kind {kind!r}; choose from {sorted(_CLASSES)}") from None


def _top_names(memory: Sequence[str]) -> list[str]:
    counts = Counter(memory)
    best = max(counts.values())
    return [n for n, c in counts.items() if c == best]
