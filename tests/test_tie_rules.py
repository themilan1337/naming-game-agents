"""Tie rules of the optimizing agent, written from the rule text.

See README, "Licence and provenance".

Memories are listed newest first.
"""

import random
from pathlib import Path

import pytest

from namegame.agents import OptimizingAgent, OptimizingRecentTieAgent

SRC = Path(__file__).resolve().parents[1] / "src" / "namegame"


def test_recent_tie_break_returns_the_newest_of_two_tied_names():
    agent = OptimizingAgent(12, ["b", "a", "a", "b"], tie_break="recent")
    assert agent.choose(random.Random(0)) == "b"
    agent = OptimizingAgent(12, ["a", "b", "b", "a"], tie_break="recent")
    assert agent.choose(random.Random(0)) == "a"


def test_random_tie_break_returns_both_tied_names():
    agent = OptimizingAgent(12, ["b", "a", "a", "b"], tie_break="random")
    rng = random.Random(1)
    assert {agent.choose(rng) for _ in range(200)} == {"a", "b"}


# (memory newest first, expected for "optimize" (unique plurality), expected for
# "optimize_recent_tie" (tie to the newest name; skipped only when the two newest distinct names
# have equal counts))
TABLE = [
    (["a", "b", "a"], "a", "a"),  # unique plurality
    (["a", "b"], None, None),  # two-way tie of the two newest names
    (["a", "b", "b", "c", "c"], None, "b"),  # b and c tie, but the two newest names a, b differ
    (["a", "a", "a"], "a", "a"),  # one name only
    (["a", "b", "c"], None, None),  # three-way tie
]


@pytest.mark.parametrize(("memory", "plain", "recent"), TABLE)
def test_predicted_name_table(memory, plain, recent):
    assert OptimizingAgent(12, memory).predicted_name() == plain
    assert OptimizingRecentTieAgent(12, memory).predicted_name() == recent


def test_recent_tie_choice_matches_prediction_where_deterministic():
    memory = ["a", "b", "b", "c", "c"]
    # choose() sends the tie between b and c to b, the name that entered memory first among the
    # tied ones in newest-first order, i.e. the newest of the tied names
    assert OptimizingRecentTieAgent(12, memory).choose(random.Random(0)) == "b"


def test_released_code_rule_is_isolated_and_flagged():
    head = (SRC / "released_code_rules.py").read_text().splitlines()[:30]
    text = "\n".join(head)
    assert "GPL-3.0" in text
    assert "scaplan/name-game" in text
    # the main agents module does not carry the looser test any more
    agents = (SRC / "agents.py").read_text()
    assert "items[1][1]" not in agents
    assert "newest_two_names_rule" in agents
