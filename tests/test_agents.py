"""Hand-checked cases for the memory model and for each choice rule."""

import random

import pytest

from namegame.agents import (
    ConfederateAgent,
    ImitatingAgent,
    LuceAgent,
    OptimizingAgent,
    OptimizingRecentTieAgent,
    ThresholdAgent,
    TwoThirdsAgent,
    make_agent,
    tp_threshold,
    two_thirds_threshold,
    update_modal_memory,
)


def test_tp_threshold_values():
    # m - m/ln(m): 12 -> 7.17 -> 8 slots (the paper's worked example); 16 -> 10.2 -> 11 of 16
    # (infants generalise with 11 of 16 but not 10 of 16, paper p. 3).
    assert tp_threshold(12) == 8
    assert tp_threshold(16) == 11
    assert tp_threshold(10) == 6
    assert tp_threshold(8) == 5
    with pytest.raises(ValueError):
        tp_threshold(1)


def test_two_thirds_equals_tp_at_12_but_not_at_11():
    assert two_thirds_threshold(12) == tp_threshold(12) == 8
    assert (
        two_thirds_threshold(10) == 7 != tp_threshold(10)
    )  # paper: they differ for M in 10, 11, 13, 14
    assert two_thirds_threshold(11) == 8 != tp_threshold(11)


# SI Tables S1-S3 (M = 5, index 0 = newest). Memory before: [A, B, A, ...]
def test_memory_match_not_full_adds_front():
    mem = ["A", "B", "A", "A"]  # Table S1
    update_modal_memory(mem, 5, own="B", heard="B")
    assert mem == ["B", "A", "B", "A", "A"]


def test_memory_match_full_drops_oldest():
    mem = ["A", "B", "A", "A", "C"]  # Table S2
    update_modal_memory(mem, 5, own="B", heard="B")
    assert mem == ["B", "A", "B", "A", "A"]


def test_memory_mismatch_removes_oldest_own_name_then_adds():
    mem = ["A", "B", "A", "C", "D"]  # produced A, heard C; oldest A is at index 2
    update_modal_memory(mem, 5, own="A", heard="C")
    assert mem == ["C", "A", "B", "C", "D"]


def test_memory_mismatch_own_name_absent_still_respects_limit():
    mem = ["A", "B", "C", "D", "E"]
    update_modal_memory(mem, 5, own="Z", heard="Y")
    assert mem == ["Y", "A", "B", "C", "D"]


def test_threshold_agent_productive_at_8_of_12():
    a = ThresholdAgent(12, ["x"] * 8 + ["y"] * 4)
    assert a.predicted_name() == "x"
    rng = random.Random(0)
    assert {a.choose(rng) for _ in range(50)} == {"x"}


def test_threshold_agent_samples_at_7_of_12():
    a = ThresholdAgent(12, ["x"] * 7 + ["y"] * 5)
    assert a.predicted_name() is None
    rng = random.Random(1)
    draws = [a.choose(rng) for _ in range(4000)]
    assert abs(draws.count("y") / 4000 - 5 / 12) < 0.03  # Luce sampling: 5/12 = 42% (paper p. 6)


def test_optimizing_agent_picks_plurality_even_at_7_of_12():
    a = OptimizingAgent(12, ["x"] * 7 + ["y"] * 5)
    rng = random.Random(2)
    assert {a.choose(rng) for _ in range(50)} == {"x"}
    assert a.predicted_name() == "x"


def test_optimizing_tie_random_vs_recent():
    mem = ["a", "b", "a", "b"]  # a and b tie, a is newest
    rnd = OptimizingAgent(12, mem)
    rng = random.Random(3)
    assert {rnd.choose(rng) for _ in range(200)} == {"a", "b"}
    assert rnd.predicted_name() is None
    rec = OptimizingRecentTieAgent(12, mem)
    assert {rec.choose(rng) for _ in range(50)} == {"a"}


def test_recent_tie_agent_reproduces_release_determinism_quirk():
    # counts a:3, b:2, c:3 in first-seen order a, b, c: the two newest names differ, so the
    # released code calls this deterministic and predicts "a" although c ties with a.
    a = OptimizingRecentTieAgent(12, ["a", "b", "c", "a", "c", "a", "b", "c"])
    assert a.predicted_name() == "a"
    # a:2, b:2 tie between the two newest names -> not deterministic
    assert OptimizingRecentTieAgent(12, ["a", "b", "a", "b"]).predicted_name() is None


def test_imitating_agent_collapses_on_known_name():
    a = ImitatingAgent(5, ["A", "B", "A", "C"])  # SI Table S6 (inventory with repeats)
    a.update("B", "B")
    assert a.memory == ["B"]
    assert a.predicted_name() == "B"


def test_imitating_agent_adds_unknown_name_with_limit():
    a = ImitatingAgent(3, ["A", "B", "C"])
    a.update("A", "Z")
    assert a.memory == ["Z", "A", "B"]
    assert a.predicted_name() is None


def test_imitating_agent_collapses_even_when_own_name_differs():
    # Table S6 text: "when an IM agent encounters a name already in memory" -> collapse
    a = ImitatingAgent(5, ["A", "B"])
    a.update("A", "B")
    assert a.memory == ["B"]


def test_luce_agent_probability_matching():
    a = LuceAgent(12, ["x"] * 3 + ["y"] * 9)
    rng = random.Random(4)
    frac_x = sum(a.choose(rng) == "x" for _ in range(6000)) / 6000
    assert abs(frac_x - 0.25) < 0.03


def test_two_thirds_agent_differs_from_tp_at_m10():
    mem = ["x"] * 6 + ["y"] * 4
    assert ThresholdAgent(10, mem).predicted_name() == "x"  # 6 >= tp_threshold(10) == 6
    assert TwoThirdsAgent(10, mem).predicted_name() is None  # needs 7


def test_confederate_never_changes():
    c = ConfederateAgent("new")
    c.update("new", "old")
    assert c.choose(random.Random(0)) == "new"


def test_make_agent_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_agent("bayes", 12)
