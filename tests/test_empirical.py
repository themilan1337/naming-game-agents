"""Replay logic on a tiny synthetic fixture (hand-checked), plus the tsv loader."""

import pytest

from namegame.empirical import (
    Interaction,
    Observation,
    accuracy,
    expected_accuracy,
    infer_conventions,
    load_interactions,
    replay,
    seven_five_check,
)


def row(rnd, a, an, b, bn, ac=False, bc=False, exp="X", source="SYN"):
    return Interaction(exp, source, a, rnd, an, ac, b, rnd, bn, bc)


# Synthetic: players p and q meet each other for 4 rounds (M = 3).
FIXTURE = [
    row(1, "p", "x", "q", "y"),
    row(2, "p", "y", "q", "y"),
    row(3, "p", "y", "q", "y"),
    row(4, "p", "y", "q", "z"),
]


def test_replay_predictions_by_hand():
    obs = replay({"X": FIXTURE}, m=3, kinds=("threshold", "optimize", "imitate", "luce"))
    p = [o for o in obs if o.name is not None]
    by = {(o.round, o.exp_id): o for o in p}
    # Round 1 has no memory yet: no observations. Round 2 players p and q: two observations.
    assert min(o.round for o in obs) == 2
    # p after round 1: said x, heard y -> memory [y]. q: said y, heard x -> memory [x]
    # (own name y was not in memory; nothing removed).
    r2 = [o for o in obs if o.round == 2]
    assert [o.predicted["optimize"] for o in r2] == ["y", "x"]  # p -> y, q -> x
    assert [o.name for o in r2] == ["y", "y"]
    assert accuracy(r2, "optimize")[0] == 1  # p right, q wrong
    # Imitation: single-name inventories predict their name as well
    assert [o.predicted["imitate"] for o in r2] == ["y", "x"]
    # Luce probability of the observed name: p holds [y] -> 1.0; q holds [x] -> 0.0
    assert [o.p_observed["luce"] for o in r2] == [1.0, 0.0]
    assert expected_accuracy(r2, "luce") == (0.5, 2)
    assert by[(2, "X")].memory_full is False


def test_threshold_prediction_needs_cutoff_slots():
    # m = 3: tp cutoff = ceil(3 - 3/ln 3) = ceil(0.27) = 1, so any non-empty memory is productive
    obs = replay({"X": FIXTURE}, m=3, kinds=("threshold",))
    assert all(o.predicted["threshold"] is not None for o in obs)


def test_out_of_order_rounds_raise():
    bad = [row(1, "p", "x", "q", "y"), row(3, "p", "x", "q", "y")]
    with pytest.raises(ValueError):
        replay({"X": bad}, m=3, kinds=("threshold",))


def test_confederate_rows_are_not_scored():
    rows = [
        row(1, "p", "x", "c", "y", bc=True),
        row(2, "p", "x", "c", "y", bc=True),
    ]
    obs = replay({"X": rows}, m=3, kinds=("optimize",))
    assert len(obs) == 1 and obs[0].round == 2


def test_load_interactions(tmp_path):
    f = tmp_path / "d.tsv"
    f.write_text(
        "LocalRowNum\tExpID\tSourcePaper\tOrigExpNum\tAgent_A_id\tAgent_A_roundnum\t"
        "Agent_A_output\tAgent_A_isconfed\tAgent_B_id\tAgent_B_roundnum\tAgent_B_output\t"
        "Agent_B_isconfed\n"
        "1\tE1\tCB2015\t1\t10\t1\temma\tFALSE\t11\t1\tlisa\tTRUE\n"
    )
    ex = load_interactions(f)
    assert ex["E1"][0].a_name == "emma" and ex["E1"][0].b_confed is True


def _committed_fixture(p_round5):
    rows = [row(r, "p", "o", "c", "o", bc=True) for r in (1, 2, 3)]
    rows += [row(4, "p", "o", "c", "a", bc=True), row(5, "p", p_round5, "c", "a", bc=True)]
    rows += [row(6, "p", "a", "c", "a", bc=True), row(7, "p", "a", "c", "a", bc=True)]
    return rows


def test_infer_conventions():
    # committed player says "o" three times, then "a" four times -> original o, alternative a
    assert infer_conventions(_committed_fixture("o")) == ("o", "a")


def _tie_fixture(p_round5, source="CBBB2018", p_committed=False):
    # m = 3. c is committed and never counted. p says o,o,o,o,<p_round5>,a,a and hears
    # o,o,o,a,a,a,a. p's memory (newest first): after round 3 [o,o,o]; round 4 (said o, heard a):
    # drop an o, push a -> [a,o,o]. Rounds 5 and 6 are the only rounds with a full memory whose
    # top name holds exactly 2 slots (see the tests for what happens in each).
    names_p = ["o", "o", "o", "o", p_round5, "a", "a"]
    names_c = ["o", "o", "o", "a", "a", "a", "a"]
    return [
        row(r, "p", names_p[r - 1], "c", names_c[r - 1], ac=p_committed, bc=True, source=source)
        for r in range(1, 8)
    ]


def test_seven_five_not_counted_before_from_round():
    # p qualifies at rounds 5 and 6 only; asking from round 7 finds nothing
    fx = {"X": _tie_fixture("o")}
    assert seven_five_check(fx, m=3, high=2, from_round=7) == (0, 0, 0)


def test_seven_five_top_name_produced_is_a_top_hit():
    # round 5: memory [a,o,o], top o (2 slots), p says o -> hit. After round 5 (said o, heard a):
    # drop an o, push a -> [a,a,o]. Round 6: top a (2 slots), p says a -> hit. Round 7: [a,a,a].
    assert seven_five_check({"X": _tie_fixture("o")}, m=3, high=2, from_round=5) == (2, 2, 0)


def test_seven_five_other_name_produced_is_other():
    # round 5: memory [a,o,o], top o, p says a -> other. After round 5 (said a, heard a):
    # push a, drop the oldest -> [a,a,o]. Round 6: top a, p says a -> hit.
    assert seven_five_check({"X": _tie_fixture("a")}, m=3, high=2, from_round=5) == (2, 1, 1)


def test_seven_five_needs_the_top_name_to_hold_exactly_high_slots():
    # one new name each round, heard = said: memory never holds a name twice, so no trial
    names = ["x", "y", "z", "w", "v", "u"]
    rows = [row(r, "p", n, "c", n, source="CBBB2018") for r, n in enumerate(names, start=1)]
    assert seven_five_check({"X": rows}, m=3, high=2, from_round=1) == (0, 0, 0)


def test_seven_five_skips_committed_players():
    # same rows, but p is committed too: nobody is counted
    fx = {"X": _tie_fixture("o", p_committed=True)}
    assert seven_five_check(fx, m=3, high=2, from_round=5) == (0, 0, 0)
    assert seven_five_check({"X": _tie_fixture("o")}, m=3, high=2, from_round=5).trials == 2


def test_seven_five_uses_only_the_named_source():
    fx = {"X": _tie_fixture("o", source="OTHER")}
    assert seven_five_check(fx, m=3, high=2, from_round=5) == (0, 0, 0)
    assert seven_five_check(fx, m=3, high=2, from_round=5, source="OTHER").trials == 2


def _obs(rnd, name, predicted, p_obs):
    return Observation("X", "SYN", rnd, name, True, {"threshold": predicted}, {"luce": p_obs})


def test_accuracy_counts_only_rounds_after_min_round():
    # round 12 predicted right, round 13 predicted wrong; "rounds > 12" keeps only round 13
    obs = [_obs(12, "a", "a", 1.0), _obs(13, "b", "a", 0.0)]
    assert accuracy(obs, "threshold", min_round=12) == (0, 1)
    assert accuracy(obs, "threshold", min_round=0) == (1, 2)


def test_expected_accuracy_counts_only_rounds_after_min_round():
    obs = [_obs(12, "a", "a", 1.0), _obs(13, "b", "a", 0.0)]
    assert expected_accuracy(obs, "luce", 12) == (0.0, 1)
