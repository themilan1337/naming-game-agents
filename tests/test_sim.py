import random

import pytest

from namegame.networks import complete_network, make_network, pair_round
from namegame.sim import NEW, n_confederates, run_emergence, run_tipping


def test_complete_pairing_is_perfect_matching():
    net = complete_network(10)
    pairs = pair_round(net, random.Random(0))
    flat = [x for p in pairs for x in p]
    assert sorted(flat) == list(range(10))


@pytest.mark.parametrize("kind", ["ring", "small_world", "random_regular", "scale_free"])
def test_sparse_pairing_uses_only_edges_and_each_node_once(kind):
    net = make_network(kind, 24, seed=3)
    rng = random.Random(1)
    for _ in range(20):
        pairs = pair_round(net, rng)
        flat = [x for p in pairs for x in p]
        assert len(flat) == len(set(flat))
        assert all(v in net.neighbors[u] for u, v in pairs)
        assert len(pairs) >= 6  # a maximal matching on these graphs pairs at least a quarter


def test_ring_has_degree_four():
    net = make_network("ring", 12)
    assert all(len(x) == 4 for x in net.neighbors)
    assert set(net.neighbors[0]) == {1, 2, 10, 11}


def test_unknown_topology():
    with pytest.raises(ValueError):
        make_network("torus", 12)


def test_all_same_initial_name_converges_in_round_one():
    net = complete_network(8)
    run = run_emergence("threshold", net, 12, random.Random(0), initial_names=["7"] * 8)
    assert run.converged_round == 1
    assert run.rounds[0].top_share == 1.0
    assert run.rounds[0].success_rate == 1.0


def test_two_players_different_names_do_not_converge_in_round_one():
    net = complete_network(2)
    run = run_emergence("optimize", net, 12, random.Random(0), initial_names=["a", "b"], tol=0)
    # round 1: they say a and b, so one deviates and tol=0 means not converged
    assert run.rounds[0].top_share == 0.5
    assert run.rounds[0].success_rate == 0.0
    assert run.converged_round != 1


def test_emergence_is_seeded():
    net = complete_network(24)
    a = run_emergence("threshold", net, 12, random.Random(5))
    b = run_emergence("threshold", net, 12, random.Random(5))
    assert a.converged_round == b.converged_round
    assert a.top_share() == b.top_share()


@pytest.mark.parametrize("kind", ["threshold", "imitate", "optimize"])
def test_emergence_converges_on_complete_network(kind):
    rng = random.Random(11)
    net = complete_network(24)
    rounds = [run_emergence(kind, net, 12, rng).converged_round for _ in range(30)]
    assert all(r is not None for r in rounds)
    assert 8 < sum(rounds) / len(rounds) < 40


def test_tipping_without_confederates_never_produces_new():
    net = complete_network(24)
    run = run_tipping("threshold", net, 12, 0, random.Random(0), max_rounds=20)
    assert all(r.new_share == 0.0 for r in run.rounds)
    assert run.converged_round is None


def test_tipping_with_majority_flips_all_kinds():
    net = complete_network(24)
    for kind in ("threshold", "optimize", "imitate"):
        run = run_tipping(kind, net, 12, 14, random.Random(2), max_rounds=75)
        assert run.rounds[-1].new_share > 0.9, kind


def test_tipping_small_minority_does_not_flip_threshold_or_optimize():
    net = complete_network(48)
    for kind in ("threshold", "optimize"):
        rng = random.Random(9)
        finals = [
            run_tipping(kind, net, 12, 4, rng, max_rounds=75).rounds[-1].new_share
            for _ in range(10)
        ]
        assert max(finals) < 0.5, kind  # 4 of 48 is 8%, below the tipping point


def test_tipping_rejects_too_many_confederates():
    with pytest.raises(ValueError):
        run_tipping("threshold", complete_network(6), 12, 6, random.Random(0))


def test_n_confederates_rounding():
    assert n_confederates(48, 0.25) == 12
    assert n_confederates(24, 0.2) == 5  # 4.8 rounds to 5
    assert NEW == "new"
