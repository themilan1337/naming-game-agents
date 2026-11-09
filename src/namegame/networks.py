"""Interaction networks and the round-by-round pairing rule.

The paper's experiments and simulations use fully connected networks (24, 48 and 96 players;
SI also 1,000 and 10,000). Other topologies here are our own extension and are labelled as such
in every result.

Pairing rule: in every round the players are visited in random order; an unpaired player picks a
uniformly random still-unpaired neighbour. On a complete graph with an even number of players
this is a uniformly random perfect matching, which is the paper's pairing. On sparse graphs some
players can be left without a partner in a round and sit out.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx

TOPOLOGIES = ("complete", "ring", "small_world", "random_regular", "scale_free")


@dataclass(frozen=True)
class Network:
    """An undirected graph on nodes 0..n-1 stored as neighbour lists."""

    kind: str
    n: int
    neighbors: tuple[tuple[int, ...], ...]

    @property
    def is_complete(self) -> bool:
        return self.kind == "complete"

    def mean_degree(self) -> float:
        return sum(len(x) for x in self.neighbors) / self.n


def complete_network(n: int) -> Network:
    nbrs = tuple(tuple(j for j in range(n) if j != i) for i in range(n))
    return Network("complete", n, nbrs)


def from_graph(kind: str, graph: nx.Graph) -> Network:
    n = graph.number_of_nodes()
    if sorted(graph.nodes) != list(range(n)):
        graph = nx.convert_node_labels_to_integers(graph)
    return Network(kind, n, tuple(tuple(sorted(graph.neighbors(i))) for i in range(n)))


def make_network(kind: str, n: int, seed: int = 0) -> Network:
    """Build one of the supported topologies.

    * complete: the paper's setting.
    * ring: ring lattice, each node joined to its 4 nearest neighbours (extension).
    * small_world: connected Watts-Strogatz, k=6, rewiring p=0.1 (extension).
    * random_regular: random 6-regular graph (extension).
    * scale_free: Barabasi-Albert, m=3 (extension).
    """
    if kind == "complete":
        return complete_network(n)
    if kind == "ring":
        return from_graph(kind, nx.watts_strogatz_graph(n, 4, 0.0, seed=seed))
    if kind == "small_world":
        return from_graph(kind, nx.connected_watts_strogatz_graph(n, 6, 0.1, seed=seed))
    if kind == "random_regular":
        return from_graph(kind, nx.random_regular_graph(6, n, seed=seed))
    if kind == "scale_free":
        return from_graph(kind, nx.barabasi_albert_graph(n, 3, seed=seed))
    raise ValueError(f"unknown topology {kind!r}; choose from {TOPOLOGIES}")


def pair_round(net: Network, rng: random.Random) -> list[tuple[int, int]]:
    """Random pairing for one round (see module docstring)."""
    if net.is_complete:
        order = list(range(net.n))
        rng.shuffle(order)
        return [(order[i], order[i + 1]) for i in range(0, net.n - 1, 2)]
    order = list(range(net.n))
    rng.shuffle(order)
    taken = [False] * net.n
    pairs: list[tuple[int, int]] = []
    for u in order:
        if taken[u]:
            continue
        free = [v for v in net.neighbors[u] if not taken[v]]
        if free:
            v = free[rng.randrange(len(free))]
            taken[u] = taken[v] = True
            pairs.append((u, v))
    return pairs
