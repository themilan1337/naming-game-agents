"""Rule taken from the authors' released code, kept apart on purpose.

The source is https://github.com/scaplan/name-game (GPL-3.0, commit
d8acbd7e2555623cf29033fbd59ba96389461413). This module implements one rule that the paper does not
state: a replayed optimizing player counts as "deterministic" unless the two newest distinct names
in memory have equal counts. It follows the logic of the upstream function
``check_unique_plurality_item``, so this repository is licensed under GPL-3.0 as well (see
LICENSE). Only with this test does the replay of the human data give the published 75.2% and 59.1%
(Fig. 3C); the paper's own sentence ("excluding ties") gives 74.8% and 57.8%.

It only matters when replaying human data (``OptimizingAgent.predicted_name`` with
``tie_break="recent"``); simulations never call it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def newest_two_names_rule(memory: Sequence[str]) -> str | None:
    """Prediction of an optimizing player whose ties go to the newest name.

    ``memory`` lists the newest entry first. Returns the most frequent name (the newest of the
    tied ones), or None when the two newest distinct names have equal counts. A tie between older
    names does not count as a tie here.
    """
    counts = Counter(memory)  # first-seen order = newest entry first
    if not counts:
        return None
    items = list(counts.items())
    if len(items) == 1 or items[0][1] != items[1][1]:
        return counts.most_common()[0][0]
    return None
