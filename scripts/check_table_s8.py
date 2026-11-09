"""Pass/fail gate for results/repro_tableS8_convergence.csv.

Bands (fixed before looking at a rerun):
- threshold, imitate, luce, optimize_recent_tie: |diff_mean| <= 0.6 rounds against SI Table S8.
- optimize (random ties, as the paper text says): diff_mean between -1.9 and -1.2. If this row
  moves out of the band the tie-rule observation is gone and a human has to look.

Usage: python scripts/check_table_s8.py results/repro_tableS8_convergence.csv
Exit code 0 on pass, 1 otherwise. Last line: "table-s8 check: pass" or "table-s8 check: FAIL";
the line before it says the random-tie optimizer row is expected to be 1.2 to 1.9 rounds faster
than published and is not a match.
"""

from __future__ import annotations

import csv
import sys

MATCHED = ("threshold", "imitate", "luce", "optimize_recent_tie")
TOL = 0.6
OPT_BAND = (-1.9, -1.2)
NOTE_OPTIMIZE = (
    "note: the random-tie optimizer row is expected to be 1.2 to 1.9 rounds faster than "
    "published and is not a match"
)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: check_table_s8.py <csv>")
        return 2
    with open(argv[0], newline="") as fh:
        rows = list(csv.DictReader(fh))
    ok = bool(rows)
    seen = set()
    for r in rows:
        agent, n, d = r["agent"], r["N"], float(r["diff_mean"])
        if agent in MATCHED:
            good = abs(d) <= TOL
            rule = f"|d| <= {TOL}"
        elif agent == "optimize":
            good = OPT_BAND[0] <= d <= OPT_BAND[1]
            rule = f"{OPT_BAND[0]} <= d <= {OPT_BAND[1]}"
        else:
            continue
        seen.add(agent)
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {agent:20s} N={n:>3s} diff_mean={d:+.3f} ({rule})")
    missing = (set(MATCHED) | {"optimize"}) - seen
    if missing:
        print(f"FAIL missing agents: {sorted(missing)}")
        ok = False
    print(NOTE_OPTIMIZE)
    print("table-s8 check: pass" if ok else "table-s8 check: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
