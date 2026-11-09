"""Command line: ``uv run python -m namegame.cli <command> --help``.

Commands: table-s8, tipping, empirical, topologies (reproduction and extension on rule-based
agents), llm-probe, llm-game, llm-tipping (LLM agents; need a key unless ``--client stub``),
charts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import empirical as emp
from . import experiments as ex
from .agents import KINDS
from .llm import (
    PROBE_NAMES,
    SYSTEM_PROMPT,
    BudgetedClient,
    BudgetExceeded,
    CallBudget,
    CallStats,
    history_prompt,
    llm_factory,
    make_client,
    memory_prompt,
    run_probe,
)
from .networks import TOPOLOGIES, make_network
from .sim import run_emergence, run_tipping

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "cache" / "ref" / "NameGame-2015-2018-all.tsv"

RULE_AGENTS = ("threshold", "imitate", "optimize", "luce")
TIPPING_AGENTS = [*RULE_AGENTS, "optimize_recent_tie"]
NAME_POOL = [
    "emma",
    "sarah",
    "jessica",
    "jennifer",
    "john",
    "mary",
    "chris",
    "samantha",
    "anna",
    "david",
]


def _shown(path: Path) -> Path:
    """Path relative to ROOT when it is inside, else the path itself."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {_shown(path)} ({len(rows)} rows)")


def cmd_table_s8(a: argparse.Namespace) -> None:
    kinds = ("threshold", "imitate", "optimize", "optimize_recent_tie", "luce")
    tasks = [(k, n, 12, a.sims, a.seed) for k in kinds for n in (24, 48, 96)]
    rows = []
    for r in ex.pool_map(ex.convergence_task, tasks, a.processes):
        pub = ex.TABLE_S8.get((r["agent"].replace("_recent_tie", ""), r["N"]))
        r["published_mean"], r["published_sd"] = pub
        r["diff_mean"] = r["mean_rounds"] - pub[0]
        r["se_mean"] = r["sd_rounds"] / r["n_converged"] ** 0.5
        rows.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    write_csv(RESULTS / "repro_tableS8_convergence.csv", rows)


def cmd_tipping(a: argparse.Namespace) -> None:
    tasks = []
    for n in a.sizes:
        for k in a.agents:
            for n_conf in ex.tipping_grid(n, step=2 if n >= 96 else 1):
                tasks.append((k, "complete", n, n_conf, 12, a.sims, 75, a.seed))
    rows = ex.pool_map(ex.tipping_task, tasks, a.processes)
    write_csv(RESULTS / "repro_tipping_complete.csv", [_round(r) for r in rows])
    summary = []
    for n in a.sizes:
        for k in a.agents:
            sub = [r for r in rows if r["N"] == n and r["agent"] == k]
            tp = ex.tipping_point(sub)
            summary.append(
                {"agent": k, "N": n, "tipping_fraction_50pct": None if tp is None else round(tp, 3)}
            )
    write_csv(RESULTS / "repro_tipping_points.csv", summary)


def _round(r: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}


def cmd_empirical(a: argparse.Namespace) -> None:
    if not DATA.exists():
        sys.exit("data missing: run scripts/fetch_data.sh first")
    experiments = emp.load_interactions(DATA)
    kinds = ("threshold", "imitate", "optimize", "optimize_recent_tie", "luce", "two_thirds")
    obs = emp.replay(experiments, 12, kinds)
    rows = []

    def add(model, subset, hits, n, published, note=""):
        rows.append(
            {
                "model": model,
                "subset": subset,
                "hits": hits,
                "n": n,
                "accuracy": round(hits / n, 4),
                "published_accuracy": published,
                "note": note,
            }
        )

    for kind, pub, note in (
        ("threshold", 0.879, "published n = 8,893"),
        ("imitate", 0.811, ""),
        ("optimize", 0.752, "unique plurality, ties at random (paper text)"),
        ("optimize_recent_tie", 0.752, "newest-name ties, looser determinism test (see README)"),
        ("two_thirds", "", ""),
    ):
        for label, min_round in (("all rounds", 0), ("rounds > 12", 12)):
            h, n = emp.accuracy(obs, kind, min_round)
            add(
                kind,
                label,
                h,
                n,
                pub if label == "rounds > 12" or kind == "threshold" else "",
                note,
            )
    for label, min_round in (("all rounds", 0), ("rounds > 12", 12)):
        acc, n = emp.expected_accuracy(obs, "luce", min_round)
        add("luce", label, round(acc * n, 1), n, 0.534 if min_round else "", "expected value")
    pre = [o for o in obs if o.predicted["threshold"] is None]
    h, n = emp.accuracy(pre, "optimize_recent_tie", 12)
    add(
        "optimize_recent_tie",
        "rounds > 12, threshold not yet reached",
        h,
        n,
        0.591,
        "published n = 10,630 (all rounds)",
    )
    h, n = emp.accuracy(pre, "optimize", 12)
    add("optimize", "rounds > 12, threshold not yet reached", h, n, 0.591, "paper-text tie rule")
    sf = emp.seven_five_check(experiments)
    rows.append(
        {
            "model": "human",
            "subset": "memory 7 top name : 5 other, CBBB2018, from round 22 (authors' definition)",
            "hits": sf.other,
            "n": sf.trials,
            "accuracy": round(sf.other / sf.trials, 4),
            "published_accuracy": 0.29,
            "note": (
                "released output file: 1343 trials, 949 top name, 394 other; the paper text "
                "gives 949 as the number of observations; replay of the human data"
            ),
        }
    )
    write_csv(RESULTS / "repro_empirical_accuracy.csv", rows)


def cmd_topologies(a: argparse.Namespace) -> None:
    curve_tasks = [
        (k, t, a.n, 12, a.sims, a.rounds, a.seed) for t in TOPOLOGIES for k in RULE_AGENTS
    ]
    curves = [
        row for rs in ex.pool_map(ex.emergence_curve_task, curve_tasks, a.processes) for row in rs
    ]
    write_csv(RESULTS / "ext_topology_curves.csv", [_round(r) for r in curves])
    grid = ex.tipping_grid(a.n, lo=0.06, hi=0.40, step=2)
    tip_tasks = [
        (k, t, a.n, c, 12, a.sims, 75, a.seed)
        for t in TOPOLOGIES
        for k in RULE_AGENTS
        for c in grid
    ]
    tips = ex.pool_map(ex.tipping_task, tip_tasks, a.processes)
    write_csv(RESULTS / "ext_topology_tipping.csv", [_round(r) for r in tips])


def _per_round_calls(n: int, topology: str) -> int:
    """Model calls per round for ``n`` players who all call the model (upper bound on sparse
    graphs, where some players may sit out a round)."""
    return n - n % 2 if topology == "complete" else n


def planned_calls(a: argparse.Namespace) -> int:
    """Calls without retries. Committed players never call the model."""
    if a.cmd == "llm-probe":
        return 13 * a.reps  # k = 0..12 slots of the target name, ``reps`` times each
    if a.cmd == "llm-game":
        return a.games * a.rounds * _per_round_calls(a.n, a.topology)
    return a.games * a.rounds * sum(a.n - c for c in a.confed)


def _max_input_tokens(a: argparse.Namespace) -> int:
    """Crude upper bound on input tokens of one call: one token per UTF-8 byte of the longest
    prompt (system + 12 longest names + the re-ask line) plus 50 for message framing. This is an
    assumption, not a measurement; the real cost is measured from the API's reported usage."""
    names = PROBE_NAMES if a.cmd == "llm-probe" else NAME_POOL
    longest = max(names, key=len)
    if a.mode == "history":
        user = history_prompt([(longest, longest)] * 12, 12)
    else:
        user = memory_prompt([longest] * 12, 12)
    return len((SYSTEM_PROMPT + user + "\nReply with exactly one name.").encode()) + 50


def _check_caps(a: argparse.Namespace) -> None:
    """Refuse bad cap flags before anything is created."""
    if a.max_cost_usd is not None and (a.price_in is None or a.price_out is None):
        sys.exit(
            "--max-cost-usd needs both --price-in and --price-out (USD per million tokens, "
            "from the provider's price page); no price is built in"
        )
    if a.client != "stub" and a.max_calls is None and a.max_cost_usd is None:
        sys.exit(
            f"refusing to run {a.client} without a cap: give --max-calls and/or --max-cost-usd "
            "(with --price-in and --price-out); look at the plan first with --dry-run"
        )


def print_plan(a: argparse.Namespace) -> None:
    """The dry run: planned calls, worst case, caps. No client, no network, no files."""
    planned = planned_calls(a)
    worst = 2 * planned
    if a.cmd == "llm-probe":
        what = f"13 memory compositions x {a.reps} repetitions"
    elif a.cmd == "llm-game":
        what = f"{a.games} games x {a.rounds} rounds x {_per_round_calls(a.n, a.topology)} players"
    else:
        what = f"{a.games} games x {a.rounds} rounds x learners summed over confed {a.confed}"
    print(f"{a.cmd}: {planned:,} planned calls ({what})")
    print(f"worst case with one re-ask per call: {worst:,} calls")
    warnings = []
    if a.price_in is not None and a.price_out is not None:
        tin = _max_input_tokens(a)
        cost = worst * (tin * a.price_in + a.max_tokens * a.price_out) / 1e6
        print(
            f"worst-case cost: {cost:.2f} USD (assumes at most {tin} input and {a.max_tokens} "
            f"output tokens per call at {a.price_in} / {a.price_out} USD per million tokens)"
        )
        if a.max_cost_usd is not None and cost > a.max_cost_usd:
            warnings.append(
                f"worst-case cost {cost:.2f} USD is above --max-cost-usd {a.max_cost_usd}"
            )
    else:
        print("worst-case cost: not computed (give --price-in and --price-out)")
    print(f"caps: --max-calls {a.max_calls}, --max-cost-usd {a.max_cost_usd}")
    if a.max_calls is not None and worst > a.max_calls:
        warnings.append(
            f"worst case {worst:,} calls is above --max-calls {a.max_calls}"
            + (f"; even the planned {planned:,} calls do not fit" if planned > a.max_calls else "")
        )
    for w in warnings:
        print(f"WARNING: {w}; the run would stop at the cap and be written as PARTIAL")
    if a.client != "stub" and a.max_calls is None and a.max_cost_usd is None:
        print("NOTE: a real run is refused without --max-calls or --max-cost-usd")
    print("dry run: no client created, no network, nothing written")


def _setup_client(a: argparse.Namespace):
    """Validate the cap flags, handle ``--dry-run`` (returns None), else the budgeted client."""
    _check_caps(a)
    if a.dry_run:
        print_plan(a)
        return None, None
    inner = make_client(a.client, a.temperature, a.max_tokens, a.token_param)
    if hasattr(inner, "raw_log"):  # claude-cli: keep every raw reply next to the results
        stem = f"{_label(a.client, a.label)}_{a.cmd.removeprefix('llm-')}.raw.jsonl"
        inner.raw_log = RESULTS / LLM_DIR / "raw" / stem
    budget = CallBudget(a.max_calls, a.max_cost_usd, a.price_in, a.price_out)
    return BudgetedClient(inner, budget), budget


LLM_DIR = "llm"


class Checkpoint:
    """Append-only JSON-lines record of finished units, one line per unit, flushed at once.

    Line 1 is ``{"config": ...}``; ``--resume`` refuses a checkpoint made with other settings.
    A torn last line (the process was killed while writing) is dropped. Stub runs are cheap and
    simply start over.
    """

    def __init__(self, path: Path, config: dict, resume: bool, is_stub: bool) -> None:
        self.path = path
        self.config = config
        self.units: dict[str, dict] = {}
        self.aborted: list[dict] = []  # spend of units a budget stop interrupted
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and (is_stub and not resume):
            path.unlink()
        if path.exists():
            if not resume:
                sys.exit(
                    f"{_shown(path)} exists from an earlier run: rerun with --resume to continue "
                    "it, or delete it to start over"
                )
            self._load()
        else:
            if resume:
                print(f"no checkpoint at {_shown(path)}: starting from the beginning")
            path.write_text(json.dumps({"config": config}) + "\n")

    def _load(self) -> None:
        good, torn = [], False
        for line in self.path.read_text().splitlines():
            try:
                good.append(json.loads(line))
            except json.JSONDecodeError:
                torn = True  # only a killed write can leave this; the unit is simply redone
        if not good or good[0].get("config") != self.config:
            sys.exit(
                f"{_shown(self.path)} was made with other settings than this command; "
                "resume only with the same flags (or delete the checkpoint)"
            )
        for rec in good[1:]:
            if rec.get("aborted"):
                self.aborted.append(rec)
            else:
                self.units[rec["unit"]] = rec
        if torn:
            self.path.write_text("".join(json.dumps(r) + "\n" for r in good))

    def add(self, unit: str, data: dict, delta: dict) -> None:
        rec = {"unit": unit, **data, "delta": delta}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        self.units[unit] = rec

    def add_aborted(self, delta: dict) -> None:
        """Record the spend of a unit a budget stop interrupted, so a resume still counts it."""
        if not any(delta.values()):
            return  # nothing was spent, nothing to remember
        rec = {"aborted": True, "delta": delta}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        self.aborted.append(rec)

    def spent(self) -> dict:
        keys = ("calls", "retries", "fallbacks", "tokens_in", "tokens_out")
        recs = [*self.units.values(), *self.aborted]
        return {k: sum(r["delta"].get(k, 0) for r in recs) for k in keys}


def _snap(stats: CallStats, budget: CallBudget) -> dict:
    return {
        "calls": stats.calls,
        "retries": stats.retries,
        "fallbacks": stats.fallbacks,
        "tokens_in": budget.tokens_in,
        "tokens_out": budget.tokens_out,
    }


def _delta(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in after}


def _restore(ck: Checkpoint, stats: CallStats, budget: CallBudget) -> int:
    """Carry the spend of the finished units into this run, so the cap covers the whole run."""
    sp = ck.spent()
    stats.add(calls=sp["calls"], retries=sp["retries"], fallbacks=sp["fallbacks"])
    budget.preload(sp["calls"], sp["tokens_in"], sp["tokens_out"])
    return len(ck.units)


def fallback_rate(stats: CallStats) -> float:
    """Share of choices that fell back to the newest remembered name (a choice is one first ask,
    plus a re-ask if the first reply was not a single word)."""
    choices = stats.calls - stats.retries + stats.fallbacks
    return stats.fallbacks / choices if choices else 0.0


def _finish(
    a: argparse.Namespace,
    client,
    budget: CallBudget,
    stats: CallStats,
    stem: Path,
    rows: list[dict],
    stopped: str | None,
    resumed: int,
    extra: dict,
) -> None:
    """Write the files of a finished, invalid or budget-stopped run and set the exit code.

    Valid: ``<stem>.csv`` / ``.json``. Invalid (too many fallbacks): ``<stem>.INVALID.*``.
    Stopped by the cap: ``<stem>.PARTIAL.*``. ``charts.py`` only reads ``<stem>.csv``.
    """
    rate = fallback_rate(stats)
    too_many = rate > a.max_fallback_rate
    if stopped:
        tag, reason = ".PARTIAL", f"PARTIAL: stopped by the budget ({stopped})"
    elif too_many:
        tag = ".INVALID"
        reason = (
            f"INVALID: fallback rate {rate:.1%} is above --max-fallback-rate "
            f"{a.max_fallback_rate:.1%}; this is not a result"
        )
    else:
        tag, reason = "", ""
    valid = not tag
    base = stem.parent / (stem.name + tag)
    if rows:
        write_csv(base.parent / (base.name + ".csv"), rows)
    cost = budget.cost_usd
    _meta(
        base.parent / (base.name + ".json"),
        client,
        a,
        stats,
        {
            **extra,
            "max_tokens": a.max_tokens,
            "fallback_rate": round(rate, 4),
            "max_fallback_rate": a.max_fallback_rate,
            "valid": valid,
            "complete": not stopped,
            "stopped_reason": stopped or "",
            "resumed_units": resumed,
            "budget": {
                "max_calls": budget.max_calls,
                "max_cost_usd": budget.max_cost_usd,
                "price_in_per_million": budget.price_in,
                "price_out_per_million": budget.price_out,
                "tokens_in": budget.tokens_in,
                "tokens_out": budget.tokens_out,
                "cost_usd": None if cost is None else round(cost, 6),
            },
        },
    )
    if valid:
        for suffix in (".csv", ".json"):  # an older stopped run of the same name is now stale
            stale = stem.parent / (stem.name + ".PARTIAL" + suffix)
            stale.unlink(missing_ok=True)
        return
    print(reason)
    if stopped:
        print("progress is saved: raise the cap and rerun the same command with --resume")
    sys.exit(3)


def _meta(path: Path, client, args: argparse.Namespace, stats: CallStats, extra: dict) -> None:
    meta = {
        "client": client.name,
        "is_stub": client.is_stub,
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "calls": stats.calls,
        "retries": stats.retries,
        "fallbacks": stats.fallbacks,
        "command": "python -m namegame.cli " + args.argv_text,
        **extra,
    }
    path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {_shown(path)}")


def _label(client_spec: str, label: str) -> str:
    base = label or client_spec.replace(":", "_").replace("/", "_")
    return f"STUB_{base}" if client_spec == "stub" and not base.startswith("STUB") else base


def _config(a: argparse.Namespace, **extra) -> dict:
    """Settings that change what a unit means; a checkpoint only resumes with the same ones."""
    return {
        "cmd": a.cmd,
        "client": a.client,
        "temperature": a.temperature,
        "mode": a.mode,
        "seed": a.seed,
        "max_tokens": a.max_tokens,
        **extra,
    }


def cmd_llm_probe(a: argparse.Namespace) -> None:
    client, budget = _setup_client(a)
    if client is None:
        return
    stem = RESULTS / LLM_DIR / f"{_label(a.client, a.label)}_probe"
    ck = Checkpoint(
        stem.with_name(stem.name + ".checkpoint.jsonl"),
        _config(a, reps=a.reps),
        a.resume,
        client.is_stub,
    )
    stats = CallStats()
    resumed = _restore(ck, stats, budget)
    done = {tuple(r["kr"]) for r in ck.units.values()}
    last = [_snap(stats, budget)]

    def on_row(row: dict) -> None:
        now = _snap(stats, budget)
        key = f"k{row['k_target']}|r{row['rep']}"
        ck.add(key, {"kr": [row["k_target"], row["rep"]], "row": row}, _delta(last[0], now))
        last[0] = now

    stopped = None
    try:
        run_probe(client, 12, a.reps, a.seed, a.mode, stats, done, on_row)
    except BudgetExceeded as e:
        stopped = str(e)
        ck.add_aborted(_delta(last[0], _snap(stats, budget)))
    rows = sorted((r["row"] for r in ck.units.values()), key=lambda r: (r["k_target"], r["rep"]))
    extra = {"reps": a.reps, "mode": a.mode, "temperature": a.temperature}
    _finish(a, client, budget, stats, stem, rows, stopped, resumed, extra)


def cmd_llm_game(a: argparse.Namespace) -> None:
    client, budget = _setup_client(a)
    if client is None:
        return
    stem = RESULTS / LLM_DIR / f"{_label(a.client, a.label)}_game_{a.topology}_N{a.n}"
    ck = Checkpoint(
        stem.with_name(stem.name + ".checkpoint.jsonl"),
        _config(a, n=a.n, games=a.games, rounds=a.rounds, topology=a.topology),
        a.resume,
        client.is_stub,
    )
    stats = CallStats()
    resumed = _restore(ck, stats, budget)
    net = make_network(a.topology, a.n, seed=a.seed)
    stopped = None
    with ThreadPoolExecutor(a.workers) as pool:
        for g in range(a.games):
            if f"g{g}" in ck.units:
                continue
            before = _snap(stats, budget)
            rng = random.Random(f"llmgame|{a.seed}|{g}")
            try:
                run = run_emergence(
                    llm_factory(client, 12, a.mode, stats),
                    net,
                    12,
                    rng,
                    max_rounds=a.rounds,
                    stop_at_convergence=False,
                    executor=pool,
                    name_pool=NAME_POOL,
                )
            except BudgetExceeded as e:
                stopped = str(e)
                ck.add_aborted(_delta(before, _snap(stats, budget)))
                break
            game_rows = [
                {
                    "game": g,
                    "round": t,
                    "top_share": round(r.top_share, 4),
                    "success_rate": round(r.success_rate, 4),
                    "n_names": r.n_names,
                    "converged_round": run.converged_round or "",
                    "client": client.name,
                    "is_stub": int(client.is_stub),
                    "topology": a.topology,
                    "N": a.n,
                    "mode": a.mode,
                }
                for t, r in enumerate(run.rounds, start=1)
            ]
            ck.add(f"g{g}", {"rows": game_rows}, _delta(before, _snap(stats, budget)))
            print(f"game {g}: converged round {run.converged_round}", flush=True)
    rows = [
        row
        for u in sorted(ck.units.values(), key=lambda r: r["rows"][0]["game"])
        for row in u["rows"]
    ]
    extra = {
        "games": a.games,
        "rounds": a.rounds,
        "N": a.n,
        "topology": a.topology,
        "mode": a.mode,
        "temperature": a.temperature,
    }
    _finish(a, client, budget, stats, stem, rows, stopped, resumed, extra)


def cmd_llm_tipping(a: argparse.Namespace) -> None:
    client, budget = _setup_client(a)
    if client is None:
        return
    stem = RESULTS / LLM_DIR / f"{_label(a.client, a.label)}_tipping_{a.topology}_N{a.n}"
    ck = Checkpoint(
        stem.with_name(stem.name + ".checkpoint.jsonl"),
        _config(a, n=a.n, confed=a.confed, games=a.games, rounds=a.rounds, topology=a.topology),
        a.resume,
        client.is_stub,
    )
    stats = CallStats()
    resumed = _restore(ck, stats, budget)
    net = make_network(a.topology, a.n, seed=a.seed)
    rows = []
    stopped = None
    with ThreadPoolExecutor(a.workers) as pool:
        for n_conf in a.confed:
            for g in range(a.games):
                key = f"c{n_conf}|g{g}"
                if key in ck.units:
                    continue
                before = _snap(stats, budget)
                rng = random.Random(f"llmtip|{a.seed}|{n_conf}|{g}")
                try:
                    run = run_tipping(
                        llm_factory(client, 12, a.mode, stats),
                        net,
                        12,
                        n_conf,
                        rng,
                        max_rounds=a.rounds,
                        executor=pool,
                        old_name="emma",
                        new_name="david",
                    )
                except BudgetExceeded as e:
                    stopped = str(e)
                    ck.add_aborted(_delta(before, _snap(stats, budget)))
                    break
                final = run.rounds[-1].new_share
                ck.add(key, {"final": final}, _delta(before, _snap(stats, budget)))
            if stopped:
                break
    for n_conf in a.confed:
        finals = [
            ck.units[f"c{n_conf}|g{g}"]["final"]
            for g in range(a.games)
            if f"c{n_conf}|g{g}" in ck.units
        ]
        if len(finals) < a.games:
            continue  # only settings with all their games are summarised
        rows.append(
            {
                "agent": client.name,
                "is_stub": int(client.is_stub),
                "topology": a.topology,
                "N": a.n,
                "n_confed": n_conf,
                "fraction": round(n_conf / a.n, 4),
                "n_sims": a.games,
                "rounds": a.rounds,
                "mean_final_new_share": round(st.mean(finals), 4),
            }
        )
        print(
            f"n_confed {n_conf}: mean final adoption {rows[-1]['mean_final_new_share']}", flush=True
        )
    extra = {
        "games": a.games,
        "rounds": a.rounds,
        "N": a.n,
        "topology": a.topology,
        "mode": a.mode,
        "temperature": a.temperature,
    }
    _finish(a, client, budget, stats, stem, rows, stopped, resumed, extra)


def cmd_charts(a: argparse.Namespace) -> None:
    from . import charts

    out = RESULTS / ("figures_with_STUB" if a.include_stub else "figures")
    charts.make_all(RESULTS, out, include_stub=a.include_stub)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="namegame", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser

    def add_parser(name, **kw):
        sp = add(name, **kw)
        sp.add_argument(
            "--results-dir",
            default=None,
            help="where result files go (default: results/ in this folder)",
        )
        return sp

    sub.add_parser = add_parser  # every command accepts --results-dir

    def common(sp, sims, processes=2):
        sp.add_argument("--sims", type=int, default=sims)
        sp.add_argument("--seed", type=int, default=2026)
        sp.add_argument("--processes", type=int, default=processes)

    s = sub.add_parser("table-s8", help="convergence rounds vs SI Table S8")
    common(s, 3000)
    s.set_defaults(fn=cmd_table_s8)
    s = sub.add_parser("tipping", help="critical-mass curves on complete networks (paper Fig. 4B)")
    common(s, 300)
    s.add_argument("--sizes", type=int, nargs="+", default=[24, 48, 96])
    s.add_argument(
        "--agents",
        nargs="+",
        choices=KINDS,
        default=TIPPING_AGENTS,
        help="agent kinds; optimize ties at random, optimize_recent_tie: newest name wins ties",
    )
    s.set_defaults(fn=cmd_tipping)
    s = sub.add_parser("empirical", help="replay the human data (paper Fig. 3C)")
    s.set_defaults(fn=cmd_empirical)
    s = sub.add_parser("topologies", help="EXTENSION: other network topologies")
    common(s, 200)
    s.add_argument("--n", type=int, default=48)
    s.add_argument("--rounds", type=int, default=100)
    s.set_defaults(fn=cmd_topologies)

    def llm_common(sp):
        sp.add_argument("--max-calls", type=int, default=None, help="hard cap on model calls")
        sp.add_argument("--max-cost-usd", type=float, default=None, help="hard cap; needs prices")
        sp.add_argument("--price-in", type=float, default=None, help="USD per million input tokens")
        sp.add_argument(
            "--price-out", type=float, default=None, help="USD per million output tokens"
        )
        sp.add_argument(
            "--resume", action="store_true", help="continue from <name>.checkpoint.jsonl"
        )
        sp.add_argument(
            "--max-fallback-rate",
            type=float,
            default=0.05,
            help="above this share of choices falling back, the run is written as INVALID",
        )
        sp.add_argument("--max-tokens", type=int, default=16, help="reply token limit per call")
        sp.add_argument(
            "--token-param",
            choices=["max_tokens", "max_completion_tokens"],
            default="max_tokens",
            help="OpenAI only: name of the reply token limit parameter",
        )
        sp.add_argument(
            "--dry-run",
            action="store_true",
            help="print planned calls and worst-case cost, create no client, write nothing",
        )
        sp.add_argument(
            "--client",
            required=True,
            help="stub | anthropic:<model> | openai:<model> | claude-cli:<model>",
        )
        sp.add_argument(
            "--temperature", type=float, default=None, help="omit to use the provider default"
        )
        sp.add_argument("--mode", choices=["memory", "history"], default="memory")
        sp.add_argument("--seed", type=int, default=2026)
        sp.add_argument("--label", default="", help="file-name prefix (default: client name)")

    s = sub.add_parser("llm-probe", help="LLM choice probability vs memory composition")
    llm_common(s)
    s.add_argument("--reps", type=int, default=20)
    s.set_defaults(fn=cmd_llm_probe)
    s = sub.add_parser("llm-game", help="LLM agents playing the naming game on a network")
    llm_common(s)
    s.add_argument("--topology", choices=TOPOLOGIES, default="complete")
    s.add_argument("--n", type=int, default=48)
    s.add_argument("--games", type=int, default=3)
    s.add_argument("--rounds", type=int, default=60)
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(fn=cmd_llm_game)
    s = sub.add_parser("llm-tipping", help="LLM agents against a committed minority")
    llm_common(s)
    s.add_argument("--topology", choices=TOPOLOGIES, default="complete")
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--confed", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    s.add_argument("--games", type=int, default=3)
    s.add_argument("--rounds", type=int, default=75)
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(fn=cmd_llm_tipping)
    s = sub.add_parser("charts", help="draw all figures from results/")
    s.add_argument(
        "--include-stub",
        action="store_true",
        help="also draw STUB runs (grey; pipeline check only)",
    )
    s.set_defaults(fn=cmd_charts)
    return p


def main(argv: list[str] | None = None) -> None:
    global RESULTS
    args = build_parser().parse_args(argv)
    args.argv_text = " ".join(sys.argv[1:] if argv is None else argv)
    previous = RESULTS
    if args.results_dir:
        RESULTS = Path(args.results_dir)
    try:
        args.fn(args)
    finally:
        RESULTS = previous  # --results-dir is per call, so in-process callers (tests) are isolated


if __name__ == "__main__":
    main()
