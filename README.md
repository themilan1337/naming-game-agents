# naming-game-agents

Naming-game simulator after Guilbeault, Caplan and Yang, "A simple threshold captures the social learning of conventions", PNAS 123(17):e2508061123 (2026). It plays threshold, imitating, optimizing and probability-matching (Luce) agents on the paper's networks, checks them against the paper's published numbers, and has an adapter for LLM agents on the same networks. The one-page summary is `NOTE.md` (PDF: `results/note/NOTE.pdf`).

Status: the reproduction is done. The LLM part is a first, small run of one model (Claude Sonnet 5.5): a probe, one 24-player game and two committed-minority games. The recorded human interaction sequences from the authors' repository have not been fed to LLM agents yet.

## What is in the repository

- `src/namegame/agents.py`: the memory model (FIFO queue of M = 12, with the paper's "remove your own name on a mismatch" update) and five choice rules: threshold (Tolerance Principle, M - M/ln M = 8 of 12 slots), imitating, optimizing (ties at random as in the paper text, or newest name wins, which fits SI Table S8), Luce, and the paper's 2/3 rule.
- `networks.py`, `sim.py`: pairing rule, emergence runs (convergence = at most two players deviate, as in SI A.10) and tipping runs with committed players. Topologies: complete (the paper), ring lattice, small world, random 6-regular, scale free (extension).
- `empirical.py`: replays the human data of Centola and Baronchelli (2015) and Centola et al. (2018) through the rules, as in the paper's Fig. 3.
- `experiments.py`, `cli.py`: batch runs and the command line.
- `llm.py`: provider-agnostic client (`stub`, `claude-cli:<model>`, `anthropic:<model>`, `openai:<model>`; keys from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`), a call and cost cap, retries, the LLM agent with two prompt modes (the 12-name memory, or a transcript of recent rounds), a memory probe (two-name memories with k of 12 slots) and a deterministic stub that is not a language model.
- `charts.py`: all figures, written for a reader who knows the paper but not this code.
- `scripts/`: `fetch_data.sh` (human data from the authors' repository), `check_table_s8.py` (pass/fail gate for Result 1), `verify_in_copy.sh` (one-command verification in a temporary copy), `llm_figure.py` (figure 7 and `results/llm/llm_summary.json`).
- `tests/`: no network; hand-checked cases (the SI's worked memory examples, threshold values, tie rules, replay of a synthetic game, prompt parsing, mocked HTTP).
- `results/`: every number in the note and below, as CSV or JSON, with the figures; `results/llm/raw/` holds every prompt and reply of the Claude runs.

Design choices: the optimizing agent has two tie rules (default `optimize` follows the paper's text; `optimize_recent_tie` gives ties to the newest name and the looser determinism test of Result 2), and both are reported wherever they differ. Real first names (Zipf-weighted over ten) are used so that the threshold question can be asked in plain language. A reply that is not exactly one name is retried once, then replaced by the newest remembered name; the counts are in the `.json` file next to every LLM result. Runs use 2 processes, and N = 96 tipping uses every second committed count to keep it under 15 minutes.

## How to run

One command checks everything in a temporary copy (the committed `results/` is never touched; needs `uv`, works on Linux and macOS):

```bash
cd yang-naming-game
bash scripts/verify_in_copy.sh           # tests, ruff, full 3000-run table-s8, table gate
bash scripts/verify_in_copy.sh --full    # also fetch_data, empirical, tipping, topologies, charts
```

It prints `verify_in_copy: all checks passed` at the end. Seconds per step, measured on a 4-core Linux container with `OMP_NUM_THREADS=2` and 2 processes (`bash scripts/verify_in_copy.sh --full`, 8 min in total): `uv sync` 1 s, pytest 9 s, ruff 1 s, `table-s8` 76 s, `fetch_data.sh` 2 s, `empirical` 1 s, `tipping` 260 s, `topologies` 127 s, `charts` 3 s. Other machines will differ.

The same steps by hand:

```bash
uv sync
export OMP_NUM_THREADS=2
./scripts/fetch_data.sh                       # human data (about 1 MB) into data/cache/ref, not committed
uv run python -m namegame.cli empirical       # seconds
uv run python -m namegame.cli table-s8        # 3000 runs x 15 settings, about 1.5 min
uv run python scripts/check_table_s8.py results/repro_tableS8_convergence.csv   # prints "table-s8 check: pass"
uv run python -m namegame.cli tipping         # critical-mass curves, about 4 to 7 min
uv run python -m namegame.cli topologies      # extension, about 2 min
uv run python -m namegame.cli charts          # figures into results/figures
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

Quick versions such as `table-s8 --sims 300`, `tipping --sims 50`, `topologies --sims 30` are noise and not a gate: with 300 runs some cells fall outside the 0.6 tolerance. All runs are seeded (`--seed 2026`); results do not depend on the number of processes. Every command accepts `--results-dir <dir>`; use it for smoke tests so that `results/` stays clean.

LLM runs need a key and cost money, so every real client is refused without a cap (`--max-calls` or `--max-cost-usd`). Prices are not built in; give them from the provider's price page. Look at the plan first (no key, no network):

```bash
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY; in the shell only, never in a file
uv run python -m namegame.cli llm-probe --client anthropic:<model> --reps 20 --dry-run --max-calls 520 --max-cost-usd <cap> --price-in <USD per million input tokens> --price-out <USD per million output tokens>
uv run python -m namegame.cli llm-probe --client anthropic:<model> --reps 1 --max-calls 30 --max-cost-usd <cap> --price-in <p> --price-out <p> --results-dir /tmp/yng_smoke
uv run python -m namegame.cli llm-probe   --client anthropic:<model> --reps 20 --max-calls 520 --max-cost-usd <cap> --price-in <p> --price-out <p>
uv run python -m namegame.cli llm-game    --client anthropic:<model> --n 48 --games 3 --rounds 60 --max-calls 17300 --max-cost-usd <cap> --price-in <p> --price-out <p>
uv run python -m namegame.cli llm-tipping --client anthropic:<model> --n 24 --confed 2 4 6 8 --games 1 --rounds 75 --max-calls 11500 --max-cost-usd <cap> --price-in <p> --price-out <p>
uv run python -m namegame.cli charts
```

Add `--resume` to continue after an interruption. For the API clients `<model>` and the cost cap are your choice. The runs of 8 to 9 Nov 2026 used the command-line client instead (no key, nothing billed; the `cost_usd` in the raw log is the CLI's notional list price):

```bash
uv run python -m namegame.cli llm-probe  --client claude-cli:claude-sonnet-5-5 --reps 6 --max-calls 200
uv run python -m namegame.cli llm-game   --client claude-cli:claude-sonnet-5-5 --n 24 --games 1 --rounds 30 --workers 16 --max-calls 1500
uv run python -m namegame.cli llm-tipping --client claude-cli:claude-sonnet-5-5 --n 24 --confed 3 6 --games 1 --rounds 30 --workers 24 --max-calls 3000 --resume
uv run python scripts/llm_figure.py claude-cli_claude-sonnet-5-5 24 30     # figure 7, llm_summary.json
```

Pipeline check without a key: `uv run python -m namegame.cli llm-probe --client stub`, likewise `llm-game` and `llm-tipping`. Stub output is written to `STUB_*` files, is made by a rule-based stand-in and is not a result; `charts --include-stub` draws it in grey.

## Results

All numbers below come from the commands above; files are in `results/`. M = 12, Zipf-weighted first names over ten types, fully connected networks, unless stated.

**1. Convergence time vs SI Table S8** (`table-s8`, 3000 runs per cell, `results/repro_tableS8_convergence.csv`, `results/figures/fig1_reproduction_convergence.png`). Tolerance used for this comparison: within 0.6 rounds of the published mean (applied by `scripts/check_table_s8.py`). Means are over runs that converged within the 75-round window, as in the released `puresimconverge.py` (MAX_ROUND = 75); the csv's `n_converged` column gives the count per cell (all 3,000 except threshold at N = 24 and 48, 2,989 and 2,997, and Luce, 2,823, 2,817 and 2,861).

| Agent | N | ours: mean (SD) | published: mean (SD) | difference |
|---|---|---|---|---|
| Threshold | 24 | 21.77 (10.32) | 21.83 (10.62) | -0.06 |
| Threshold | 48 | 24.19 (8.53) | 23.81 (8.01) | +0.38 |
| Threshold | 96 | 24.44 (5.64) | 24.13 (5.31) | +0.31 |
| Imitating | 24 | 15.59 (7.33) | 15.09 (6.61) | +0.50 |
| Imitating | 48 | 16.55 (6.09) | 17.00 (6.85) | -0.45 |
| Imitating | 96 | 16.92 (4.65) | 16.99 (4.72) | -0.07 |
| Optimizing, random ties (paper text) | 24 | 12.80 (4.97) | 14.44 (5.50) | -1.64 |
| Optimizing, random ties (paper text) | 48 | 13.46 (4.16) | 15.20 (4.60) | -1.74 |
| Optimizing, random ties (paper text) | 96 | 13.53 (2.83) | 14.89 (3.12) | -1.36 |
| Optimizing, newest name wins ties | 24 | 14.61 (5.38) | 14.44 (5.50) | +0.17 |
| Optimizing, newest name wins ties | 48 | 15.06 (4.38) | 15.20 (4.60) | -0.14 |
| Optimizing, newest name wins ties | 96 | 14.99 (2.90) | 14.89 (3.12) | +0.10 |
| Luce | 24 | 24.40 (13.81) | 24.36 (14.39) | +0.04 |
| Luce | 48 | 31.74 (13.17) | 31.39 (13.13) | +0.35 |
| Luce | 96 | 40.57 (10.58) | 40.17 (10.38) | +0.40 |

What matches: all threshold, imitating, Luce cells and the optimizing agent with newest-name tie-breaking are within about half a round (largest gap 0.50, imitating at N = 24), inside the 0.6 tolerance. The ordering in the paper (optimizing and imitating converge too fast, Luce too slowly, only the threshold agent lands in the human range of 20 to 30 rounds) is reproduced. What does not match: the optimizing agent with random ties, as the paper's text describes it, converges 1.4 to 1.7 rounds faster than Table S8. The table is matched only when ties go to the most recently remembered name. The fit to the published Table S8 is the evidence for this choice; the paper text says random ties, and we have not asked the authors which one they used. A cross-check run of their released simulation script for this agent at N = 24 gave 14.68 (ours 14.61).

**2. Next-choice accuracy on the human data vs Fig. 3C** (`empirical`, 23,852 replayed human choices from 21 networks, `results/repro_empirical_accuracy.csv`, fig. 2). Each model's memory is rebuilt from what each player actually experienced; a model is scored where it makes a deterministic prediction. The number of choices scored matches the paper only for the threshold row (8,893 over all rounds). The other rows score a different n in our replay: imitating 12,575 (rounds > 12), optimizing 15,843 (newest-name rule) or 15,716 (random ties), Luce 17,366 (expected value), optimizing before the threshold is reached 6,994, against 10,630 in the paper for that row. The paper's exact subsets are not known to us, and we have not traced the differences.

| Model | ours | published |
|---|---|---|
| Threshold | 87.9% (7,815 of 8,893, all rounds; same n as the paper) | 87.9% (n = 8,893) |
| Imitating, rounds > 12 | 81.1% | 81.1% |
| Optimizing, rounds > 12, newest-name ties with the flagged determinism test | 75.2% | 75.2% |
| Optimizing, rounds > 12, unique plurality with random ties | 74.8% | 75.2% |
| Luce, rounds > 12 (expected value) | 53.4% | 53.4% |
| Optimizing before the threshold is reached, rounds > 12, same rule | 59.1% | 59.1% |

The percentages match the published ones to the printed digit. The optimizing figure is matched only with newest-name ties and a player counted as "deterministic" unless the two newest distinct names have equal counts (a looser test than a unique plurality, kept in `released_code_rules.py`, see Licence and provenance). The paper's text states unique plurality; with it we get 74.8% (and 57.8% before the threshold), a gap of 0.4 and 1.3 points.

**3. Tipping by a committed minority vs Fig. 4B** (`tipping`, 300 runs per point, 75 rounds, `results/repro_tipping_complete.csv`, `results/repro_tipping_points.csv`, fig. 3). Committed share at which learners end up on the new name in half of the cases (linear interpolation):

| Agent | N = 24 | N = 48 | N = 96 |
|---|---|---|---|
| Threshold | 20.5% | 21.2% | 21.5% |
| Optimizing (random ties) | 23.4% | 24.4% | 24.7% |
| Optimizing (newest name wins ties) | 22.8% | 23.3% | 23.6% |
| Imitating | 6.4% | 7.2% | 7.2% |
| Luce | 9.4% | 9.7% | 9.7% |

The paper reports about 25% for the human data (range 20 to 25%), says threshold and optimizing agents are consistent with it, and says imitation flips at 10% or less. We reproduce this qualitatively (no tolerance was stated in the paper for this figure, so we do not claim a numerical match). The simulation also gives the sharp, size-independent step for threshold and optimizing agents.

**4. The 7:5 memory check, with the authors' definition** (`empirical`, last row of `results/repro_empirical_accuracy.csv`). Definition (taken from the column names and numbers of the authors' released output file for this check, implemented independently): experiments of the 2018 data (CBBB2018), non-committed players, from round 22, memory full with the most frequent name in exactly 7 of 12 slots. Our replay finds 1,343 such trials; the top name was produced in 949 and another name in 394 (29.3%), against 29% in the paper. The released output file lists the same three numbers (1,343, 949, 394). The paper text gives 949 as the number of observations; we read 949 as the top-name count (our inference from the file's column names, not checked with the authors). The file names CBBB2018 only, while the paper text says "across all ten networks"; this is an observation to raise with the authors, not a correction. An earlier, looser definition of ours did not match and is not used.

**5. Extension, not in the paper: other topologies** (`topologies`, N = 48, 200 runs per setting, `results/ext_topology_curves.csv`, `results/ext_topology_tipping.csv`, figs. 4 and 5). On the ring lattice a shared name is reached in only some runs within 100 rounds: 38 (threshold), 47 (imitating), 29 (optimizing) and 46 (Luce) of 200 runs converged by round 100 (`n_converged_by_end`), and the most common name is used by 68 to 72% of players at round 100. The other graphs get closer to full agreement, more slowly than on the complete graph. The committed share at which half of the learners end on the new name (linear interpolation of `mean_final_new_share`) is 20.8% (threshold) and 24.8% (optimizing) on the complete graph, and between 18.6% and 21.9% for those two agents on the other four graphs, where the curves are flatter. Imitating crosses 50% at 6.8 to 7.8% on four of the five graphs; on the small-world graph it has no upward crossing, because even the lowest grid point (6.25%) already gives a mean final new share of 0.5035. These are simulation results for rule-based agents under a pairing rule of our own (see Limitations), not a statement about humans.

**6. LLM agents: a first, small real run.** Model `claude-sonnet-5-5` through the Claude command-line client (headless, no tools, no key), run on 8 and 9 Nov 2026; default sampling (not controllable from the command line). Prompt mode `memory`: the model sees the same 12-name memory a rule agent has and answers with one name; real first names, Zipf-weighted initial names. Files in `results/llm/`: `claude-cli_claude-sonnet-5-5_probe.{csv,json}`, `..._game_complete_N24.{csv,json}`, `..._tipping_complete_N24.{csv,json}`, `llm_summary.json` (built by `scripts/llm_figure.py`, which plays the rule agents on the same network: 300 runs each), `rule_games_N24.csv`, figure `results/figures/fig7_llm_probe_and_game.png`, raw replies in `results/llm/raw/`. All three runs are valid: 0 fallbacks (a reply that was not exactly one name) in 78 + 720 + 1,170 = 1,968 calls.

- **Probe** (name held in k of 12 memory slots, the rest another name, 6 repeats for each k from 0 to 12, 78 calls). The model wrote the name held in k slots in 6 of 6 repeats for every k from 7 to 12, and in 0 of 6 for every k from 0 to 5. At k = 6 (a 6 to 6 tie) it wrote the name listed first in 6 of 6 repeats (the name listed first was maria in 2 repeats and david in 4, so this is a position effect, not a preference for one name). The choice is a step at a simple majority with no probabilistic region. The threshold agent writes the name with probability about 0.58 at k = 7 (a sample of one slot) and 1.0 from k = 8; the optimizing agent is a step at a majority too. So in this probe the model resembles the optimizing agent more than the threshold agent. Six repeats per k cannot show small probabilities; a probability between 0 and 1 at one k is not excluded for all k.
- **Game** (complete network, N = 24, 30 rounds, one game, 720 calls). All 24 players used one name from round 7 (`converged_round` 7, `top_share` 1.0 from round 7). Rule agents on the same network and settings (300 runs each, `llm_summary.json`), mean convergence round among runs that converged within 30 rounds: threshold 18.4 (245 of 300 converged), imitating 14.6 (286), optimizing with random ties 12.8 (299), Luce 18.1 (210). The model was faster than the mean of every rule agent in this one game. One game gives no estimate of the model's spread, and why it was faster was not tested.
- **Committed minority** (N = 24, 3 and 6 committed players who always write the new name, one 30-round game each, 1,170 calls). Share of the other players writing the new name in round 30: 0 of 21 (0.0) with 3 committed and 3 of 18 (0.167) with 6 committed. Same setting, rule agents, 300 runs each: threshold 0.006 and 0.345, optimizing 0.002 and 0.122, imitating 0.942 and 1.0, Luce 0.292 and 0.646 (`tipping_rule_agents_same_setting` in `llm_summary.json`). No flip in either game; this says nothing about where the model's critical mass lies.
- **Not run:** the probe with more repeats, games with N = 48 or 96, the `history` prompt mode, other models, the recorded human interaction sequences, other topologies (see Limitations). Figures 3, 4 and 6 show "LLM series: not run yet"; figure 7 is the LLM figure.

## Limitations

- Fully connected networks only are from the paper; every other topology is an extension with our own pairing rule (random greedy matching; on sparse graphs some players sit out a round), parameters chosen by us (ring k = 4, small world k = 6 and p = 0.1, 6-regular, Barabasi-Albert m = 3) and a single N.
- The reproduction uses the paper's text, SI and the released data; we could not obtain the authors' own simulation output. First-round name distribution (Zipf over ten names), the convergence rule and the 75-round window come from SI Table S8 and the released code.
- The mind-reading game (Fig. 5 of the paper) is not reproduced: it is a single-agent task, and a first attempt with a plain memory-window model did not match SI Table S9 (its output was not kept), so the exact simulation protocol was not recovered. It is not in this repository.
- The 7:5 reproduction (result 4) rests on our reading of the released output file's columns; the subset in the paper's own text ("across all ten networks") differs from the file (CBBB2018 only).
- On sparse graphs convergence is judged on the players who produced a name in a round, because some players sit out under our pairing rule (result 5).
- The LLM runs are small and from one model: Claude Sonnet 5.5 through the Claude command-line client, which has no temperature or token-limit flag (sampling is the default and unknown to us), and runs on a subscription, so the `cost_usd` in the raw log is a notional list price. The probe has 6 repeats per memory composition (78 calls), the game is one 30-round game of 24 players (720 calls). Neither supports an estimate of how often or how reliably the model behaves like this. Why the model converges faster than every rule agent was not tested.
- The probe lists the memory without order; in the game it is listed newest first. The tie result (the name listed first, 6 of 6 times) was seen in the probe only; in the game, "first" means "newest".
- The `claude-cli` client starts one process per call in an empty directory with no settings sources; the raw log keeps every call. Interrupted runs were resumed with `--resume`; a game cut off midway restarts from round 1, so `results/llm/raw/*tipping.raw.jsonl` also holds the calls of aborted attempts (it has more lines than the 630 calls of each finished game) and the per-run `calls` counts in the `.json` files are the ones to use.
- The API clients (`anthropic:`, `openai:`) were written from the public documentation and have run only against the stub and mocked HTTP, never against a live endpoint. Prompt design (single names, 12-slot memory, no scores) differs from earlier LLM naming-game work (Ashery, Aiello, Baronchelli 2025, which gives agents a history with scores and uses letter names), so results are not comparable to it without care; we have not compared against it, and the note does not cite it.
- Third-party data and papers are not committed.

## Licence and provenance

Copyright (c) 2026 Milan Gorislavets and Kamilya Mukh. GPL-3.0, see `LICENSE`.

The simulator is written from the paper text and SI. The authors' released repository (https://github.com/scaplan/name-game, GPL-3.0, commit d8acbd7) was read for details the paper leaves open: the Zipf first-round name distribution, the 75-round window and how a replayed prediction counts as deterministic. We compared the code with the released files (line similarity at most 0.044, token similarity at most 0.168) and found no copied text, but one rule follows the logic of the released function `check_unique_plurality_item`: the test that a prediction counts as deterministic unless the two newest distinct names have equal counts. It is isolated in `src/namegame/released_code_rules.py`, and because of it this repository is under GPL-3.0 too. Without it the replay gives 74.8% and 57.8% (the paper's own wording, unique plurality) in place of 75.2% and 59.1%. The human data is fetched from the authors' repository by `scripts/fetch_data.sh` and is not stored here.

## References

`refs.bib` lists the references; the note cites Guilbeault, Caplan and Yang (2026), Centola and Baronchelli (2015) and Centola, Becker, Brackbill and Baronchelli (2018).
