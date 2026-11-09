from namegame.experiments import (
    TABLE_S8,
    convergence_task,
    emergence_curve_task,
    tipping_grid,
    tipping_point,
    tipping_task,
)


def test_tipping_point_interpolates():
    rows = [
        {"fraction": 0.1, "mean_final_new_share": 0.0},
        {"fraction": 0.2, "mean_final_new_share": 0.2},
        {"fraction": 0.3, "mean_final_new_share": 0.8},
    ]
    # crossing 0.5 between 0.2 (0.2) and 0.3 (0.8): halfway
    assert abs(tipping_point(rows) - 0.25) < 1e-9
    assert tipping_point([{"fraction": 0.1, "mean_final_new_share": 0.1}]) is None
    assert tipping_point([{"fraction": 0.1, "mean_final_new_share": 0.9}]) == 0.1


def test_tipping_grid_bounds():
    ks = tipping_grid(24)
    assert ks[0] == 1 and ks[-1] == 9  # 1/24 = 4.2% and 9/24 = 37.5%
    assert tipping_grid(96, step=2)[:2] == [4, 6]


def test_convergence_task_is_deterministic_and_close_to_table_s8():
    a = convergence_task(("threshold", 24, 12, 150, 1))
    b = convergence_task(("threshold", 24, 12, 150, 1))
    assert a == b
    mean_pub, sd_pub = TABLE_S8[("threshold", 24)]
    # 150 runs: standard error about 10.6/sqrt(150) = 0.9 rounds; allow 3.5 standard errors
    assert abs(a["mean_rounds"] - mean_pub) < 3.5


def test_tipping_task_shape():
    row = tipping_task(("threshold", "complete", 24, 3, 12, 20, 40, 1))
    assert row["fraction"] == 3 / 24
    assert row["mean_final_new_share"] < 0.3  # 12.5% committed: below the tipping point
    row2 = tipping_task(("threshold", "complete", 24, 9, 12, 20, 75, 1))
    assert row2["mean_final_new_share"] > 0.9  # 37.5% committed


def test_emergence_curve_rows_and_monotone_start():
    rows = emergence_curve_task(("optimize", "ring", 16, 12, 10, 30, 1))
    assert len(rows) == 30
    assert rows[0]["round"] == 1
    assert rows[-1]["mean_top_share"] > rows[0]["mean_top_share"]
