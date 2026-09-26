"""
Forecasting stage - Step 11: two-way (spatial x temporal) block bootstrap.

Every confidence interval in steps 4-8 came from a SPATIAL-ONLY block bootstrap (2 deg x 2 deg
tiles), which accounts for neighbouring cells being correlated but treats different days within
the test period as independent draws. They are not: forecast errors on consecutive days share
weather-regime persistence. This script re-computes the key PCMCI-vs-alternative comparisons
with a two-way block bootstrap that resamples BOTH spatial tiles AND contiguous time blocks,
so the resulting intervals reflect both kinds of dependence. It needs no new model fitting -
it reuses the daily_predictions.csv.gz already saved by step 6 (or step 7's no-cloud version,
or a step10 subset), so it runs in minutes, not hours.

Method: each bootstrap replicate independently resamples (a) the spatial tiles, with
replacement, and (b) the time blocks (contiguous chunks of BLOCK_DAYS calendar days spanning
the full test period), with replacement. The replicate's statistic is the weighted mean of the
per-(cell,day) squared error, where the weight of a (cell,day) pair is (number of times its
tile was drawn) x (number of times its time block was drawn). This is the natural nonparametric
extension of the spatial-only block bootstrap to two dependence dimensions; with only spatial
resampling (weights collapsing to 0/1 x 1) or only temporal resampling it reduces to the
single-way bootstraps used earlier.

Usage:
    python forecast_step11_spatiotemporal_bootstrap.py --daily step6_full\\daily_predictions.csv.gz \\
        --out step11_out --spatial_block 2.0 --time_block_days 30 --n_boot 2000
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

METHODS = ["PCMCI", "LASSO", "ALL"]


def safe_csv(df, path, **kw):
    try:
        df.to_csv(path, index=False, **kw)
    except PermissionError:
        alt = path.replace(".csv", f"_{int(time.time())}.csv")
        df.to_csv(alt, index=False, **kw)
        print(f"{path} is locked - saved as {alt}")


def two_way_block_bootstrap_ci(err_a, err_b, tile_id, block_id, n_boot=2000, seed=0):
    """err_a, err_b: 1-D arrays of per-(cell,day) squared errors for methods A and B, aligned.
    tile_id, block_id: 1-D arrays giving each row's spatial tile and time-block label.
    Returns (mean_pct, ci_low, ci_high) for the % RMSE reduction of A relative to B, and, for
    comparison, the single-way (spatial-only and temporal-only) CIs on the same data."""
    rng = np.random.default_rng(seed)
    tiles = np.unique(tile_id)
    blocks = np.unique(block_id)
    tile_idx = {t: np.flatnonzero(tile_id == t) for t in tiles}
    block_idx = {b: np.flatnonzero(block_id == b) for b in blocks}

    def stat(weights):
        w = weights
        rmse_a = np.sqrt(np.average(err_a, weights=w))
        rmse_b = np.sqrt(np.average(err_b, weights=w))
        return 100 * (1 - rmse_a / rmse_b)

    def run(resample_space, resample_time):
        vals = np.empty(n_boot)
        n = len(err_a)
        for i in range(n_boot):
            w = np.ones(n)
            if resample_space:
                w_tile = np.zeros(n)
                picks = rng.choice(tiles, size=len(tiles), replace=True)
                counts = pd.Series(picks).value_counts()
                for t, c in counts.items():
                    w_tile[tile_idx[t]] = c
                w = w * w_tile
            if resample_time:
                w_block = np.zeros(n)
                picks = rng.choice(blocks, size=len(blocks), replace=True)
                counts = pd.Series(picks).value_counts()
                for b, c in counts.items():
                    w_block[block_idx[b]] = c
                w = w * w_block
            vals[i] = stat(w)
        return vals

    point = stat(np.ones(len(err_a)))
    two_way = run(True, True)
    spatial_only = run(True, False)
    temporal_only = run(False, True)
    pct = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return dict(point=point,
               two_way_ci=pct(two_way), two_way_mean=two_way.mean(),
               spatial_only_ci=pct(spatial_only),
               temporal_only_ci=pct(temporal_only))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", required=True, help="daily_predictions.csv.gz from step 6 (or step 7/10)")
    ap.add_argument("--out", default="step11_out")
    ap.add_argument("--spatial_block", type=float, default=2.0, help="deg x deg spatial tile size")
    ap.add_argument("--time_block_days", type=int, default=30, help="contiguous time-block size (days)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--models", nargs="+", default=["LIN", "KRR"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = pd.read_csv(a.daily)
    d["date"] = pd.to_datetime(d["date"])
    d["lat"], d["lon"] = d["lat"].round(4), d["lon"].round(4)
    ncell = d[["lat", "lon"]].drop_duplicates().shape[0]
    print(f"{ncell} cells | {d['date'].nunique()} distinct test days | "
          f"{d['date'].min().date()} to {d['date'].max().date()}")

    d["tile"] = [f"{np.floor(la / a.spatial_block)}_{np.floor(lo / a.spatial_block)}"
                for la, lo in zip(d["lat"], d["lon"])]
    t0 = d["date"].min()
    d["time_block"] = ((d["date"] - t0).dt.days // a.time_block_days).astype(str)
    n_tiles = d["tile"].nunique()
    n_blocks = d["time_block"].nunique()
    print(f"{n_tiles} spatial tiles ({a.spatial_block}x{a.spatial_block} deg) | "
          f"{n_blocks} time blocks ({a.time_block_days} days each)")

    comparisons = []
    for mo in a.models:
        for alt in ["LASSO", "ALL"]:
            comparisons.append((f"PCMCI_{mo}", f"{alt}_{mo}", f"PCMCI vs {alt} ({mo})"))
        comparisons.append((f"PCMCI_{mo}", "PERSIST_KC", f"PCMCI vs persistence ({mo})"))
        comparisons.append((f"PCMCI_{mo}", "CLIM", f"PCMCI vs climatology ({mo})"))

    rows, t_start = [], time.time()
    for a_col, b_col, label in comparisons:
        if a_col not in d or b_col not in d:
            print(f"skip {label}: column missing")
            continue
        sub = d.dropna(subset=[a_col, b_col, "ghi"])
        err_a = (sub[a_col].values - sub["ghi"].values) ** 2
        err_b = (sub[b_col].values - sub["ghi"].values) ** 2
        res = two_way_block_bootstrap_ci(err_a, err_b, sub["tile"].values, sub["time_block"].values,
                                          n_boot=a.n_boot)
        rows.append(dict(comparison=label, a=a_col, b=b_col, n_rows=len(sub), point_pct=res["point"],
                         two_way_lo=res["two_way_ci"][0], two_way_hi=res["two_way_ci"][1],
                         spatial_only_lo=res["spatial_only_ci"][0], spatial_only_hi=res["spatial_only_ci"][1],
                         temporal_only_lo=res["temporal_only_ci"][0], temporal_only_hi=res["temporal_only_ci"][1]))
        print(f"  {label:32s}: point {res['point']:+6.2f}% | two-way CI [{res['two_way_ci'][0]:+6.2f}, "
              f"{res['two_way_ci'][1]:+6.2f}] | spatial-only CI [{res['spatial_only_ci'][0]:+6.2f}, "
              f"{res['spatial_only_ci'][1]:+6.2f}] | temporal-only CI [{res['temporal_only_ci'][0]:+6.2f}, "
              f"{res['temporal_only_ci'][1]:+6.2f}]  ({time.time() - t_start:.0f}s elapsed)")

    out_df = pd.DataFrame(rows)
    safe_csv(out_df, os.path.join(a.out, "spatiotemporal_bootstrap.csv"))
    if len(out_df):
        widen = (out_df["two_way_hi"] - out_df["two_way_lo"]) / (out_df["spatial_only_hi"] - out_df["spatial_only_lo"])
        print(f"\nMean CI width, two-way vs spatial-only: {widen.mean():.2f}x "
              f"(1.0x = temporal dependence adds nothing beyond spatial)")
        flips = ((out_df["two_way_lo"] <= 0) & (out_df["two_way_hi"] >= 0) &
                 ~((out_df["spatial_only_lo"] <= 0) & (out_df["spatial_only_hi"] >= 0)))
        if flips.any():
            print(f"\n{flips.sum()} comparison(s) that excluded 0 under the spatial-only CI now include 0 "
                  f"under the two-way CI - re-check these before reporting them as significant:")
            print(out_df.loc[flips, "comparison"].to_string(index=False))
        else:
            print("\nNo comparison flips from 'significant' to 'not distinguishable from zero' under the "
                  "two-way bootstrap - the paper's significance claims are robust to adding temporal blocking.")
    print(f"\ndone in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
