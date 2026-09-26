"""
Forecasting stage - Step 7: ablation without CLOUD_AMT.

Question: how much of the forecast skill (and of the PCMCI selection result) depends on the cloud-amount product,
which is the strongest lagged predictor in all cells and is missing from 1 Oct 2025 onward in the downloaded file?

Inputs
    --prepared      prepared cell CSVs (any step 1 run; they contain all anomalies)
    --links_nocloud links file from a step 1 run made with  --exclude CLOUD_AMT   (PCMCI on 7 variables)
    --with_daily    daily_predictions.csv.gz from step 6 (WITH cloud amount: PCMCI / LASSO / ALL, LIN and KRR)
For every cell this script fits, on the same training rows (<= 2022-12-31) and with the same fixed models:
    PCMCI_nc   parents found by PCMCI without CLOUD_AMT
    LASSO_nc   the same number of inputs picked by the LASSO path (candidate pool without CLOUD_AMT)
    ALL_nc     all remaining lagged inputs (7 variables x 7 lags = 49)
It then merges the WITH-cloud predictions from step 6 by (cell, day), so both variants are compared on IDENTICAL days,
and it also scores the no-cloud models on the additional days (Oct-Dec 2025) that only the no-cloud variant can use.

Outputs (in --out): daily_predictions_nocloud.csv.gz, ablation_table.csv, ablation_paired.csv.
Confidence intervals: spatial block bootstrap over BLOCK x BLOCK degree tiles.

Usage:
    python forecast_step7_nocloud_ablation.py --prepared forecast_full\\prepared ^
        --links_nocloud forecast_nocloud\\links_into_kc_all_cells.csv ^
        --with_daily step6_full\\daily_predictions.csv.gz --out step7_out --n_jobs 4 --limit 3
Needs forecast_step3_fit.py and forecast_step4_selection_comparison.py in the same folder.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from forecast_step3_fit import VARS, TRAIN_END, args_key, build_features, fit_predict
from forecast_step4_selection_comparison import select_lasso_path, block_bootstrap_ci

warnings.filterwarnings("ignore")
H = 1
DROP = "CLOUD_AMT"
METHODS = ["PCMCI", "LASSO", "ALL"]


def safe_csv(df, path, **kw):
    try:
        df.to_csv(path, index=False, **kw)
    except PermissionError:
        alt = path.replace(".csv", f"_{int(time.time())}.csv")
        df.to_csv(alt, index=False, **kw)
        print(f"{path} is locked - saved as {alt}")


# ----------------------------------------------------------------------------- per cell
def run_cell(args):
    path, links, models = args
    lat, lon = args_key(path)
    P = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    ref, kcc, ghi, kcraw = P["ref_clr"], P["kc_clim"], P["ghi"], P["kc_raw"]
    tau_max = max(int(links["lag"].max()) if len(links) else 7, 7)

    pool = [(v, l) for v in VARS if v != DROP for l in range(H, tau_max + 1)]
    pool_names = [f"{v}_L{l}" for v, l in pool]
    X_pool = build_features(P, pool, H)
    ok = X_pool.notna().all(axis=1) & P["kc"].notna() & kcraw.notna() & ghi.notna() & ref.notna()
    ok &= kcraw.shift(H).notna()
    idx = ok[ok].index
    tr, te = idx[idx <= TRAIN_END], idx[idx > TRAIN_END]
    if len(te) < 100 or len(tr) < 500:
        return None, (lat, lon, "not enough rows")

    pc = links[(links["method"] == "PCMCI") & (links["lag"] >= H) & (links["p"] <= 0.0101)]
    names_pc = sorted({f"{r.driver}_L{int(r.lag)}" for r in pc.itertuples()})
    if any(n.startswith(DROP) for n in names_pc):
        return None, (lat, lon, "links file contains CLOUD_AMT - run step 1 with --exclude CLOUD_AMT")
    k = len(names_pc)
    if k == 0:
        return None, (lat, lon, "no PCMCI parents")
    A = X_pool[pool_names].loc[tr].values
    y = P["kc"].loc[tr].values
    sets = {"PCMCI": names_pc,
            "LASSO": [pool_names[i] for i in select_lasso_path(A, y, k)],
            "ALL": pool_names}
    cols_sc = ["sin_doy", "cos_doy"]

    out = pd.DataFrame(index=te)
    out["lat"], out["lon"] = lat, lon
    out["k_nc"] = k
    out["ghi"] = ghi.loc[te].values
    out["CLIM"] = kcc.loc[te].values * ref.loc[te].values
    out["PERSIST_KC"] = kcraw.shift(H).loc[te].values * ref.loc[te].values
    for m, names in sets.items():
        X = X_pool[list(names) + cols_sc]
        for model in models:
            p = fit_predict(model, X.loc[tr].values, y, X.loc[te].values)
            out[f"{m}_{model}_nc"] = (kcc.loc[te].values + p) * ref.loc[te].values
    out.index.name = "date"
    return out.reset_index(), None


# ----------------------------------------------------------------------------- summary
def rmse_cells(d, cols, min_days):
    g = d.groupby(["lat", "lon"])
    n = g.size()
    res = pd.DataFrame({c: np.sqrt(((d[c] - d["ghi"]) ** 2).groupby([d["lat"], d["lon"]]).mean()) for c in cols})
    res["n"] = n
    return res[res["n"] >= min_days]


def summarise(daily, out, block, models, links_nc):
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    with_cols = [f"{m}_{mo}" for mo in models for m in METHODS]
    nc_cols = [f"{m}_{mo}_nc" for mo in models for m in METHODS]
    common = daily.dropna(subset=with_cols + nc_cols + ["CLIM", "PERSIST_KC"])
    extra = daily[daily[with_cols].isna().any(axis=1)].dropna(subset=nc_cols + ["CLIM", "PERSIST_KC", "ghi"])
    ncell = common[["lat", "lon"]].drop_duplicates().shape[0]
    print(f"\n{ncell} cells | common window (with cloud usable): {common['date'].min().date()} to {common['date'].max().date()}"
          f" ({common['date'].nunique()} days)")
    if len(extra):
        print(f"extra days usable only without CLOUD_AMT: {extra['date'].min().date()} to {extra['date'].max().date()} "
              f"({extra['date'].nunique()} distinct days, {len(extra) / max(ncell, 1):.0f} per cell on average)")

    allcols = ["CLIM", "PERSIST_KC"] + with_cols + nc_cols
    R = rmse_cells(common, allcols, 100)
    skill = {c: (100 * (1 - R[c] / R["CLIM"])) for c in allcols if c != "CLIM"}
    tab = pd.DataFrame({"mean_rmse": R[allcols].mean(), "skill_vs_clim_pct": pd.Series({c: s.mean() for c, s in skill.items()})})
    tab.loc["CLIM", "skill_vs_clim_pct"] = 0.0
    print("\n--- COMMON window: mean RMSE and skill vs climatology (mean over cells) ---")
    print(tab.round(4).to_string())
    safe_csv(tab.reset_index().rename(columns={"index": "method"}), os.path.join(out, "ablation_table.csv"))

    blocks = np.array([f"{np.floor(a / block)}_{np.floor(b / block)}" for a, b in R.index])
    rows = []

    def paired(A, B, label, RR, bl):
        diff = 100 * (1 - RR[A] / RR[B])
        lo, hi = block_bootstrap_ci(diff.values, bl, n_boot=1000)
        rows.append(dict(comparison=label, A=A, B=B, mean_pct=diff.mean(), ci_low=lo, ci_high=hi,
                         pct_cells_A_better=100 * (RR[A] < RR[B]).mean(), n_cells=len(RR)))

    for mo in models:
        for m in METHODS:
            paired(f"{m}_{mo}_nc", f"{m}_{mo}", f"cost of removing cloud amount ({m},{mo})", R, blocks)
        paired(f"PCMCI_{mo}_nc", f"LASSO_{mo}_nc", f"no-cloud: PCMCI vs LASSO ({mo})", R, blocks)
        paired(f"PCMCI_{mo}_nc", f"ALL_{mo}_nc", f"no-cloud: PCMCI vs ALL ({mo})", R, blocks)
        paired(f"PCMCI_{mo}_nc", "PERSIST_KC", f"no-cloud PCMCI vs persistence ({mo})", R, blocks)
        paired(f"PCMCI_{mo}_nc", "CLIM", f"no-cloud PCMCI vs climatology ({mo})", R, blocks)
    pr = pd.DataFrame(rows)
    safe_csv(pr, os.path.join(out, "ablation_paired.csv"))
    print("\n--- COMMON window: per-cell % RMSE reduction of A relative to B (>0 = A better); 95% CI spatial block bootstrap ---")
    for r in pr.itertuples():
        print(f"  {r.comparison:46s}: {r.mean_pct:+6.2f}% [{r.ci_low:+6.2f}, {r.ci_high:+6.2f}] | A better in {r.pct_cells_A_better:5.1f}% of {r.n_cells} cells")

    if len(extra):
        RX = rmse_cells(extra, ["CLIM", "PERSIST_KC"] + nc_cols, 20)
        print(f"\n--- EXTRA days only (no-cloud models, {len(RX)} cells with >=20 such days): skill vs climatology ---")
        print(pd.Series({c: (100 * (1 - RX[c] / RX["CLIM"])).mean() for c in ["PERSIST_KC"] + nc_cols}).round(2).to_string())
        print("(a short window of ~3 months: descriptive only)")

    if links_nc is not None:
        pcl = links_nc[(links_nc["method"] == "PCMCI") & (links_nc["lag"] >= H)]
        n = pcl[["lat", "lon"]].drop_duplicates().shape[0]
        fr = (pcl.groupby(["driver", "lag"]).size() / n * 100).sort_values(ascending=False).head(12)
        print(f"\n--- PCMCI parents WITHOUT cloud amount: % of {n} cells (top 12) ---")
        print(fr.round(1).to_string())
        print(f"mean number of parents per cell: {len(pcl) / n:.2f}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--links_nocloud", required=True)
    ap.add_argument("--with_daily", required=True, help="step 6 daily_predictions.csv.gz (with cloud amount)")
    ap.add_argument("--out", default="step7_out")
    ap.add_argument("--models", nargs="+", default=["LIN", "KRR"], choices=["LIN", "POLY2", "KRR", "RF"])
    ap.add_argument("--block", type=float, default=2.0)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    links = pd.read_csv(a.links_nocloud)
    if (links["driver"] == DROP).any():
        raise SystemExit("The no-cloud links file contains CLOUD_AMT - rerun step 1 with --exclude CLOUD_AMT.")
    links["key"] = [(round(float(x), 4), round(float(y), 4)) for x, y in zip(links["lat"], links["lon"])]
    by_cell = {k: g for k, g in links.groupby("key")}
    files_all = sorted(glob.glob(os.path.join(a.prepared, "cell_*.csv")))
    files = [f for f in files_all if args_key(f) in by_cell]
    print(f"{len(files_all)} prepared cells | {len(by_cell)} cells in links file | {len(files)} in both")
    if not files:
        raise SystemExit("No prepared cell matches the links file.")
    if len(files) < len(files_all):
        print(f"WARNING: {len(files_all) - len(files)} prepared cells have no links and are skipped.")
    if a.limit:
        files = files[: a.limit]
    jobs = [(f, by_cell[args_key(f)], a.models) for f in files]
    print(f"running {len(files)} cells | models {a.models} | variables without {DROP}")

    frames, errs, t0 = [], [], time.time()
    it = ProcessPoolExecutor(max_workers=a.n_jobs).map(run_cell, jobs) if a.n_jobs > 1 else map(run_cell, jobs)
    for k, (df, e) in enumerate(it, 1):
        if e:
            errs.append(e)
        else:
            frames.append(df)
        print(f"[{k}/{len(jobs)}] {time.time() - t0:.0f}s")
    if errs:
        safe_csv(pd.DataFrame(errs, columns=["lat", "lon", "error"]), os.path.join(a.out, "errors.csv"))
        print(f"{len(errs)} cells skipped - see errors.csv")
    nc = pd.concat(frames, ignore_index=True)
    num = nc.select_dtypes("float").columns
    nc[num] = nc[num].round(5)
    nc["date"] = pd.to_datetime(nc["date"])

    wd = pd.read_csv(a.with_daily)
    wd["date"] = pd.to_datetime(wd["date"])
    wd["lat"], wd["lon"] = wd["lat"].round(4), wd["lon"].round(4)
    nc["lat"], nc["lon"] = nc["lat"].round(4), nc["lon"].round(4)
    keep = ["lat", "lon", "date"] + [c for c in wd.columns if c.split("_")[0] in METHODS and c.split("_")[1] in a.models]
    merged = nc.merge(wd[keep], on=["lat", "lon", "date"], how="left")
    # sanity check: baselines from both runs must coincide on shared days
    chk = nc.merge(wd[["lat", "lon", "date", "ghi", "CLIM"]], on=["lat", "lon", "date"], suffixes=("", "_w"))
    if len(chk):
        d1, d2 = (chk["ghi"] - chk["ghi_w"]).abs().max(), (chk["CLIM"] - chk["CLIM_w"]).abs().max()
        print(f"consistency check on {len(chk)} shared cell-days: max |ghi diff| = {d1:.2g}, max |climatology diff| = {d2:.2g}")
        if max(d1, d2) > 1e-3:
            print("WARNING: the with-cloud file does not match this run's data - use the same prepared folder.")
    safe_csv(merged, os.path.join(a.out, "daily_predictions_nocloud.csv.gz"), compression="gzip")
    summarise(merged, a.out, a.block, a.models, links)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
