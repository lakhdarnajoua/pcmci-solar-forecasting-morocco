"""
Forecasting stage - Step 8a: fit and evaluate ONE rolling-origin fold.

A "fold" = a training-period end date (must match the --train_end used for the step1 run that produced
--prepared/--links, or the split here won't match the causal-discovery split -> leakage) plus a fixed test
window. Use this once per fold; forecast_step8b_rolling_summary.py then combines the folds.

Why causal discovery must be rerun per fold: the training-year day-of-year climatology (kc_clim, ref_clr in
the prepared files) and the PCMCI parents both depend on which years count as "training". Reusing a single
step1 run's prepared/links for an earlier test window would leak later years into both.

Fits fixed models (LIN, KRR by default) on:
    PCMCI   lagged parents of this fold (BH-FDR 0.01)
    LASSO   same number of inputs, chosen by the LASSO path on this fold's training rows
    ALL     all 56 lagged inputs
plus CLIM and PERSIST_KC baselines. Metrics are on the GHI scale, computed only over [--test_start, --test_end].

Usage (repeat once per fold, each from its own step1 --train_end run):
    python forecast_step8a_origin_fit.py --prepared forecast_orig2020\\prepared --links forecast_orig2020\\links_into_kc_all_cells.csv ^
        --train_end 2020-12-31 --test_start 2021-01-01 --test_end 2021-12-31 --fold 2021 --out step8_folds --n_jobs 4 --limit 3
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

from forecast_step3_fit import VARS, args_key, build_features, fit_predict, metrics
from forecast_step4_selection_comparison import select_lasso_path

warnings.filterwarnings("ignore")
H = 1
METHODS = ["PCMCI", "LASSO", "ALL"]


def safe_csv(df, path, **kw):
    try:
        df.to_csv(path, index=False, **kw)
    except PermissionError:
        alt = path.replace(".csv", f"_{int(time.time())}.csv")
        df.to_csv(alt, index=False, **kw)
        print(f"{path} is locked - saved as {alt}")


def run_cell(args):
    path, links, models, train_end, test_start, test_end = args
    lat, lon = args_key(path)
    P = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    ref, kcc, ghi, kcraw = P["ref_clr"], P["kc_clim"], P["ghi"], P["kc_raw"]
    tau_max = max(int(links["lag"].max()) if len(links) else 7, 7)

    pool = [(v, l) for v in VARS for l in range(H, tau_max + 1)]
    pool_names = [f"{v}_L{l}" for v, l in pool]
    X_pool = build_features(P, pool, H)
    ok = X_pool.notna().all(axis=1) & P["kc"].notna() & kcraw.notna() & ghi.notna() & ref.notna()
    ok &= kcraw.shift(H).notna()
    idx = ok[ok].index
    tr = idx[idx <= train_end]
    te = idx[(idx >= test_start) & (idx <= test_end)]
    if len(te) < 30 or len(tr) < 500:
        return None, (lat, lon, "not enough rows")

    pc = links[(links["method"] == "PCMCI") & (links["lag"] >= H) & (links["p"] <= 0.0101)]
    names_pc = sorted({f"{r.driver}_L{int(r.lag)}" for r in pc.itertuples()})
    k = len(names_pc)
    A = X_pool[pool_names].loc[tr].values
    y = P["kc"].loc[tr].values
    sets = {"ALL": pool_names}
    if k > 0:
        sets["PCMCI"] = names_pc
        sets["LASSO"] = [pool_names[i] for i in select_lasso_path(A, y, k)]
    cols_sc = ["sin_doy", "cos_doy"]

    row = dict(lat=lat, lon=lon, k=k, n_train=len(tr), n_test=len(te))
    clim_pred = kcc.loc[te].values * ref.loc[te].values
    pers_pred = kcraw.shift(H).loc[te].values * ref.loc[te].values
    obs = ghi.loc[te].values
    m = metrics(obs, clim_pred); row.update({f"CLIM_{k2}": v for k2, v in m.items()})
    m = metrics(obs, pers_pred); row.update({f"PERSIST_KC_{k2}": v for k2, v in m.items()})
    for meth, names in sets.items():
        X = X_pool[list(names) + cols_sc]
        for model in models:
            p = fit_predict(model, X.loc[tr].values, y, X.loc[te].values)
            pred = (kcc.loc[te].values + p) * ref.loc[te].values
            m = metrics(obs, pred)
            row.update({f"{meth}_{model}_{k2}": v for k2, v in m.items()})
    return row, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--links", required=True)
    ap.add_argument("--train_end", required=True, help="must match the --train_end used for the step1 run above")
    ap.add_argument("--test_start", required=True)
    ap.add_argument("--test_end", required=True)
    ap.add_argument("--fold", required=True, help="short label for this fold, e.g. 2021")
    ap.add_argument("--out", default="step8_folds")
    ap.add_argument("--models", nargs="+", default=["LIN", "KRR"], choices=["LIN", "POLY2", "KRR", "RF"])
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    train_end, test_start, test_end = pd.Timestamp(a.train_end), pd.Timestamp(a.test_start), pd.Timestamp(a.test_end)

    links = pd.read_csv(a.links)
    links["key"] = [(round(float(x), 4), round(float(y), 4)) for x, y in zip(links["lat"], links["lon"])]
    by_cell = {k: g for k, g in links.groupby("key")}
    files_all = sorted(glob.glob(os.path.join(a.prepared, "cell_*.csv")))
    files = [f for f in files_all if args_key(f) in by_cell]
    print(f"fold {a.fold} | train <= {train_end.date()} | test {test_start.date()} to {test_end.date()}")
    print(f"{len(files_all)} prepared cells | {len(by_cell)} cells in links file | {len(files)} in both")
    if not files:
        raise SystemExit("No prepared cell matches the links file.")
    if a.limit:
        files = files[: a.limit]
    jobs = [(f, by_cell[args_key(f)], a.models, train_end, test_start, test_end) for f in files]

    rows, errs, t0 = [], [], time.time()
    it = ProcessPoolExecutor(max_workers=a.n_jobs).map(run_cell, jobs) if a.n_jobs > 1 else map(run_cell, jobs)
    for i, (row, e) in enumerate(it, 1):
        if e:
            errs.append(e)
        else:
            rows.append(row)
        if i % 25 == 0 or i == len(jobs):
            print(f"[{i}/{len(jobs)}] {time.time() - t0:.0f}s")
    if errs:
        safe_csv(pd.DataFrame(errs, columns=["lat", "lon", "error"]), os.path.join(a.out, f"errors_{a.fold}.csv"))
        print(f"{len(errs)} cells skipped - see errors_{a.fold}.csv")
    res = pd.DataFrame(rows)
    res.insert(0, "fold", a.fold)
    outpath = os.path.join(a.out, f"fold_{a.fold}_results.csv")
    safe_csv(res, outpath)
    n = len(res)
    print(f"\n{n} cells | mean k = {res['k'].mean():.2f} (cells with 0 PCMCI parents: {(res['k'] == 0).sum()})")
    for c in ["CLIM_rmse", "PERSIST_KC_rmse"] + [f"{m}_{mo}_rmse" for mo in a.models for m in METHODS]:
        if c in res:
            skill = 100 * (1 - res[c] / res["CLIM_rmse"])
            print(f"  {c:20s} mean rmse {res[c].mean():.4f} | skill vs CLIM {skill.mean():+6.2f}%")
    print(f"saved: {outpath}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
