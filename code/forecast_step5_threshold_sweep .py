"""
Forecasting stage - Step 5: do stricter PCMCI thresholds give a better (smaller) predictor set?

No re-run of causal discovery is needed: step 2 stored, for every PCMCI link that passed the BH-FDR 0.01 level,
the BH-adjusted p-value (column p, exactly as returned by tigramite when fdr_method="fdr_bh") and the partial
correlation (val). Thresholding column p at a stricter level gives exactly the graph a stricter alpha_level
would have produced (same MCI p-values, same conditioning sets). Column q, present in some older links
files as a redundant secondary FDR correction, is not used.

Variants (lagged PCMCI parents into kc, lag >= 1):
    fdr<=0.01    baseline = the PCMCI sets used in steps 3-4
    fdr<=0.001   stricter FDR level
    fdr<=0.0001  much stricter FDR level
    |val|>=0.075 effect-size filter on top of fdr<=0.01 (a 0.05 filter would change nothing: with ~2,900 days
    |val|>=0.10  every link that passes fdr<=0.01 already has |val| >= ~0.05)
    |val|>=0.15  strongest effect-size filter

For every variant and cell the script fits the same fixed models as step 4 on
    PCMCI   the parents kept by the variant
    LASSO   the same number k of inputs chosen by the LASSO path (training years only)
and reports test-period (2023-2025) metrics on the GHI scale. ALL (56 inputs) and climatology are references.
Identical input sets are fitted only once (cache). Cells where a variant keeps 0 parents are fitted with the
day-of-year terms only and are excluded from the PCMCI-vs-LASSO comparison.
CIs come from the same spatial block bootstrap as step 4.

Usage:
    python forecast_step5_threshold_sweep.py --prepared forecast_full\\prepared \\
        --links forecast_full\\links_into_kc_all_cells.csv --out step5_out --n_jobs 4 --limit 3
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

from forecast_step3_fit import VARS, TRAIN_END, args_key, build_features, fit_predict, metrics
from forecast_step4_selection_comparison import select_lasso_path, block_bootstrap_ci

warnings.filterwarnings("ignore")
H = 1
VARIANTS = [("fdr<=0.01", 0.01, 0.0), ("fdr<=0.001", 0.001, 0.0), ("fdr<=0.0001", 0.0001, 0.0),
            ("|val|>=0.075", 0.01, 0.075), ("|val|>=0.10", 0.01, 0.10), ("|val|>=0.15", 0.01, 0.15)]


def safe_csv(df, path):
    """Write a CSV; if the file is locked (e.g. open in Excel) save under a timestamped name instead."""
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        alt = path.replace(".csv", f"_{int(time.time())}.csv")
        df.to_csv(alt, index=False)
        print(f"{path} is locked - saved as {alt}")


def run_cell(args):
    path, links, models = args
    lat, lon = args_key(path)
    P = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    ref, kcc, ghi, kcraw = P["ref_clr"], P["kc_clim"], P["ghi"], P["kc_raw"]
    tau_max = max(int(links["lag"].max()) if len(links) else 7, 7)

    pool = [(v, l) for v in VARS for l in range(H, tau_max + 1)]
    pool_names = [f"{v}_L{l}" for v, l in pool]
    X_pool = build_features(P, pool, H)
    ok = X_pool.notna().all(axis=1) & P["kc"].notna() & kcraw.notna() & ghi.notna() & ref.notna()
    idx = ok[ok].index
    tr, te = idx[idx <= TRAIN_END], idx[idx > TRAIN_END]
    if len(te) < 100 or len(tr) < 500:
        return [], [(lat, lon, "not enough rows")]

    A = X_pool[pool_names].loc[tr].values
    y = P["kc"].loc[tr].values
    cols_sc = ["sin_doy", "cos_doy"]
    cache, rows = {}, []

    def fit_eval(names, model):
        key = (tuple(sorted(names)), model)
        if key not in cache:
            X = X_pool[list(names) + cols_sc]
            p = fit_predict(model, X.loc[tr].values, y, X.loc[te].values)
            cache[key] = metrics(ghi.loc[te].values, (kcc.loc[te].values + p) * ref.loc[te].values)
        return cache[key]

    def add(variant, method, names, model):
        m = fit_eval(names, model)
        rows.append(dict(lat=lat, lon=lon, variant=variant, method=method, model=model, k=len(names),
                         test_r2=m["r2"], test_rmse=m["rmse"], test_mae=m["mae"], test_mbe=m["mbe"]))

    clim = metrics(ghi.loc[te].values, kcc.loc[te].values * ref.loc[te].values)
    rows.append(dict(lat=lat, lon=lon, variant="-", method="CLIMATOLOGY", model="-", k=0, test_r2=clim["r2"],
                     test_rmse=clim["rmse"], test_mae=clim["mae"], test_mbe=clim["mbe"]))
    for model in models:
        add("-", "ALL", pool_names, model)

    pc = links[(links["method"] == "PCMCI") & (links["lag"] >= H)]
    for label, qthr, vmin in VARIANTS:
        keep = pc[(pc["p"] <= qthr) & (pc["val"].abs() >= vmin)]
        names = sorted({f"{r.driver}_L{int(r.lag)}" for r in keep.itertuples()})
        k = len(names)
        lasso_names = [pool_names[i] for i in select_lasso_path(A, y, k)] if k > 0 else None
        for model in models:
            add(label, "PCMCI", names, model)
            if lasso_names is not None:
                add(label, "LASSO", lasso_names, model)
    return rows, []


def summarise(res, out, block):
    res = res.copy()
    clim = res[res["method"] == "CLIMATOLOGY"].set_index(["lat", "lon"])["test_rmse"]
    res = res[res["method"] != "CLIMATOLOGY"]
    res["skill_vs_clim_pct"] = [100 * (1 - r / clim.loc[(a, b)]) for r, a, b in zip(res["test_rmse"], res["lat"], res["lon"])]
    order = [v[0] for v in VARIANTS]

    tab = (res.groupby(["model", "method", "variant"])
           .agg(k=("k", "mean"), n_cells=("k", "size"), cells_k0=("k", lambda s: int((s == 0).sum())),
                r2=("test_r2", "mean"), rmse=("test_rmse", "mean"), skill_vs_clim_pct=("skill_vs_clim_pct", "mean"))
           .reset_index())
    tab["o"] = tab["variant"].map({v: i for i, v in enumerate(order)}).fillna(-1)
    tab = tab.sort_values(["model", "method", "o"]).drop(columns="o")
    print(f"\n=== Threshold sweep | {res[['lat', 'lon']].drop_duplicates().shape[0]} cells | test 2023-2025 ===")
    print(tab.round(4).to_string(index=False))
    safe_csv(tab, os.path.join(out, "threshold_sweep_table.csv"))

    print("\nPer-cell % RMSE reduction of A relative to B (>0 = A better); 95% CI = spatial block bootstrap "
          f"({block} x {block} deg tiles)")
    lines = []
    for model in sorted(res["model"].unique()):
        d = res[res["model"] == model]
        piv = d.assign(col=d["method"] + "|" + d["variant"]).pivot_table(index=["lat", "lon"], columns="col", values="test_rmse")
        blocks = np.array([f"{np.floor(a / block)}_{np.floor(b / block)}" for a, b in piv.index])
        comps = []
        for v in order[1:]:
            comps.append((f"PCMCI|{v}", "PCMCI|fdr<=0.01"))
        for v in order:
            comps.append((f"PCMCI|{v}", f"LASSO|{v}"))
        for v in order:
            comps.append((f"PCMCI|{v}", "ALL|-"))
        for a, b in comps:
            if a not in piv or b not in piv:
                continue
            x = piv[[a, b]].dropna()
            if len(x) < 10:
                continue
            bl = np.array([f"{np.floor(i / block)}_{np.floor(j / block)}" for i, j in x.index])
            diff = 100 * (1 - x[a].values / x[b].values)
            lo, hi = block_bootstrap_ci(diff, bl)
            win = 100 * (x[a].values < x[b].values).mean()
            print(f"  {model:4s} {a:18s} vs {b:16s}: {diff.mean():+6.2f}% [{lo:+6.2f}, {hi:+6.2f}] | A better in {win:5.1f}% of {len(x)} cells")
            lines.append(dict(model=model, A=a, B=b, mean_pct=diff.mean(), ci_low=lo, ci_high=hi,
                              pct_cells_A_better=win, n_cells=len(x)))
    safe_csv(pd.DataFrame(lines), os.path.join(out, "threshold_sweep_comparisons.csv"))
    print("\nIf the interval contains 0, the two are not distinguishable.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--links", required=True)
    ap.add_argument("--out", default="step5_out")
    ap.add_argument("--models", nargs="+", default=["LIN", "KRR"], choices=["LIN", "POLY2", "KRR", "RF"])
    ap.add_argument("--block", type=float, default=2.0)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    links = pd.read_csv(a.links)
    pm = links.loc[links["method"] == "PCMCI", "p"]
    if pm.empty:
        raise SystemExit("No PCMCI links in the links file - it must come from forecast_step1_prep_pcmci.py.")
    if pm.max() > 0.0101:
        print(f"WARNING: max stored PCMCI p = {pm.max():.4f} > 0.01 - the links file was not produced with alpha 0.01 "
              f"and BH-FDR; the fdr<=0.01 variant will not match the step 3-4 sets.")
    links["key"] = [(round(float(x), 4), round(float(y), 4)) for x, y in zip(links["lat"], links["lon"])]
    by_cell = {k: g for k, g in links.groupby("key")}
    files_all = sorted(glob.glob(os.path.join(a.prepared, "cell_*.csv")))
    files = [f for f in files_all if args_key(f) in by_cell]
    print(f"{len(files_all)} prepared cells | {len(by_cell)} cells in links file | {len(files)} in both")
    if not files:
        raise SystemExit("No prepared cell matches the links file - wrong --prepared / --links pair.")
    if len(files) < len(files_all):
        print(f"WARNING: {len(files_all) - len(files)} prepared cells have no links and are skipped.")
    if a.limit:
        files = files[: a.limit]
    jobs = [(f, by_cell[args_key(f)], a.models) for f in files]
    print(f"running {len(files)} cells | models {a.models} | variants {[v[0] for v in VARIANTS]}")

    R, E, t0 = [], [], time.time()
    if a.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=a.n_jobs) as ex:
            for k, (r, e) in enumerate(ex.map(run_cell, jobs), 1):
                R += r; E += e
                print(f"[{k}/{len(jobs)}] {time.time() - t0:.0f}s")
    else:
        for k, j in enumerate(jobs, 1):
            r, e = run_cell(j)
            R += r; E += e
            print(f"[{k}/{len(jobs)}] {time.time() - t0:.0f}s")

    res = pd.DataFrame(R)
    safe_csv(res, os.path.join(a.out, "results_long.csv"))
    if E:
        safe_csv(pd.DataFrame(E, columns=["lat", "lon", "error"]), os.path.join(a.out, "errors.csv"))
        print(f"{len(E)} cells skipped - see errors.csv")
    if len(res):
        summarise(res, a.out, a.block)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()