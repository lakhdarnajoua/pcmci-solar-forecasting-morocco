"""
Forecasting stage - Step 4: does PCMCI selection beat simple, non-causal selection at EQUAL size?

For every cell, k = number of PCMCI parents (lag >= 1) found in step 2. Each alternative method then
picks exactly k inputs from the same pool of 56 candidates (8 variables x lags 1..7), using the
TRAINING years (2015-2022) only:
    PCMCI       parents from step 2 (reference)
    TOPK_CORR   k largest |Pearson correlation| with the target anomaly
    TOPK_MI     k largest mutual information with the target anomaly
    LASSO_PATH  first k inputs to enter the LASSO (LARS) regularisation path
    OMP         greedy forward selection (orthogonal matching pursuit), k inputs
    RANDOM      k inputs drawn at random (mean of 20 draws, linear model only) = null reference
    ALL         all 56 inputs (no selection)
All sets also get day-of-year sin/cos. Same protocol, rows and metrics as step 3: train <= 2022-12-31,
test 2023-2025, metrics on the GHI scale. Models are fixed in advance (no choice made on the test set).

Inference: per-cell differences are NOT independent (neighbouring 0.5 deg cells are strongly correlated),
so confidence intervals come from a spatial block bootstrap (cells grouped in BLOCK x BLOCK degree tiles,
tiles resampled with replacement).

Usage:
    python forecast_step4_selection_comparison.py --prepared forecast_full\\prepared \\
        --links forecast_full\\links_into_kc_all_cells.csv --out step4_out --n_jobs 4 --limit 3
Needs forecast_step3_fit.py in the same folder (imports its helpers).
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
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import OrthogonalMatchingPursuit, lars_path
from sklearn.preprocessing import StandardScaler

from forecast_step3_fit import VARS, TRAIN_END, args_key, build_features, fit_predict, metrics

warnings.filterwarnings("ignore")
H = 1
N_RANDOM = 20


# ----------------------------------------------------------------------------- selectors
def select_topk_corr(X, y, k):
    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / y.std()
    c = np.abs(Xs.T @ ys) / len(y)
    return list(np.argsort(-c)[:k])


def select_topk_mi(X, y, k):
    mi = mutual_info_regression(X, y, n_neighbors=5, random_state=0)
    return list(np.argsort(-mi)[:k])


def select_lasso_path(X, y, k):
    Xs = (X - X.mean(0)) / X.std(0)
    _, _, coefs = lars_path(Xs, y - y.mean(), method="lasso")
    nz = (coefs != 0).sum(axis=0)
    hit = np.where(nz >= k)[0]
    step = hit[0] if len(hit) else coefs.shape[1] - 1
    c = np.abs(coefs[:, step])
    return list(np.argsort(-c)[:k])


def select_omp(X, y, k):
    Xs = (X - X.mean(0)) / X.std(0)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k).fit(Xs, y - y.mean())
    return list(np.flatnonzero(omp.coef_))


# ----------------------------------------------------------------------------- per cell
def run_cell(args):
    path, links, models = args
    lat, lon = args_key(path)
    P = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    ref, kcc, ghi, kcraw = P["ref_clr"], P["kc_clim"], P["ghi"], P["kc_raw"]
    tau_max = int(links["lag"].max()) if len(links) else 7
    tau_max = max(tau_max, 7)

    pool = [(v, l) for v in VARS for l in range(H, tau_max + 1)]
    pool_names = [f"{v}_L{l}" for v, l in pool]
    X_pool = build_features(P, pool, H)
    ok = X_pool.notna().all(axis=1) & P["kc"].notna() & kcraw.notna() & ghi.notna() & ref.notna()
    idx = ok[ok].index
    tr, te = idx[idx <= TRAIN_END], idx[idx > TRAIN_END]
    if len(te) < 100 or len(tr) < 500:
        return [], [], [(lat, lon, "not enough rows")]

    pc = sorted({(r.driver, int(r.lag)) for r in links.itertuples() if r.method == "PCMCI" and r.lag >= H})
    k = len(pc)
    if k == 0:
        return [], [], [(lat, lon, "no PCMCI parents")]
    pc_names = [f"{v}_L{l}" for v, l in pc]

    A = X_pool[pool_names].loc[tr].values
    y = P["kc"].loc[tr].values
    sel = {
        "PCMCI": [pool_names.index(n) for n in pc_names],
        "TOPK_CORR": select_topk_corr(A, y, k),
        "TOPK_MI": select_topk_mi(A, y, k),
        "LASSO_PATH": select_lasso_path(A, y, k),
        "OMP": select_omp(A, y, k),
        "ALL": list(range(len(pool_names))),
    }

    rows, feats = [], []
    cols_sc = ["sin_doy", "cos_doy"]

    def run(method, cols_idx, model, seed_tag=None):
        names = [pool_names[i] for i in cols_idx] + cols_sc
        X = X_pool.reindex(columns=pool_names + cols_sc)[names]
        p = fit_predict(model, X.loc[tr].values, y, X.loc[te].values)
        ghi_hat = (kcc.loc[te].values + p) * ref.loc[te].values
        m = metrics(ghi.loc[te].values, ghi_hat)
        return m

    # climatology reference (same rows)
    clim = metrics(ghi.loc[te].values, kcc.loc[te].values * ref.loc[te].values)
    rows.append(dict(lat=lat, lon=lon, method="CLIMATOLOGY", model="-", k=0, test_r2=clim["r2"],
                     test_rmse=clim["rmse"], test_mae=clim["mae"], test_mbe=clim["mbe"]))

    for method, ci in sel.items():
        feats.append(dict(lat=lat, lon=lon, method=method, k=len(ci),
                          features=";".join(pool_names[i] for i in ci) if method != "ALL" else "all"))
        for model in models:
            m = run(method, ci, model)
            rows.append(dict(lat=lat, lon=lon, method=method, model=model, k=len(ci), test_r2=m["r2"],
                             test_rmse=m["rmse"], test_mae=m["mae"], test_mbe=m["mbe"]))

    # random-k null (linear model, mean over draws)
    rng = np.random.default_rng(int(abs(lat) * 1e4) * 100003 + int(abs(lon) * 1e4))
    rr = [run("RANDOM", list(rng.choice(len(pool_names), k, replace=False)), "LIN") for _ in range(N_RANDOM)]
    rows.append(dict(lat=lat, lon=lon, method="RANDOM", model="LIN", k=k,
                     test_r2=np.mean([m["r2"] for m in rr]), test_rmse=np.mean([m["rmse"] for m in rr]),
                     test_mae=np.mean([m["mae"] for m in rr]), test_mbe=np.mean([m["mbe"] for m in rr])))
    return rows, feats, []


# ----------------------------------------------------------------------------- summary
def block_bootstrap_ci(diff, blocks, n_boot=2000, seed=0):
    ub = np.unique(blocks)
    groups = [np.flatnonzero(blocks == b) for b in ub]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(ub), len(ub))
        means[i] = diff[np.concatenate([groups[j] for j in pick])].mean()
    return np.percentile(means, [2.5, 97.5])


def summarise(res, feats, out, block):
    res = res.copy()
    clim = res[res["method"] == "CLIMATOLOGY"].set_index(["lat", "lon"])["test_rmse"]
    res = res[res["method"] != "CLIMATOLOGY"]
    res["skill_vs_clim_pct"] = [100 * (1 - r / clim.loc[(a, b)]) for r, a, b in zip(res["test_rmse"], res["lat"], res["lon"])]

    # overlap of each method's features with PCMCI parents
    f = feats.copy()
    f["set"] = f["features"].apply(lambda s: set(s.split(";")))
    pc = f[f["method"] == "PCMCI"].set_index(["lat", "lon"])["set"]
    jac = []
    for r in f.itertuples():
        if r.method in ("PCMCI", "ALL"):
            continue
        a, b = r.set, pc.loc[(r.lat, r.lon)]
        jac.append((r.method, len(a & b) / len(a | b)))
    jac = pd.DataFrame(jac, columns=["method", "jaccard"]).groupby("method")["jaccard"].mean()

    tab = (res.groupby(["model", "method"]).agg(k=("k", "mean"), r2=("test_r2", "mean"), rmse=("test_rmse", "mean"),
                                                 skill_vs_clim_pct=("skill_vs_clim_pct", "mean")).reset_index())
    tab["jaccard_with_PCMCI"] = tab["method"].map(jac)
    print(f"\n=== Equal-size selection comparison | {res[['lat', 'lon']].drop_duplicates().shape[0]} cells | test 2023-2025 ===")
    print(tab.round(4).to_string(index=False))
    tab.to_csv(os.path.join(out, "selection_comparison_table.csv"), index=False)

    print(f"\nPCMCI vs alternative (per-cell % RMSE reduction of PCMCI relative to the alternative; >0 = PCMCI better)")
    print(f"95% CI = spatial block bootstrap ({block} x {block} deg tiles, 2000 resamples)")
    lines = []
    for model in sorted(res["model"].unique()):
        d = res[res["model"] == model].pivot_table(index=["lat", "lon"], columns="method", values="test_rmse")
        blocks = np.array([f"{np.floor(a / block)}_{np.floor(b / block)}" for a, b in d.index])
        for alt in [c for c in d.columns if c != "PCMCI"]:
            diff = 100 * (1 - d["PCMCI"].values / d[alt].values)
            lo, hi = block_bootstrap_ci(diff, blocks)
            win = 100 * (d["PCMCI"].values < d[alt].values).mean()
            print(f"  {model:4s} PCMCI vs {alt:10s}: {diff.mean():+6.2f}% [{lo:+6.2f}, {hi:+6.2f}] | PCMCI better in {win:5.1f}% of cells")
            lines.append(dict(model=model, alternative=alt, mean_pct=diff.mean(), ci_low=lo, ci_high=hi, pct_cells_pcmci_better=win))
    pd.DataFrame(lines).to_csv(os.path.join(out, "pcmci_vs_alternatives.csv"), index=False)
    print("\nRule of thumb: if the interval contains 0, PCMCI is not distinguishable from that alternative.")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--links", required=True)
    ap.add_argument("--out", default="step4_out")
    ap.add_argument("--models", nargs="+", default=["LIN", "KRR"], choices=["LIN", "POLY2", "KRR", "RF"])
    ap.add_argument("--block", type=float, default=2.0, help="tile size (deg) for the block bootstrap")
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    links = pd.read_csv(a.links)
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
    print(f"running {len(files)} cells | models {a.models} + RANDOM(LIN)")

    R, F, E, t0 = [], [], [], time.time()
    if a.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=a.n_jobs) as ex:
            for k, (r, f, e) in enumerate(ex.map(run_cell, jobs), 1):
                R += r; F += f; E += e
                print(f"[{k}/{len(jobs)}] {time.time() - t0:.0f}s")
    else:
        for k, j in enumerate(jobs, 1):
            r, f, e = run_cell(j)
            R += r; F += f; E += e
            print(f"[{k}/{len(jobs)}] {time.time() - t0:.0f}s")

    res, feats = pd.DataFrame(R), pd.DataFrame(F)
    res.to_csv(os.path.join(a.out, "results_long.csv"), index=False)
    feats.to_csv(os.path.join(a.out, "selected_features.csv"), index=False)
    if E:
        pd.DataFrame(E, columns=["lat", "lon", "error"]).to_csv(os.path.join(a.out, "errors.csv"), index=False)
        print(f"{len(E)} cells skipped - see errors.csv")
    if len(res):
        summarise(res, feats, a.out, a.block)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
