"""
Forecasting stage - Step 10: does a NONLINEAR conditional-independence test change which
parents PCMCI finds? PCMCI's ParCorr test assumes linear-Gaussian dependence; cloud-precipitation-
irradiance relationships are known to be nonlinear. This is a targeted, not exhaustive, check
(re-running full PCMCI with CMIknn on all 250 cells is computationally infeasible - a single
CMIknn conditional-independence test at n~2,900 training days takes ~100-400s depending on the
number of shuffle-test samples; see the runtime note below).

Design (per selected cell):
    - Take the existing PCMCI (ParCorr) parent set P for kc, already computed in step 2.
    - For each parent (v, lag) in P: leave-one-out MCI-style test, i.e. test whether
      kc(t) is conditionally independent of v(t-lag) given the OTHER parents in P, using
      CMIknn instead of ParCorr. This mirrors PCMCI's own final (MCI) test, just with a
      nonlinear test statistic.
    - Optionally (--check_runnerups), also test the top-2 candidate (variable, lag) pairs
      that were NOT selected by ParCorr but had the next-highest |partial correlation| with
      kc, conditioning on the same parent set P, to see whether nonlinearity reveals structure
      ParCorr missed.
    - Refit LIN/KRR on (a) the original PCMCI (ParCorr) parent set and (b) the subset of those
      parents that also clear p < 0.05 under CMIknn ("CMIknn-confirmed"), to see whether
      dropping the CMIknn-unconfirmed parents changes forecast accuracy on this cell subset.

Cell selection: a spatially stratified sample (k-means on lat/lon into --n_cells clusters,
nearest real cell to each cluster centroid) so the subset spans Morocco's climate gradient
rather than clustering in one region.

Runtime: with the defaults (12 cells, ~6-8 tests/cell, sig_samples=199), budget roughly
1.5-2.5 hours with --n_jobs 4 on a 4+ core machine. ALWAYS run --limit 1 first to time it on
your own machine before committing to the full run. Reduce --sig_samples (coarser p-value
resolution) or --n_cells to shorten it.

Usage:
    python forecast_step10_nonlinear_ci.py --prepared forecast_full\\prepared \\
        --links forecast_full\\links_into_kc_all_cells.csv --out step10_out --n_cells 12 \\
        --sig_samples 199 --n_jobs 4 --limit 1
Needs forecast_step3_fit.py in the same folder.
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

warnings.filterwarnings("ignore")
H = 1


def safe_csv(df, path, **kw):
    try:
        df.to_csv(path, index=False, **kw)
    except PermissionError:
        alt = path.replace(".csv", f"_{int(time.time())}.csv")
        df.to_csv(alt, index=False, **kw)
        print(f"{path} is locked - saved as {alt}")


def select_stratified_cells(files, n_cells, seed=0):
    """K-means on (lat, lon) into n_cells clusters; keep the real cell nearest each centroid."""
    from sklearn.cluster import KMeans
    coords = np.array([args_key(f) for f in files])
    km = KMeans(n_clusters=n_cells, random_state=seed, n_init=10).fit(coords)
    chosen = []
    for c in km.cluster_centers_:
        d = np.sum((coords - c) ** 2, axis=1)
        chosen.append(files[int(np.argmin(d))])
    return sorted(set(chosen), key=lambda f: args_key(f))


def run_cell(args):
    path, links, sig_samples, knn, check_runnerups, models = args
    from tigramite.independence_tests.cmiknn import CMIknn

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
        return None, (lat, lon, "not enough rows")

    pc = links[(links["method"] == "PCMCI") & (links["lag"] >= H) & (links["p"] <= 0.0101)]
    parents = sorted({(r.driver, int(r.lag)) for r in pc.itertuples()})
    if not parents:
        return None, (lat, lon, "no PCMCI parents")
    parent_names = [f"{v}_L{l}" for v, l in parents]

    y = P["kc"].loc[tr].values
    A = X_pool[pool_names].loc[tr].values  # standardised implicitly via ranks in CMIknn
    cmi = CMIknn(knn=knn, shuffle_neighbors=5, significance="shuffle_test",
                 sig_samples=sig_samples, workers=1)

    tested, t0 = [], time.time()
    print(f"  cell ({lat},{lon}): {len(parent_names)} parents to test: {parent_names}", flush=True)
    for i, name in enumerate(parent_names, 1):
        t_test = time.time()
        cond = [n for n in parent_names if n != name]
        cols = [name, "__kc__"] + cond
        data = np.column_stack([X_pool[name].loc[tr].values, y] +
                               [X_pool[c].loc[tr].values for c in cond])
        var_idx = np.array([0, 1] + [2] * len(cond))
        val = cmi.get_dependence_measure(data.T, var_idx)
        p = cmi.get_shuffle_significance(data.T, var_idx, val)
        tested.append(dict(lat=lat, lon=lon, driver_lag=name, role="PCMCI_parent",
                           cmi_val=float(val), cmi_p=float(p)))
        print(f"    [{i}/{len(parent_names)}] {name}: p={p:.4f} ({time.time() - t_test:.0f}s, "
              f"{time.time() - t0:.0f}s elapsed for this cell)", flush=True)

    if check_runnerups:
        # rank remaining candidates by |raw partial correlation with kc| (simple linear proxy,
        # just to pick which 2 non-parents to spend the expensive nonlinear test on)
        others = [n for n in pool_names if n not in parent_names]
        corrs = {}
        yk = (y - y.mean()) / y.std()
        for n in others:
            xn = X_pool[n].loc[tr].values
            xn = (xn - xn.mean()) / xn.std()
            corrs[n] = abs(np.nanmean(xn * yk))
        top2 = sorted(corrs, key=corrs.get, reverse=True)[:2]
        for j, name in enumerate(top2, 1):
            t_test = time.time()
            cond = parent_names
            data = np.column_stack([X_pool[name].loc[tr].values, y] +
                                   [X_pool[c].loc[tr].values for c in cond])
            var_idx = np.array([0, 1] + [2] * len(cond))
            val = cmi.get_dependence_measure(data.T, var_idx)
            p = cmi.get_shuffle_significance(data.T, var_idx, val)
            tested.append(dict(lat=lat, lon=lon, driver_lag=name, role="runner_up",
                               cmi_val=float(val), cmi_p=float(p)))
            print(f"    [runner-up {j}/2] {name}: p={p:.4f} ({time.time() - t_test:.0f}s, "
                  f"{time.time() - t0:.0f}s elapsed for this cell)", flush=True)

    confirmed = [t["driver_lag"] for t in tested if t["role"] == "PCMCI_parent" and t["cmi_p"] < 0.05]

    fit_rows = []
    cols_sc = ["sin_doy", "cos_doy"]
    for label, names in [("PCMCI_all", parent_names), ("PCMCI_cmiknn_confirmed", confirmed)]:
        if not names:
            continue
        X = X_pool[names + cols_sc]
        for model in models:
            p_ = fit_predict(model, X.loc[tr].values, y, X.loc[te].values)
            pred = (kcc.loc[te].values + p_) * ref.loc[te].values
            m = metrics(ghi.loc[te].values, pred)
            fit_rows.append(dict(lat=lat, lon=lon, feature_set=label, model=model, k=len(names),
                                 test_r2=m["r2"], test_rmse=m["rmse"]))
    clim_m = metrics(ghi.loc[te].values, kcc.loc[te].values * ref.loc[te].values)
    fit_rows.append(dict(lat=lat, lon=lon, feature_set="CLIM", model="-", k=0,
                         test_r2=clim_m["r2"], test_rmse=clim_m["rmse"]))

    print(f"  cell ({lat},{lon}) done in {time.time() - t0:.0f}s | "
          f"{len(parent_names)} parents, {sum(1 for t in tested if t['role'] == 'PCMCI_parent' and t['cmi_p'] < 0.05)} "
          f"confirmed at p<0.05")
    return (tested, fit_rows), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--links", required=True)
    ap.add_argument("--out", default="step10_out")
    ap.add_argument("--n_cells", type=int, default=12)
    ap.add_argument("--sig_samples", type=int, default=199,
                    help="CMIknn shuffle-test samples; resolution floor is ~1/(sig_samples+1)")
    ap.add_argument("--knn", type=float, default=0.2)
    ap.add_argument("--check_runnerups", action="store_true",
                    help="also test the top-2 ParCorr-rejected candidates (2 extra, expensive tests/cell)")
    ap.add_argument("--models", nargs="+", default=["LIN", "KRR"], choices=["LIN", "POLY2", "KRR", "RF"])
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="test on only the first N stratified cells")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    links = pd.read_csv(a.links)
    links["key"] = [(round(float(x), 4), round(float(y), 4)) for x, y in zip(links["lat"], links["lon"])]
    by_cell = {k: g for k, g in links.groupby("key")}
    files_all = sorted(glob.glob(os.path.join(a.prepared, "cell_*.csv")))
    files_all = [f for f in files_all if args_key(f) in by_cell]
    if len(files_all) < a.n_cells:
        raise SystemExit(f"Only {len(files_all)} cells have links - fewer than --n_cells {a.n_cells}.")
    chosen = select_stratified_cells(files_all, a.n_cells, a.seed)
    print(f"{len(files_all)} cells available | selected {len(chosen)} spatially stratified cells:")
    for f in chosen:
        print("  ", args_key(f))
    if a.limit:
        chosen = chosen[: a.limit]
    n_tests_per_cell = None  # printed after we know parent counts, informational only
    print(f"\nsig_samples={a.sig_samples} (~{100 * a.sig_samples / 50:.0f}s per test at n~2900, "
          f"single-threaded, based on empirical benchmarking) | running {len(chosen)} cells | "
          f"--check_runnerups {'ON (+2 tests/cell)' if a.check_runnerups else 'OFF'}")
    jobs = [(f, by_cell[args_key(f)], a.sig_samples, a.knn, a.check_runnerups, a.models) for f in chosen]

    tested_all, fit_all, errs, t0 = [], [], [], time.time()
    it = ProcessPoolExecutor(max_workers=a.n_jobs).map(run_cell, jobs) if a.n_jobs > 1 else map(run_cell, jobs)
    for i, (res, e) in enumerate(it, 1):
        if e:
            errs.append(e)
            print(f"[{i}/{len(jobs)}] FAILED: {e}")
        else:
            tested, fit_rows = res
            tested_all += tested
            fit_all += fit_rows
            print(f"[{i}/{len(jobs)}] {time.time() - t0:.0f}s elapsed")
    if errs:
        safe_csv(pd.DataFrame(errs, columns=["lat", "lon", "error"]), os.path.join(a.out, "errors.csv"))

    tested_df = pd.DataFrame(tested_all)
    fit_df = pd.DataFrame(fit_all)
    safe_csv(tested_df, os.path.join(a.out, "cmiknn_tests.csv"))
    safe_csv(fit_df, os.path.join(a.out, "cmiknn_refit_results.csv"))

    if len(tested_df):
        parent_tests = tested_df[tested_df["role"] == "PCMCI_parent"]
        n_cells_done = tested_df[["lat", "lon"]].drop_duplicates().shape[0]
        confirmed_pct = 100 * (parent_tests["cmi_p"] < 0.05).mean()
        print(f"\n=== Nonlinear (CMIknn) re-test of ParCorr-selected parents | {n_cells_done} cells, "
              f"{len(parent_tests)} parent-level tests ===")
        print(f"Parents that also clear p<0.05 under CMIknn: {confirmed_pct:.1f}%")
        print(f"Parents that also clear p<0.01 under CMIknn: {100 * (parent_tests['cmi_p'] < 0.01).mean():.1f}%")
        if a.check_runnerups:
            ru = tested_df[tested_df["role"] == "runner_up"]
            print(f"ParCorr-rejected runner-up candidates newly significant under CMIknn "
                  f"(p<0.05): {100 * (ru['cmi_p'] < 0.05).mean():.1f}% of {len(ru)} tested")
        if len(fit_df):
            summ = fit_df.groupby(["feature_set", "model"])[["test_r2", "test_rmse"]].mean()
            clim_rmse = fit_df.loc[fit_df["feature_set"] == "CLIM", "test_rmse"].mean()
            summ["skill_vs_clim_pct"] = 100 * (1 - summ["test_rmse"] / clim_rmse)
            print(f"\n=== Accuracy on this {n_cells_done}-cell subset: full PCMCI parents vs. "
                  f"CMIknn-confirmed-only subset ===")
            print(summ.round(4).to_string())
            print("\nIf accuracy barely changes when unconfirmed parents are dropped, the linearity "
                  "assumption is not doing much work in this dataset. If it changes a lot, the ParCorr "
                  "parent set may include spurious (nonlinearly-unsupported) links, or CMIknn power at "
                  "this n and sig_samples is too low to confirm real linear links - the sig_samples "
                  "resolution floor (~1/(sig_samples+1)) matters here.")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()