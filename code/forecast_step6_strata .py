"""
Forecasting stage - Step 6: where does the skill come from? Breakdown by season, cloud regime and test year.

Per cell, fits fixed models on the training years (2015-2022) and predicts every test day (2023-2025):
    PCMCI   lagged parents kept by BH-FDR 0.01 (the sets of steps 3-5)
    LASSO   the same number of inputs, chosen by the LASSO path (training data only)
    ALL     all 56 candidate inputs
    models  LIN and KRR (fixed in advance, no choice made on the test period)
plus two baselines: CLIM (kc climatology x clear-sky reference) and PERSIST_KC (kc of the previous day).
All errors are on the GHI scale (same units as the NASA POWER file).

Strata (each day of the test period falls in exactly one class of every family):
    season       DJF, MAM, JJA, SON
    year         2023, 2024, 2025            <- shows how much the result moves between test years
    kc_obs       low / mid / high clearness of the OBSERVED day (cell-specific training terciles of kc)
                 NB: conditions on the outcome (common in the solar literature, but it mechanically favours
                 forecasts close to the mean on mid days and penalises them on the extremes)
    kc_prev      low / mid / high clearness of the PREVIOUS day (known at forecast time -> no outcome conditioning)
    kc_obs_p10   only the 10% cloudiest days of each cell (training 10th percentile of kc): extreme-cloud days

Outputs (in --out): daily_predictions.csv.gz (reusable), strata_skill.csv, strata_mbe.csv, strata_paired.csv.
Confidence intervals: spatial block bootstrap over BLOCK x BLOCK degree tiles, as in steps 4-5.

Usage:
    python forecast_step6_strata.py --prepared forecast_full\\prepared --links forecast_full\\links_into_kc_all_cells.csv ^
        --out step6_out --n_jobs 4 --limit 3
    python forecast_step6_strata.py --from_daily step6_out\\daily_predictions.csv.gz --out step6_out   (re-summarise only)
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
MIN_DAYS = 20          # minimum days of a cell in a stratum to keep that cell-stratum
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

    pool = [(v, l) for v in VARS for l in range(H, tau_max + 1)]
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
    out["ghi"] = ghi.loc[te].values
    out["kc_obs"] = kcraw.loc[te].values
    out["kc_prev"] = kcraw.shift(H).loc[te].values
    out["CLIM"] = kcc.loc[te].values * ref.loc[te].values
    out["PERSIST_KC"] = kcraw.shift(H).loc[te].values * ref.loc[te].values
    for m, names in sets.items():
        X = X_pool[list(names) + cols_sc]
        for model in models:
            p = fit_predict(model, X.loc[tr].values, y, X.loc[te].values)
            out[f"{m}_{model}"] = (kcc.loc[te].values + p) * ref.loc[te].values
    # cell-specific regime thresholds from TRAINING days only
    kt = kcraw.loc[tr]
    out["q33"], out["q67"], out["q10"] = kt.quantile(1 / 3), kt.quantile(2 / 3), kt.quantile(0.10)
    out["k"] = k
    out.index.name = "date"
    return out.reset_index(), None


# ----------------------------------------------------------------------------- strata
def add_strata(d):
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"])
    mon = d["date"].dt.month
    d["season"] = np.select([mon.isin([12, 1, 2]), mon.isin([3, 4, 5]), mon.isin([6, 7, 8])], ["DJF", "MAM", "JJA"], "SON")
    d["year"] = d["date"].dt.year.astype(str)

    def cls(v):
        return np.select([v <= d["q33"], v <= d["q67"]], ["low", "mid"], "high")
    d["kc_obs_cls"] = cls(d["kc_obs"])
    d["kc_prev_cls"] = cls(d["kc_prev"])
    d["kc_obs_p10"] = np.where(d["kc_obs"] <= d["q10"], "cloudiest10%", None)
    return d


FAMILIES = [("season", "season", ["DJF", "MAM", "JJA", "SON"]),
            ("year", "year", ["2023", "2024", "2025"]),
            ("kc_obs", "kc_obs_cls", ["low", "mid", "high"]),
            ("kc_prev", "kc_prev_cls", ["low", "mid", "high"]),
            ("kc_obs_p10", "kc_obs_p10", ["cloudiest10%"])]


def summarise(daily, out, block, models):
    d = add_strata(daily)
    preds = ["CLIM", "PERSIST_KC"] + [f"{m}_{mo}" for mo in models for m in METHODS]
    for c in preds:
        e = d[c] - d["ghi"]
        d[f"se_{c}"], d[f"e_{c}"] = e ** 2, e
    ncells = d[["lat", "lon"]].drop_duplicates().shape[0]
    print(f"\n{ncells} cells | {d['date'].nunique()} test days from {d['date'].min().date()} to {d['date'].max().date()} "
          f"| strata skill = mean over cells of 100*(1 - RMSE/RMSE_climatology) within the stratum")
    per_year = d.groupby(d["date"].dt.year)["date"].nunique().to_dict()
    print(f"distinct test days per calendar year: {per_year}  (a partial year means its seasons are under-represented)")

    cell_rows = []
    for fam, col, levels in FAMILIES:
        g = d.dropna(subset=[col]).groupby(["lat", "lon", col])
        agg = g[[f"se_{c}" for c in preds] + [f"e_{c}" for c in preds]].sum()
        agg["n"] = g.size()
        agg = agg[agg["n"] >= MIN_DAYS].reset_index().rename(columns={col: "level"})
        for c in preds:
            agg[f"rmse_{c}"] = np.sqrt(agg[f"se_{c}"] / agg["n"])
            agg[f"mbe_{c}"] = agg[f"e_{c}"] / agg["n"]
        agg["family"] = fam
        cell_rows.append(agg)
    cells = pd.concat(cell_rows)

    skill_rows, mbe_rows, paired_rows = [], [], []
    for fam, col, levels in FAMILIES:
        for lv in levels:
            a = cells[(cells["family"] == fam) & (cells["level"] == lv)]
            if a.empty:
                continue
            row = dict(family=fam, level=lv, n_cells=len(a), mean_days=a["n"].mean(),
                       rmse_clim=a["rmse_CLIM"].mean())
            mrow = dict(family=fam, level=lv)
            for c in preds:
                if c != "CLIM":
                    row[c] = (100 * (1 - a[f"rmse_{c}"] / a["rmse_CLIM"])).mean()
                mrow[c] = a[f"mbe_{c}"].mean()
            skill_rows.append(row)
            mbe_rows.append(mrow)
            blocks = np.array([f"{np.floor(x / block)}_{np.floor(y / block)}" for x, y in zip(a["lat"], a["lon"])])
            for mo in models:
                for alt in ["LASSO", "ALL"]:
                    diff = 100 * (1 - a[f"rmse_PCMCI_{mo}"].values / a[f"rmse_{alt}_{mo}"].values)
                    lo, hi = block_bootstrap_ci(diff, blocks, n_boot=1000)
                    paired_rows.append(dict(family=fam, level=lv, model=mo, alternative=alt, n_cells=len(a),
                                            mean_pct=diff.mean(), ci_low=lo, ci_high=hi))
    skill, mbe, paired = pd.DataFrame(skill_rows), pd.DataFrame(mbe_rows), pd.DataFrame(paired_rows)
    safe_csv(skill, os.path.join(out, "strata_skill.csv"))
    safe_csv(mbe, os.path.join(out, "strata_mbe.csv"))
    safe_csv(paired, os.path.join(out, "strata_paired.csv"))

    pd.set_option("display.width", 250)
    print("\n--- Skill vs climatology (% RMSE reduction; >0 = better than climatology) ---")
    show = ["family", "level", "n_cells", "mean_days", "rmse_clim", "PERSIST_KC"] + [f"{m}_{mo}" for mo in models for m in METHODS]
    print(skill[show].round(2).to_string(index=False))
    print("\n--- Mean bias MBE = forecast - observed (GHI units), per stratum ---")
    print(mbe[["family", "level", "CLIM", "PERSIST_KC"] + [f"PCMCI_{mo}" for mo in models]].round(3).to_string(index=False))
    print(f"\n--- PCMCI vs alternative: per-cell % RMSE reduction of PCMCI (>0 = PCMCI better), 95% CI spatial block bootstrap ---")
    paired["txt"] = [f"{m:+.2f} [{l:+.2f},{h:+.2f}]" for m, l, h in zip(paired["mean_pct"], paired["ci_low"], paired["ci_high"])]
    wide = paired.pivot_table(index=["family", "level"], columns=["model", "alternative"], values="txt", aggfunc="first")
    order = [(f, l) for f, _, ls in FAMILIES for l in ls]
    print(wide.reindex(order).dropna(how="all").to_string())
    print("\nkc_obs strata condition on the outcome; kc_prev does not. Compare the two before drawing conclusions.")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared")
    ap.add_argument("--links")
    ap.add_argument("--from_daily", help="re-summarise a saved daily_predictions.csv.gz without refitting")
    ap.add_argument("--out", default="step6_out")
    ap.add_argument("--models", nargs="+", default=["LIN", "KRR"], choices=["LIN", "POLY2", "KRR", "RF"])
    ap.add_argument("--block", type=float, default=2.0)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--years", type=int, nargs="+", help="only summarise these test years (e.g. 2023 2024 = complete years)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.from_daily:
        daily = pd.read_csv(a.from_daily)
        if a.years:
            daily = daily[pd.to_datetime(daily["date"]).dt.year.isin(a.years)]
        models = sorted({c.split("_")[1] for c in daily.columns if c.startswith("PCMCI_")}, key=["LIN", "POLY2", "KRR", "RF"].index)
        summarise(daily, a.out, a.block, models)
        return
    if not a.prepared or not a.links:
        raise SystemExit("Give --prepared and --links (or --from_daily).")

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
    print(f"running {len(files)} cells | models {a.models}")

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
    daily = pd.concat(frames, ignore_index=True)
    num = daily.select_dtypes("float").columns
    daily[num] = daily[num].round(5)
    safe_csv(daily, os.path.join(a.out, "daily_predictions.csv.gz"), compression="gzip")
    summarise(daily, a.out, a.block, a.models)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()