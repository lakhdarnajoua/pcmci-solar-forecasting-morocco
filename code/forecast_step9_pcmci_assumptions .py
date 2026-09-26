"""
Forecasting stage - Step 9: two checks on the assumptions behind PCMCI / PCMCI+ (step 2).

PCMCI's partial correlations (ParCorr) assume the series are stationary and that predictors aren't so
collinear that the conditional independence tests become unstable. This script checks both, on the TRAINING
anomaly series only (the same rows PCMCI/PCMCI+ actually ran on).

A) Collinearity: for each cell, the correlation matrix of the 7 predictor anomalies (excluding kc) on the
   training rows; reports the single most-correlated pair per cell and how often each pair is the worst one
   (this is what prep_diagnostics.csv's "max_abs_corr_predictors" column summarised as one number per cell -
   this script identifies WHICH pair).

B) Stationarity, per cell x variable (target kc and all 7 predictor anomalies), on the training rows:
     - Augmented Dickey-Fuller test (statsmodels): low p-value = evidence AGAINST a unit root (good)
     - Linear trend: OLS slope of the anomaly against time, in units per year, with its p-value
       (a trend here means the day-of-year climatology removal did not fully capture a longer-term drift -
       e.g. a warming trend in T2M anomalies, or a sensor/product change)
     - Split-half comparison: first half of training years vs second half - difference in mean, ratio of
       variance, difference in lag-1 autocorrelation (a large shift suggests the series behaves differently
       in different sub-periods, which ADF and a linear trend can both miss)

Usage:
    python forecast_step9_pcmci_assumptions.py --prepared forecast_full\\prepared --train_end 2022-12-31 --out step9_out --n_jobs 4 --limit 3
Requires: pip install statsmodels
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

VARS = ["kc", "T2M", "DTR", "RH2M", "WS10M", "PRECT", "PS", "CLOUD_AMT"]
PRED_VARS = VARS[1:]


def safe_csv(df, path, **kw):
    try:
        df.to_csv(path, index=False, **kw)
    except PermissionError:
        alt = path.replace(".csv", f"_{int(time.time())}.csv")
        df.to_csv(alt, index=False, **kw)
        print(f"{path} is locked - saved as {alt}")


def args_key(path):
    import re
    m = re.search(r"cell_(-?[\d.]+)_(-?[\d.]+)\.csv$", os.path.basename(path))
    return round(float(m.group(1)), 4), round(float(m.group(2)), 4)


def run_cell(args):
    path, train_end = args
    from statsmodels.tsa.stattools import adfuller
    lat, lon = args_key(path)
    P = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    tr = P[P.index <= train_end][VARS].dropna()
    if len(tr) < 500:
        return None, None, (lat, lon, "not enough training rows")

    # A) collinearity among predictors (excluding kc), matches prep_diagnostics.csv's definition
    C = tr[PRED_VARS].corr().to_numpy()
    iu = np.triu_indices(len(PRED_VARS), k=1)
    vals = C[iu]
    j = np.argmax(np.abs(vals))
    i1, i2 = iu[0][j], iu[1][j]
    pair_row = dict(lat=lat, lon=lon, var_a=PRED_VARS[i1], var_b=PRED_VARS[i2], corr=float(vals[j]))

    # B) stationarity per variable
    t = np.arange(len(tr))
    years = len(tr) / 365.25
    n_half = len(tr) // 2
    stat_rows = []
    for v in VARS:
        x = tr[v].to_numpy()
        try:
            adf_stat, adf_p = adfuller(x, maxlag=10, autolag="AIC")[:2]
        except Exception:
            adf_stat, adf_p = np.nan, np.nan
        slope, intercept, r, p_trend, se = stats.linregress(t, x)
        trend_per_year = slope * 365.25
        h1, h2 = x[:n_half], x[len(x) - n_half:]

        def acf1(a):
            a = a - a.mean()
            return float(np.corrcoef(a[:-1], a[1:])[0, 1]) if len(a) > 2 else np.nan
        stat_rows.append(dict(lat=lat, lon=lon, variable=v, n=len(x), years=years,
                              adf_stat=adf_stat, adf_p=adf_p,
                              trend_per_year=trend_per_year, trend_p=p_trend,
                              half1_mean=h1.mean(), half2_mean=h2.mean(), mean_shift=h2.mean() - h1.mean(),
                              half1_std=h1.std(), half2_std=h2.std(),
                              var_ratio=(h2.std() ** 2) / max(h1.std() ** 2, 1e-12),
                              half1_acf1=acf1(h1), half2_acf1=acf1(h2)))
    return pair_row, stat_rows, None


def summarise(pairs, stat, out):
    print(f"\n=== A) Most collinear predictor pair per cell (n={len(pairs)} cells) ===")
    pairs["pair"] = pairs.apply(lambda r: " & ".join(sorted([r.var_a, r.var_b])), axis=1)
    freq = pairs.groupby("pair").agg(n_cells=("pair", "size"), mean_abs_corr=("corr", lambda s: s.abs().mean()),
                                     max_abs_corr=("corr", lambda s: s.abs().max())).sort_values("n_cells", ascending=False)
    freq["pct_cells"] = 100 * freq["n_cells"] / len(pairs)
    print(freq.round(3).to_string())
    safe_csv(freq.reset_index(), os.path.join(out, "collinear_pairs_summary.csv"))
    top = pairs.reindex(pairs["corr"].abs().sort_values(ascending=False).index).head(5)
    print(f"\nTop 5 single most collinear cells overall:")
    print(top[["lat", "lon", "var_a", "var_b", "corr"]].round(3).to_string(index=False))

    ncells = stat[["lat", "lon"]].drop_duplicates().shape[0]
    print(f"\n=== B) Stationarity of training-period anomalies, by variable (n={ncells} cells) ===")
    g = stat.groupby("variable")
    tab = pd.DataFrame({
        "pct_cells_ADF_stationary_p<0.05": g["adf_p"].apply(lambda s: 100 * (s < 0.05).mean()),
        "pct_cells_significant_trend_p<0.05": g["trend_p"].apply(lambda s: 100 * (s < 0.05).mean()),
        "median_trend_per_year": g["trend_per_year"].median(),
        "median_|mean_shift|_half2_minus_half1": g["mean_shift"].apply(lambda s: s.abs().median()),
        "median_var_ratio_half2/half1": g["var_ratio"].median(),
        "pct_cells_var_ratio_outside_0.7_1.4": g["var_ratio"].apply(lambda s: 100 * ((s < 0.7) | (s > 1.4)).mean()),
    }).reindex(VARS)
    print(tab.round(3).to_string())
    safe_csv(tab.reset_index().rename(columns={"index": "variable"}), os.path.join(out, "stationarity_by_variable.csv"))
    safe_csv(stat, os.path.join(out, "stationarity_by_cell.csv"))

    print("\nReading guide:")
    print("- ADF: high pct_cells_ADF_stationary means the null of a unit root is rejected almost everywhere (good).")
    print("- Trend: if pct_cells_significant_trend is high AND median_trend_per_year is not ~0, the day-of-year")
    print("  climatology removal (fit on training years only) left a longer-term drift in the anomalies -")
    print("  PCMCI would then partly be picking up a shared trend rather than a genuine lagged dependency.")
    print("- Split-half: a var_ratio far from 1, or a large mean_shift, flags a variable that behaves differently")
    print("  in the first vs second half of the training period (e.g. a product change) even if ADF/trend look fine.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--train_end", default="2022-12-31", help="must match the step1 run that produced --prepared")
    ap.add_argument("--out", default="step9_out")
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    train_end = pd.Timestamp(a.train_end)

    files = sorted(glob.glob(os.path.join(a.prepared, "cell_*.csv")))
    if a.limit:
        files = files[: a.limit]
    print(f"{len(files)} cells | train <= {train_end.date()}")
    jobs = [(f, train_end) for f in files]

    pair_rows, stat_rows, errs, t0 = [], [], [], time.time()
    it = ProcessPoolExecutor(max_workers=a.n_jobs).map(run_cell, jobs) if a.n_jobs > 1 else map(run_cell, jobs)
    for i, (pr, sr, e) in enumerate(it, 1):
        if e:
            errs.append(e)
        else:
            pair_rows.append(pr)
            stat_rows += sr
        if i % 25 == 0 or i == len(jobs):
            print(f"[{i}/{len(jobs)}] {time.time() - t0:.0f}s")
    if errs:
        safe_csv(pd.DataFrame(errs, columns=["lat", "lon", "error"]), os.path.join(a.out, "errors.csv"))
        print(f"{len(errs)} cells skipped - see errors.csv")
    pairs = pd.DataFrame(pair_rows)
    stat = pd.DataFrame(stat_rows)
    if len(pairs):
        summarise(pairs, stat, a.out)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()