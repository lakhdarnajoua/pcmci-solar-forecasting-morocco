"""
Forecasting stage - Step 1 (data preparation) + Step 2 (causal discovery: PCMCI and PCMCI+)

What it does, per grid cell:
  1. Cleans the NASA POWER daily series (-999 -> NaN, short gaps interpolated).
  2. Builds the variable set  [kc, T2M, DTR, RH2M, WS10M, PRECT, PS, CLOUD_AMT]
       kc   = ALLSKY_SFC_SW_DWN / clear-sky reference (DOY climatology of CLRSKY over TRAIN years)
       DTR  = T2M_MAX - T2M_MIN
       PRECT= log1p(PRECTOTCORR)
  3. Removes the seasonal cycle (DOY climatology, smoothed, TRAIN years only) -> anomalies.
  4. Runs PCMCI (lagged links only, tau_min=1; BH-FDR at --alpha, column p = BH-adjusted p) and PCMCI+ (lag 0 + lagged, oriented; column p = raw p)
     with ParCorr, on the TRAIN period only (default 2015-2022), so that the choice of
     predictors never sees the test years.
  5. Saves: prepared series per cell (for the fitting step), all detected links into kc,
     a frequency table of (variable, lag) across cells, and simple diagnostics.

Requires: numpy, pandas, tigramite (tested with 5.2.10.1)
Usage:
    python forecast_step1_prep_pcmci.py --input path/to/data.csv_or_folder --out forecast_out --tau_max 7 --n_jobs 4
    python forecast_step1_prep_pcmci.py --input ... --limit 3     # quick test on 3 cells first
"""
import argparse
import glob
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- settings
TRAIN_END = "2022-12-31"          # same split as the prediction stage (train 2015-2022 / test 2023-2025)
GHI = "ALLSKY_SFC_SW_DWN"
CLR = "CLRSKY_SFC_SW_DWN"
VARS = ["kc", "T2M", "DTR", "RH2M", "WS10M", "PRECT", "PS", "CLOUD_AMT"]   # kc = index 0 = target
MISSING = 9999.0                  # flag passed to tigramite for remaining NaNs
SMOOTH_WIN = 15                   # days, circular moving average for the DOY climatology
RAW_NEEDED = [GHI, CLR, "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS10M", "PRECTOTCORR", "PS", "CLOUD_AMT"]


# ----------------------------------------------------------------------------- data loading
def load_all(path, dayfirst=False):
    files = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
    frames = []
    for f in files:
        d = pd.read_csv(f)
        date_col = d.columns[0]                      # first column = date (header may be blank)
        d = d.rename(columns={date_col: "date"})
        d["date"] = pd.to_datetime(d["date"], dayfirst=dayfirst)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    missing_cols = [c for c in RAW_NEEDED + ["lat", "lon"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input: {missing_cols}. Found: {list(df.columns)}")
    return df


# ----------------------------------------------------------------------------- preparation
def doy_climatology(s, train_mask, win=SMOOTH_WIN):
    """365-value seasonal climatology from TRAIN years only, circularly smoothed."""
    doy = np.minimum(s.index.dayofyear.values, 365)     # 29 Feb / 31 Dec of leap years fold onto neighbours
    tmp = pd.Series(s.values[train_mask.values], index=doy[train_mask.values])
    clim = tmp.groupby(level=0).mean().reindex(range(1, 366)).interpolate(limit_direction="both").values
    k = win // 2
    padded = np.r_[clim[-k:], clim, clim[:k]]
    sm = pd.Series(padded).rolling(win, center=True).mean().values[k:-k]
    return sm                                           # index 0 <-> DOY 1


def prepare_cell(d):
    d = d.sort_values("date").drop_duplicates("date").set_index("date")
    if (d.index.to_series().diff().dropna() != pd.Timedelta(days=1)).any():
        # reindex to a complete daily calendar; gaps become NaN
        d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
    raw = d[RAW_NEEDED].replace(-999, np.nan).replace(-999.0, np.nan)
    raw = raw.interpolate(limit=3, limit_area="inside")

    train = pd.Series(raw.index <= pd.Timestamp(TRAIN_END), index=raw.index)
    doy = np.minimum(raw.index.dayofyear.values, 365) - 1

    ref_clim = doy_climatology(raw[CLR], train)         # clear-sky reference known in advance (train climatology)
    ref = ref_clim[doy]
    if (ref <= 0).any():
        raise ValueError("Non-positive clear-sky reference")

    X = pd.DataFrame(index=raw.index)
    X["kc"] = raw[GHI] / ref
    X["T2M"] = raw["T2M"]
    X["DTR"] = raw["T2M_MAX"] - raw["T2M_MIN"]
    X["RH2M"] = raw["RH2M"]
    X["WS10M"] = raw["WS10M"]
    X["PRECT"] = np.log1p(raw["PRECTOTCORR"].clip(lower=0))
    X["PS"] = raw["PS"]
    X["CLOUD_AMT"] = raw["CLOUD_AMT"]

    clim = {v: doy_climatology(X[v], train) for v in VARS}
    A = pd.DataFrame({v: X[v].values - clim[v][doy] for v in VARS}, index=X.index)

    out = A.copy()
    out["kc_raw"] = X["kc"]
    out["kc_clim"] = clim["kc"][doy]
    out["ref_clr"] = ref
    out["ghi"] = raw[GHI]
    out["is_train"] = train.values
    return out


# ----------------------------------------------------------------------------- causal discovery
def _links_into_target(graph, val, pv, qv, tau_max, method, lag0_allowed, var_list):
    rows = []
    N = graph.shape[0]
    for i in range(N):
        for tau in range(0 if lag0_allowed else 1, tau_max + 1):
            g = graph[i, 0, tau]
            if g in ("", None):
                continue
            if tau == 0 and g not in ("-->", "o-o", "x-x", "o->", "<--", "<-o", "<->"):
                continue
            if tau > 0 and g != "-->":
                continue
            if tau == 0 and i == 0:
                continue
            rows.append(dict(method=method, driver=var_list[i], lag=tau, edge=g,
                             val=float(val[i, 0, tau]), p=float(pv[i, 0, tau])))
    return rows


def run_cell(args):
    key, A, tau_max, alpha, pc_alpha_pcmci, pc_alpha_plus, var_list = args
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    data = A[var_list].to_numpy(dtype=float)
    data = np.where(np.isnan(data), MISSING, data)
    rows = []
    t0 = time.time()
    try:
        df = pp.DataFrame(data, var_names=var_list, missing_flag=MISSING)

        pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(significance="analytic", mask_type=None), verbosity=0)
        r1 = pcmci.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha_pcmci,
                             alpha_level=alpha, fdr_method="fdr_bh")
        # With fdr_method="fdr_bh", tigramite returns BH-adjusted p-values directly in p_matrix and builds
        # the graph from them; these values must not be corrected a second time downstream.
        rows += _links_into_target(r1["graph"], r1["val_matrix"], r1["p_matrix"], None, tau_max, "PCMCI", False, var_list)

        pcmci2 = PCMCI(dataframe=df, cond_ind_test=ParCorr(significance="analytic", mask_type=None), verbosity=0)
        r2 = pcmci2.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha_plus)
        rows += _links_into_target(r2["graph"], r2["val_matrix"], r2["p_matrix"], None, tau_max, "PCMCI+", True, var_list)
    except Exception as e:                                  # keep going, report the failure
        return key, [], f"{type(e).__name__}: {e}", time.time() - t0
    return key, rows, "", time.time() - t0


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV file (all cells) or folder of CSVs")
    ap.add_argument("--out", default="forecast_out")
    ap.add_argument("--tau_max", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.01, help="FDR level for PCMCI links")
    ap.add_argument("--pc_alpha_pcmci", type=float, default=0.2)
    ap.add_argument("--pc_alpha_plus", type=float, default=0.01)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only run first N cells (test)")
    ap.add_argument("--dayfirst", action="store_true", help="dates stored as DD/MM/YYYY")
    ap.add_argument("--exclude", nargs="*", default=[], help="variables left out of the causal discovery (e.g. CLOUD_AMT)")
    ap.add_argument("--train_end", default=None, help="override the training-period end date (YYYY-MM-DD), default 2022-12-31 - use this for rolling-origin runs")
    a = ap.parse_args()

    global TRAIN_END
    if a.train_end:
        TRAIN_END = a.train_end
    var_list = [v for v in VARS if v not in a.exclude]
    if "kc" not in var_list or var_list[0] != "kc":
        raise SystemExit("kc (the target) cannot be excluded")
    bad = [v for v in a.exclude if v not in VARS]
    if bad:
        raise SystemExit(f"Unknown variable(s) in --exclude: {bad}. Choices: {VARS[1:]}")
    print(f"variables in causal discovery: {var_list}")
    os.makedirs(os.path.join(a.out, "prepared"), exist_ok=True)
    df = load_all(a.input, a.dayfirst)
    cells = sorted(df.groupby(["lat", "lon"]).groups.keys())
    if a.limit:
        cells = cells[: a.limit]
    print(f"{len(cells)} cells | train <= {TRAIN_END} | tau_max={a.tau_max}")

    jobs, diag = [], []
    for (lat, lon) in cells:
        d = df[(df["lat"] == lat) & (df["lon"] == lon)]
        P = prepare_cell(d)
        tag = f"{lat:.4f}_{lon:.4f}"
        P.to_csv(os.path.join(a.out, "prepared", f"cell_{tag}.csv"), index_label="date")
        tr = P[P["is_train"]][VARS]
        c = tr.corr().abs().to_numpy().copy()
        np.fill_diagonal(c, 0)
        diag.append(dict(lat=lat, lon=lon, n_days=len(P), n_train=int(P["is_train"].sum()),
                         n_nan_train=int(tr.isna().any(axis=1).sum()),
                         max_abs_corr_predictors=float(np.nanmax(c[1:, 1:])),
                         acf1_kc_anom=float(tr["kc"].autocorr(1)),
                         acf2_kc_anom=float(tr["kc"].autocorr(2)),
                         acf7_kc_anom=float(tr["kc"].autocorr(7)),
                         share_kc_gt_1p2=float((P["kc_raw"] > 1.2).mean())))
        jobs.append(((lat, lon), P[P["is_train"]], a.tau_max, a.alpha, a.pc_alpha_pcmci, a.pc_alpha_plus, var_list))
    pd.DataFrame(diag).to_csv(os.path.join(a.out, "prep_diagnostics.csv"), index=False)

    all_rows, fails = [], []
    t0 = time.time()
    if a.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=a.n_jobs) as ex:
            results = ex.map(run_cell, jobs)
            for k, (key, rows, err, dt) in enumerate(results, 1):
                _collect(key, rows, err, all_rows, fails)
                print(f"[{k}/{len(jobs)}] {key} {dt:.1f}s {err}")
    else:
        for k, j in enumerate(jobs, 1):
            key, rows, err, dt = run_cell(j)
            _collect(key, rows, err, all_rows, fails)
            print(f"[{k}/{len(jobs)}] {key} {dt:.1f}s {err}")

    links = pd.DataFrame(all_rows)
    links.to_csv(os.path.join(a.out, "links_into_kc_all_cells.csv"), index=False)
    if len(links):
        n_cells = len(cells) - len(fails)
        freq = (links.groupby(["method", "driver", "lag"])
                .agg(n_cells=("val", "size"), mean_val=("val", "mean"), mean_abs_val=("val", lambda s: s.abs().mean()))
                .reset_index())
        freq["pct_cells"] = 100 * freq["n_cells"] / max(n_cells, 1)
        freq.sort_values(["method", "pct_cells"], ascending=[True, False]).to_csv(
            os.path.join(a.out, "link_frequency.csv"), index=False)
        print(freq.sort_values(["method", "pct_cells"], ascending=[True, False]).groupby("method").head(12).to_string(index=False))
    if fails:
        pd.DataFrame(fails, columns=["cell", "error"]).to_csv(os.path.join(a.out, "failed_cells.csv"), index=False)
        print(f"{len(fails)} cells failed - see failed_cells.csv")
    print(f"done in {time.time() - t0:.0f}s")


def _collect(key, rows, err, all_rows, fails):
    if err:
        fails.append((key, err))
    for r in rows:
        r["lat"], r["lon"] = key
        all_rows.append(r)


if __name__ == "__main__":
    main()