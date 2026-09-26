"""
Forecasting stage - Step 3: fit forecasting models on the parents found by PCMCI / PCMCI+ (step 2)

Forecast definition (direct approach): the value for day t is forecast with information available
up to day t-h. For horizon h, only parents with lag >= h are usable, so PCMCI+ lag-0 links are never used.

Feature sets compared (all get the same day-of-year sin/cos as known covariates):
    PCMCI      : (variable, lag) parents from PCMCI, lag >= h
    PCMCIplus  : (variable, lag) parents from PCMCI+, lag >= h   (lagged links only)
    ALL        : every variable at every lag h..tau_max (no causal selection)

Model families F:
    LIN     linear regression
    POLY2   degree-2 polynomial, ridge-regularised (RidgeCV)
    LOGLIN  linear model on log(kc)  (log link; prediction = exp(.))
    KRR     kernel ridge, Gaussian (RBF) kernel, hyper-parameters by time-ordered CV inside the training years
    RF      random forest

Baselines: PERSIST_KC (kc of day t-h x clear-sky reference), PERSIST_GHI (GHI of day t-h), CLIMATOLOGY.

Protocol: train <= 2022-12-31, test = 2023-2025 (never used for any choice).
Model choice per cell (SELECTED) uses a validation year (2022) inside the training period:
fit on <= 2021, score on 2022, pick the lowest RMSE, then refit on all training years.
All metrics are on the GHI scale (kWh/m2/day, as in the NASA POWER file). R2 = 1 - SSE/SST (true R2).

Usage:
    python forecast_step3_fit.py --prepared forecast_out2/prepared --links forecast_out2/links_into_kc_all_cells.csv \
        --out step3_out --horizons 1 --n_jobs 4 --limit 3
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

VARS = ["kc", "T2M", "DTR", "RH2M", "WS10M", "PRECT", "PS", "CLOUD_AMT"]
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_START = pd.Timestamp("2022-01-01")
ALL_MODELS = ["LIN", "POLY2", "LOGLIN", "KRR", "RF"]


# ----------------------------------------------------------------------------- helpers
def metrics(obs, pred):
    e = pred - obs
    sse = float(np.sum(e ** 2))
    sst = float(np.sum((obs - obs.mean()) ** 2))
    return dict(r2=1 - sse / sst, rmse=float(np.sqrt(np.mean(e ** 2))),
                mae=float(np.mean(np.abs(e))), mbe=float(np.mean(e)))


def make_model(name):
    if name in ("LIN", "LOGLIN"):
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "POLY2":
        return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(),
                             RidgeCV(alphas=[1, 10, 100, 1000, 10000]))
    if name == "RF":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5,
                                     random_state=0, n_jobs=1)
    raise ValueError(name)


def fit_predict(name, Xtr, ytr, Xte):
    """ytr already on the scale the model works on (anomaly, or log kc for LOGLIN)."""
    if name == "KRR":
        nf = Xtr.shape[1]
        pipe = make_pipeline(StandardScaler(),
                             TransformedTargetRegressor(regressor=KernelRidge(kernel="rbf"),
                                                        transformer=StandardScaler()))
        grid = {"transformedtargetregressor__regressor__alpha": [0.3, 3.0],
                "transformedtargetregressor__regressor__gamma": [0.5 / nf, 2.0 / nf]}
        gs = GridSearchCV(pipe, grid, cv=TimeSeriesSplit(3), scoring="neg_mean_squared_error", n_jobs=1)
        gs.fit(Xtr, ytr)
        return gs.predict(Xte)
    m = make_model(name)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def build_features(P, pairs, h):
    cols = {}
    for var, lag in pairs:
        cols[f"{var}_L{lag}"] = P[var].shift(lag)
    X = pd.DataFrame(cols, index=P.index)
    doy = P.index.dayofyear.values
    X["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    X["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    return X


# ----------------------------------------------------------------------------- per cell
def run_cell(args):
    path, links, horizons, tau_max, models = args
    P = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    lat, lon = args_key(path)
    rows, errors = [], []
    ref, kcc, ghi = P["ref_clr"], P["kc_clim"], P["ghi"]
    kcraw = P["kc_raw"]

    for h in horizons:
        sets = {
            "PCMCI": sorted({(r.driver, int(r.lag)) for r in links.itertuples()
                             if r.method == "PCMCI" and r.lag >= h}),
            "PCMCIplus": sorted({(r.driver, int(r.lag)) for r in links.itertuples()
                                 if r.method == "PCMCI+" and r.lag >= h}),
            "ALL": [(v, l) for v in VARS for l in range(h, tau_max + 1)],
        }
        X_all = build_features(P, sets["ALL"], h)
        ok = X_all.notna().all(axis=1) & P["kc"].notna() & kcraw.notna() & ghi.notna() & ref.notna()
        ok &= kcraw.shift(h).notna() & ghi.shift(h).notna()               # same rows for baselines
        idx = ok[ok].index
        tr_all = idx[idx <= TRAIN_END]
        tr_a = idx[idx < VAL_START]
        val = idx[(idx >= VAL_START) & (idx <= TRAIN_END)]
        te = idx[idx > TRAIN_END]
        if len(te) < 100 or len(tr_a) < 500:
            errors.append((lat, lon, h, "not enough rows"))
            continue

        def ghi_from_kc(kc_hat, index):
            return kc_hat * ref.loc[index].values

        def add_row(fs, model, nfeat, val_pred_ghi, te_pred_ghi):
            vm = metrics(ghi.loc[val].values, val_pred_ghi) if val_pred_ghi is not None else {"rmse": np.nan}
            tm = metrics(ghi.loc[te].values, te_pred_ghi)
            rows.append(dict(lat=lat, lon=lon, horizon=h, featureset=fs, model=model, n_feat=nfeat,
                             val_rmse=vm["rmse"], test_r2=tm["r2"], test_rmse=tm["rmse"],
                             test_mae=tm["mae"], test_mbe=tm["mbe"], n_test=len(te)))

        # baselines
        add_row("-", "PERSIST_KC", 0, ghi_from_kc(kcraw.shift(h).loc[val].values, val),
                ghi_from_kc(kcraw.shift(h).loc[te].values, te))
        add_row("-", "PERSIST_GHI", 0, ghi.shift(h).loc[val].values, ghi.shift(h).loc[te].values)
        add_row("-", "CLIMATOLOGY", 0, ghi_from_kc(kcc.loc[val].values, val), ghi_from_kc(kcc.loc[te].values, te))

        y_anom = P["kc"]
        y_log = np.log(kcraw.clip(lower=1e-3))
        for fs, pairs in sets.items():
            X = build_features(P, pairs, h)
            for name in models:
                try:
                    y = y_log if name == "LOGLIN" else y_anom
                    # validation fit (train <= 2021) -> score on 2022
                    p_val = fit_predict(name, X.loc[tr_a].values, y.loc[tr_a].values, X.loc[val].values)
                    # final fit (train <= 2022) -> test
                    p_te = fit_predict(name, X.loc[tr_all].values, y.loc[tr_all].values, X.loc[te].values)
                    if name == "LOGLIN":
                        kv, kt = np.exp(p_val), np.exp(p_te)
                    else:
                        kv, kt = kcc.loc[val].values + p_val, kcc.loc[te].values + p_te
                    add_row(fs, name, X.shape[1] - 2, ghi_from_kc(kv, val), ghi_from_kc(kt, te))
                except Exception as e:
                    errors.append((lat, lon, h, f"{fs}/{name}: {type(e).__name__}: {e}"))
    return rows, errors


def args_key(path):
    m = re.search(r"cell_(-?[\d.]+)_(-?[\d.]+)\.csv$", os.path.basename(path))
    return round(float(m.group(1)), 4), round(float(m.group(2)), 4)


# ----------------------------------------------------------------------------- summary
def summarise(res, out):
    lines = []
    for h, d in res.groupby("horizon"):
        # SELECTED: per cell and feature set, model with lowest validation RMSE (no test-set choice)
        sel = (d[d["featureset"] != "-"].sort_values("val_rmse")
               .groupby(["lat", "lon", "featureset"], as_index=False).first())
        sel["model"] = "SELECTED(" + sel["model"] + ")"
        base = d[d["featureset"] == "-"]
        per_cell = {}
        for fs in ["PCMCI", "PCMCIplus", "ALL"]:
            per_cell[fs] = sel[sel["featureset"] == fs].set_index(["lat", "lon"])["test_rmse"]
        for b in ["PERSIST_KC", "PERSIST_GHI", "CLIMATOLOGY"]:
            per_cell[b] = base[base["model"] == b].set_index(["lat", "lon"])["test_rmse"]
        tab = (pd.concat([d, sel.assign(model="SELECTED")]).groupby(["featureset", "model"])
               [["test_r2", "test_rmse", "test_mae", "test_mbe"]].mean().reset_index())
        tab.insert(0, "horizon", h)
        lines.append(tab)
        print(f"\n=== horizon {h} d | mean over {d[['lat', 'lon']].drop_duplicates().shape[0]} cells | test 2023-2025 ===")
        print(tab.drop(columns="horizon").round(4).to_string(index=False))

        print("\nPaired comparison of per-cell test RMSE (selected models vs baselines):")
        pairs = [("PCMCI", "PERSIST_KC"), ("PCMCI", "CLIMATOLOGY"), ("PCMCIplus", "PERSIST_KC"),
                 ("ALL", "PERSIST_KC"), ("PCMCI", "ALL"), ("PCMCI", "PCMCIplus")]
        for a, b in pairs:
            x = pd.concat([per_cell[a], per_cell[b]], axis=1, keys=["a", "b"]).dropna()
            if len(x) < 5:
                continue
            gain = 100 * (1 - x["a"].mean() / x["b"].mean())
            wins = 100 * (x["a"] < x["b"]).mean()
            try:
                p = stats.wilcoxon(x["a"], x["b"]).pvalue
            except Exception:
                p = np.nan
            print(f"  {a:9s} vs {b:12s}: RMSE reduction {gain:+6.2f}% | {a} better in {wins:5.1f}% of cells | Wilcoxon p={p:.2e}")
    pd.concat(lines).to_csv(os.path.join(out, "summary_by_horizon.csv"), index=False)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True, help="folder forecast_out/prepared")
    ap.add_argument("--links", required=True, help="links_into_kc_all_cells.csv from step 2")
    ap.add_argument("--out", default="step3_out")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1])
    ap.add_argument("--tau_max", type=int, default=7, help="must equal the tau_max used in step 2")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
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
        raise SystemExit("No prepared cell matches the links file - wrong --prepared / --links pair "
                         "(both must come from the same step 2 run).")
    if len(files) < len(files_all):
        print(f"WARNING: {len(files_all) - len(files)} prepared cells have no links and are skipped "
              f"(links file probably comes from a different / partial step 2 run).")
    if a.limit:
        files = files[: a.limit]
    print(f"running {len(files)} cells | horizons {a.horizons} | models {a.models}")

    jobs = [(f, by_cell[args_key(f)], a.horizons, a.tau_max, a.models) for f in files]

    all_rows, all_err, t0 = [], [], time.time()
    if a.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=a.n_jobs) as ex:
            for k, (rows, errs) in enumerate(ex.map(run_cell, jobs), 1):
                all_rows += rows
                all_err += errs
                print(f"[{k}/{len(jobs)}] {time.time() - t0:.0f}s")
    else:
        for k, j in enumerate(jobs, 1):
            rows, errs = run_cell(j)
            all_rows += rows
            all_err += errs
            print(f"[{k}/{len(jobs)}] {time.time() - t0:.0f}s")

    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(a.out, "results_long.csv"), index=False)
    if all_err:
        pd.DataFrame(all_err, columns=["lat", "lon", "horizon", "error"]).to_csv(
            os.path.join(a.out, "errors.csv"), index=False)
        print(f"{len(all_err)} model fits failed - see errors.csv")
    if len(res):
        summarise(res, a.out)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()