"""
Forecasting stage - Step 8b: combine the per-fold results from forecast_step8a_origin_fit.py into one
rolling-origin summary, so the forecast comparison no longer rests on a single fixed 2023-2025 split.

Usage:
    python forecast_step8b_rolling_summary.py --folds step8_folds\\fold_2021_results.csv step8_folds\\fold_2022_results.csv step8_folds\\fold_2023_results.csv ^
        --out step8_summary --block 2.0
or, simpler, point it at the folder and it picks up every fold_*_results.csv:
    python forecast_step8b_rolling_summary.py --folds_dir step8_folds --out step8_summary
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

from forecast_step4_selection_comparison import block_bootstrap_ci

METHODS = ["PCMCI", "LASSO", "ALL"]


def safe_csv(df, path, **kw):
    try:
        df.to_csv(path, index=False, **kw)
    except PermissionError:
        alt = path.replace(".csv", f"_{int(time.time())}.csv")
        df.to_csv(alt, index=False, **kw)
        print(f"{path} is locked - saved as {alt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", default=[])
    ap.add_argument("--folds_dir", help="folder containing fold_*_results.csv - used if --folds is empty")
    ap.add_argument("--out", default="step8_summary")
    ap.add_argument("--models", nargs="+", default=["LIN", "KRR"])
    ap.add_argument("--block", type=float, default=2.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    paths = a.folds or sorted(glob.glob(os.path.join(a.folds_dir or ".", "fold_*_results.csv")))
    if not paths:
        raise SystemExit("No fold result files given (use --folds or --folds_dir).")
    print(f"combining {len(paths)} folds: {[os.path.basename(p) for p in paths]}")
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    d["lat"], d["lon"] = d["lat"].round(4), d["lon"].round(4)
    folds = list(d["fold"].unique())
    have_pcmci = [f for f in folds if d.loc[d["fold"] == f, "k"].fillna(0).gt(0).any()]
    if len(have_pcmci) < len(folds):
        print(f"note: fold(s) {[f for f in folds if f not in have_pcmci]} have no PCMCI parents in some/all cells "
              f"and are excluded from PCMCI comparisons where empty.")

    # --- per-fold skill vs climatology, per model/method ---
    rows = []
    for f in folds:
        g = d[d["fold"] == f]
        row = dict(fold=f, n_cells=len(g), n_test_days=g["n_test"].mean(), mean_k=g["k"].mean())
        for mo in a.models:
            for m in ["PERSIST_KC"] + METHODS:
                col = f"{m}_rmse" if m == "PERSIST_KC" else f"{m}_{mo}_rmse"
                if col in g:
                    row[f"{m if m == 'PERSIST_KC' else m + '_' + mo}_skill"] = (100 * (1 - g[col] / g["CLIM_rmse"])).mean()
        rows.append(row)
    per_fold = pd.DataFrame(rows)
    safe_csv(per_fold, os.path.join(a.out, "rolling_skill_by_fold.csv"))
    print(f"\n=== Skill vs climatology by fold (mean over cells, % RMSE reduction) ===")
    print(per_fold.round(2).to_string(index=False))
    print("\nIf skill varies a lot across folds, the single 2023-2025 split understates the real uncertainty.")

    # --- pooled skill (mean and range across folds, not a single bootstrap since folds are few and non-iid) ---
    skill_cols = [c for c in per_fold.columns if c.endswith("_skill")]
    pooled = per_fold[skill_cols].agg(["mean", "std", "min", "max"]).T
    pooled.index.name = "method"
    safe_csv(pooled.reset_index(), os.path.join(a.out, "rolling_skill_pooled.csv"))
    print(f"\n=== Pooled across {len(folds)} folds: mean / std / min / max skill vs climatology ===")
    print(pooled.round(2).to_string())

    # --- PCMCI vs LASSO / ALL, per fold (spatial block bootstrap within fold) and pooled cells across folds ---
    comp_rows = []
    for f in folds:
        g = d[d["fold"] == f]
        if not (g["k"] > 0).all():
            g = g[g["k"] > 0]
        if len(g) < 10:
            continue
        blocks = np.array([f"{np.floor(a_ / a.block)}_{np.floor(b_ / a.block)}" for a_, b_ in zip(g["lat"], g["lon"])])
        for mo in a.models:
            for alt in ["LASSO", "ALL"]:
                ca, cb = f"PCMCI_{mo}_rmse", f"{alt}_{mo}_rmse"
                if ca not in g or cb not in g:
                    continue
                diff = 100 * (1 - g[ca].values / g[cb].values)
                lo, hi = block_bootstrap_ci(diff, blocks, n_boot=1000)
                comp_rows.append(dict(fold=f, model=mo, alternative=alt, n_cells=len(g), mean_pct=diff.mean(),
                                      ci_low=lo, ci_high=hi, pct_cells_pcmci_better=100 * (g[ca].values < g[cb].values).mean()))
    comp = pd.DataFrame(comp_rows)
    safe_csv(comp, os.path.join(a.out, "rolling_pcmci_vs_alt.csv"))
    print(f"\n=== PCMCI vs alternative, per fold (per-cell %% RMSE reduction, >0 = PCMCI better; 95%% CI spatial block bootstrap) ===")
    for r in comp.itertuples():
        print(f"  fold {str(r.fold):6s} {r.model:4s} PCMCI vs {r.alternative:6s}: {r.mean_pct:+6.2f}% [{r.ci_low:+6.2f}, {r.ci_high:+6.2f}] | PCMCI better in {r.pct_cells_pcmci_better:5.1f}% of {r.n_cells} cells")

    # --- pooled PCMCI vs alt: pool ALL (cell, fold) pairs, block by (spatial tile, fold) ---
    print(f"\n=== PCMCI vs alternative, pooled across folds (each cell-fold pair as one unit) ===")
    dd = d[d["k"] > 0] if (d["k"] <= 0).any() else d
    blocks_all = np.array([f"{np.floor(a_ / a.block)}_{np.floor(b_ / a.block)}_{ff}"
                           for a_, b_, ff in zip(dd["lat"], dd["lon"], dd["fold"])])
    for mo in a.models:
        for alt in ["LASSO", "ALL"]:
            ca, cb = f"PCMCI_{mo}_rmse", f"{alt}_{mo}_rmse"
            if ca not in dd or cb not in dd:
                continue
            diff = 100 * (1 - dd[ca].values / dd[cb].values)
            lo, hi = block_bootstrap_ci(diff, blocks_all, n_boot=2000)
            print(f"  {mo:4s} PCMCI vs {alt:6s}: {diff.mean():+6.2f}% [{lo:+6.2f}, {hi:+6.2f}] | n = {len(dd)} cell-fold pairs across {len(folds)} folds")
    print("\nThis pooled bootstrap treats each fold's tiles as separate blocks; it does not account for folds sharing overlapping training years.")


if __name__ == "__main__":
    main()
