# src/evaluation/statistical_tests.py
# Statistical significance tests for thesis model comparison
#
# Tests run for each (ticker, horizon, model_a vs model_b) triple:
#   1. Paired t-test on Q50 predictions
#      H0: mean(Q50_a) == mean(Q50_b)
#      Interpretation: does sentiment shift the predicted price level?
#
#   2. Paired t-test on absolute forecast errors |actual - Q50|
#      H0: mean(|error_a|) == mean(|error_b|)
#      Interpretation: does sentiment improve forecast accuracy?
#
#   3. Diebold-Mariano test on squared errors
#      H0: equal forecast accuracy (Newey-West variance, 0 lags)
#      Interpretation: is the accuracy difference statistically significant?
#
# Input:  results/revised_experiments/predictions/*.csv
# Output: results/revised_experiments/statistical_tests.csv
#
# Usage:
#   python -m src.evaluation.statistical_tests

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR   = "results/revised_experiments"
PRED_DIR      = os.path.join(RESULTS_DIR, "predictions")
OUTPUT_PATH   = os.path.join(RESULTS_DIR, "statistical_tests.csv")

TICKERS   = ["AAPL", "MSFT", "NVDA"]
HORIZONS  = [1, 5, 10]

COMPARISONS = [
    ("price_only_lstm", "sentiment_lstm"),
    ("price_only_lstm", "sentiment_shift_lstm"),
    ("sentiment_lstm",  "sentiment_shift_lstm"),
]


# ── Statistical test helpers ──────────────────────────────────────

def _paired_ttest(a: np.ndarray, b: np.ndarray) -> tuple:
    """Paired t-test. Returns (t_stat, p_value)."""
    if len(a) < 3 or len(b) < 3 or len(a) != len(b):
        return float("nan"), float("nan")
    t_stat, p_value = stats.ttest_rel(a, b)
    return float(t_stat), float(p_value)


def _diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray) -> tuple:
    """
    Diebold-Mariano test (Harvey, Leybourne & Newbold, 1997 small-sample correction).
    Uses squared error loss: L(e) = e^2.
    H0: equal predictive accuracy.
    Returns (dm_stat, p_value).
    """
    d = errors_a ** 2 - errors_b ** 2
    n = len(d)
    if n < 4:
        return float("nan"), float("nan")
    d_bar = np.mean(d)
    sigma2 = np.var(d, ddof=1) / n
    if sigma2 < 1e-14:
        return 0.0, 1.0
    dm_stat = d_bar / np.sqrt(sigma2)
    # Harvey et al. small-sample correction: DM ~ t(n-1)
    dm_corr = dm_stat * np.sqrt((n + 1 - 2 + 1 / n) / n)
    p_value  = 2.0 * (1.0 - stats.t.cdf(abs(dm_corr), df=n - 1))
    return float(dm_stat), float(p_value)


def _interpret(p_q50: float, p_err: float, dm_p: float) -> str:
    parts = []
    if not np.isnan(p_q50):
        if p_q50 < 0.01:
            parts.append("Q50 highly significant (p<0.01)")
        elif p_q50 < 0.05:
            parts.append("Q50 significant (p<0.05)")
        elif p_q50 < 0.10:
            parts.append("Q50 marginal (p<0.10)")
        else:
            parts.append("Q50 not significant")
    if not np.isnan(p_err):
        if p_err < 0.05:
            parts.append("error difference significant")
        else:
            parts.append("error difference not significant")
    if not np.isnan(dm_p):
        if dm_p < 0.05:
            parts.append("DM significant")
        else:
            parts.append("DM not significant")
    return "; ".join(parts) if parts else "insufficient data"


# ── Load prediction file ──────────────────────────────────────────

def _load_preds(model_name: str) -> pd.DataFrame:
    """Load and concatenate prediction CSVs for one model across all tickers."""
    frames = []
    for ticker in TICKERS:
        path = os.path.join(PRED_DIR, f"{model_name}_{ticker}_predictions.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            frames.append(df)
        else:
            print(f"  Warning: prediction file missing: {path}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────

def run_statistical_tests() -> pd.DataFrame:
    print(f"\n{'='*62}")
    print("  Statistical Tests - Thesis Model Comparison")
    print(f"{'='*62}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load all prediction files once
    all_preds = {}
    for model_name in set(m for pair in COMPARISONS for m in pair):
        df = _load_preds(model_name)
        if df.empty:
            print(f"  No predictions found for '{model_name}'. "
                  f"Run walk_forward_eval.py first.")
        else:
            all_preds[model_name] = df
            print(f"  Loaded {len(df)} rows for '{model_name}'")

    rows = []

    for model_a_name, model_b_name in COMPARISONS:
        if model_a_name not in all_preds or model_b_name not in all_preds:
            print(f"\n  Skipping {model_a_name} vs {model_b_name} "
                  f"(data missing)")
            continue

        df_a_all = all_preds[model_a_name]
        df_b_all = all_preds[model_b_name]

        print(f"\n  {model_a_name}  vs  {model_b_name}")
        print(f"  {'-'*56}")

        for ticker in TICKERS:
            for H in HORIZONS:
                df_a = df_a_all[
                    (df_a_all["ticker"] == ticker) &
                    (df_a_all["horizon"] == H)
                ].copy()
                df_b = df_b_all[
                    (df_b_all["ticker"] == ticker) &
                    (df_b_all["horizon"] == H)
                ].copy()

                if df_a.empty or df_b.empty:
                    continue

                # Align on date (inner join) so paired tests are valid
                merged = df_a[["date", "actual_price", "predicted_Q50"]].merge(
                    df_b[["date", "actual_price", "predicted_Q50"]],
                    on="date",
                    suffixes=("_a", "_b"),
                )
                if len(merged) < 5:
                    continue

                q50_a   = merged["predicted_Q50_a"].values
                q50_b   = merged["predicted_Q50_b"].values
                actual  = merged["actual_price_a"].values

                err_a   = np.abs(actual - q50_a)
                err_b   = np.abs(actual - q50_b)

                t_q50, p_q50   = _paired_ttest(q50_a, q50_b)
                t_err, p_err   = _paired_ttest(err_a, err_b)
                dm_stat, dm_p  = _diebold_mariano(actual - q50_a, actual - q50_b)

                interp = _interpret(p_q50, p_err, dm_p)

                rows.append({
                    "ticker":               ticker,
                    "horizon":              H,
                    "model_a":              model_a_name,
                    "model_b":              model_b_name,
                    "n_paired":             len(merged),
                    "mean_Q50_model_a":     round(float(q50_a.mean()),  4),
                    "mean_Q50_model_b":     round(float(q50_b.mean()),  4),
                    "mean_Q50_difference":  round(float(q50_b.mean() - q50_a.mean()), 4),
                    "t_stat_Q50":           round(t_q50, 4) if not np.isnan(t_q50) else None,
                    "p_value_Q50":          round(p_q50, 4) if not np.isnan(p_q50) else None,
                    "mean_abs_error_model_a": round(float(err_a.mean()), 4),
                    "mean_abs_error_model_b": round(float(err_b.mean()), 4),
                    "mean_abs_error_difference": round(float(err_b.mean() - err_a.mean()), 4),
                    "t_stat_error":         round(t_err, 4) if not np.isnan(t_err) else None,
                    "p_value_error":        round(p_err, 4) if not np.isnan(p_err) else None,
                    "dm_stat":              round(dm_stat, 4) if not np.isnan(dm_stat) else None,
                    "dm_p_value":           round(dm_p,   4) if not np.isnan(dm_p)   else None,
                    "interpretation":       interp,
                })

                sig = "*" if (not np.isnan(p_q50) and p_q50 < 0.05) else " "
                print(f"  {ticker} H={H:>2}d  n={len(merged):>3}  "
                      f"dQ50={q50_b.mean()-q50_a.mean():+.2f}  "
                      f"p_q50={p_q50:.3f}{sig}  "
                      f"p_err={p_err:.3f}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Results saved -> {OUTPUT_PATH}  ({len(result_df)} rows)")
    return result_df


if __name__ == "__main__":
    run_statistical_tests()
