# src/evaluation/distribution_shift_analysis.py
# Distribution Shift Analysis — Thesis Revision
#
# Quantifies how much sentiment-aware models shift the predicted price
# distribution relative to the price-only baseline.
#
# For each (ticker, horizon, model_a vs model_b) triple:
#   - mean_Q10/Q50/Q90 for each model
#   - mean_Q10/Q50/Q90 shift (model_b minus model_a)
#   - average prediction interval width and its change
#   - direction_of_shift: "upward" / "downward" / "no_clear_shift"
#   - approximate Gaussian sigma estimated from (Q90-Q10)/2.5632
#     (the distance between the 10th and 90th percentiles of N(0,1)
#     equals 2 × 1.2816 = 2.5632 standard deviations)
#
# Input:  results/revised_experiments/predictions/*.csv
# Output: results/revised_experiments/distribution_shift.csv
#
# Usage:
#   python -m src.evaluation.distribution_shift_analysis

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

RESULTS_DIR = "results/revised_experiments"
PRED_DIR    = os.path.join(RESULTS_DIR, "predictions")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "distribution_shift.csv")

TICKERS  = ["AAPL", "MSFT", "NVDA"]
HORIZONS = [1, 5, 10]

COMPARISONS = [
    ("price_only_lstm",  "sentiment_lstm"),
    ("price_only_lstm",  "sentiment_shift_lstm"),
    ("sentiment_lstm",   "sentiment_shift_lstm"),
]

# Q90 - Q10 = 2 * 1.2816 * sigma  =>  sigma = (Q90 - Q10) / 2.5632
_GAUSSIAN_SCALE = 2.5632


def _load_preds(model_name: str) -> pd.DataFrame:
    frames = []
    for ticker in TICKERS:
        path = os.path.join(PRED_DIR, f"{model_name}_{ticker}_predictions.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
        else:
            print(f"  Warning: missing {path}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _shift_direction(delta: float, threshold: float = 0.10) -> str:
    """Classify Q50 shift direction relative to a price-normalised threshold."""
    if delta > threshold:
        return "upward"
    if delta < -threshold:
        return "downward"
    return "no_clear_shift"


def run_distribution_shift_analysis() -> pd.DataFrame:
    print(f"\n{'='*62}")
    print("  Distribution Shift Analysis - Thesis")
    print(f"{'='*62}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_preds = {}
    for model_name in set(m for pair in COMPARISONS for m in pair):
        df = _load_preds(model_name)
        if not df.empty:
            all_preds[model_name] = df
            print(f"  Loaded {len(df)} rows for '{model_name}'")
        else:
            print(f"  No predictions for '{model_name}'. "
                  f"Run walk_forward_eval.py first.")

    rows = []

    for model_a_name, model_b_name in COMPARISONS:
        if model_a_name not in all_preds or model_b_name not in all_preds:
            print(f"\n  Skipping {model_a_name} vs {model_b_name} (data missing)")
            continue

        df_a_all = all_preds[model_a_name]
        df_b_all = all_preds[model_b_name]

        print(f"\n  {model_a_name}  vs  {model_b_name}")
        print(f"  {'-'*56}")

        for ticker in TICKERS:
            for H in HORIZONS:
                df_a = df_a_all[
                    (df_a_all["ticker"] == ticker) & (df_a_all["horizon"] == H)
                ]
                df_b = df_b_all[
                    (df_b_all["ticker"] == ticker) & (df_b_all["horizon"] == H)
                ]

                if df_a.empty or df_b.empty:
                    continue

                # Align on date for paired comparison
                merged = df_a[["date", "predicted_Q10", "predicted_Q50",
                                "predicted_Q90"]].merge(
                    df_b[["date", "predicted_Q10", "predicted_Q50",
                          "predicted_Q90"]],
                    on="date",
                    suffixes=("_a", "_b"),
                )
                if len(merged) < 3:
                    continue

                # Per-model means
                mean_q10_a = merged["predicted_Q10_a"].mean()
                mean_q50_a = merged["predicted_Q50_a"].mean()
                mean_q90_a = merged["predicted_Q90_a"].mean()

                mean_q10_b = merged["predicted_Q10_b"].mean()
                mean_q50_b = merged["predicted_Q50_b"].mean()
                mean_q90_b = merged["predicted_Q90_b"].mean()

                width_a = (merged["predicted_Q90_a"] - merged["predicted_Q10_a"]).mean()
                width_b = (merged["predicted_Q90_b"] - merged["predicted_Q10_b"]).mean()

                # Approximate Gaussian parameters from quantile spread
                sigma_a = width_a / _GAUSSIAN_SCALE
                sigma_b = width_b / _GAUSSIAN_SCALE

                q50_shift = mean_q50_b - mean_q50_a
                direction = _shift_direction(q50_shift)

                rows.append({
                    "ticker":                   ticker,
                    "horizon":                  H,
                    "model_a":                  model_a_name,
                    "model_b":                  model_b_name,
                    "n_paired":                 len(merged),
                    # Model A distribution
                    "mean_Q10_model_a":         round(mean_q10_a, 4),
                    "mean_Q50_model_a":         round(mean_q50_a, 4),
                    "mean_Q90_model_a":         round(mean_q90_a, 4),
                    "approx_sigma_model_a":     round(sigma_a,    4),
                    "interval_width_model_a":   round(width_a,    4),
                    # Model B distribution
                    "mean_Q10_model_b":         round(mean_q10_b, 4),
                    "mean_Q50_model_b":         round(mean_q50_b, 4),
                    "mean_Q90_model_b":         round(mean_q90_b, 4),
                    "approx_sigma_model_b":     round(sigma_b,    4),
                    "interval_width_model_b":   round(width_b,    4),
                    # Shifts
                    "mean_Q10_shift":           round(mean_q10_b - mean_q10_a, 4),
                    "mean_Q50_shift":           round(q50_shift,               4),
                    "mean_Q90_shift":           round(mean_q90_b - mean_q90_a, 4),
                    "interval_width_change":    round(width_b - width_a,       4),
                    "sigma_change":             round(sigma_b - sigma_a,       4),
                    "direction_of_shift":       direction,
                })

                print(f"  {ticker} H={H:>2}d  n={len(merged):>3}  "
                      f"dQ50={q50_shift:+.2f}  "
                      f"sigma_a={sigma_a:.2f} sigma_b={sigma_b:.2f}  "
                      f"direction={direction}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Results saved -> {OUTPUT_PATH}  ({len(result_df)} rows)")
    return result_df


if __name__ == "__main__":
    run_distribution_shift_analysis()
