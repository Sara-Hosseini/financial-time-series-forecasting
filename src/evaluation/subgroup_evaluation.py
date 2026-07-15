# src/evaluation/subgroup_evaluation.py
# Subgroup Evaluation — Thesis Event Feature Analysis
#
# Evaluates model performance separately for 6 subgroups:
#   all_days             — every test observation
#   news_days            — news_count > 0
#   no_news_days         — news_count == 0
#   high_sentiment_days  — abs(avg_sentiment_score) >= 0.5
#   event_days           — event_count > 0
#   high_event_intensity — event_intensity >= 75th percentile (among event days)
#
# Subgroup feature sources (intentionally separated):
#   news_count, avg_sentiment_score  -> outputs/{ticker}_price_sentiment_merged.csv
#       This file is always up-to-date because the sentiment pipeline writes to it.
#       Loading from here prevents stale data if the event dataset was built before
#       the sentiment pipeline was run for all tickers.
#
#   event_count, event_intensity     -> event_features_daily.csv
#       From keyword-based extraction on news_raw.csv.
#
# Input:
#   results/revised_experiments/predictions/<model>_<ticker>_predictions.csv
#   outputs/{ticker}_price_sentiment_merged.csv
#   results/revised_experiments/event_features/event_features_daily.csv
#
# Output:
#   results/revised_experiments/event_features/subgroup_evaluation.csv
#
# Usage:
#   python -m src.evaluation.subgroup_evaluation

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.utils import mape, dir_acc, coverage, interval_width

RESULTS_DIR = "results/revised_experiments"
PRED_DIR    = os.path.join(RESULTS_DIR, "predictions")
EVENT_CSV   = os.path.join(RESULTS_DIR, "event_features", "event_features_daily.csv")
OUTPUT_DIR  = os.path.join(RESULTS_DIR, "event_features")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "subgroup_evaluation.csv")

TICKERS  = ["AAPL", "MSFT", "NVDA"]
HORIZONS = [1, 5, 10]

ALL_MODELS = [
    "price_only_lstm",
    "sentiment_lstm",
    "sentiment_shift_lstm",
    "sentiment_lstm_event",
    "sentiment_shift_lstm_event",
    "xgboost_baseline",
    "xgboost_event_baseline",
]

HIGH_SENTIMENT_THRESH = 0.5
HIGH_EVENT_PERCENTILE = 75


# ── Subgroup feature loader ───────────────────────────────────────

def _build_subgroup_features(ticker: str) -> pd.DataFrame:
    """
    Build a date-indexed feature table for subgroup filtering.

    news_count / avg_sentiment_score come from the CURRENT merged CSV
    (outputs/{ticker}_price_sentiment_merged.csv), which is always
    up-to-date with the latest sentiment pipeline run.

    event_count / event_intensity come from event_features_daily.csv
    (keyword extraction result).

    Returns columns: date, news_count, avg_sentiment_score,
                     event_count, event_intensity
    """
    # ── Sentiment features: always read from current merged CSV ───
    merged_path = f"outputs/{ticker}_price_sentiment_merged.csv"
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"Merged CSV not found: {merged_path}")

    df_m = pd.read_csv(merged_path)
    dc   = "Date" if "Date" in df_m.columns else "date"
    df_m["date"] = (pd.to_datetime(df_m[dc], errors="coerce")
                      .dt.strftime("%Y-%m-%d"))
    df_m = df_m.dropna(subset=["date"])

    for col in ["news_count", "avg_sentiment_score"]:
        if col in df_m.columns:
            df_m[col] = pd.to_numeric(df_m[col], errors="coerce").fillna(0.0)
        else:
            df_m[col] = 0.0

    sent = df_m[["date", "news_count", "avg_sentiment_score"]].drop_duplicates("date")

    # ── Event features: from keyword extraction CSV ────────────────
    if os.path.exists(EVENT_CSV):
        ev_all = pd.read_csv(EVENT_CSV)
        ev_t   = ev_all[ev_all["ticker"] == ticker][
            ["date", "event_count", "event_intensity"]
        ].copy()
        ev_t["date"] = (pd.to_datetime(ev_t["date"], errors="coerce")
                          .dt.strftime("%Y-%m-%d"))
        ev_t = ev_t.dropna(subset=["date"]).drop_duplicates("date")
    else:
        ev_t = pd.DataFrame(columns=["date", "event_count", "event_intensity"])

    # ── Merge: all trading days keep their sentiment values ────────
    result = sent.merge(ev_t, on="date", how="left")
    result["event_count"]     = result["event_count"].fillna(0.0)
    result["event_intensity"]  = result["event_intensity"].fillna(0.0)

    return result


# ── Metric helpers ────────────────────────────────────────────────

def _compute_metrics(sub: pd.DataFrame) -> dict:
    if len(sub) == 0:
        return {
            "n": 0, "mape_pct": None, "coverage_pct": None,
            "dir_acc_pct": None, "interval_width": None,
            "mean_abs_error": None, "median_abs_error": None,
        }

    actual = sub["actual_price"].values
    q10    = sub["predicted_Q10"].values
    q50    = sub["predicted_Q50"].values
    q90    = sub["predicted_Q90"].values

    abs_errors = np.abs(actual - q50)

    if "actual_return" in sub.columns:
        base_closes = actual / np.exp(sub["actual_return"].values)
        dir_acc_val = dir_acc(base_closes, actual, q50)
    else:
        dir_acc_val = float("nan")

    return {
        "n":                len(sub),
        "mape_pct":         round(mape(actual, q50),             4),
        "coverage_pct":     round(coverage(actual, q10, q90),    2),
        "dir_acc_pct":      round(dir_acc_val,                   2),
        "interval_width":   round(interval_width(q10, q90),      4),
        "mean_abs_error":   round(float(abs_errors.mean()),      4),
        "median_abs_error": round(float(np.median(abs_errors)),  4),
    }


# ── Subgroup mask definitions ─────────────────────────────────────

def _get_subgroup_masks(df: pd.DataFrame) -> dict:
    n = len(df)
    masks = {}

    masks["all_days"]            = pd.Series([True] * n, index=df.index)
    masks["news_days"]           = df["news_count"] > 0
    masks["no_news_days"]        = df["news_count"] == 0
    masks["high_sentiment_days"] = df["avg_sentiment_score"].abs() >= HIGH_SENTIMENT_THRESH
    masks["event_days"]          = df["event_count"] > 0

    event_mask = df["event_count"] > 0
    if event_mask.sum() > 0:
        threshold = df.loc[event_mask, "event_intensity"].quantile(
            HIGH_EVENT_PERCENTILE / 100
        )
        masks["high_event_intensity"] = event_mask & (df["event_intensity"] >= threshold)
    else:
        masks["high_event_intensity"] = pd.Series([False] * n, index=df.index)

    return masks


# ── Main ──────────────────────────────────────────────────────────

def run_subgroup_evaluation() -> pd.DataFrame:
    print(f"\n{'='*62}")
    print("  Subgroup Evaluation - Event Feature Analysis")
    print(f"{'='*62}")
    print(f"  Sentiment features source : outputs/{{ticker}}_price_sentiment_merged.csv")
    print(f"  Event features source     : {EVENT_CSV}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []

    for model_name in ALL_MODELS:
        for ticker in TICKERS:
            pred_path = os.path.join(PRED_DIR, f"{model_name}_{ticker}_predictions.csv")
            if not os.path.exists(pred_path):
                continue

            pred_df = pd.read_csv(pred_path)
            print(f"\n  {model_name} | {ticker}  ({len(pred_df)} predictions)")

            # Build subgroup features from authoritative sources
            try:
                sg_features = _build_subgroup_features(ticker)
            except FileNotFoundError as e:
                print(f"  ERROR loading subgroup features: {e}")
                continue

            # Quick sanity check
            n_news  = (sg_features["news_count"] > 0).sum()
            n_event = (sg_features["event_count"] > 0).sum()
            print(f"  subgroup features: {len(sg_features)} rows  "
                  f"news_days={n_news}  event_days={n_event}")

            # Join predictions with subgroup features on date
            merged = pred_df.merge(sg_features, on="date", how="left")
            for col in ["news_count", "event_count", "event_intensity"]:
                merged[col] = merged[col].fillna(0.0)
            merged["avg_sentiment_score"] = merged["avg_sentiment_score"].fillna(0.0)

            for H in HORIZONS:
                sub_h = merged[merged["horizon"] == H].copy()
                if sub_h.empty:
                    continue

                masks = _get_subgroup_masks(sub_h)

                for sg_name, mask in masks.items():
                    sub_sg  = sub_h[mask]
                    metrics = _compute_metrics(sub_sg)

                    rows.append({
                        "model_name": model_name,
                        "ticker":     ticker,
                        "horizon":    H,
                        "subgroup":   sg_name,
                        **metrics,
                    })

                    n_v    = metrics["n"]
                    mape_v = metrics["mape_pct"]
                    dir_v  = metrics["dir_acc_pct"]
                    mape_s = f"{mape_v:.2f}%" if mape_v is not None else "N/A"
                    dir_s  = f"{dir_v:.1f}%"  if dir_v  is not None else "N/A"
                    print(f"    H={H:>2}d  {sg_name:<25}  n={n_v:>4}  "
                          f"MAPE={mape_s:>7}  DirAcc={dir_s:>6}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Subgroup results saved -> {OUTPUT_CSV}  ({len(result_df)} rows)")
    return result_df


if __name__ == "__main__":
    run_subgroup_evaluation()
