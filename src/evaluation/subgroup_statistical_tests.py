# src/evaluation/subgroup_statistical_tests.py
# Subgroup Statistical Tests — Thesis Event Feature Analysis
#
# Extends the global statistical_tests.py to run within each subgroup,
# enabling thesis claims like:
#   "Sentiment features significantly shift predicted price levels on
#    event days (p<0.05) but not on no-news days (p=0.38)."
#
# Tests per (ticker, horizon, subgroup, model_a vs model_b):
#   1. Paired t-test on Q50 predictions
#      H0: mean Q50 predictions are equal
#      Thesis relevance: does sentiment shift the predicted distribution?
#
#   2. Paired t-test on absolute errors |actual - Q50|
#      H0: mean absolute errors are equal
#      Thesis relevance: does sentiment improve accuracy on this subgroup?
#
#   3. Diebold-Mariano test (Harvey et al. small-sample correction)
#      H0: equal predictive accuracy
#      Thesis relevance: is the accuracy difference statistically robust?
#
# Input:
#   results/revised_experiments/predictions/<model>_<ticker>_predictions.csv
#   results/revised_experiments/event_features/price_sentiment_event_dataset.csv
#
# Output:
#   results/revised_experiments/event_features/subgroup_statistical_tests.csv
#
# Usage:
#   python -m src.evaluation.subgroup_statistical_tests

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "results/revised_experiments"
PRED_DIR    = os.path.join(RESULTS_DIR, "predictions")
EVENT_CSV   = os.path.join(RESULTS_DIR, "event_features", "event_features_daily.csv")
OUTPUT_DIR  = os.path.join(RESULTS_DIR, "event_features")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "subgroup_statistical_tests.csv")

TICKERS   = ["AAPL", "MSFT", "NVDA"]
HORIZONS  = [1, 5, 10]

SUBGROUPS = [
    "all_days",
    "news_days",
    "no_news_days",
    "high_sentiment_days",
    "event_days",
    "high_event_intensity",
]

COMPARISONS = [
    ("price_only_lstm",       "sentiment_lstm"),
    ("price_only_lstm",       "sentiment_shift_lstm"),
    ("price_only_lstm",       "sentiment_lstm_event"),
    ("price_only_lstm",       "sentiment_shift_lstm_event"),
    ("sentiment_lstm",        "sentiment_lstm_event"),
    ("sentiment_shift_lstm",  "sentiment_shift_lstm_event"),
]

HIGH_SENTIMENT_THRESH = 0.5
HIGH_EVENT_PERCENTILE = 75


# ── Statistical helpers ───────────────────────────────────────────

def _paired_ttest(a: np.ndarray, b: np.ndarray) -> tuple:
    if len(a) < 4:
        return float("nan"), float("nan")
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p)


def _diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray) -> tuple:
    """Harvey-Leybourne-Newbold small-sample DM test, squared-error loss."""
    d = errors_a ** 2 - errors_b ** 2
    n = len(d)
    if n < 4:
        return float("nan"), float("nan")
    d_bar  = np.mean(d)
    sigma2 = np.var(d, ddof=1) / n
    if sigma2 < 1e-14:
        return 0.0, 1.0
    dm_raw  = d_bar / np.sqrt(sigma2)
    dm_corr = dm_raw * np.sqrt((n + 1 - 2 + 1 / n) / n)
    p_val   = 2.0 * (1.0 - stats.t.cdf(abs(dm_corr), df=n - 1))
    return float(dm_raw), float(p_val)


def _interpret(p_q50, p_err, dm_p) -> str:
    parts = []
    for p_val, label in [(p_q50, "Q50"), (p_err, "error"), (dm_p, "DM")]:
        if p_val is None or (isinstance(p_val, float) and np.isnan(p_val)):
            continue
        if p_val < 0.01:
            parts.append(f"{label} highly sig (p<0.01)")
        elif p_val < 0.05:
            parts.append(f"{label} sig (p<0.05)")
        elif p_val < 0.10:
            parts.append(f"{label} marginal (p<0.10)")
        else:
            parts.append(f"{label} not sig")
    return "; ".join(parts) if parts else "insufficient data"


# ── Subgroup masking (same logic as subgroup_evaluation.py) ───────

def _apply_subgroup(df: pd.DataFrame, subgroup: str) -> pd.DataFrame:
    """Filter df to rows belonging to the given subgroup."""
    if subgroup == "all_days":
        return df

    if subgroup == "news_days":
        return df[df["news_count"] > 0]

    if subgroup == "no_news_days":
        return df[df["news_count"] == 0]

    if subgroup == "high_sentiment_days":
        return df[df["avg_sentiment_score"].abs() >= HIGH_SENTIMENT_THRESH]

    if subgroup == "event_days":
        return df[df["event_count"] > 0]

    if subgroup == "high_event_intensity":
        event_rows = df[df["event_count"] > 0]
        if event_rows.empty:
            return event_rows
        threshold = event_rows["event_intensity"].quantile(
            HIGH_EVENT_PERCENTILE / 100
        )
        return event_rows[event_rows["event_intensity"] >= threshold]

    raise ValueError(f"Unknown subgroup: {subgroup}")


# ── Data loading ──────────────────────────────────────────────────

def _load_preds(model_name: str) -> pd.DataFrame:
    frames = []
    for ticker in TICKERS:
        path = os.path.join(PRED_DIR, f"{model_name}_{ticker}_predictions.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_subgroup_features(ticker: str) -> pd.DataFrame:
    """
    Build the subgroup feature table for one ticker using authoritative sources:
      news_count / avg_sentiment_score -> outputs/{ticker}_price_sentiment_merged.csv
      event_count / event_intensity    -> event_features_daily.csv
    """
    merged_path = f"outputs/{ticker}_price_sentiment_merged.csv"
    df_m = pd.read_csv(merged_path)
    dc   = "Date" if "Date" in df_m.columns else "date"
    df_m["date"] = pd.to_datetime(df_m[dc], errors="coerce").dt.strftime("%Y-%m-%d")
    df_m = df_m.dropna(subset=["date"])
    for col in ["news_count", "avg_sentiment_score"]:
        df_m[col] = pd.to_numeric(df_m.get(col, 0), errors="coerce").fillna(0.0)
    sent = df_m[["date", "news_count", "avg_sentiment_score"]].drop_duplicates("date")

    if os.path.exists(EVENT_CSV):
        ev_all = pd.read_csv(EVENT_CSV)
        ev_t   = ev_all[ev_all["ticker"] == ticker][
            ["date", "event_count", "event_intensity"]
        ].copy()
        ev_t["date"] = pd.to_datetime(ev_t["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        ev_t = ev_t.dropna(subset=["date"]).drop_duplicates("date")
    else:
        ev_t = pd.DataFrame(columns=["date", "event_count", "event_intensity"])

    result = sent.merge(ev_t, on="date", how="left")
    result["event_count"]    = result["event_count"].fillna(0.0)
    result["event_intensity"] = result["event_intensity"].fillna(0.0)
    return result


# ── Main ──────────────────────────────────────────────────────────

def run_subgroup_statistical_tests() -> pd.DataFrame:
    print(f"\n{'='*62}")
    print("  Subgroup Statistical Tests - Thesis")
    print(f"{'='*62}")
    print(f"  Sentiment source: outputs/{{ticker}}_price_sentiment_merged.csv")
    print(f"  Event source    : {EVENT_CSV}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pre-build subgroup features for each ticker (avoids repeated file reads)
    sg_cache: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        try:
            sg_cache[ticker] = _build_subgroup_features(ticker)
            n_news  = (sg_cache[ticker]["news_count"] > 0).sum()
            n_event = (sg_cache[ticker]["event_count"] > 0).sum()
            print(f"  {ticker} subgroup features: {len(sg_cache[ticker])} rows  "
                  f"news_days={n_news}  event_days={n_event}")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            sg_cache[ticker] = pd.DataFrame(
                columns=["date", "news_count", "avg_sentiment_score",
                         "event_count", "event_intensity"]
            )

    # Load all needed prediction files once
    all_preds = {}
    for model_name in set(m for pair in COMPARISONS for m in pair):
        df = _load_preds(model_name)
        if not df.empty:
            all_preds[model_name] = df
            print(f"  Loaded {len(df):>5} rows for '{model_name}'")
        else:
            print(f"  [missing] '{model_name}' - will skip comparisons involving it")

    rows = []

    for model_a_name, model_b_name in COMPARISONS:
        if model_a_name not in all_preds or model_b_name not in all_preds:
            continue

        df_a_all = all_preds[model_a_name]
        df_b_all = all_preds[model_b_name]

        print(f"\n  {model_a_name}  vs  {model_b_name}")

        for ticker in TICKERS:
            sg_feats = sg_cache[ticker]

            for H in HORIZONS:
                df_a = df_a_all[
                    (df_a_all["ticker"] == ticker) & (df_a_all["horizon"] == H)
                ].copy()
                df_b = df_b_all[
                    (df_b_all["ticker"] == ticker) & (df_b_all["horizon"] == H)
                ].copy()

                if df_a.empty or df_b.empty:
                    continue

                # Join both prediction sets with authoritative subgroup features
                def _attach(df):
                    out = df.merge(sg_feats, on="date", how="left")
                    for c in ["news_count", "event_count", "event_intensity"]:
                        out[c] = out[c].fillna(0.0)
                    out["avg_sentiment_score"] = out["avg_sentiment_score"].fillna(0.0)
                    return out

                df_a = _attach(df_a)
                df_b = _attach(df_b)

                for subgroup in SUBGROUPS:
                    sg_a = _apply_subgroup(df_a, subgroup)
                    sg_b = _apply_subgroup(df_b, subgroup)

                    # Align on date (inner join) so tests are truly paired
                    merged = sg_a[["date", "actual_price", "predicted_Q50"]].merge(
                        sg_b[["date", "actual_price", "predicted_Q50"]],
                        on="date",
                        suffixes=("_a", "_b"),
                    )
                    if len(merged) < 5:
                        continue

                    q50_a  = merged["predicted_Q50_a"].values
                    q50_b  = merged["predicted_Q50_b"].values
                    actual = merged["actual_price_a"].values

                    err_a = np.abs(actual - q50_a)
                    err_b = np.abs(actual - q50_b)

                    t_q50, p_q50 = _paired_ttest(q50_a, q50_b)
                    t_err, p_err = _paired_ttest(err_a, err_b)
                    dm_s,  dm_p  = _diebold_mariano(actual - q50_a, actual - q50_b)

                    def _r(v):
                        return round(v, 4) if not np.isnan(v) else None

                    rows.append({
                        "ticker":                    ticker,
                        "horizon":                   H,
                        "subgroup":                  subgroup,
                        "model_a":                   model_a_name,
                        "model_b":                   model_b_name,
                        "n_paired":                  len(merged),
                        "mean_Q50_model_a":          round(float(q50_a.mean()), 4),
                        "mean_Q50_model_b":          round(float(q50_b.mean()), 4),
                        "mean_Q50_difference":       round(float(q50_b.mean() - q50_a.mean()), 4),
                        "t_stat_Q50":                _r(t_q50),
                        "p_value_Q50":               _r(p_q50),
                        "mean_abs_error_model_a":    round(float(err_a.mean()), 4),
                        "mean_abs_error_model_b":    round(float(err_b.mean()), 4),
                        "mean_abs_error_difference": round(float(err_b.mean() - err_a.mean()), 4),
                        "t_stat_error":              _r(t_err),
                        "p_value_error":             _r(p_err),
                        "dm_stat":                   _r(dm_s),
                        "dm_p_value":                _r(dm_p),
                        "interpretation":            _interpret(p_q50, p_err, dm_p),
                    })

                    # Quick console summary for event_days subgroup
                    if subgroup == "event_days":
                        sig = "*" if (p_q50 is not None and
                                      not np.isnan(p_q50) and p_q50 < 0.05) else " "
                        print(f"  {ticker} H={H:>2}d {subgroup:<22}  "
                              f"n={len(merged):>3}  "
                              f"dQ50={q50_b.mean()-q50_a.mean():+.2f}  "
                              f"p_q50={p_q50:.3f}{sig}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Results saved -> {OUTPUT_CSV}  ({len(result_df)} rows)")
    return result_df


if __name__ == "__main__":
    run_subgroup_statistical_tests()
