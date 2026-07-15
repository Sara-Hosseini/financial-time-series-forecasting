# src/models/xgboost_baseline.py
# XGBoost / GradientBoosting Quantile Baseline
#
# Trains 3 separate quantile models (Q10, Q50, Q90) for each
# ticker × horizon combination using walk-forward year-based folds
# identical to the LSTM evaluation.
#
# Feature strategy: for each time-step t, creates a flat feature vector
# using current-day features + 1-day lag + 5-day lag (39 features total).
# This gives tree models a temporal view without requiring sequences.
#
# Backend priority:
#   1. xgboost >= 1.7  (objective='reg:quantileerror')  if installed
#   2. sklearn GradientBoostingRegressor (loss='quantile')  fallback
#
# Usage:
#   python -m src.models.xgboost_baseline --ticker AAPL
#   python -m src.models.xgboost_baseline          # runs all tickers

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.utils import (
    add_features, make_target,
    mape, dir_acc, coverage, interval_width, ensure_dirs,
)

# ── Feature column definitions (thesis: 9 price + 4 sentiment = 13) ──
PRICE_FEATURE_COLS = [
    "log_ret_1", "ret_5", "ret_10",
    "ma_ratio_5", "ma_ratio_10",
    "hl_range", "vol_10", "vol_20", "rsi_14",
]

CORE_SENTIMENT_COLS = [
    "avg_sentiment_score", "positive_ratio", "negative_ratio", "news_count",
]

ALL_SENTIMENT_COLS = [
    "news_count", "avg_sentiment_score", "avg_positive_prob", "avg_negative_prob",
    "avg_neutral_prob", "max_positive_prob", "max_negative_prob", "pos_news_count",
    "neg_news_count", "neu_news_count", "sentiment_balance", "positive_ratio",
    "negative_ratio", "neutral_ratio", "has_news",
]

EVENT_FEATURE_COLS = [
    "earnings_event", "analyst_upgrade", "analyst_downgrade",
    "product_launch", "lawsuit_event", "regulatory_event",
    "partnership_event", "management_change_event", "macro_event",
    "event_count", "event_intensity",
]

BASE_FEATURE_COLS  = PRICE_FEATURE_COLS + CORE_SENTIMENT_COLS               # 13 features
EVENT_BASE_COLS    = PRICE_FEATURE_COLS + CORE_SENTIMENT_COLS + EVENT_FEATURE_COLS  # 24 features
LAGS               = [0, 1, 5]   # days of lag; lag_0 = current day
EVENT_DATASET_CSV  = "results/revised_experiments/event_features/price_sentiment_event_dataset.csv"

QUANTILES     = [0.1, 0.5, 0.9]
TEST_YEARS    = (2021, 2022, 2023, 2024, 2025)
HORIZONS      = (1, 5, 10)
TICKERS       = ("AAPL", "MSFT", "NVDA")
RESULTS_DIR   = "results/revised_experiments"
MIN_TRAIN_ROWS = 300  # lower than LSTM since no sequences needed


# ── Backend detection ─────────────────────────────────────────────
def _detect_backend():
    try:
        import xgboost as xgb
        major, minor = (int(x) for x in xgb.__version__.split(".")[:2])
        if major > 1 or (major == 1 and minor >= 7):
            return "xgboost_quantile"
        return "xgboost_approx"
    except ImportError:
        return "sklearn"


BACKEND = _detect_backend()


def _print_backend_info():
    if BACKEND == "xgboost_quantile":
        import xgboost as xgb
        print(f"  Backend: XGBoost {xgb.__version__} (quantile objective)")
    elif BACKEND == "xgboost_approx":
        import xgboost as xgb
        print(f"  Backend: XGBoost {xgb.__version__} "
              f"(no quantile objective; using 3× squared-error models)")
    else:
        from sklearn import __version__ as skv
        print(f"  Backend: scikit-learn {skv} GradientBoostingRegressor "
              f"(loss='quantile')")
        print("  To use XGBoost: pip install 'xgboost>=1.7'")


# ── Model builders ─────────────────────────────────────────────────
def _make_sklearn_quantile(alpha: float):
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )


def _make_xgb_quantile(alpha: float):
    import xgboost as xgb
    if BACKEND == "xgboost_quantile":
        return xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    else:
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbosity=0,
        )


def _make_model(alpha: float):
    if BACKEND == "sklearn":
        return _make_sklearn_quantile(alpha)
    return _make_xgb_quantile(alpha)


# ── Feature engineering: lagged flat features ─────────────────────
def _build_lagged_features(df: pd.DataFrame, feature_cols: list, lags: list) -> pd.DataFrame:
    """
    For each feature in feature_cols, create lag_0, lag_1, lag_5 columns.
    Returns a DataFrame with len(feature_cols) × len(lags) columns.
    """
    frames = []
    for lag in lags:
        shifted = df[feature_cols].shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in feature_cols]
        frames.append(shifted)
    return pd.concat(frames, axis=1)


# ── Data loading ──────────────────────────────────────────────────
def _load_features(ticker: str) -> pd.DataFrame:
    merged_path = f"outputs/{ticker}_price_sentiment_merged.csv"
    price_path  = f"data/raw/{ticker}_daily_10y_yahoo.csv"

    if os.path.exists(merged_path):
        df = pd.read_csv(merged_path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = (df.dropna(subset=["Date"])
                .sort_values("Date")
                .drop_duplicates(subset=["Date"], keep="last")
                .set_index("Date"))
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = add_features(df)
        for col in ALL_SENTIMENT_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df[ALL_SENTIMENT_COLS] = (
            df[ALL_SENTIMENT_COLS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        return df.dropna(subset=["Close"])
    else:
        from src.utils import load_prices
        df = load_prices(price_path)
        df = add_features(df)
        for col in ALL_SENTIMENT_COLS:
            df[col] = 0.0
        print(f"  Warning: merged CSV missing for {ticker}. Sentiment = 0.")
        return df


# ── Walk-forward eval ─────────────────────────────────────────────
def run_xgboost_baseline(ticker: str) -> tuple:
    print(f"\n{'='*62}")
    print(f"  XGBoost Baseline | {ticker}")
    print(f"{'='*62}")
    _print_backend_info()

    ensure_dirs(
        os.path.join(RESULTS_DIR, "predictions"),
        RESULTS_DIR,
    )

    df_raw = _load_features(ticker)

    # Build lagged feature matrix
    lag_df   = _build_lagged_features(df_raw, BASE_FEATURE_COLS, LAGS)
    df_work  = pd.concat([df_raw[["Close"]], lag_df], axis=1).dropna()
    lag_cols = list(lag_df.columns)

    print(f"  Dataset after lag: {len(df_work)} rows | {len(lag_cols)} features")

    pred_rows    = []
    metrics_rows = []

    for H in HORIZONS:
        print(f"\n  Horizon H={H}d  {'─'*40}")

        target  = make_target(df_work, H)
        df_h    = df_work.loc[target.dropna().index].copy()
        y_raw   = target.loc[df_h.index].values
        X_raw   = df_h[lag_cols].values
        dates   = df_h.index
        closes  = df_h["Close"].values

        for year in TEST_YEARS:
            test_mask  = dates.year == year
            train_mask = dates.year < year
            train_idx  = np.where(train_mask)[0]
            test_idx   = np.where(test_mask)[0]

            if len(train_idx) < MIN_TRAIN_ROWS or len(test_idx) == 0:
                print(f"    Fold {year}: skipped")
                continue

            scaler_X = RobustScaler().fit(X_raw[train_idx])
            X_s      = scaler_X.transform(X_raw)

            X_tr = X_s[train_idx]
            y_tr = y_raw[train_idx]
            X_te = X_s[test_idx]
            y_te = y_raw[test_idx]

            print(f"    Fold {year}: train={len(X_tr)} test={len(X_te)}", end="")

            models_q = {}
            for alpha in QUANTILES:
                m = _make_model(alpha)
                m.fit(X_tr, y_tr)
                models_q[alpha] = m

            # Predict log-returns then convert to prices
            q10_log = models_q[0.1].predict(X_te) if BACKEND == "sklearn" or BACKEND == "xgboost_quantile" else models_q[0.1].predict(X_te)
            q50_log = models_q[0.5].predict(X_te)
            q90_log = models_q[0.9].predict(X_te)

            base_closes = closes[test_idx]
            q10_price   = base_closes * np.exp(q10_log)
            q50_price   = base_closes * np.exp(q50_log)
            q90_price   = base_closes * np.exp(q90_log)
            true_price  = base_closes * np.exp(y_te)

            fold_mape  = mape(true_price, q50_price)
            fold_cov   = coverage(true_price, q10_price, q90_price)
            fold_width = interval_width(q10_price, q90_price)
            fold_dir   = dir_acc(base_closes, true_price, q50_price)

            print(f"  MAPE={fold_mape:.2f}%  Cov={fold_cov:.1f}%  "
                  f"Width=${fold_width:.2f}  DirAcc={fold_dir:.1f}%")

            for i, row_i in enumerate(test_idx):
                pred_rows.append({
                    "date":                  dates[row_i].strftime("%Y-%m-%d"),
                    "ticker":                ticker,
                    "horizon":               H,
                    "fold":                  year,
                    "model_name":            "xgboost_baseline",
                    "actual_price":          round(float(true_price[i]),  4),
                    "predicted_Q10":         round(float(q10_price[i]),   4),
                    "predicted_Q50":         round(float(q50_price[i]),   4),
                    "predicted_Q90":         round(float(q90_price[i]),   4),
                    "actual_return":         round(float(y_te[i]),        6),
                    "predicted_return_Q50":  round(float(q50_log[i]),     6),
                })

            metrics_rows.append({
                "model_name":     "xgboost_baseline",
                "ticker":         ticker,
                "horizon":        H,
                "fold":           year,
                "mape_pct":       round(fold_mape,  4),
                "coverage_pct":   round(fold_cov,   2),
                "interval_width": round(fold_width, 4),
                "dir_acc_pct":    round(fold_dir,   2),
                "n_test":         len(test_idx),
            })

    # ── Save predictions CSV ──────────────────────────────────────
    pred_df   = pd.DataFrame(pred_rows)
    pred_path = os.path.join(
        RESULTS_DIR, "predictions", f"xgboost_baseline_{ticker}_predictions.csv"
    )
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  Predictions saved  -> {pred_path}  ({len(pred_df)} rows)")

    # ── Save dedicated results CSV ────────────────────────────────
    results_path = os.path.join(RESULTS_DIR, "xgboost_baseline_results.csv")
    new_df       = pd.DataFrame(metrics_rows)
    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        existing = existing[existing["ticker"] != ticker]
        new_df   = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(results_path, index=False)
    print(f"  XGB metrics saved  -> {results_path}")

    # ── Also append to shared evaluation_metrics.csv ─────────────
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")
    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        mask = (existing["model_name"] == "xgboost_baseline") & \
               (existing["ticker"] == ticker)
        existing = existing[~mask]
        new_df   = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(metrics_path, index=False)
    print(f"  Metrics appended   -> {metrics_path}")

    return pred_df, pd.DataFrame(metrics_rows)


# ── Event-Enhanced XGBoost Baseline ──────────────────────────────
def _load_event_features(ticker: str) -> pd.DataFrame:
    """Load the 24-feature event dataset for one ticker."""
    if not os.path.exists(EVENT_DATASET_CSV):
        raise FileNotFoundError(
            f"Event dataset not found: {EVENT_DATASET_CSV}\n"
            "Run first: python -m src.features.merge_event_features"
        )
    df_all = pd.read_csv(EVENT_DATASET_CSV)
    df = df_all[df_all["ticker"] == ticker].copy()
    if df.empty:
        raise ValueError(f"No rows for ticker '{ticker}' in {EVENT_DATASET_CSV}")

    date_col = "date" if "date" in df.columns else "Date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = (df.dropna(subset=[date_col])
            .sort_values(date_col)
            .drop_duplicates(subset=[date_col], keep="last")
            .set_index(date_col))
    df.index.name = "Date"

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = add_features(df)

    for col in ALL_SENTIMENT_COLS + EVENT_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    for col_group in [ALL_SENTIMENT_COLS, EVENT_FEATURE_COLS]:
        df[col_group] = (
            df[col_group].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        )

    return df.dropna(subset=["Close"])


def run_xgboost_event_baseline(ticker: str) -> tuple:
    """
    XGBoost baseline using the event-enhanced 24-feature dataset.
    Identical walk-forward structure to run_xgboost_baseline but
    uses EVENT_BASE_COLS (24 features) with lag_0/lag_1/lag_5.
    """
    model_name = "xgboost_event_baseline"

    print(f"\n{'='*62}")
    print(f"  XGBoost Event Baseline | {ticker}")
    print(f"{'='*62}")
    _print_backend_info()

    ensure_dirs(
        os.path.join(RESULTS_DIR, "predictions"),
        RESULTS_DIR,
    )

    df_raw  = _load_event_features(ticker)
    lag_df  = _build_lagged_features(df_raw, EVENT_BASE_COLS, LAGS)
    df_work = pd.concat([df_raw[["Close"]], lag_df], axis=1).dropna()
    lag_cols = list(lag_df.columns)

    print(f"  Dataset after lag: {len(df_work)} rows | {len(lag_cols)} features")

    pred_rows    = []
    metrics_rows = []

    for H in HORIZONS:
        print(f"\n  Horizon H={H}d  {'─'*40}")

        target  = make_target(df_work, H)
        df_h    = df_work.loc[target.dropna().index].copy()
        y_raw   = target.loc[df_h.index].values
        X_raw   = df_h[lag_cols].values
        dates   = df_h.index
        closes  = df_h["Close"].values

        for year in TEST_YEARS:
            test_mask  = dates.year == year
            train_mask = dates.year < year
            train_idx  = np.where(train_mask)[0]
            test_idx   = np.where(test_mask)[0]

            if len(train_idx) < MIN_TRAIN_ROWS or len(test_idx) == 0:
                print(f"    Fold {year}: skipped")
                continue

            scaler_X = RobustScaler().fit(X_raw[train_idx])
            X_s = scaler_X.transform(X_raw)

            X_tr, y_tr = X_s[train_idx], y_raw[train_idx]
            X_te, y_te = X_s[test_idx],  y_raw[test_idx]

            print(f"    Fold {year}: train={len(X_tr)} test={len(X_te)}", end="")

            models_q = {}
            for alpha in QUANTILES:
                m = _make_model(alpha)
                m.fit(X_tr, y_tr)
                models_q[alpha] = m

            q10_log = models_q[0.1].predict(X_te)
            q50_log = models_q[0.5].predict(X_te)
            q90_log = models_q[0.9].predict(X_te)

            base_closes = closes[test_idx]
            q10_price   = base_closes * np.exp(q10_log)
            q50_price   = base_closes * np.exp(q50_log)
            q90_price   = base_closes * np.exp(q90_log)
            true_price  = base_closes * np.exp(y_te)

            fold_mape  = mape(true_price, q50_price)
            fold_cov   = coverage(true_price, q10_price, q90_price)
            fold_width = interval_width(q10_price, q90_price)
            fold_dir   = dir_acc(base_closes, true_price, q50_price)

            print(f"  MAPE={fold_mape:.2f}%  Cov={fold_cov:.1f}%  "
                  f"Width=${fold_width:.2f}  DirAcc={fold_dir:.1f}%")

            for i, row_i in enumerate(test_idx):
                pred_rows.append({
                    "date":                 dates[row_i].strftime("%Y-%m-%d"),
                    "ticker":               ticker,
                    "horizon":              H,
                    "fold":                 year,
                    "model_name":           model_name,
                    "actual_price":         round(float(true_price[i]), 4),
                    "predicted_Q10":        round(float(q10_price[i]),  4),
                    "predicted_Q50":        round(float(q50_price[i]),  4),
                    "predicted_Q90":        round(float(q90_price[i]),  4),
                    "actual_return":        round(float(y_te[i]),       6),
                    "predicted_return_Q50": round(float(q50_log[i]),    6),
                })

            metrics_rows.append({
                "model_name":     model_name,
                "ticker":         ticker,
                "horizon":        H,
                "fold":           year,
                "mape_pct":       round(fold_mape,  4),
                "coverage_pct":   round(fold_cov,   2),
                "interval_width": round(fold_width, 4),
                "dir_acc_pct":    round(fold_dir,   2),
                "n_test":         len(test_idx),
            })

    pred_df   = pd.DataFrame(pred_rows)
    pred_path = os.path.join(
        RESULTS_DIR, "predictions", f"{model_name}_{ticker}_predictions.csv"
    )
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  Predictions saved  -> {pred_path}  ({len(pred_df)} rows)")

    results_path = os.path.join(
        RESULTS_DIR, "event_features", "xgboost_event_baseline_results.csv"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    new_df = pd.DataFrame(metrics_rows)
    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        existing = existing[existing["ticker"] != ticker]
        new_df   = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(results_path, index=False)
    print(f"  Event XGB metrics  -> {results_path}")

    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")
    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        mask = (existing["model_name"] == model_name) & \
               (existing["ticker"] == ticker)
        existing = existing[~mask]
        new_df   = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(metrics_path, index=False)
    print(f"  Metrics appended   -> {metrics_path}")

    return pred_df, pd.DataFrame(metrics_rows)


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XGBoost quantile baseline")
    parser.add_argument(
        "--ticker",
        default=None,
        choices=["AAPL", "MSFT", "NVDA"],
        help="Single ticker to run (omit to run all three)",
    )
    parser.add_argument(
        "--model",
        default="baseline",
        choices=["baseline", "event"],
        help="'baseline' = 13-feature model, 'event' = 24-feature event model",
    )
    args = parser.parse_args()

    tickers   = [args.ticker] if args.ticker else list(TICKERS)
    run_func  = run_xgboost_event_baseline if args.model == "event" \
                else run_xgboost_baseline

    for t in tickers:
        run_func(t)

    print("\nXGBoost baseline complete.")
