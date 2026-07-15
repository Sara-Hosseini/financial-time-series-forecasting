# src/models/walk_forward_eval.py
# Walk-Forward Evaluation Framework — Thesis Revision
#
# Supported model variants:
#   price_only_lstm           — 9 price features
#   sentiment_lstm            — 9 price + 4 core sentiment  (13 total)
#   sentiment_shift_lstm      — 13 features, dual-branch SentimentShift
#   sentiment_lstm_event      — 9 price + 4 sentiment + 11 event  (24 total)
#   sentiment_shift_lstm_event — 24 features, dual-branch + wider event MLP
#
# For each variant × ticker × horizon × fold:
#   - Trains a fresh model on all prior years (walk-forward)
#   - Saves row-level predictions to results/revised_experiments/predictions/
#   - Appends fold metrics to results/revised_experiments/evaluation_metrics.csv
#
# Usage:
#   python -m src.models.walk_forward_eval --model price_only_lstm --ticker AAPL
#   python -m src.models.walk_forward_eval --model sentiment_lstm_event --ticker MSFT
#   python -m src.models.walk_forward_eval --model sentiment_shift_lstm_event --ticker NVDA
#                                           [--skip-training]

import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.utils import (
    add_features, make_target, make_sequences,
    mape, dir_acc, coverage, interval_width, ensure_dirs,
)
from src.models.sentiment_shift_lstm import (
    build_sentiment_shift_model,
    build_sentiment_shift_event_model,
    pinball_loss,
    PRICE_FEATURE_COLS,
    CORE_SENTIMENT_COLS,
    EVENT_FEATURE_COLS,
    N_PRICE,
    N_SENT,
    N_SENT_EVENT,
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── All 15 original sentiment columns (for completeness checks) ───
ALL_SENTIMENT_COLS = [
    "news_count", "avg_sentiment_score", "avg_positive_prob", "avg_negative_prob",
    "avg_neutral_prob", "max_positive_prob", "max_negative_prob", "pos_news_count",
    "neg_news_count", "neu_news_count", "sentiment_balance", "positive_ratio",
    "negative_ratio", "neutral_ratio", "has_news",
]

# ── Feature sets per model variant ───────────────────────────────
FEATURE_MAP = {
    "price_only_lstm":            PRICE_FEATURE_COLS,
    "sentiment_lstm":             PRICE_FEATURE_COLS + CORE_SENTIMENT_COLS,
    "sentiment_shift_lstm":       PRICE_FEATURE_COLS + CORE_SENTIMENT_COLS,
    "sentiment_lstm_event":       PRICE_FEATURE_COLS + CORE_SENTIMENT_COLS + EVENT_FEATURE_COLS,
    "sentiment_shift_lstm_event": PRICE_FEATURE_COLS + CORE_SENTIMENT_COLS + EVENT_FEATURE_COLS,
}

# Models that require the event-enhanced dataset
EVENT_MODELS      = {"sentiment_lstm_event", "sentiment_shift_lstm_event"}
EVENT_DATASET_CSV = "results/revised_experiments/event_features/price_sentiment_event_dataset.csv"

# ── Default walk-forward config ───────────────────────────────────
DEFAULT_CFG = {
    "window":         30,
    "quantiles":      (0.1, 0.5, 0.9),
    "units1":         64,
    "units2":         32,
    "dropout":        0.25,
    "lr":             5e-4,
    "epochs":         150,
    "batch_size":     32,
    "es_patience":    20,
    "es_min_delta":   1e-5,
    "lr_patience":    6,
    "lr_factor":      0.5,
    "lr_min":         1e-6,
    "val_ratio":      0.10,
    "min_train_rows": 500,
    "test_years":     (2021, 2022, 2023, 2024, 2025),
    "horizons":       (1, 5, 10),
    "results_dir":    "results/revised_experiments",
}


# ── Standard LSTM builder ─────────────────────────────────────────
def _build_standard_lstm(window: int, n_features: int, cfg: dict) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(window, n_features)),
        LSTM(cfg["units1"], return_sequences=True),
        Dropout(cfg["dropout"]),
        LSTM(cfg["units2"]),
        Dropout(cfg["dropout"]),
        Dense(32, activation="relu"),
        Dense(3),   # Q10, Q50, Q90
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"], clipnorm=1.0),
        loss=pinball_loss(cfg["quantiles"]),
    )
    return model


def _callbacks(cfg: dict):
    return [
        EarlyStopping(
            monitor="val_loss", patience=cfg["es_patience"],
            min_delta=cfg["es_min_delta"], restore_best_weights=True, verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", patience=cfg["lr_patience"],
            factor=cfg["lr_factor"], min_lr=cfg["lr_min"], verbose=0,
        ),
    ]


# ── Data loading ──────────────────────────────────────────────────
def _load_features(ticker: str, model_name: str) -> tuple:
    """
    Load the correct dataset for the given model variant and ticker.

    - Event models  → load from EVENT_DATASET_CSV  (multi-ticker, has all 24 cols)
    - Other models  → load from outputs/{ticker}_price_sentiment_merged.csv

    Returns (df_indexed_by_Date, feature_col_list).
    """
    feature_cols = FEATURE_MAP[model_name]

    if model_name in EVENT_MODELS:
        # Event-enhanced dataset
        if not os.path.exists(EVENT_DATASET_CSV):
            raise FileNotFoundError(
                f"Event dataset not found: {EVENT_DATASET_CSV}\n"
                "Run first: python -m src.features.merge_event_features"
            )
        df_all = pd.read_csv(EVENT_DATASET_CSV)
        # Filter to this ticker
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

        # Ensure all sentiment and event columns exist (fill missing with 0)
        for col in ALL_SENTIMENT_COLS + EVENT_FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        for col_group in [ALL_SENTIMENT_COLS, EVENT_FEATURE_COLS]:
            df[col_group] = (
                df[col_group].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            )
        df = df.dropna(subset=["Close"])

    else:
        # Original price + sentiment dataset
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
            df = df.dropna(subset=["Close"])
        else:
            from src.utils import load_prices
            df = load_prices(price_path)
            df = add_features(df)
            for col in ALL_SENTIMENT_COLS:
                df[col] = 0.0
            print(f"  Warning: merged CSV missing for {ticker}. Sentiment = 0.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for {ticker}/{model_name}: {missing}")

    return df, feature_cols


# ── Main walk-forward function ────────────────────────────────────
def run_walk_forward_eval(
    model_name: str,
    ticker: str,
    cfg: dict = None,
    skip_training: bool = False,
) -> tuple:
    """
    Run walk-forward evaluation for one model variant and one ticker.

    Parameters
    ----------
    model_name     : one of the keys in FEATURE_MAP
    ticker         : "AAPL", "MSFT", or "NVDA"
    cfg            : optional config dict (defaults to DEFAULT_CFG)
    skip_training  : if True AND prediction CSV already exists, skip training
                     and return the cached predictions immediately.

    Saves:
      results/revised_experiments/predictions/<model>_<ticker>_predictions.csv
      results/revised_experiments/evaluation_metrics.csv  (appended)

    Returns (pred_df, metrics_df).
    """
    if cfg is None:
        cfg = DEFAULT_CFG

    print(f"\n{'='*62}")
    print(f"  Walk-Forward Eval: {model_name} | {ticker}")
    print(f"{'='*62}")

    pred_dir  = os.path.join(cfg["results_dir"], "predictions")
    pred_path = os.path.join(pred_dir, f"{model_name}_{ticker}_predictions.csv")
    ensure_dirs(pred_dir)

    # ── Skip-training fast path ───────────────────────────────────
    if skip_training and os.path.exists(pred_path):
        print(f"  [skip-training] Predictions already exist: {pred_path}")
        print(f"  [skip-training] Skipping training. Load cached predictions.")
        pred_df = pd.read_csv(pred_path)
        return pred_df, pd.DataFrame()

    if skip_training:
        print(f"  [skip-training] No cached predictions found. Running training.")

    # ── Load data ─────────────────────────────────────────────────
    df_feat, feature_cols = _load_features(ticker, model_name)
    n_feats = len(feature_cols)

    print(f"  Dataset: {len(df_feat)} rows  "
          f"({df_feat.index[0].date()} -> {df_feat.index[-1].date()})")
    print(f"  Features: {n_feats}  |  model: {model_name}")

    pred_rows    = []
    metrics_rows = []

    for H in cfg["horizons"]:
        print(f"\n  Horizon H={H}d  {'─'*45}")

        target  = make_target(df_feat, H)
        df_work = df_feat.loc[target.dropna().index].copy()
        y_raw   = target.loc[df_work.index].values
        X_raw   = df_work[feature_cols].values
        dates   = df_work.index
        closes  = df_work["Close"].values

        for year in cfg["test_years"]:
            test_mask  = dates.year == year
            train_mask = dates.year < year
            train_idx  = np.where(train_mask)[0]
            test_idx   = np.where(test_mask)[0]

            if len(train_idx) < cfg["min_train_rows"] or len(test_idx) == 0:
                print(f"    Fold {year}: skipped "
                      f"(train={len(train_idx)} test={len(test_idx)})")
                continue

            val_cut   = int(len(train_idx) * (1 - cfg["val_ratio"]))
            train_end = train_idx[val_cut]
            val_end   = train_idx[-1] + 1

            scaler_X = RobustScaler().fit(X_raw[train_idx[:val_cut]])
            scaler_y = StandardScaler().fit(
                y_raw[train_idx[:val_cut]].reshape(-1, 1)
            )

            X_s = scaler_X.transform(X_raw)
            y_s = scaler_y.transform(y_raw.reshape(-1, 1)).flatten()

            X_tr, y_tr, _ = make_sequences(X_s, y_s, 0,
                                            train_end, cfg["window"])
            X_va, y_va, _ = make_sequences(X_s, y_s, train_end,
                                            val_end,   cfg["window"])
            X_te, y_te_s, te_idx = make_sequences(
                X_s, y_s, test_idx[0], test_idx[-1] + 1, cfg["window"]
            )

            if len(X_tr) == 0 or len(X_te) == 0:
                print(f"    Fold {year}: skipped (empty sequences)")
                continue

            print(f"    Fold {year}: train={len(X_tr)} "
                  f"val={len(X_va)} test={len(X_te)}")

            # ── Build model for this fold ──────────────────────────
            tf.keras.backend.clear_session()
            if model_name == "sentiment_shift_lstm":
                model = build_sentiment_shift_model(
                    cfg["window"], N_PRICE, N_SENT, cfg
                )
            elif model_name == "sentiment_shift_lstm_event":
                model = build_sentiment_shift_event_model(
                    cfg["window"], N_PRICE, N_SENT_EVENT, cfg
                )
            else:
                # price_only_lstm, sentiment_lstm, sentiment_lstm_event
                model = _build_standard_lstm(cfg["window"], n_feats, cfg)

            model.fit(
                X_tr, y_tr,
                validation_data=(X_va, y_va),
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                callbacks=_callbacks(cfg),
                verbose=0,
                shuffle=False,
            )

            # ── Predict and inverse-scale ──────────────────────────
            preds_s   = model.predict(X_te, verbose=0)
            preds_log = scaler_y.inverse_transform(preds_s)
            y_te_log  = scaler_y.inverse_transform(
                y_te_s.reshape(-1, 1)
            ).flatten()

            base_closes = closes[te_idx]
            q10_price   = base_closes * np.exp(preds_log[:, 0])
            q50_price   = base_closes * np.exp(preds_log[:, 1])
            q90_price   = base_closes * np.exp(preds_log[:, 2])
            true_price  = base_closes * np.exp(y_te_log)

            fold_mape  = mape(true_price, q50_price)
            fold_cov   = coverage(true_price, q10_price, q90_price)
            fold_width = interval_width(q10_price, q90_price)
            fold_dir   = dir_acc(base_closes, true_price, q50_price)

            print(f"    MAPE={fold_mape:.2f}%  Cov={fold_cov:.1f}%  "
                  f"Width=${fold_width:.2f}  DirAcc={fold_dir:.1f}%")

            # ── Row-level predictions ──────────────────────────────
            for i, row_i in enumerate(te_idx):
                pred_rows.append({
                    "date":                 dates[row_i].strftime("%Y-%m-%d"),
                    "ticker":               ticker,
                    "horizon":              H,
                    "fold":                 year,
                    "model_name":           model_name,
                    "actual_price":         round(float(true_price[i]),    4),
                    "predicted_Q10":        round(float(q10_price[i]),     4),
                    "predicted_Q50":        round(float(q50_price[i]),     4),
                    "predicted_Q90":        round(float(q90_price[i]),     4),
                    "actual_return":        round(float(y_te_log[i]),      6),
                    "predicted_return_Q50": round(float(preds_log[i, 1]), 6),
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
                "n_test":         len(X_te),
            })

    # ── Save predictions CSV ──────────────────────────────────────
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  Predictions saved  -> {pred_path}  ({len(pred_df)} rows)")

    # ── Append to cumulative evaluation_metrics.csv ───────────────
    metrics_path = os.path.join(cfg["results_dir"], "evaluation_metrics.csv")
    new_metrics  = pd.DataFrame(metrics_rows)
    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        mask = (existing["model_name"] == model_name) & \
               (existing["ticker"] == ticker)
        existing    = existing[~mask]
        new_metrics = pd.concat([existing, new_metrics], ignore_index=True)
    new_metrics.to_csv(metrics_path, index=False)
    print(f"  Metrics appended   -> {metrics_path}")

    return pred_df, new_metrics


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Thesis walk-forward evaluation"
    )
    parser.add_argument(
        "--model",
        default="price_only_lstm",
        choices=list(FEATURE_MAP.keys()),
        help="Model variant to evaluate",
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        choices=["AAPL", "MSFT", "NVDA"],
        help="Ticker symbol",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Return cached predictions if CSV exists; skip training.",
    )
    args = parser.parse_args()

    run_walk_forward_eval(
        args.model,
        args.ticker,
        skip_training=args.skip_training,
    )
