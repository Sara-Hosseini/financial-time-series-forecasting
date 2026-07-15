# src/models/tft_quantile.py
# Lightweight Transformer Encoder with Quantile Output
# ─────────────────────────────────────────────────────
# Architecture:
#   Input (batch, window=30, n_features=24)
#     → Linear projection  → (batch, 30, d_model=64)
#     + Learnable positional embedding
#     → TransformerEncoder (2 layers, 4 heads, FFN=128, GELU)
#     → Last time-step      → (batch, 64)
#     → Dense(64, GELU)
#     → Dense(3) with monotone quantile trick
#   Output: Q10, Q50, Q90   (ordering guaranteed: Q10 < Q50 < Q90 always)
#   Loss: Pinball (quantile regression loss)
#
# Key design choices:
#   - No new package installs — PyTorch is already available
#   - Monotone trick: Q50 = Q10 + softplus(b); Q90 = Q50 + softplus(c)
#   - Same walk-forward folds as LSTM (2021–2025, year-based)
#   - Same 24-feature event-enhanced input as sentiment_lstm_event
#   - Results saved separately; evaluation_metrics.csv is NOT modified
#
# Usage:
#   python -m src.models.tft_quantile --ticker AAPL
#   python -m src.models.tft_quantile --ticker MSFT
#   python -m src.models.tft_quantile --ticker NVDA
#   python -m src.models.tft_quantile --ticker AAPL --features 13  # price+sentiment only

import os
import sys
import math
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.utils import (
    add_features, make_target, make_sequences,
    mape, dir_acc, coverage, interval_width, ensure_dirs,
)

# ── Reproducibility ───────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Feature column definitions ────────────────────────────────────
PRICE_FEATURE_COLS = [
    "log_ret_1", "ret_5", "ret_10",
    "ma_ratio_5", "ma_ratio_10",
    "hl_range", "vol_10", "vol_20", "rsi_14",
]
CORE_SENTIMENT_COLS = [
    "avg_sentiment_score", "positive_ratio", "negative_ratio", "news_count",
]
EVENT_FEATURE_COLS = [
    "earnings_event", "analyst_upgrade", "analyst_downgrade",
    "product_launch", "lawsuit_event", "regulatory_event",
    "partnership_event", "management_change_event", "macro_event",
    "event_count", "event_intensity",
]
ALL_SENTIMENT_COLS = [
    "news_count", "avg_sentiment_score", "avg_positive_prob", "avg_negative_prob",
    "avg_neutral_prob", "max_positive_prob", "max_negative_prob", "pos_news_count",
    "neg_news_count", "neu_news_count", "sentiment_balance", "positive_ratio",
    "negative_ratio", "neutral_ratio", "has_news",
]

FEATURES_24 = PRICE_FEATURE_COLS + CORE_SENTIMENT_COLS + EVENT_FEATURE_COLS  # 24
FEATURES_13 = PRICE_FEATURE_COLS + CORE_SENTIMENT_COLS                        # 13

EVENT_DATASET  = "results/revised_experiments/event_features/price_sentiment_event_dataset.csv"
RESULTS_DIR    = "results/revised_experiments"
PRED_DIR       = os.path.join(RESULTS_DIR, "predictions")
RESULTS_CSV    = os.path.join(RESULTS_DIR, "tft_quantile_results.csv")

# ── Training hyper-parameters ─────────────────────────────────────
DEFAULT_CFG = dict(
    window        = 30,
    d_model       = 64,
    nhead         = 4,
    num_layers    = 2,
    dim_ff        = 128,
    dropout       = 0.1,
    lr            = 1e-4,
    weight_decay  = 1e-4,
    epochs        = 150,
    batch_size    = 32,
    es_patience   = 20,
    lr_patience   = 6,
    lr_factor     = 0.5,
    lr_min        = 1e-6,
    val_ratio     = 0.10,
    min_train_rows= 500,
    test_years    = (2021, 2022, 2023, 2024, 2025),
    horizons      = (1, 5, 10),
    quantiles     = (0.1, 0.5, 0.9),
)


# ── Model ─────────────────────────────────────────────────────────

class QuantileTransformer(nn.Module):
    """
    Transformer Encoder → Q10/Q50/Q90 quantile forecaster.

    Input : (batch, window, n_features)
    Output: (batch, 3) — [Q10, Q50, Q90] with guaranteed ordering
    """

    def __init__(self, n_features: int, cfg: dict):
        super().__init__()
        d   = cfg["d_model"]
        win = cfg["window"]

        # Project raw features into model dimension
        self.input_proj = nn.Linear(n_features, d)

        # Learnable positional encoding (one embedding per time step)
        self.pos_emb = nn.Embedding(win, d)

        # Transformer encoder stack
        enc_layer = nn.TransformerEncoderLayer(
            d_model        = d,
            nhead          = cfg["nhead"],
            dim_feedforward= cfg["dim_ff"],
            dropout        = cfg["dropout"],
            activation     = "gelu",
            batch_first    = True,
            norm_first     = True,   # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg["num_layers"])

        # Output head
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(d, 3),   # raw outputs — processed by monotone trick below
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        batch, seq_len, _ = x.shape
        device = x.device

        # Feature projection + positional encoding
        x = self.input_proj(x)                                      # (B, T, d)
        pos = torch.arange(seq_len, device=device)
        x   = x + self.pos_emb(pos)                                 # broadcast over batch

        # Transformer
        x = self.encoder(x)                                         # (B, T, d)

        # Aggregate: use last time step (most recent is richest in temporal context)
        x = x[:, -1, :]                                             # (B, d)

        # Head → raw 3-vector
        raw = self.head(x)                                          # (B, 3)

        # Monotone quantile trick: ensures Q10 < Q50 < Q90 everywhere
        q10 = raw[:, 0]
        q50 = q10 + F.softplus(raw[:, 1])
        q90 = q50 + F.softplus(raw[:, 2])

        return torch.stack([q10, q50, q90], dim=1)                  # (B, 3)


# ── Loss ──────────────────────────────────────────────────────────

def pinball_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    quantiles: tuple = (0.1, 0.5, 0.9),
) -> torch.Tensor:
    """
    Quantile (pinball) loss.
    y_pred: (batch, 3)
    y_true: (batch, 1) or (batch,)
    """
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(1)
    y_true = y_true.expand_as(y_pred)
    q = torch.tensor(quantiles, dtype=y_pred.dtype, device=y_pred.device)
    errors = y_true - y_pred
    return torch.mean(torch.max(q * errors, (q - 1.0) * errors))


# ── Early stopping ────────────────────────────────────────────────

class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 1e-6):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ── Data loading ──────────────────────────────────────────────────

def _load_features(ticker: str, n_features: int) -> tuple:
    """
    Load the correct dataset for the chosen feature count.

    n_features=24 → event-enhanced dataset (price + sentiment + event)
    n_features=13 → original merged dataset (price + sentiment only)

    Returns (df, feature_col_list)
    """
    if n_features == 24:
        feature_cols = FEATURES_24
        if not os.path.exists(EVENT_DATASET):
            raise FileNotFoundError(
                f"Event dataset not found: {EVENT_DATASET}\n"
                "Run: python -m src.features.merge_event_features"
            )
        df_all = pd.read_csv(EVENT_DATASET)
        df = df_all[df_all["ticker"] == ticker].copy()
        if df.empty:
            raise ValueError(f"No rows for ticker '{ticker}' in {EVENT_DATASET}")
        date_col = "date" if "date" in df.columns else "Date"
    else:
        feature_cols = FEATURES_13
        merged_path  = f"outputs/{ticker}_price_sentiment_merged.csv"
        if not os.path.exists(merged_path):
            raise FileNotFoundError(f"Merged CSV not found: {merged_path}")
        df       = pd.read_csv(merged_path)
        date_col = "Date" if "Date" in df.columns else "date"

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
    df = df.dropna(subset=["Close"])

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for {ticker}: {missing}")

    return df, feature_cols


# ── Training helper ───────────────────────────────────────────────

def _train_one_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    cfg: dict,
    device: torch.device,
    n_features: int,
) -> nn.Module:
    """Train a fresh QuantileTransformer on one walk-forward fold."""

    model = QuantileTransformer(n_features=n_features, cfg=cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg["lr_patience"],
        factor=cfg["lr_factor"], min_lr=cfg["lr_min"],
    )
    stopper = EarlyStopper(patience=cfg["es_patience"])

    # Convert to tensors
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).squeeze()
    Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
    yv = torch.tensor(y_va, dtype=torch.float32).squeeze().to(device)

    loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    for epoch in range(cfg["epochs"]):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = pinball_loss(pred, yb, cfg["quantiles"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = pinball_loss(val_pred, yv, cfg["quantiles"]).item()

        scheduler.step(val_loss)
        if stopper.step(val_loss, model):
            break

    stopper.restore_best(model)
    return model


# ── Walk-forward evaluation ───────────────────────────────────────

def run_tft_walk_forward(
    ticker: str,
    n_features: int = 24,
    cfg: dict = None,
) -> tuple:
    """
    Walk-forward evaluation for the Transformer model.

    Saves:
      results/revised_experiments/predictions/tft_quantile_{ticker}_predictions.csv
      results/revised_experiments/tft_quantile_results.csv  (appended)

    Returns (pred_df, metrics_df).
    """
    if cfg is None:
        cfg = DEFAULT_CFG

    model_name = f"tft_quantile" if n_features == 24 else "tft_quantile_13feat"
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*62}")
    print(f"  Transformer Quantile | {ticker} | {n_features} features | {device}")
    print(f"{'='*62}")

    ensure_dirs(PRED_DIR, RESULTS_DIR)

    df_feat, feature_cols = _load_features(ticker, n_features)
    print(f"  Dataset: {len(df_feat)} rows "
          f"({df_feat.index[0].date()} -> {df_feat.index[-1].date()})")

    pred_rows    = []
    metrics_rows = []

    for H in cfg["horizons"]:
        print(f"\n  Horizon H={H}d  {'-'*45}")

        target  = make_target(df_feat, H)
        df_work = df_feat.loc[target.dropna().index].copy()
        y_raw   = target.loc[df_work.index].values
        X_raw   = df_work[feature_cols].values
        dates   = df_work.index
        closes  = df_work["Close"].values

        for year in cfg["test_years"]:
            t0 = time.time()

            test_mask  = dates.year == year
            train_mask = dates.year < year
            train_idx  = np.where(train_mask)[0]
            test_idx   = np.where(test_mask)[0]

            if len(train_idx) < cfg["min_train_rows"] or len(test_idx) == 0:
                print(f"    Fold {year}: skipped")
                continue

            val_cut   = int(len(train_idx) * (1 - cfg["val_ratio"]))
            train_end = train_idx[val_cut]
            val_end   = train_idx[-1] + 1

            # Scale (fit on train only)
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
                  f"val={len(X_va)} test={len(X_te)} ...", end="", flush=True)

            # Train
            torch.manual_seed(SEED)
            model = _train_one_fold(X_tr, y_tr, X_va, y_va, cfg, device, n_features)

            # Predict on test set
            model.eval()
            with torch.no_grad():
                Xte_t = torch.tensor(X_te, dtype=torch.float32).to(device)
                preds_s = model(Xte_t).cpu().numpy()      # (n_test, 3) — scaled log-returns

            # Inverse-scale
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

            elapsed = time.time() - t0
            print(f"  MAPE={fold_mape:.2f}%  Cov={fold_cov:.1f}%  "
                  f"Width=${fold_width:.2f}  DirAcc={fold_dir:.1f}%  "
                  f"[{elapsed:.0f}s]")

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

    # ── Save predictions ──────────────────────────────────────────
    pred_df   = pd.DataFrame(pred_rows)
    pred_path = os.path.join(PRED_DIR, f"{model_name}_{ticker}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  Predictions saved  -> {pred_path}  ({len(pred_df)} rows)")

    # ── Save / append TFT-specific results (NOT evaluation_metrics.csv) ──
    new_metrics = pd.DataFrame(metrics_rows)
    if os.path.exists(RESULTS_CSV):
        existing = pd.read_csv(RESULTS_CSV)
        mask = (existing["model_name"] == model_name) & (existing["ticker"] == ticker)
        existing    = existing[~mask]
        new_metrics = pd.concat([existing, new_metrics], ignore_index=True)
    new_metrics.to_csv(RESULTS_CSV, index=False)
    print(f"  Metrics saved      -> {RESULTS_CSV}")
    print(f"  NOTE: evaluation_metrics.csv was NOT modified.")

    return pred_df, new_metrics


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transformer Encoder quantile forecaster (experimental)"
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        choices=["AAPL", "MSFT", "NVDA"],
    )
    parser.add_argument(
        "--features",
        type=int,
        default=24,
        choices=[13, 24],
        help="Number of input features: 24 (price+sentiment+event) or 13 (price+sentiment)",
    )
    # Optional hyper-parameter overrides
    parser.add_argument("--d-model",    type=int,   default=DEFAULT_CFG["d_model"])
    parser.add_argument("--num-layers", type=int,   default=DEFAULT_CFG["num_layers"])
    parser.add_argument("--nhead",      type=int,   default=DEFAULT_CFG["nhead"])
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_CFG["epochs"])
    parser.add_argument("--lr",         type=float, default=DEFAULT_CFG["lr"])
    args = parser.parse_args()

    cfg = {**DEFAULT_CFG,
           "d_model":    args.d_model,
           "num_layers": args.num_layers,
           "nhead":      args.nhead,
           "epochs":     args.epochs,
           "lr":         args.lr}

    run_tft_walk_forward(args.ticker, n_features=args.features, cfg=cfg)
