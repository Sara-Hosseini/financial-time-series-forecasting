# src/utils.py
# ─────────────────────────────────────────────────────────────
# توابع مشترک — همه فایل‌های دیگه از اینجا import میکنن
# load_prices, add_features, make_sequences, metrics
# ─────────────────────────────────────────────────────────────

import os
import json
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ─────────────────────────────────────────────
# ستون‌های feature — یه بار تعریف میشه
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "log_ret_1", "ret_5", "ret_10",
    "ma_ratio_5", "ma_ratio_10",
    "hl_range", "vol_10", "vol_20", "rsi_14",
]


# ─────────────────────────────────────────────
# بارگذاری داده
# ─────────────────────────────────────────────
def load_prices(csv_path: str, date_format: str = "%Y-%m-%d") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    first_cell = str(df.iloc[0, 0]).strip().lower() if len(df) >= 1 else ""
    if first_cell == "ticker":
        df = df.iloc[2:].copy()
    if "Price" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Price": "Date"})
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], format=date_format, errors="coerce")
    df = (
        df.dropna(subset=["Date"])
          .sort_values("Date")
          .drop_duplicates(subset=["Date"], keep="last")
          .set_index("Date")
    )
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"])
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].dropna()


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_ret_1"]   = np.log(out["Close"] / out["Close"].shift(1))
    out["ret_5"]       = np.log(out["Close"] / out["Close"].shift(5))
    out["ret_10"]      = np.log(out["Close"] / out["Close"].shift(10))
    out["ma_ratio_5"]  = out["Close"].rolling(5).mean()  / (out["Close"] + 1e-12) - 1
    out["ma_ratio_10"] = out["Close"].rolling(10).mean() / (out["Close"] + 1e-12) - 1
    out["hl_range"]    = (out["High"] - out["Low"]) / (out["Close"] + 1e-12)
    out["vol_10"]      = out["log_ret_1"].rolling(10).std()
    out["vol_20"]      = out["log_ret_1"].rolling(20).std()
    delta = out["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-12)))
    return out.dropna()


def make_target(df: pd.DataFrame, H: int) -> pd.Series:
    return np.log(df["Close"].shift(-H) / (df["Close"] + 1e-12)).rename("target")


# ─────────────────────────────────────────────
# ساخت sequence برای LSTM
# ─────────────────────────────────────────────
def make_sequences(
    X_scaled: np.ndarray,
    y_1d: np.ndarray,
    idx_start: int,
    idx_end: int,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, ys, idxs = [], [], []
    for t in range(idx_start, idx_end):
        if t < window:
            continue
        Xs.append(X_scaled[t - window: t, :])
        ys.append(y_1d[t])
        idxs.append(t)
    return np.array(Xs), np.array(ys).reshape(-1, 1), np.array(idxs)


# ─────────────────────────────────────────────
# معیارهای ارزیابی
# ─────────────────────────────────────────────
def rmse(a, b) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))

def mae_m(a, b) -> float:
    return float(mean_absolute_error(a, b))

def mape(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    mask = np.abs(a) > 1e-8
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((b[mask] - a[mask]) / a[mask])) * 100.0)

def dir_acc(c_t, c_true, c_pred) -> float:
    c_t, c_true, c_pred = map(np.asarray, (c_t, c_true, c_pred))
    actual = np.sign(c_true - c_t)
    pred   = np.sign(c_pred - c_t)
    mask   = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(actual[mask] == pred[mask]) * 100.0)

def coverage(y_true, y_low, y_high) -> float:
    """درصد مقادیر واقعی داخل بازه — برای Quantile Regression"""
    inside = np.sum((y_true >= y_low) & (y_true <= y_high))
    return float(inside / len(y_true) * 100)

def interval_width(y_low, y_high) -> float:
    """میانگین عرض بازه — برای Quantile Regression"""
    return float(np.mean(y_high - y_low))


# ─────────────────────────────────────────────
# ابزارهای کمکی
# ─────────────────────────────────────────────
def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def next_business_days(start_date, n: int):
    days = []
    current = start_date
    while len(days) < n:
        current += pd.Timedelta(days=1)
        if current.weekday() < 5:
            days.append(current)
    return days