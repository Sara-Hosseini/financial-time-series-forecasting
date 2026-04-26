# Load trained models and predict future prices for AAPL, MSFT, NVDA

import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import date

TICKERS  = ["AAPL", "MSFT", "NVDA"]
HORIZONS = [5, 10]
WINDOW   = 30

OUT_MODELS  = "outputs/models"
OUT_SCALERS = "outputs/scalers"
OUT_PLOTS   = "outputs/plots"

FEATURE_COLS = [
    "log_ret_1", "ret_5", "ret_10",
    "ma_ratio_5", "ma_ratio_10",
    "hl_range", "vol_10", "vol_20", "rsi_14",
]

def load_prices(csv_path):
    df = pd.read_csv(csv_path)
    first_cell = str(df.iloc[0, 0]).strip().lower() if len(df) >= 1 else ""
    if first_cell == "ticker":
        df = df.iloc[2:].copy()
    if "Price" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Price": "Date"})
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date").set_index("Date")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Close"])

def add_features(df):
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

def next_business_days(start_date, n):
    days = []
    current = start_date
    while len(days) < n:
        current += pd.Timedelta(days=1)
        if current.weekday() < 5:
            days.append(current)
    return days

def save_prediction_plot(ticker, hist_dates, hist_prices, results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hist_dates, hist_prices, color="steelblue", linewidth=2,
            label="Actual Price (last 30 days)")
    last_date  = hist_dates[-1]
    last_price = hist_prices[-1]
    ax.scatter(last_date, last_price, color="steelblue", s=80, zorder=5)
    colors = {5: "green", 10: "darkorange"}
    for r in results:
        H           = r["horizon"]
        pred_price  = r["pred_price"]
        target_date = pd.Timestamp(r["target_date"])
        direction   = r["direction"]
        change_pct  = r["change_pct"]
        ax.plot([last_date, target_date], [last_price, pred_price],
                color=colors[H], linewidth=1.5, linestyle="--", alpha=0.7)
        ax.scatter(target_date, pred_price, color=colors[H], s=120, zorder=5,
                   label=f"H={H}d -> ${pred_price:.2f} ({change_pct:+.2f}%) {direction}")
        ax.annotate(f"${pred_price:.2f}\n({change_pct:+.2f}%)",
                    xy=(target_date, pred_price), xytext=(10, 10),
                    textcoords="offset points", fontsize=10, color=colors[H],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=colors[H], alpha=0.8))
    ax.axvline(last_date, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title(f"{ticker} — Price Prediction (data up to {last_date.date()})", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price ($)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{ticker}_prediction_chart.png")
    plt.savefig(plot_path, dpi=180)
    plt.close()
    print(f"   Chart saved -> {plot_path}")
    return plot_path

if __name__ == "__main__":
    all_results = {}

    for TICKER in TICKERS:
        CSV_PATH = f"data/raw/{TICKER}_daily_10y_yahoo.csv"
        if not os.path.exists(CSV_PATH):
            print(f"\nSkipping {TICKER} — data file not found: {CSV_PATH}")
            continue

        print(f"\nLoading data: {CSV_PATH}")
        df_raw  = load_prices(CSV_PATH)
        df_feat = add_features(df_raw)

        last_date  = df_feat.index[-1]
        last_close = df_feat["Close"].iloc[-1]
        hist_dates  = df_feat.index[-30:]
        hist_prices = df_feat["Close"].values[-30:]
        X_last      = df_feat[FEATURE_COLS].values[-WINDOW:]

        print(f"\nLast trading day : {last_date.date()}")
        print(f"Last close price : ${last_close:.2f}")
        print(f"Today            : {date.today()}")

        print(f"\n{'='*50}")
        print(f"  PREDICTIONS FOR {TICKER}")
        print(f"{'='*50}")

        results = []
        for H in HORIZONS:
            model_path    = os.path.join(OUT_MODELS,  f"{TICKER}_H{H}_final.keras")
            scaler_X_path = os.path.join(OUT_SCALERS, f"{TICKER}_H{H}_scaler_X.joblib")
            scaler_y_path = os.path.join(OUT_SCALERS, f"{TICKER}_H{H}_scaler_y.joblib")

            if not os.path.exists(model_path):
                print(f"\n   H={H}: Model not found. Run train_final.py first.")
                continue

            model    = tf.keras.models.load_model(model_path)
            scaler_X = joblib.load(scaler_X_path)
            scaler_y = joblib.load(scaler_y_path)
            X_scaled = scaler_X.transform(X_last)
            X_input  = X_scaled.reshape(1, WINDOW, -1)

            y_pred_scaled = model.predict(X_input, verbose=0)
            y_pred        = scaler_y.inverse_transform(y_pred_scaled).flatten()[0]
            pred_price    = last_close * np.exp(y_pred)
            target_date   = next_business_days(last_date, H)[-1]
            change_pct    = (pred_price - last_close) / last_close * 100
            direction     = "UP" if pred_price > last_close else "DOWN"

            results.append({
                "horizon":     H,
                "target_date": target_date.date(),
                "last_close":  last_close,
                "pred_price":  pred_price,
                "change_pct":  change_pct,
                "direction":   direction,
            })

            print(f"\n  Horizon = {H} days")
            print(f"  Target date  : {target_date.date()}")
            print(f"  Current price: ${last_close:.2f}")
            print(f"  Pred price   : ${pred_price:.2f}")
            print(f"  Change       : {change_pct:+.2f}%  {direction}")

        if results:
            save_prediction_plot(TICKER, hist_dates, hist_prices, results, OUT_PLOTS)
            all_results[TICKER] = results

    # Cross-ticker comparison
    print(f"\n{'='*60}")
    print(f"  MULTI-TICKER COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Ticker':<8} {'H':>4} {'Pred Price':>12} {'Change':>10} {'Dir':>6}")
    print(f"  {'-'*44}")
    for ticker, res in all_results.items():
        for r in res:
            print(f"  {ticker:<8} {str(r['horizon'])+'d':>4} "
                  f"${r['pred_price']:<11.2f} "
                  f"{r['change_pct']:+.2f}%{'':4} "
                  f"{r['direction']}")