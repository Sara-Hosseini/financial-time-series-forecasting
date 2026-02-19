# ============================================================
# update_price_yahoo.py
# Downloads latest daily OHLCV from Yahoo Finance and
# MERGES with the existing CSV so no old data is lost.
#
# Usage:
#   python src/update_price_yahoo.py            → updates AAPL
#   python src/update_price_yahoo.py MSFT NVDA  → updates multiple tickers
# ============================================================

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEFAULT_TICKERS  = ["AAPL"]
DATA_DIR         = Path("data/raw")
HISTORY_START    = "2015-01-01"   # used only when no existing file found


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and index, drop bad rows."""
    # yfinance sometimes returns MultiIndex columns → flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()

    # Rename 'index' or 'Datetime' → 'Date'
    for alias in ("Datetime", "index", "Price"):
        if alias in df.columns and "Date" not in df.columns:
            df = df.rename(columns={alias: "Date"})

    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"])

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"])
    keep = ["Date"] + [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def _load_existing(path: Path) -> pd.DataFrame:
    """Load existing CSV; returns empty DataFrame if file doesn't exist."""
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Drop Yahoo Finance meta rows if present
    if len(df) >= 1 and str(df.iloc[0, 0]).strip().lower() == "ticker":
        df = df.iloc[2:].copy()

    if "Price" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Price": "Date"})
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"])
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = ["Date"] + [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[[c for c in keep if c in df.columns]]
    return df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)


# ─────────────────────────────────────────────
# Core update function
# ─────────────────────────────────────────────
def update_price_data(
    ticker: str,
    *,
    out_path: Path | None = None,
    history_start: str = HISTORY_START,
) -> pd.DataFrame:
    """
    1. Load existing CSV (if any).
    2. Determine the start date for the new download:
       - If existing data found → day after last date
       - Otherwise             → history_start
    3. Download from Yahoo Finance up to today.
    4. Merge old + new, deduplicate, sort, save.

    Returns the final merged DataFrame.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Default output path: same folder as the 10y file so LSTM can read it
    if out_path is None:
        out_path = DATA_DIR / f"{ticker}_daily_10y_yahoo.csv"

    existing = _load_existing(out_path)

    today = date.today().isoformat()

    if existing.empty:
        fetch_start = history_start
        print(f"[{ticker}] No existing file found. Downloading full history from {fetch_start}.")
    else:
        last_date = existing["Date"].max()
        fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"[{ticker}] Existing data up to {last_date.date()}. "
              f"Fetching {fetch_start} → {today} ...")

    if fetch_start > today:
        print(f"[{ticker}] Already up to date. Nothing to download.")
        return existing

    # ── Download
    new_df = yf.download(
        ticker,
        start=fetch_start,
        end=today,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if new_df is None or new_df.empty:
        print(f"[{ticker}] ⚠️  Yahoo returned no new data.")
        return existing

    new_df = _clean_df(new_df)
    print(f"[{ticker}] Downloaded {len(new_df)} new rows "
          f"({new_df['Date'].min().date()} → {new_df['Date'].max().date()})")

    # ── Merge
    if existing.empty:
        merged = new_df
    else:
        merged = (
            pd.concat([existing, new_df], ignore_index=True)
              .drop_duplicates(subset=["Date"])
              .sort_values("Date")
              .reset_index(drop=True)
        )

    # ── Save
    merged.to_csv(out_path, index=False)
    print(f"[{ticker}] ✅ Saved {len(merged)} rows → {out_path}")
    print(f"[{ticker}] Date range: {merged['Date'].min().date()} → {merged['Date'].max().date()}")
    print(merged.tail(3).to_string(index=False))

    return merged


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS

    for t in tickers:
        print(f"\n{'='*55}")
        update_price_data(t)

    print("\n✅  All tickers updated.")
