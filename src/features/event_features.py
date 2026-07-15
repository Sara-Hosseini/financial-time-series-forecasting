# src/features/event_features.py
# Market-Moving Event Feature Extraction
#
# Reads article-level news from data/processed/sentiment/news_raw.csv,
# applies 9 keyword rule sets to the pre-combined `text` column (title + summary),
# and produces a daily ticker-level feature table with:
#   - 9 binary event flags  (1 = at least one article that day triggered that rule)
#   - event_count           (number of distinct event types detected that day, 0-9)
#   - event_intensity       (weighted event score + 0.5 * log(1 + n_raw_articles))
#
# All outputs: results/revised_experiments/event_features/event_features_daily.csv
#
# Usage:
#   python -m src.features.event_features

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

NEWS_PATH  = "data/processed/sentiment/news_raw.csv"
OUTPUT_DIR = "results/revised_experiments/event_features"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "event_features_daily.csv")

TICKERS = ["AAPL", "MSFT", "NVDA"]


# ── Keyword rule sets ─────────────────────────────────────────────
# Each value is a list of lowercase keyword strings.
# Single-word tokens use word-boundary matching (\b...\b).
# Multi-word phrases use plain substring matching (specific enough to avoid noise).

_KEYWORD_RULES = {
    "earnings_event": [
        "earnings", "quarterly result", "quarter result", "eps",
        "revenue beat", "revenue miss", "profit", "guidance", "outlook",
        "beats estimate", "misses estimate", "reports result",
        "beat expectation", "missed expectation", "above consensus",
        "below consensus", "quarterly earnings", "net income",
        "gross margin", "fiscal quarter", "annual result",
    ],
    "analyst_upgrade": [
        "upgrade", "upgraded", "raises rating", "buy rating", "outperform",
        "price target raised", "target raised", "raises target", "overweight",
        "strong buy", "initiates coverage", "raised price target",
        "increases target", "bullish on",
    ],
    "analyst_downgrade": [
        "downgrade", "downgraded", "cuts rating", "sell rating", "underperform",
        "price target cut", "target cut", "lowers target", "underweight",
        "reduces target", "cuts price target", "lowered price target",
        "bearish on",
    ],
    "product_launch": [
        "launch", "launches", "launched", "unveils", "unveiled",
        "introduces", "introduced", "releases new", "new product",
        "ai chip", "new gpu", "new iphone", "cloud product",
        "new platform", "new model", "new version", "new service",
        "announces new", "product announcement",
    ],
    "lawsuit_event": [
        "lawsuit", "sued", "sues", "legal action", "settlement",
        "class action", "antitrust lawsuit", "investigation", "probe",
        "litigation", "court ruling", "legal challenge", "legal dispute",
        "jury verdict",
    ],
    "regulatory_event": [
        "ftc", "doj", "eu commission", "antitrust",
        "fine", "penalty", "regulatory approval", "export ban",
        "sanctions", "compliance issue", "regulatory", "regulator",
        "banned", "trade restriction", "data privacy",
    ],
    "partnership_event": [
        "partnership", "partners with", "collaboration", "collaborates",
        "acquisition", "acquires", "merger", "new deal", "contract",
        "new agreement", "joint venture", "strategic alliance",
        "signs deal", "signs agreement",
    ],
    "management_change_event": [
        "new ceo", "new cfo", "new cto", "new coo",
        "resignation", "resigns", "appointed as", "appoints new",
        "management change", "board of directors", "leadership change",
        "steps down", "new chief", "named president", "executive departure",
    ],
    "macro_event": [
        "federal reserve", "rate hike", "rate cut", "inflation",
        "cpi report", "gdp report", "recession", "jobs report",
        "unemployment rate", "bond yields", "treasury yields",
        "fomc", "monetary policy", "interest rate decision",
        "fed raises", "fed cuts",
    ],
}

# Higher weight = more market-moving event type.
# Used in event_intensity calculation.
_EVENT_WEIGHTS = {
    "earnings_event":          2.0,
    "regulatory_event":        1.8,
    "lawsuit_event":           1.5,
    "macro_event":             1.5,
    "analyst_upgrade":         1.2,
    "analyst_downgrade":       1.2,
    "management_change_event": 1.0,
    "partnership_event":       0.8,
    "product_launch":          0.8,
}

EVENT_COLS = list(_KEYWORD_RULES.keys())   # 9 binary flag columns


# ── Compile regex patterns once at import time ────────────────────
# Multi-word phrases → plain substring check (fast, specific enough).
# Single-word tokens → \b word-boundary regex (avoids subword matches).

def _compile(keywords: list) -> re.Pattern:
    parts = []
    for kw in keywords:
        if " " in kw:
            # multi-word: plain substring, escaped for safety
            parts.append(re.escape(kw))
        else:
            parts.append(r"\b" + re.escape(kw) + r"\b")
    return re.compile("|".join(parts))

_PATTERNS = {evt: _compile(kws) for evt, kws in _KEYWORD_RULES.items()}


# ── Article-level detection ───────────────────────────────────────

def detect_events(text: str) -> dict:
    """
    Given a lowercased article text, return a dict of binary flags
    (1 if any keyword in that rule set was found, else 0).
    """
    if not isinstance(text, str) or not text.strip():
        return {evt: 0 for evt in EVENT_COLS}
    return {
        evt: int(bool(_PATTERNS[evt].search(text)))
        for evt in EVENT_COLS
    }


# ── Daily aggregation ─────────────────────────────────────────────

def build_event_features(df_articles: pd.DataFrame) -> pd.DataFrame:
    """
    Takes article-level DataFrame (must have columns: ticker, date, text).
    Returns daily ticker-level DataFrame with event flags + event_count + event_intensity.
    """
    df = df_articles[["ticker", "date", "text"]].copy()

    # Lowercase text for matching; fill nulls with empty string.
    df["text_lower"] = df["text"].fillna("").str.lower()

    # Detect event flags per article
    flag_df = df["text_lower"].apply(detect_events).apply(pd.Series)
    df = pd.concat([df[["ticker", "date"]], flag_df], axis=1)

    # Count raw articles per ticker-day (before aggregation)
    n_articles = (
        df.groupby(["ticker", "date"])
        .size()
        .reset_index(name="n_raw_articles")
    )

    # Aggregate to daily level
    # Binary flags: max() → 1 if at least one article triggered the rule
    daily = (
        df.groupby(["ticker", "date"])[EVENT_COLS]
        .max()
        .reset_index()
    )

    daily = daily.merge(n_articles, on=["ticker", "date"], how="left")

    # event_count: number of distinct event types detected that day (0–9)
    daily["event_count"] = daily[EVENT_COLS].sum(axis=1)

    # event_intensity: weighted event score + 0.5 * log(1 + n_articles)
    # Earnings/regulatory/lawsuit events count more than product/partnership.
    weighted_score = sum(
        _EVENT_WEIGHTS.get(col, 1.0) * daily[col]
        for col in EVENT_COLS
    )
    daily["event_intensity"] = (
        weighted_score + 0.5 * np.log1p(daily["n_raw_articles"])
    ).round(4)

    # Drop the helper column; not needed downstream
    daily.drop(columns=["n_raw_articles"], inplace=True)

    return daily.sort_values(["ticker", "date"]).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────

def run_event_feature_extraction() -> pd.DataFrame:
    print(f"\n{'='*62}")
    print("  Event Feature Extraction")
    print(f"{'='*62}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load article-level news
    print(f"\n  Loading {NEWS_PATH} ...")
    df_raw = pd.read_csv(NEWS_PATH)
    print(f"  {len(df_raw)} total articles | columns: {list(df_raw.columns)}")

    # Filter to the 3 model tickers only
    df_raw = df_raw[df_raw["ticker"].isin(TICKERS)].copy()
    print(f"  {len(df_raw)} articles after filtering to {TICKERS}")

    # Validate required columns
    for col in ["ticker", "date", "text"]:
        if col not in df_raw.columns:
            raise ValueError(
                f"Required column '{col}' not found in {NEWS_PATH}. "
                f"Available: {list(df_raw.columns)}"
            )

    # Show article count per ticker
    print("\n  Articles per ticker:")
    for t, n in df_raw["ticker"].value_counts().items():
        print(f"    {t}: {n}")

    # Build event features
    print("\n  Detecting event types (regex keyword matching)...")
    daily_df = build_event_features(df_raw)

    # Coverage stats
    print("\n  Daily event coverage per ticker:")
    for t in TICKERS:
        sub = daily_df[daily_df["ticker"] == t]
        event_days = (sub["event_count"] > 0).sum()
        total_days  = len(sub)
        print(f"    {t}: {total_days} news-days, "
              f"{event_days} ({100*event_days/max(total_days,1):.1f}%) "
              f"have >=1 event type")

    # Per-event-type frequency
    print("\n  Event type frequencies (any-ticker total):")
    for col in EVENT_COLS:
        n = daily_df[col].sum()
        print(f"    {col:<30} {int(n):>4} days")

    # Save
    daily_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved {len(daily_df)} daily rows -> {OUTPUT_CSV}")

    return daily_df


if __name__ == "__main__":
    run_event_feature_extraction()
