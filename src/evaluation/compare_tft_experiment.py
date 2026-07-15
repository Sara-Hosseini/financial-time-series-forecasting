# src/evaluation/compare_tft_experiment.py
# TFT Experiment Comparison
# ─────────────────────────────────────────────────────────────────
# Compares the Transformer Encoder (tft_quantile) against:
#   - sentiment_lstm_event
#   - sentiment_shift_lstm_event
#   - xgboost_baseline
#
# Reads ONLY from separate result files — does NOT touch evaluation_metrics.csv.
#
# Outputs:
#   results/revised_experiments/plots/tft_comparison_mape.png
#   results/revised_experiments/plots/tft_comparison_diracc.png
#   results/revised_experiments/plots/tft_comparison_coverage.png
#   results/revised_experiments/tft_comparison_table.csv
#
# Usage:
#   python -m src.evaluation.compare_tft_experiment

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import mape, dir_acc, coverage, interval_width

RESULTS_DIR   = "results/revised_experiments"
PRED_DIR      = os.path.join(RESULTS_DIR, "predictions")
TFT_CSV       = os.path.join(RESULTS_DIR, "tft_quantile_results.csv")
EVAL_CSV      = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")
PLOTS_DIR     = os.path.join(RESULTS_DIR, "plots")
OUTPUT_TABLE  = os.path.join(RESULTS_DIR, "tft_comparison_table.csv")

TICKERS  = ["AAPL", "MSFT", "NVDA"]
HORIZONS = [1, 5, 10]

# Models to include in the comparison
COMPARE_MODELS = [
    "tft_quantile",
    "sentiment_lstm_event",
    "sentiment_shift_lstm_event",
    "xgboost_baseline",
    "sentiment_lstm",           # kept for reference
    "price_only_lstm",          # kept for reference
]

MODEL_DISPLAY = {
    "tft_quantile":              "Transformer (24f)",
    "tft_quantile_13feat":       "Transformer (13f)",
    "sentiment_lstm_event":      "Sent+Event LSTM",
    "sentiment_shift_lstm_event":"SentShift+Ev LSTM",
    "xgboost_baseline":          "XGBoost",
    "sentiment_lstm":            "Sentiment LSTM",
    "price_only_lstm":           "Price-Only LSTM",
}

MODEL_COLORS = {
    "tft_quantile":              "#FF6B35",   # orange-red — new model stands out
    "tft_quantile_13feat":       "#FFB347",
    "sentiment_lstm_event":      "#CCBB44",
    "sentiment_shift_lstm_event":"#AA3377",
    "xgboost_baseline":          "#66CCEE",
    "sentiment_lstm":            "#EE6677",
    "price_only_lstm":           "#4477AA",
}


# ── Load data ─────────────────────────────────────────────────────

def _load_all_metrics() -> pd.DataFrame:
    """
    Load metrics from both the TFT-specific CSV and the main evaluation_metrics.csv.
    TFT results take precedence for TFT model names.
    """
    frames = []

    if os.path.exists(TFT_CSV):
        tft_df = pd.read_csv(TFT_CSV)
        frames.append(tft_df)
        print(f"  TFT results     : {len(tft_df)} rows ({TFT_CSV})")
    else:
        print(f"  WARNING: TFT results not found: {TFT_CSV}")
        print("  Run: python -m src.models.tft_quantile --ticker AAPL (and MSFT, NVDA)")

    if os.path.exists(EVAL_CSV):
        eval_df = pd.read_csv(EVAL_CSV)
        frames.append(eval_df)
        print(f"  Existing models : {len(eval_df)} rows ({EVAL_CSV})")
    else:
        print(f"  WARNING: evaluation_metrics.csv not found: {EVAL_CSV}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # Deduplicate: keep first occurrence per (model, ticker, horizon, fold)
    combined = combined.drop_duplicates(
        subset=["model_name", "ticker", "horizon", "fold"], keep="first"
    )
    return combined


def _load_pred(model_name: str, ticker: str) -> pd.DataFrame:
    path = os.path.join(PRED_DIR, f"{model_name}_{ticker}_predictions.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


# ── Summary table ─────────────────────────────────────────────────

def build_comparison_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, ticker, horizon): average metrics across all folds.
    """
    if metrics_df.empty:
        return pd.DataFrame()

    rows = []
    for model in COMPARE_MODELS:
        sub_m = metrics_df[metrics_df["model_name"] == model]
        if sub_m.empty:
            continue
        for ticker in TICKERS:
            sub_t = sub_m[sub_m["ticker"] == ticker]
            if sub_t.empty:
                continue
            for H in HORIZONS:
                sub_h = sub_t[sub_t["horizon"] == H]
                if sub_h.empty:
                    continue
                rows.append({
                    "model":          MODEL_DISPLAY.get(model, model),
                    "model_name":     model,
                    "ticker":         ticker,
                    "horizon":        H,
                    "avg_mape_pct":   round(sub_h["mape_pct"].mean(),       3),
                    "avg_coverage":   round(sub_h["coverage_pct"].mean(),    2),
                    "avg_dir_acc":    round(sub_h["dir_acc_pct"].mean(),     2),
                    "avg_width":      round(sub_h["interval_width"].mean(),  4),
                    "n_folds":        len(sub_h),
                })
    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────

def _bar_chart(
    table: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
):
    models    = [m for m in COMPARE_MODELS if m in table["model_name"].unique()]
    if not models:
        print(f"  [skip] No data for {title}")
        return

    x         = np.arange(len(HORIZONS))
    bar_width  = 0.7 / len(models)

    fig, axes = plt.subplots(1, len(TICKERS), figsize=(5.5 * len(TICKERS), 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    if len(TICKERS) == 1:
        axes = [axes]

    for ax, ticker in zip(axes, TICKERS):
        for j, model in enumerate(models):
            vals = []
            for H in HORIZONS:
                row = table[
                    (table["model_name"] == model) &
                    (table["ticker"]     == ticker) &
                    (table["horizon"]    == H)
                ]
                vals.append(row[metric].values[0] if len(row) else np.nan)

            offset = (j - len(models) / 2 + 0.5) * bar_width
            bars   = ax.bar(
                x + offset, vals, bar_width,
                label  = MODEL_DISPLAY.get(model, model),
                color  = MODEL_COLORS.get(model, "#999"),
                alpha  = 0.87,
                edgecolor="white",
            )
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2,
                            b.get_height() + 0.005,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=6)

        ax.set_title(ticker, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"H={h}d" for h in HORIZONS])
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=6.5, ncol=1)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def _print_summary_table(table: pd.DataFrame):
    """Print a console-friendly comparison for MAPE and DirAcc."""
    if table.empty:
        return
    print(f"\n  {'Model':<25}  {'Ticker':<5}  {'H':>4}  "
          f"{'MAPE%':>7}  {'Cov%':>6}  {'DirAcc%':>8}  {'Folds':>5}")
    print(f"  {'-'*70}")
    for _, row in table.iterrows():
        highlight = " <<" if row["model_name"] == "tft_quantile" else ""
        print(f"  {row['model']:<25}  {row['ticker']:<5}  "
              f"H={row['horizon']:>2}  "
              f"{row['avg_mape_pct']:>6.2f}%  "
              f"{row['avg_coverage']:>5.1f}%  "
              f"{row['avg_dir_acc']:>7.1f}%  "
              f"{row['n_folds']:>5}{highlight}")


# ── Quantile ordering check ───────────────────────────────────────

def check_quantile_ordering(ticker: str = "AAPL") -> dict:
    """
    Verify the monotone trick works: report % of rows where Q10<Q50<Q90.
    """
    pred_df = _load_pred("tft_quantile", ticker)
    if pred_df.empty:
        return {}

    q10 = pred_df["predicted_Q10"].values
    q50 = pred_df["predicted_Q50"].values
    q90 = pred_df["predicted_Q90"].values

    ordered = ((q10 < q50) & (q50 < q90)).mean() * 100
    return {"ticker": ticker, "pct_ordered": round(ordered, 2)}


# ── Main ──────────────────────────────────────────────────────────

def run_comparison():
    print(f"\n{'='*62}")
    print("  TFT Experiment Comparison")
    print(f"  TFT  vs  sentiment_lstm_event, xgboost_baseline, ...")
    print(f"{'='*62}")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    metrics_df = _load_all_metrics()
    if metrics_df.empty:
        print("  No data to compare. Train models first.")
        return

    # Quantile ordering check
    print("\n  Quantile ordering check (monotone trick):")
    for t in TICKERS:
        result = check_quantile_ordering(t)
        if result:
            pct = result["pct_ordered"]
            print(f"    {t}: {pct:.2f}% of rows satisfy Q10 < Q50 < Q90"
                  + (" [PERFECT]" if pct == 100.0 else f" [WARNING: {100-pct:.2f}% violations]"))

    # Build and print summary table
    table = build_comparison_table(metrics_df)
    if table.empty:
        print("  No comparable models found.")
        return

    _print_summary_table(table)

    # Save table
    table.to_csv(OUTPUT_TABLE, index=False)
    print(f"\n  Table saved -> {OUTPUT_TABLE}")

    # Generate plots
    print("\n  Generating comparison plots...")
    _bar_chart(
        table, "avg_mape_pct", "MAPE (%)",
        "MAPE Comparison: Transformer vs LSTM/XGBoost",
        "tft_comparison_mape.png",
    )
    _bar_chart(
        table, "avg_dir_acc", "Directional Accuracy (%)",
        "Directional Accuracy: Transformer vs LSTM/XGBoost",
        "tft_comparison_diracc.png",
    )
    _bar_chart(
        table, "avg_coverage", "Coverage (%)",
        "Quantile Coverage: Transformer vs LSTM/XGBoost",
        "tft_comparison_coverage.png",
    )

    # Quick verdict
    print("\n  Quick verdict (MAPE, all tickers, H=5d average):")
    h5 = table[table["horizon"] == 5]
    for _, row in h5.sort_values("avg_mape_pct").iterrows():
        marker = " << TFT" if row["model_name"] == "tft_quantile" else ""
        print(f"    {row['model']:<28} {row['ticker']}  "
              f"MAPE={row['avg_mape_pct']:.2f}%{marker}")


if __name__ == "__main__":
    run_comparison()
