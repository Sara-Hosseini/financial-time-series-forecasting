# src/visualization/make_revised_plots.py
# Thesis Revised Plots
#
# Generates:
#   1. Distribution shift plots — KDE + approximate Gaussian curves
#      comparing price_only_lstm vs sentiment_shift_lstm (and others)
#   2. MAPE comparison bar chart (all models × horizons)
#   3. Directional Accuracy comparison bar chart
#   4. Coverage comparison bar chart
#   5. Statistical test summary table as PNG
#
# Input:
#   results/revised_experiments/predictions/*.csv
#   results/revised_experiments/evaluation_metrics.csv
#   results/revised_experiments/statistical_tests.csv  (if available)
#
# Output:
#   results/revised_experiments/plots/
#
# Usage:
#   python -m src.visualization.make_revised_plots

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats

RESULTS_DIR = "results/revised_experiments"
PRED_DIR    = os.path.join(RESULTS_DIR, "predictions")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")

TICKERS   = ["AAPL", "MSFT", "NVDA"]
HORIZONS  = [1, 5, 10]

MODEL_DISPLAY = {
    "price_only_lstm":      "Price-Only LSTM",
    "sentiment_lstm":       "Sentiment LSTM",
    "sentiment_shift_lstm": "SentimentShift LSTM",
    "xgboost_baseline":     "XGBoost Baseline",
}

MODEL_COLORS = {
    "price_only_lstm":      "#4477AA",
    "sentiment_lstm":       "#EE6677",
    "sentiment_shift_lstm": "#228833",
    "xgboost_baseline":     "#AA3377",
}

# Q90-Q10 = 2*1.2816*sigma for Gaussian
_GAUSSIAN_SCALE = 2.5632


# ── Loaders ───────────────────────────────────────────────────────

def _load_pred(model_name: str, ticker: str) -> pd.DataFrame:
    path = os.path.join(PRED_DIR, f"{model_name}_{ticker}_predictions.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def _load_metrics() -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def _load_stat_tests() -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, "statistical_tests.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


# ── Plot 1: Distribution Shift ────────────────────────────────────

def plot_distribution_shift():
    """
    For each ticker × horizon, compare Q50 prediction distributions
    of price_only_lstm vs sentiment_lstm vs sentiment_shift_lstm.
    Shows:
      - Kernel density estimate (KDE) of Q50 predictions
      - Approximate Gaussian curve (mean=Q50_mean, sigma from Q90-Q10 spread)
      - Vertical line at mean Q50
      - Shaded ±σ region
    """
    comparison_pairs = [
        ("price_only_lstm",  "sentiment_lstm",       "Price-Only vs Sentiment"),
        ("price_only_lstm",  "sentiment_shift_lstm",  "Price-Only vs SentShift"),
    ]

    for ticker in TICKERS:
        for H in HORIZONS:
            fig, axes = plt.subplots(1, len(comparison_pairs),
                                     figsize=(7 * len(comparison_pairs), 5),
                                     sharey=False)
            if len(comparison_pairs) == 1:
                axes = [axes]

            fig.suptitle(
                f"Predicted Q50 Distribution — {ticker}  H={H}d",
                fontsize=13, fontweight="bold",
            )

            for ax, (model_a, model_b, subtitle) in zip(axes, comparison_pairs):
                df_a = _load_pred(model_a, ticker)
                df_b = _load_pred(model_b, ticker)

                if df_a.empty or df_b.empty:
                    ax.text(0.5, 0.5, "Data not available",
                            ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(subtitle)
                    continue

                q50_a = df_a[df_a["horizon"] == H]["predicted_Q50"].values
                q50_b = df_b[df_b["horizon"] == H]["predicted_Q50"].values

                if len(q50_a) < 5 or len(q50_b) < 5:
                    ax.text(0.5, 0.5, "Insufficient data",
                            ha="center", va="center", transform=ax.transAxes)
                    continue

                # Determine common x range
                x_min = min(q50_a.min(), q50_b.min()) * 0.995
                x_max = max(q50_a.max(), q50_b.max()) * 1.005
                x     = np.linspace(x_min, x_max, 300)

                # For Gaussian approximation use average Q10/Q90 spread
                def _sigma_from_spread(df, H):
                    sub = df[df["horizon"] == H]
                    if sub.empty:
                        return None
                    w = (sub["predicted_Q90"] - sub["predicted_Q10"]).mean()
                    return w / _GAUSSIAN_SCALE

                sigma_a = _sigma_from_spread(df_a, H)
                sigma_b = _sigma_from_spread(df_b, H)
                mu_a    = q50_a.mean()
                mu_b    = q50_b.mean()

                col_a = MODEL_COLORS.get(model_a, "#4477AA")
                col_b = MODEL_COLORS.get(model_b, "#EE6677")
                lbl_a = MODEL_DISPLAY.get(model_a, model_a)
                lbl_b = MODEL_DISPLAY.get(model_b, model_b)

                # KDE
                kde_a = scipy_stats.gaussian_kde(q50_a)
                kde_b = scipy_stats.gaussian_kde(q50_b)
                ax.plot(x, kde_a(x), color=col_a, linewidth=2,
                        label=f"{lbl_a} (KDE)")
                ax.plot(x, kde_b(x), color=col_b, linewidth=2, linestyle="--",
                        label=f"{lbl_b} (KDE)")

                # Gaussian approximation
                if sigma_a and sigma_a > 0:
                    gauss_a = scipy_stats.norm.pdf(x, mu_a, sigma_a)
                    ax.plot(x, gauss_a, color=col_a, linewidth=1,
                            linestyle=":", alpha=0.7, label=f"{lbl_a} (Gauss)")
                if sigma_b and sigma_b > 0:
                    gauss_b = scipy_stats.norm.pdf(x, mu_b, sigma_b)
                    ax.plot(x, gauss_b, color=col_b, linewidth=1,
                            linestyle=":", alpha=0.7, label=f"{lbl_b} (Gauss)")

                # Mean vertical lines
                ax.axvline(mu_a, color=col_a, linewidth=1.5, linestyle="-",
                           alpha=0.8, label=f"μ_a = {mu_a:.1f}")
                ax.axvline(mu_b, color=col_b, linewidth=1.5, linestyle="--",
                           alpha=0.8, label=f"μ_b = {mu_b:.1f}")

                # Annotate shift
                shift = mu_b - mu_a
                ax.annotate(
                    f"ΔQ50 = {shift:+.2f}",
                    xy=(0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
                )

                ax.set_title(subtitle, fontsize=10)
                ax.set_xlabel("Predicted Closing Price ($)", fontsize=9)
                ax.set_ylabel("Density", fontsize=9)
                ax.legend(fontsize=7, loc="upper left")
                ax.grid(alpha=0.3)

            plt.tight_layout()
            out_path = os.path.join(
                PLOTS_DIR, f"distribution_shift_{ticker}_H{H}.png"
            )
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved -> {out_path}")


# ── Plot 2-4: Metric Comparison Bar Charts ────────────────────────

def _metric_bar_chart(metrics_df: pd.DataFrame,
                      metric_col: str,
                      ylabel: str,
                      title: str,
                      filename: str):
    """Generic grouped bar chart: horizons × models, one panel per ticker."""
    model_names = [m for m in MODEL_DISPLAY.keys() if m in metrics_df["model_name"].unique()]
    if not model_names:
        print(f"  No data for {title}")
        return

    n_models   = len(model_names)
    bar_width  = 0.7 / n_models
    x          = np.arange(len(HORIZONS))

    fig, axes = plt.subplots(1, len(TICKERS),
                             figsize=(5 * len(TICKERS), 5),
                             sharey=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    if len(TICKERS) == 1:
        axes = [axes]

    for ax, ticker in zip(axes, TICKERS):
        for i, model in enumerate(model_names):
            vals = []
            for H in HORIZONS:
                sub = metrics_df[
                    (metrics_df["model_name"] == model) &
                    (metrics_df["ticker"] == ticker) &
                    (metrics_df["horizon"] == H)
                ]
                if not sub.empty:
                    vals.append(sub[metric_col].mean())
                else:
                    vals.append(np.nan)

            offset = (i - n_models / 2 + 0.5) * bar_width
            bars   = ax.bar(x + offset, vals, bar_width,
                            color=MODEL_COLORS.get(model, "#888888"),
                            label=MODEL_DISPLAY.get(model, model),
                            alpha=0.85, edgecolor="white")

            # Value labels
            for bar_rect, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(
                        bar_rect.get_x() + bar_rect.get_width() / 2,
                        bar_rect.get_height() + 0.001,
                        f"{val:.1f}",
                        ha="center", va="bottom", fontsize=6.5,
                    )

        ax.set_title(ticker, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"H={h}d" for h in HORIZONS])
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_mape_comparison(metrics_df: pd.DataFrame):
    _metric_bar_chart(
        metrics_df, "mape_pct", "MAPE (%)",
        "MAPE Comparison — All Models",
        "mape_comparison.png",
    )


def plot_diracc_comparison(metrics_df: pd.DataFrame):
    _metric_bar_chart(
        metrics_df, "dir_acc_pct", "Directional Accuracy (%)",
        "Directional Accuracy Comparison",
        "diracc_comparison.png",
    )


def plot_coverage_comparison(metrics_df: pd.DataFrame):
    _metric_bar_chart(
        metrics_df, "coverage_pct", "Coverage (%)",
        "Quantile Interval Coverage Comparison",
        "coverage_comparison.png",
    )


# ── Plot 5: Statistical Test Summary Table ────────────────────────

def plot_stat_test_table(stat_df: pd.DataFrame):
    if stat_df.empty:
        print("  No statistical test results. Skipping table plot.")
        return

    display_cols = [
        "ticker", "horizon", "model_a", "model_b",
        "mean_Q50_difference", "p_value_Q50", "p_value_error", "dm_p_value",
    ]
    tbl = stat_df[[c for c in display_cols if c in stat_df.columns]].copy()

    # Shorten model names for table display
    short = {
        "price_only_lstm":      "PriceOnly",
        "sentiment_lstm":       "Sentiment",
        "sentiment_shift_lstm": "SentShift",
        "xgboost_baseline":     "XGBoost",
    }
    for col in ["model_a", "model_b"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].map(short).fillna(tbl[col])

    # Colour cells by significance
    def _cell_colors(df):
        colors = []
        for _, row in df.iterrows():
            row_c = []
            for col in df.columns:
                val = row[col]
                if col in ("p_value_Q50", "p_value_error", "dm_p_value"):
                    try:
                        v = float(val)
                        if v < 0.01:
                            row_c.append("#c8e6c9")   # green
                        elif v < 0.05:
                            row_c.append("#fff9c4")   # yellow
                        elif v < 0.10:
                            row_c.append("#ffe0b2")   # orange
                        else:
                            row_c.append("white")
                    except (TypeError, ValueError):
                        row_c.append("white")
                else:
                    row_c.append("white")
            colors.append(row_c)
        return colors

    fig_h = max(3.5, len(tbl) * 0.35 + 1.2)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    col_labels = [c.replace("_", " ").title() for c in tbl.columns]
    cell_text  = tbl.astype(str).values.tolist()
    cell_cols  = _cell_colors(tbl)

    tbl_obj = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_cols,
        loc="center",
        cellLoc="center",
    )
    tbl_obj.auto_set_font_size(False)
    tbl_obj.set_fontsize(8)
    tbl_obj.auto_set_column_width(list(range(len(col_labels))))

    for (r, c), cell in tbl_obj.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a3a5c")
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#cccccc")

    # Legend
    patches = [
        mpatches.Patch(color="#c8e6c9", label="p < 0.01"),
        mpatches.Patch(color="#fff9c4", label="p < 0.05"),
        mpatches.Patch(color="#ffe0b2", label="p < 0.10"),
        mpatches.Patch(color="white",   label="p ≥ 0.10"),
    ]
    ax.legend(handles=patches, loc="upper right",
              fontsize=7, title="Significance", title_fontsize=7)

    fig.suptitle("Statistical Test Summary (p-value colouring by significance)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "statistical_tests_table.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────

def make_all_plots():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"\n{'='*62}")
    print("  Generating Thesis Revised Plots")
    print(f"{'='*62}")

    metrics_df  = _load_metrics()
    stat_df     = _load_stat_tests()

    print("\n  [1/5] Distribution shift plots...")
    plot_distribution_shift()

    if not metrics_df.empty:
        print("\n  [2/5] MAPE comparison...")
        plot_mape_comparison(metrics_df)

        print("\n  [3/5] Directional Accuracy comparison...")
        plot_diracc_comparison(metrics_df)

        print("\n  [4/5] Coverage comparison...")
        plot_coverage_comparison(metrics_df)
    else:
        print("\n  No evaluation_metrics.csv found. "
              "Skipping metric bar charts.")

    print("\n  [5/5] Statistical test summary table...")
    plot_stat_test_table(stat_df)

    print(f"\n  All plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    make_all_plots()
