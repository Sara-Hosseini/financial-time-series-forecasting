# src/visualization/make_event_plots.py
# Event Feature Visualizations — Thesis
#
# Generates 5 plots saved to results/revised_experiments/plots/event_features/:
#   1. event_coverage_by_ticker.png   — % of trading days with each event type
#   2. event_type_frequency.png       — article-level event counts by ticker
#   3. subgroup_mape_comparison.png   — MAPE on all/news/event/sentiment subgroups
#   4. subgroup_diracc_comparison.png — Directional accuracy by subgroup
#   5. distribution_shift_event_days.png — Q50 KDE on event days only
#
# Usage:
#   python -m src.visualization.make_event_plots

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

RESULTS_DIR   = "results/revised_experiments"
PRED_DIR      = os.path.join(RESULTS_DIR, "predictions")
EVENT_DIR     = os.path.join(RESULTS_DIR, "event_features")
PLOTS_DIR     = os.path.join(RESULTS_DIR, "plots", "event_features")

EVENT_CSV     = os.path.join(EVENT_DIR, "event_features_daily.csv")
SUBGROUP_CSV  = os.path.join(EVENT_DIR, "subgroup_evaluation.csv")

TICKERS  = ["AAPL", "MSFT", "NVDA"]
HORIZONS = [1, 5, 10]

EVENT_COLS = [
    "earnings_event", "analyst_upgrade", "analyst_downgrade",
    "product_launch", "lawsuit_event", "regulatory_event",
    "partnership_event", "management_change_event", "macro_event",
]

MODEL_DISPLAY = {
    "price_only_lstm":            "Price-Only",
    "sentiment_lstm":             "Sentiment",
    "sentiment_shift_lstm":       "SentShift",
    "sentiment_lstm_event":       "Sent+Event",
    "sentiment_shift_lstm_event": "SentShift+Ev",
    "xgboost_baseline":           "XGBoost",
    "xgboost_event_baseline":     "XGB+Event",
}

MODEL_COLORS = {
    "price_only_lstm":            "#4477AA",
    "sentiment_lstm":             "#EE6677",
    "sentiment_shift_lstm":       "#228833",
    "sentiment_lstm_event":       "#CCBB44",
    "sentiment_shift_lstm_event": "#AA3377",
    "xgboost_baseline":           "#66CCEE",
    "xgboost_event_baseline":     "#BB5566",
}

SUBGROUP_LABELS = {
    "all_days":             "All Days",
    "news_days":            "News Days",
    "no_news_days":         "No-News Days",
    "high_sentiment_days":  "High Sentiment",
    "event_days":           "Event Days",
    "high_event_intensity": "High Intensity",
}


# ── Plot 1: Event coverage by ticker ─────────────────────────────

def plot_event_coverage():
    """Bar chart: % of total trading days covered by each event type, per ticker."""
    if not os.path.exists(EVENT_CSV):
        print(f"  [skip] Event features CSV not found: {EVENT_CSV}")
        return

    ev_df = pd.read_csv(EVENT_CSV)

    # We need total trading days per ticker from the merged dataset
    # Approximate: use news days count as denominator base, scaled by coverage rate
    # Better: load price data row count. Use ev_df unique dates as proxy.
    # Show % of news days that had each event type.
    # Label x-axis clearly.

    event_labels = {
        "earnings_event":          "Earnings",
        "analyst_upgrade":         "Ana. Upgrade",
        "analyst_downgrade":       "Ana. Downgrade",
        "product_launch":          "Product Launch",
        "lawsuit_event":           "Lawsuit",
        "regulatory_event":        "Regulatory",
        "partnership_event":       "Partnership",
        "management_change_event": "Mgmt Change",
        "macro_event":             "Macro",
    }

    fig, axes = plt.subplots(1, len(TICKERS), figsize=(14, 5), sharey=False)
    fig.suptitle("Event Type Coverage (% of News-Days per Ticker)",
                 fontsize=13, fontweight="bold")

    colors = plt.cm.Set2(np.linspace(0, 1, len(EVENT_COLS)))

    for ax, ticker in zip(axes, TICKERS):
        sub  = ev_df[ev_df["ticker"] == ticker]
        n    = len(sub)
        if n == 0:
            ax.set_title(ticker); continue

        pcts  = [(sub[col].sum() / n * 100) for col in EVENT_COLS]
        labs  = [event_labels[c] for c in EVENT_COLS]
        bars  = ax.barh(labs, pcts, color=colors, edgecolor="white", height=0.65)

        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=8)

        ax.set_title(f"{ticker}  (n={n} news-days)", fontsize=10)
        ax.set_xlabel("% of news-days", fontsize=9)
        ax.set_xlim(0, max(pcts) * 1.25 + 5)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "event_coverage_by_ticker.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {out}")


# ── Plot 2: Event type frequency ──────────────────────────────────

def plot_event_type_frequency():
    """Grouped bar chart: total event-day counts by type and ticker."""
    if not os.path.exists(EVENT_CSV):
        print(f"  [skip] {EVENT_CSV}")
        return

    ev_df = pd.read_csv(EVENT_CSV)

    event_labels = {
        "earnings_event":          "Earnings",
        "analyst_upgrade":         "Upgrade",
        "analyst_downgrade":       "Downgrade",
        "product_launch":          "Product",
        "lawsuit_event":           "Lawsuit",
        "regulatory_event":        "Regulatory",
        "partnership_event":       "Partnership",
        "management_change_event": "Mgmt",
        "macro_event":             "Macro",
    }

    x         = np.arange(len(EVENT_COLS))
    width     = 0.25
    tkr_colors = ["#4477AA", "#EE6677", "#228833"]

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, ticker in enumerate(TICKERS):
        sub    = ev_df[ev_df["ticker"] == ticker]
        counts = [int(sub[col].sum()) for col in EVENT_COLS]
        offset = (i - 1) * width
        bars   = ax.bar(x + offset, counts, width, label=ticker,
                        color=tkr_colors[i], alpha=0.85, edgecolor="white")
        for b, c in zip(bars, counts):
            if c > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                        str(c), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([event_labels[c] for c in EVENT_COLS], rotation=30, ha="right")
    ax.set_ylabel("Number of news-days", fontsize=9)
    ax.set_title("Event Type Frequency by Ticker (total news-days with event)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "event_type_frequency.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {out}")


# ── Plots 3 & 4: Subgroup MAPE and DirAcc comparison ─────────────

def _subgroup_metric_chart(sg_df: pd.DataFrame, metric: str,
                            ylabel: str, title: str, filename: str,
                            horizon: int = 5):
    """
    Bar chart comparing a metric across subgroups for a given horizon.
    One panel per ticker; bars grouped by subgroup.
    """
    models = [m for m in MODEL_DISPLAY if m in sg_df["model_name"].unique()]
    if not models:
        print(f"  [skip] No model data for {title}")
        return

    sub_names = [s for s in SUBGROUPS_ORDER
                 if s in sg_df["subgroup"].unique()]

    x     = np.arange(len(sub_names))
    width = 0.7 / len(models)

    fig, axes = plt.subplots(1, len(TICKERS), figsize=(6 * len(TICKERS), 5))
    fig.suptitle(f"{title}  (H={horizon}d)", fontsize=12, fontweight="bold")
    if len(TICKERS) == 1:
        axes = [axes]

    for ax, ticker in zip(axes, TICKERS):
        for j, model in enumerate(models):
            vals = []
            for sg in sub_names:
                row = sg_df[
                    (sg_df["model_name"] == model) &
                    (sg_df["ticker"]     == ticker) &
                    (sg_df["horizon"]    == horizon) &
                    (sg_df["subgroup"]   == sg)
                ]
                vals.append(row[metric].values[0] if len(row) else np.nan)

            offset = (j - len(models) / 2 + 0.5) * width
            bars   = ax.bar(x + offset, vals, width,
                            label=MODEL_DISPLAY.get(model, model),
                            color=MODEL_COLORS.get(model, "#888"),
                            alpha=0.85, edgecolor="white")

            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2,
                            b.get_height() + 0.002,
                            f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)

        ax.set_title(ticker, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SUBGROUP_LABELS.get(s, s) for s in sub_names],
            rotation=25, ha="right", fontsize=8
        )
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=6.5, ncol=2)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {out}")


SUBGROUPS_ORDER = [
    "all_days", "news_days", "no_news_days",
    "high_sentiment_days", "event_days", "high_event_intensity",
]


def plot_subgroup_mape(sg_df: pd.DataFrame):
    for H in HORIZONS:
        _subgroup_metric_chart(
            sg_df, "mape_pct", "MAPE (%)",
            "Subgroup MAPE Comparison",
            f"subgroup_mape_H{H}.png",
            horizon=H,
        )


def plot_subgroup_diracc(sg_df: pd.DataFrame):
    for H in HORIZONS:
        _subgroup_metric_chart(
            sg_df, "dir_acc_pct", "Directional Accuracy (%)",
            "Subgroup Directional Accuracy",
            f"subgroup_diracc_H{H}.png",
            horizon=H,
        )


# ── Plot 5: Distribution shift on event days ──────────────────────

def plot_distribution_shift_event_days():
    """
    KDE + approximate Gaussian comparison of predicted Q50 distributions
    specifically for event_days, comparing price_only vs sentiment_shift_lstm_event.
    """
    model_pairs = [
        ("price_only_lstm",       "sentiment_shift_lstm_event",
         "Price-Only vs SentShift+Event  (event days)"),
        ("price_only_lstm",       "sentiment_lstm_event",
         "Price-Only vs Sent+Event  (event days)"),
    ]

    for model_a, model_b, subtitle in model_pairs:
        for H in HORIZONS:
            fig, axes = plt.subplots(1, len(TICKERS),
                                     figsize=(5 * len(TICKERS), 4.5))
            fig.suptitle(f"{subtitle}  H={H}d", fontsize=11, fontweight="bold")
            if len(TICKERS) == 1:
                axes = [axes]

            any_data = False
            for ax, ticker in zip(axes, TICKERS):
                # Load predictions for event days only via the event dataset
                ev_path = os.path.join(
                    RESULTS_DIR, "event_features", "price_sentiment_event_dataset.csv"
                )
                if not os.path.exists(ev_path):
                    ax.text(0.5, 0.5, "Event dataset\nnot found",
                            ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(ticker)
                    continue

                ev_df    = pd.read_csv(ev_path)
                date_col = "date" if "date" in ev_df.columns else "Date"
                ev_df[date_col] = pd.to_datetime(ev_df[date_col], errors="coerce") \
                                    .dt.strftime("%Y-%m-%d")
                ev_df = ev_df.rename(columns={date_col: "date"})
                ev_ticker = ev_df[ev_df["ticker"] == ticker][
                    ["date", "event_count"]
                ].drop_duplicates("date")

                def _load_event_preds(model_name):
                    path = os.path.join(PRED_DIR, f"{model_name}_{ticker}_predictions.csv")
                    if not os.path.exists(path):
                        return None
                    df = pd.read_csv(path)
                    df = df[df["horizon"] == H].merge(ev_ticker, on="date", how="left")
                    df["event_count"] = df["event_count"].fillna(0)
                    return df[df["event_count"] > 0]

                df_a = _load_event_preds(model_a)
                df_b = _load_event_preds(model_b)

                if df_a is None or df_b is None or len(df_a) < 5 or len(df_b) < 5:
                    ax.text(0.5, 0.5, "Insufficient data",
                            ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{ticker}")
                    continue

                any_data = True
                q50_a = df_a["predicted_Q50"].values
                q50_b = df_b["predicted_Q50"].values
                mu_a, mu_b = q50_a.mean(), q50_b.mean()

                x_min = min(q50_a.min(), q50_b.min()) * 0.995
                x_max = max(q50_a.max(), q50_b.max()) * 1.005
                x_rng = np.linspace(x_min, x_max, 300)

                col_a = MODEL_COLORS.get(model_a, "#4477AA")
                col_b = MODEL_COLORS.get(model_b, "#AA3377")
                lbl_a = MODEL_DISPLAY.get(model_a, model_a)
                lbl_b = MODEL_DISPLAY.get(model_b, model_b)

                kde_a = scipy_stats.gaussian_kde(q50_a)
                kde_b = scipy_stats.gaussian_kde(q50_b)

                ax.plot(x_rng, kde_a(x_rng), color=col_a, lw=2,
                        label=f"{lbl_a}")
                ax.plot(x_rng, kde_b(x_rng), color=col_b, lw=2,
                        linestyle="--", label=f"{lbl_b}")
                ax.axvline(mu_a, color=col_a, lw=1.5, alpha=0.7)
                ax.axvline(mu_b, color=col_b, lw=1.5, alpha=0.7,
                           linestyle="--")

                ax.annotate(
                    f"ΔQ50={mu_b - mu_a:+.2f}  n={len(df_a)}",
                    xy=(0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
                )
                ax.set_title(f"{ticker}  (event days, n={len(df_a)})", fontsize=9)
                ax.set_xlabel("Predicted Q50 ($)", fontsize=8)
                ax.set_ylabel("Density", fontsize=8)
                ax.legend(fontsize=7)
                ax.grid(alpha=0.3)

            if not any_data:
                plt.close()
                continue

            plt.tight_layout()
            safe_a = model_a.replace("_", "")[:12]
            safe_b = model_b.replace("_", "")[:12]
            out = os.path.join(
                PLOTS_DIR, f"dist_shift_event_{safe_a}_vs_{safe_b}_H{H}.png"
            )
            plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
            print(f"  Saved -> {out}")


# ── Main ──────────────────────────────────────────────────────────

def make_all_event_plots():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"\n{'='*62}")
    print("  Generating Event Feature Plots")
    print(f"{'='*62}")

    print("\n  [1/5] Event coverage by ticker...")
    plot_event_coverage()

    print("\n  [2/5] Event type frequency...")
    plot_event_type_frequency()

    if os.path.exists(SUBGROUP_CSV):
        sg_df = pd.read_csv(SUBGROUP_CSV)
        print(f"\n  [3/5] Subgroup MAPE charts  ({len(sg_df)} rows)...")
        plot_subgroup_mape(sg_df)
        print("\n  [4/5] Subgroup DirAcc charts...")
        plot_subgroup_diracc(sg_df)
    else:
        print(f"\n  [3-4/5] Subgroup CSV not found: {SUBGROUP_CSV}")
        print("  Run subgroup_evaluation.py first.")

    print("\n  [5/5] Distribution shift on event days...")
    plot_distribution_shift_event_days()

    print(f"\n  All event plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    make_all_event_plots()
