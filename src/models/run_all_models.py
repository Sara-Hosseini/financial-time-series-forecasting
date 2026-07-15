# src/models/run_all_models.py
# Thesis Model Runner — orchestrates all evaluation experiments
#
# Base LSTM models (original):
#   price_only_lstm, sentiment_lstm, sentiment_shift_lstm
#
# Event-enhanced models (new):
#   sentiment_lstm_event, sentiment_shift_lstm_event
#
# Usage:
#   # Run only original models for all tickers
#   python -m src.models.run_all_models
#
#   # Also build event features + run event models
#   python -m src.models.run_all_models --event
#
#   # Run a single model / ticker
#   python -m src.models.run_all_models --model sentiment_lstm --ticker AAPL
#
#   # Skip training (use cached predictions) and only run post-processing
#   python -m src.models.run_all_models --skip-training --post-only
#   python -m src.models.run_all_models --event --skip-training --post-only
#
#   # Skip XGBoost
#   python -m src.models.run_all_models --no-xgboost
#
# Runtime estimate (CPU):
#   Base:   3 models × 3 tickers × 15 folds × ~3 min  ≈ 4–7 h
#   Event:  2 models × 3 tickers × 15 folds × ~3 min  ≈ 3–5 h extra

import os
import argparse
import time

BASE_MODELS  = ["price_only_lstm", "sentiment_lstm", "sentiment_shift_lstm"]
EVENT_MODELS = ["sentiment_lstm_event", "sentiment_shift_lstm_event"]
TICKERS      = ["AAPL", "MSFT", "NVDA"]
ALL_MODELS   = BASE_MODELS + EVENT_MODELS


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


# ── LSTM walk-forward training ────────────────────────────────────

def run_lstm_models(models: list, tickers: list, skip_training: bool = False):
    from src.models.walk_forward_eval import run_walk_forward_eval

    total = len(models) * len(tickers)
    done  = 0
    for model_name in models:
        for ticker in tickers:
            done += 1
            print(f"\n[{done}/{total}] {model_name} | {ticker}"
                  + (" [skip-training]" if skip_training else ""))
            t0 = time.time()
            try:
                run_walk_forward_eval(
                    model_name, ticker,
                    skip_training=skip_training,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()
            print(f"  Duration: {_fmt_duration(time.time() - t0)}")


# ── XGBoost baselines ─────────────────────────────────────────────

def run_xgboost(tickers: list, event: bool = False):
    from src.models.xgboost_baseline import (
        run_xgboost_baseline, run_xgboost_event_baseline,
    )
    run_func = run_xgboost_event_baseline if event else run_xgboost_baseline
    label    = "XGBoost-Event" if event else "XGBoost"

    for ticker in tickers:
        print(f"\n[{label}] {ticker}")
        t0 = time.time()
        try:
            run_func(ticker)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
        print(f"  Duration: {_fmt_duration(time.time() - t0)}")


# ── Event feature pipeline ────────────────────────────────────────

def run_event_pipeline(rebuild: bool = False):
    """Build event features CSV and merge into model dataset."""
    print("\n" + "="*62)
    print("  Event Feature Pipeline")
    print("="*62)

    print("\n[Event 1/2] Extracting keyword event features...")
    try:
        from src.features.event_features import run_event_feature_extraction
        run_event_feature_extraction()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return False

    print("\n[Event 2/2] Merging event features with price/sentiment dataset...")
    try:
        from src.features.merge_event_features import merge_event_features
        merge_event_features(rebuild_events=rebuild)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return False

    return True


# ── Post-processing (stats + plots) ──────────────────────────────

def run_postprocessing(include_event: bool = False):
    print("\n" + "="*62)
    print("  Post-processing: statistical tests + plots")
    print("="*62)

    # 1. Global statistical tests (original models)
    print("\n[Post 1/5] Global statistical tests...")
    try:
        from src.evaluation.statistical_tests import run_statistical_tests
        run_statistical_tests()
    except Exception as e:
        print(f"  Failed: {e}")

    # 2. Global distribution shift
    print("\n[Post 2/5] Distribution shift analysis...")
    try:
        from src.evaluation.distribution_shift_analysis import run_distribution_shift_analysis
        run_distribution_shift_analysis()
    except Exception as e:
        print(f"  Failed: {e}")

    # 3. Revised comparison plots
    print("\n[Post 3/5] Revised comparison plots...")
    try:
        from src.visualization.make_revised_plots import make_all_plots
        make_all_plots()
    except Exception as e:
        print(f"  Failed: {e}")

    if include_event:
        # 4. Subgroup evaluation
        print("\n[Post 4/5] Subgroup evaluation...")
        try:
            from src.evaluation.subgroup_evaluation import run_subgroup_evaluation
            run_subgroup_evaluation()
        except Exception as e:
            print(f"  Failed: {e}")

        # 5. Subgroup statistical tests
        print("\n[Post 5/5] Subgroup statistical tests + event plots...")
        try:
            from src.evaluation.subgroup_statistical_tests import run_subgroup_statistical_tests
            run_subgroup_statistical_tests()
        except Exception as e:
            print(f"  Failed: {e}")

        try:
            from src.visualization.make_event_plots import make_all_event_plots
            make_all_event_plots()
        except Exception as e:
            print(f"  Event plots failed: {e}")
    else:
        print("\n  [Post 4-5/5] Event post-processing skipped "
              "(add --event to enable)")


# ── Entry point ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Thesis model runner"
    )
    parser.add_argument(
        "--model", default=None,
        choices=ALL_MODELS,
        help="Run only this model variant (default: all base models)",
    )
    parser.add_argument(
        "--ticker", default=None,
        choices=TICKERS,
        help="Run only this ticker (default: AAPL MSFT NVDA)",
    )
    parser.add_argument(
        "--event", action="store_true",
        help="Build event features and run event-enhanced models",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Use cached prediction CSVs; skip model training",
    )
    parser.add_argument(
        "--no-xgboost", action="store_true",
        help="Skip XGBoost baseline(s)",
    )
    parser.add_argument(
        "--no-post", action="store_true",
        help="Skip statistical tests and plots after training",
    )
    parser.add_argument(
        "--post-only", action="store_true",
        help="Skip training; run only post-processing",
    )
    parser.add_argument(
        "--rebuild-events", action="store_true",
        help="Force re-extraction of event features even if cached",
    )
    args = parser.parse_args()

    # Determine which models and tickers to run
    if args.model:
        models = [args.model]
    elif args.event:
        models = BASE_MODELS + EVENT_MODELS
    else:
        models = BASE_MODELS

    tickers = [args.ticker] if args.ticker else TICKERS

    total_t0 = time.time()

    print("="*62)
    print("  Thesis Walk-Forward Evaluation Suite")
    print(f"  Models:        {models}")
    print(f"  Tickers:       {tickers}")
    print(f"  Event mode:    {args.event}")
    print(f"  Skip training: {args.skip_training}")
    print("="*62)

    # Rebuild event dataset if requested — runs regardless of --post-only
    # so that subgroup evaluation always gets fresh features.
    if args.event and args.rebuild_events:
        print("\n  [--rebuild-events] Rebuilding event feature dataset ...")
        run_event_pipeline(rebuild=True)

    if not args.post_only:
        # Build event features first if event models are requested and not already rebuilt
        if args.event and not args.rebuild_events and any(m in EVENT_MODELS for m in models):
            ok = run_event_pipeline(rebuild=False)
            if not ok:
                print("  Event pipeline failed. Skipping event model training.")
                models = [m for m in models if m not in EVENT_MODELS]

        # Run LSTM models
        run_lstm_models(models, tickers, skip_training=args.skip_training)

        # Run XGBoost
        if not args.no_xgboost:
            run_xgboost(tickers, event=False)
            if args.event:
                run_xgboost(tickers, event=True)

    if not args.no_post:
        run_postprocessing(include_event=args.event)

    print(f"\n{'='*62}")
    print(f"  Total wall-clock time: {_fmt_duration(time.time() - total_t0)}")
    print(f"  Results saved to:      results/revised_experiments/")
    print("="*62)


if __name__ == "__main__":
    main()
