"""
run_pipeline.py
==============================================================
Automated end-to-end pipeline for AAPL stock price prediction

Steps:
  1. Update price data (Yahoo Finance)
  2. Collect news (yfinance)
  3. Fine-tune FinBERT (skipped if model missing -> price-only)
  4. Score news with FinBERT+LoRA
  5. Merge price + sentiment
  6. Train final LSTM model
  7. Train Quantile LSTM model
  8. Walk-Forward evaluation
  9. Final price prediction
  10. Sentiment baseline comparison

Usage:
  python run_pipeline.py                  <- full pipeline
  python run_pipeline.py --skip-sentiment <- price-only (no sentiment)
  python run_pipeline.py --from-step 6   <- start from step 6
  python run_pipeline.py --only-predict  <- predict only (model must exist)
==============================================================
"""

import os
import sys
import time
import argparse
import traceback
from datetime import datetime
from pathlib import Path

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def log(msg, color=RESET):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{ts}] {msg}{RESET}")

def log_step(n, title):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}  Step {n}: {title}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")

def log_ok(msg):   log(f"[OK]   {msg}", GREEN)
def log_skip(msg): log(f"[SKIP] {msg}", YELLOW)
def log_warn(msg): log(f"[WARN] {msg}", YELLOW)
def log_err(msg):  log(f"[FAIL] {msg}", RED)

results = {}

def mark(step, status):
    results[step] = status


TICKERS = ["AAPL", "MSFT", "NVDA"]

def step1_update_price():
    log_step(1, "Update price data (Yahoo Finance) — AAPL, MSFT, NVDA")
    try:
        from src.update_price_yahoo import update_price_data
        for ticker in TICKERS:
            df = update_price_data(ticker)
            log_ok(f"{ticker}: {len(df)} rows updated")
        mark("price_data", "ok")
        return True
    except Exception as e:
        log_err(f"Price update failed: {e}")
        traceback.print_exc()
        mark("price_data", "fail")
        return False


def step2_collect_news(skip=False):
    log_step(2, "Collect news (yfinance)")
    if skip:
        log_skip("Sentiment disabled — skipped")
        mark("news_collect", "skip")
        return True
    try:
        from src.sentiment.collector import collect_and_label
        df = collect_and_label()
        n = len(df) if df is not None else 0
        log_ok(f"News collected — {n} articles")
        mark("news_collect", "ok")
        return True
    except Exception as e:
        log_warn(f"News collection failed: {e}")
        log_warn("Continuing with price-only...")
        mark("news_collect", "fail")
        return False


def step3_finetune_finbert(skip=False):
    log_step(3, "Fine-tune FinBERT with LoRA")
    lora_dir = Path("outputs/models/finbert_lora_best")
    if lora_dir.exists() and any(lora_dir.iterdir()):
        log_ok(f"LoRA model already exists: {lora_dir}")
        mark("finbert", "ok")
        return True
    if skip:
        log_skip("--skip-sentiment active — fine-tuning skipped")
        mark("finbert", "skip")
        return True
    log_warn("LoRA model not found! Run finbert_finetune_kaggle.ipynb on Kaggle first.")
    log_warn("Continuing with price-only for now...")
    mark("finbert", "skip")
    return True


def step4_score_news(skip=False):
    log_step(4, "Score news with FinBERT+LoRA")
    lora_dir  = Path("outputs/models/finbert_lora_best")
    news_file = Path("data/processed/sentiment/news_raw.csv")
    if skip or not (lora_dir.exists() and any(lora_dir.iterdir())):
        log_skip("LoRA model or news file missing — skipped")
        mark("news_score", "skip")
        return True
    if not news_file.exists():
        log_warn("News file not found — skipped")
        mark("news_score", "skip")
        return True
    try:
        from src.sentiment.sentiment_model import (
            load_news, load_finbert_lora, score_news, build_daily_sentiment
        )
        df_news = load_news(str(news_file))
        tokenizer, model = load_finbert_lora()
        df_scored = score_news(df_news, tokenizer, model)
        Path("outputs").mkdir(exist_ok=True)
        df_scored.to_csv("outputs/AAPL_news_scored.csv", index=False, encoding="utf-8-sig")
        daily = build_daily_sentiment(df_scored)
        daily.to_csv("outputs/AAPL_news_daily.csv", index=False, encoding="utf-8-sig")
        log_ok(f"Scoring done — {len(df_scored)} articles, {len(daily)} days")
        mark("news_score", "ok")
        return True
    except Exception as e:
        log_warn(f"Scoring failed: {e}")
        mark("news_score", "fail")
        return False


def step5_merge(skip=False):
    log_step(5, "Merge price + sentiment data")

    if skip:
        log_skip("Merge skipped")
        mark("merge", "skip")
        return True

    ok_all = True

    try:
        from src.merge_sentiment_price import (
            load_price_data,
            load_sentiment_data,
            merge_price_and_sentiment,
        )

        for ticker in TICKERS:
            sentiment_file = Path(f"outputs/{ticker}_news_daily.csv")
            price_file = Path(f"data/raw/{ticker}_daily_10y_yahoo.csv")
            out_file = Path(f"outputs/{ticker}_price_sentiment_merged.csv")

            if not price_file.exists():
                log_warn(f"{ticker}: price file missing — skipping merge")
                ok_all = False
                continue

            if not sentiment_file.exists():
                log_warn(f"{ticker}: sentiment file missing — creating price-only merged file")
                price_df = load_price_data(str(price_file))
                price_df.to_csv(out_file, index=False, encoding="utf-8-sig")
                log_ok(f"{ticker}: price-only merge done — {len(price_df)} days")
                continue

            price_df = load_price_data(str(price_file))
            sent_df = load_sentiment_data(str(sentiment_file))
            merged = merge_price_and_sentiment(price_df, sent_df)

            out_file.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(out_file, index=False, encoding="utf-8-sig")

            log_ok(f"{ticker}: merge done — {len(merged)} days → {out_file}")

        mark("merge", "ok" if ok_all else "partial")
        return True

    except Exception as e:
        log_warn(f"Merge failed: {e}")
        mark("merge", "fail")
        return False

def step6_train_lstm(from_step):
    log_step(6, "Train final LSTM model (H=5 and H=10)")

    try:
        import subprocess
        import sys
        import os

        ok_all = True

        for ticker in TICKERS:
            model_5 = Path(f"outputs/models/{ticker}_H5_final.keras")
            model_10 = Path(f"outputs/models/{ticker}_H10_final.keras")

            if model_5.exists() and model_10.exists() and from_step <= 5:
                log_skip(f"{ticker}: LSTM models already exist — skipped")
                continue

            log(f"{ticker}: Training LSTM...")

            env = os.environ.copy()
            env["TICKER"] = ticker

            result = subprocess.run(
                [sys.executable, "src/train_final.py"],
                env=env,
                check=False
            )

            if result.returncode != 0:
                log_err(f"{ticker}: LSTM training failed")
                ok_all = False
            else:
                log_ok(f"{ticker}: LSTM training complete")

        mark("train_lstm", "ok" if ok_all else "partial")
        return ok_all

    except Exception as e:
        log_err(f"LSTM training failed: {e}")
        traceback.print_exc()
        mark("train_lstm", "fail")
        return False


def step7_train_quantile(from_step):
    log_step(7, "Train Quantile LSTM model")
    q5  = Path("outputs/models/AAPL_quantile_H5.keras")
    q10 = Path("outputs/models/AAPL_quantile_H10.keras")
    if q5.exists() and q10.exists() and from_step <= 6:
        log_skip("Quantile models already exist — skipped")
        mark("train_quantile", "skip")
        return True
    try:
        log("Training Quantile LSTM... (this may take several minutes)")
        from src.models.lstm_quantile import QuantileConfig, run
        cfg = QuantileConfig()
        run(cfg, force_train=True)
        log_ok("Quantile LSTM training complete")
        mark("train_quantile", "ok")
        return True
    except Exception as e:
        log_err(f"Quantile training failed: {e}")
        traceback.print_exc()
        mark("train_quantile", "fail")
        return False


def step8_walk_forward():
    log_step(8, "Walk-Forward Evaluation — AAPL, MSFT, NVDA")
    try:
        from src.models.lstm_quantile import QuantileConfig, run_walk_forward

        for ticker in TICKERS:
            log(f"  Walk-Forward for {ticker}...")
            cfg = QuantileConfig(ticker=ticker)
            run_walk_forward(cfg)

        log_ok("Walk-Forward evaluation complete for all tickers")
        mark("walk_forward", "ok")
        return True

    except Exception as e:
        log_warn(f"Walk-Forward failed: {e}")
        mark("walk_forward", "fail")
        return False


def step9_predict():
    log_step(9, "Final price prediction")
    try:
        import runpy
        runpy.run_path("src/predict.py", run_name="__main__")
        log_ok("Prediction complete — chart saved to outputs/plots/")
        mark("predict", "ok")
        return True
    except Exception as e:
        log_err(f"Prediction failed: {e}")
        traceback.print_exc()
        mark("predict", "fail")
        return False


def step10_sentiment_baseline(skip=False):
    log_step(10, "Sentiment baseline: Zero-shot / Few-shot / Fine-tuned")
    lora_dir = Path("outputs/models/finbert_lora_best")
    if skip or not (lora_dir.exists() and any(lora_dir.iterdir())):
        log_skip("LoRA model not found — skipped")
        mark("baseline", "skip")
        return True
    try:
        from src.sentiment.zero_few_shot import main as zfs_main
        zfs_main()
        log_ok("Baseline comparison complete — outputs/metrics/")
        mark("baseline", "ok")
        return True
    except Exception as e:
        log_warn(f"Baseline comparison failed: {e}")
        mark("baseline", "fail")
        return False


def print_summary(t_start):
    elapsed = time.time() - t_start
    mins    = int(elapsed // 60)
    secs    = int(elapsed % 60)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Pipeline Summary  ({mins}m {secs}s){RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    icons  = {"ok": "[OK]  ", "skip": "[SKIP]", "fail": "[FAIL]"}
    colors = {"ok": GREEN,    "skip": YELLOW,   "fail": RED}
    labels = {
        "price_data":     " 1. Price data update",
        "news_collect":   " 2. News collection",
        "finbert":        " 3. FinBERT fine-tuning",
        "news_score":     " 4. News scoring",
        "merge":          " 5. Price + sentiment merge",
        "train_lstm":     " 6. LSTM training",
        "train_quantile": " 7. Quantile LSTM training",
        "walk_forward":   " 8. Walk-Forward evaluation",
        "predict":        " 9. Final prediction",
        "baseline":       "10. Sentiment baseline",
    }

    failed = []
    for key, label in labels.items():
        status = results.get(key, "skip")
        icon   = icons.get(status, "[SKIP]")
        color  = colors.get(status, YELLOW)
        print(f"  {color}{icon}  {label}{RESET}")
        if status == "fail":
            failed.append(label)

    print(f"{BOLD}{'='*60}{RESET}")

    if not failed:
        print(f"\n{GREEN}{BOLD}Pipeline completed successfully!{RESET}")
        print(f"\nOutputs:")
        print(f"  Prediction chart : outputs/plots/AAPL_prediction_chart.png")
        print(f"  Trained models   : outputs/models/")
        print(f"  Metrics          : outputs/metrics/")
    else:
        print(f"\n{YELLOW}Pipeline finished with some errors:{RESET}")
        for f in failed:
            print(f"  [FAIL] {f}")


def parse_args():
    p = argparse.ArgumentParser(description="Run the full AAPL stock prediction pipeline")
    p.add_argument("--skip-sentiment", action="store_true",
                   help="Skip sentiment steps — price-only mode")
    p.add_argument("--from-step", type=int, default=1, metavar="N",
                   help="Start from step N (1-10)")
    p.add_argument("--only-predict", action="store_true",
                   help="Run prediction only (model must already exist)")
    return p.parse_args()


def main():
    args      = parse_args()
    t_start   = time.time()
    skip_sent = args.skip_sentiment
    from_step = args.from_step

    if args.only_predict:
        from_step = 9
        skip_sent = True

    print(f"{BOLD}{GREEN}")
    print("╔══════════════════════════════════════════╗")
    print("║   AAPL Stock Prediction — Full Pipeline  ║")
    print("╚══════════════════════════════════════════╝")
    print(f"{RESET}")

    if skip_sent:
        log_warn("Price-only mode: sentiment steps will be skipped")
    if from_step > 1:
        log_warn(f"Starting from step {from_step}")

    if from_step <= 1:
        if not step1_update_price():
            log_err("Price data is required — pipeline stopped")
            print_summary(t_start)
            sys.exit(1)

    if from_step <= 2:  step2_collect_news(skip=skip_sent)
    if from_step <= 3:  step3_finetune_finbert(skip=skip_sent)
    if from_step <= 4:  step4_score_news(skip=skip_sent)
    if from_step <= 5:  step5_merge(skip=skip_sent)

    if from_step <= 6:
        if not step6_train_lstm(from_step):
            log_err("LSTM training failed — prediction not possible")
            print_summary(t_start)
            sys.exit(1)

    if from_step <= 7:  step7_train_quantile(from_step)
    if from_step <= 8:  step8_walk_forward()
    if from_step <= 9:  step9_predict()
    if from_step <= 10: step10_sentiment_baseline(skip=skip_sent)

    print_summary(t_start)


if __name__ == "__main__":
    main()