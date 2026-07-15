# A Hybrid Deep Learning Framework for Stock Price Prediction
## Integrating Quantile Regression and Sentiment Analysis

**Master's Thesis Project — Sara Hosseini Ghalehjough**

---

## Overview

This project implements a hybrid stock price prediction framework that combines:
- **Quantile LSTM** — predicts price intervals (Q10, Q50, Q90) using quantile regression with pinball loss
- **FinBERT+LoRA** — extracts daily sentiment features from financial news using fine-tuned BERT
- **SentimentShift LSTM** — a novel dual-branch architecture where sentiment directly shifts the predicted price distribution
- **XGBoost baseline** — strong non-LSTM comparison model using quantile regression
- **Statistical tests** — paired t-tests and Diebold-Mariano tests to validate sentiment's effect

**Assets**: AAPL, MSFT, NVDA  
**Horizons**: 1-day, 5-day, 10-day  
**Validation**: 5-fold walk-forward (2021–2025)  
**Metrics**: MAPE, Coverage, Directional Accuracy, Interval Width

---

## Project Structure

```
stock-transformer-project/
├── src/
│   ├── models/
│   │   ├── lstm_quantile.py          # Original Quantile LSTM (price + 15 sentiment cols)
│   │   ├── lstm_quantile_ablation.py # Original price-only ablation
│   │   ├── sentiment_shift_lstm.py   # NEW: SentimentShift dual-branch architecture
│   │   ├── walk_forward_eval.py      # NEW: Shared walk-forward evaluation framework
│   │   ├── xgboost_baseline.py       # NEW: XGBoost/sklearn quantile baseline
│   │   └── run_all_models.py         # NEW: Full experiment runner
│   ├── evaluation/
│   │   ├── statistical_tests.py      # NEW: Paired t-test, Diebold-Mariano
│   │   └── distribution_shift_analysis.py  # NEW: Q10/Q50/Q90 shift metrics
│   ├── visualization/
│   │   └── make_revised_plots.py     # NEW: All thesis plots
│   ├── sentiment/
│   │   ├── collector.py              # Multi-source news collector
│   │   ├── sentiment_model.py        # FinBERT+LoRA scorer
│   │   ├── train_finbert_phrasebank.py
│   │   └── knowledge_distillation_plan.md  # NEW: Future work plan
│   ├── utils.py                      # Shared utilities (features, metrics, sequences)
│   ├── merge_sentiment_price.py      # Merge price + sentiment CSVs
│   └── run_pipeline.py               # Original end-to-end pipeline
│
├── results/
│   └── revised_experiments/          # NEW: All thesis revision outputs
│       ├── predictions/              # Row-level predictions per model/ticker
│       ├── evaluation_metrics.csv    # Aggregated metrics across all folds
│       ├── statistical_tests.csv     # Statistical test results
│       ├── distribution_shift.csv    # Distribution shift metrics
│       ├── xgboost_baseline_results.csv
│       └── plots/                    # All thesis plots
│
├── outputs/                          # Original pipeline outputs (untouched)
│   ├── models/                       # Trained Keras models + LoRA adapter
│   ├── scalers/
│   ├── metrics/
│   └── predictions/
│
├── data/
│   ├── raw/                          # AAPL/MSFT/NVDA 10-year daily price CSVs
│   └── processed/sentiment/          # Raw news articles
│
├── results/sentiment_finetuning_summary.md  # NEW: FinBERT training documentation
├── requirements.txt
└── run_pipeline.py
```

---

## Setup

### Local (Windows)

```powershell
# Navigate to project directory
cd "D:\lessons\season 4\code\stock-transformer-project"

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies (if not already done)
pip install -r requirements.txt

# Optional: install XGBoost for native quantile support
pip install "xgboost>=1.7"
```

### Google Colab

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone or upload project
import sys
sys.path.insert(0, '/content/drive/MyDrive/stock-transformer-project')

# Install dependencies
!pip install tensorflow transformers peft yfinance scikit-learn joblib
!pip install "xgboost>=1.7"  # optional but recommended
```

---

## Running the Models

### Original Pipeline (Unchanged)

```powershell
# Full pipeline: update data → score news → train LSTM → predict
python run_pipeline.py

# Skip sentiment collection (price-only mode)
python run_pipeline.py --skip-sentiment

# Force retrain models
python -m src.models.lstm_quantile --train --ticker AAPL

# Run walk-forward evaluation
python -m src.models.lstm_quantile --eval --ticker AAPL
```

### Thesis Revised Experiments

#### Option 1 — Run Everything at Once

```powershell
# All 3 LSTM variants + XGBoost + stats + plots (3–8 hours on CPU)
python -m src.models.run_all_models

# Only one model variant and one ticker (faster testing)
python -m src.models.run_all_models --model price_only_lstm --ticker AAPL

# Skip XGBoost
python -m src.models.run_all_models --no-xgboost

# Only run post-processing (assumes predictions already exist)
python -m src.models.run_all_models --post-only
```

#### Option 2 — Run Models Individually

```powershell
# Price-only LSTM (9 price features)
python -m src.models.walk_forward_eval --model price_only_lstm --ticker AAPL
python -m src.models.walk_forward_eval --model price_only_lstm --ticker MSFT
python -m src.models.walk_forward_eval --model price_only_lstm --ticker NVDA

# Sentiment LSTM (9 price + 4 core sentiment features)
python -m src.models.walk_forward_eval --model sentiment_lstm --ticker AAPL
python -m src.models.walk_forward_eval --model sentiment_lstm --ticker MSFT
python -m src.models.walk_forward_eval --model sentiment_lstm --ticker NVDA

# SentimentShift LSTM (dual-branch: price LSTM + sentiment MLP)
python -m src.models.walk_forward_eval --model sentiment_shift_lstm --ticker AAPL
python -m src.models.walk_forward_eval --model sentiment_shift_lstm --ticker MSFT
python -m src.models.walk_forward_eval --model sentiment_shift_lstm --ticker NVDA

# XGBoost baseline (all tickers)
python -m src.models.xgboost_baseline
# Or single ticker:
python -m src.models.xgboost_baseline --ticker AAPL
```

#### Option 3 — Post-Processing Only

```powershell
# Statistical tests (requires prediction CSVs to exist)
python -m src.evaluation.statistical_tests

# Distribution shift analysis
python -m src.evaluation.distribution_shift_analysis

# Generate all thesis plots
python -m src.visualization.make_revised_plots
```

---

## Event Feature Pipeline (New)

### Step 1 — Generate Event Features

```powershell
# Extract keyword-based event features from news_raw.csv
python -m src.features.event_features

# Merge event features with price/sentiment datasets
python -m src.features.merge_event_features
```

Outputs:
- `results/revised_experiments/event_features/event_features_daily.csv` — daily event flags per ticker
- `results/revised_experiments/event_features/price_sentiment_event_dataset.csv` — 24-feature model input

### Step 2 — Run Event-Enhanced Models

```powershell
# Sentiment LSTM + event features (24 features)
python -m src.models.walk_forward_eval --model sentiment_lstm_event --ticker AAPL
python -m src.models.walk_forward_eval --model sentiment_lstm_event --ticker MSFT
python -m src.models.walk_forward_eval --model sentiment_lstm_event --ticker NVDA

# SentimentShift LSTM + event features (dual-branch, 24 features)
python -m src.models.walk_forward_eval --model sentiment_shift_lstm_event --ticker AAPL
python -m src.models.walk_forward_eval --model sentiment_shift_lstm_event --ticker MSFT
python -m src.models.walk_forward_eval --model sentiment_shift_lstm_event --ticker NVDA

# Skip training if predictions already exist
python -m src.models.walk_forward_eval --model sentiment_lstm_event --ticker AAPL --skip-training

# XGBoost with event features
python -m src.models.xgboost_baseline --model event --ticker AAPL
```

Or run everything at once:
```powershell
# Full event pipeline: build features + train event models + post-processing
python -m src.models.run_all_models --event

# With skip-training (re-use cached predictions)
python -m src.models.run_all_models --event --skip-training

# Event post-processing only (assumes predictions exist)
python -m src.models.run_all_models --event --skip-training --post-only
```

### Step 3 — Subgroup Evaluation

```powershell
# Evaluate all models on 6 subgroups (all_days, news_days, event_days, ...)
python -m src.evaluation.subgroup_evaluation

# Subgroup statistical tests (paired t-test + DM test within each subgroup)
python -m src.evaluation.subgroup_statistical_tests

# Event feature plots
python -m src.visualization.make_event_plots
```

---

## Model Descriptions

### price_only_lstm
- **Features**: 9 price-derived technical indicators
- **Architecture**: LSTM(64) → Dropout → LSTM(32) → Dense(32) → Dense(3)
- **Output**: Q10, Q50, Q90 price predictions
- **Purpose**: Baseline without any sentiment information

### sentiment_lstm
- **Features**: 9 price features + 4 core sentiment features (13 total)
  - `avg_sentiment_score`, `positive_ratio`, `negative_ratio`, `news_count`
- **Architecture**: Same as price_only_lstm but with wider input
- **Purpose**: Standard fusion model — sentiment as additional input features

### sentiment_shift_lstm *(Novel Architecture)*
- **Features**: Same 13 features
- **Architecture**: Dual-branch
  - Price branch: LSTM over full 30-day sequence → base Q10/Q50/Q90
  - Sentiment branch: MLP over last-day's 4 sentiment features → scalar shift
  - Final output: `base_quantiles + sentiment_shift`
- **Purpose**: Explicit distribution shift — sentiment directly pushes the entire predicted price distribution up or down
- **Thesis relevance**: Supports the hypothesis that sentiment causes a shift analogous to two Gaussians with different means

### sentiment_lstm_event *(New)*
- **Features**: 9 price + 4 sentiment + 11 event = 24 features
- **Architecture**: Same standard LSTM as `sentiment_lstm` but wider input
- **Event features**: 9 binary keyword flags + `event_count` + `event_intensity`
- **Purpose**: Tests whether structured event information improves over raw sentiment

### sentiment_shift_lstm_event *(New)*
- **Features**: Same 24 features
- **Architecture**: Dual-branch: price LSTM (9) + sentiment/event MLP (15 = 4 sentiment + 11 event)
  - MLP branch is wider (32→16→8→1 units) to accommodate 15-dim input
- **Purpose**: Event signals directly shift the predicted price distribution

### xgboost_baseline
- **Features**: 13 features × 3 lags (current, 1-day, 5-day) = 39 features
- **Backend**: XGBoost ≥ 1.7 (quantile objective) or sklearn GradientBoostingRegressor
- **Purpose**: Strong non-LSTM comparison for MAPE and directional accuracy

### xgboost_event_baseline *(New)*
- **Features**: 24 features × 3 lags = 72 features
- **Same backend as xgboost_baseline**
- **Purpose**: Tree-based event-enhanced comparison

---

## Output Files

All revised experiment outputs go to `results/revised_experiments/`:

| File | Description |
|------|-------------|
| `predictions/<model>_<ticker>_predictions.csv` | Row-level predictions: date, fold, actual_price, Q10, Q50, Q90 |
| `evaluation_metrics.csv` | Per-fold MAPE, Coverage, DirAcc, Interval Width for all models |
| `statistical_tests.csv` | Paired t-tests and DM test results for all model comparisons |
| `distribution_shift.csv` | Mean Q10/Q50/Q90 shifts and Gaussian σ estimates |
| `xgboost_baseline_results.csv` | XGBoost-specific metrics |
| `plots/distribution_shift_<TICKER>_H<n>.png` | Distribution shift visualizations |
| `plots/mape_comparison.png` | MAPE bar chart across all models |
| `plots/diracc_comparison.png` | Directional Accuracy bar chart |
| `plots/coverage_comparison.png` | Coverage bar chart |
| `plots/statistical_tests_table.png` | Statistical test summary table |

---

## Feature Columns

### Price Features (9)
| Feature | Description |
|---------|-------------|
| `log_ret_1` | Daily log return |
| `ret_5` | 5-day log return |
| `ret_10` | 10-day log return |
| `ma_ratio_5` | 5-day MA deviation |
| `ma_ratio_10` | 10-day MA deviation |
| `hl_range` | (High - Low) / Close |
| `vol_10` | 10-day rolling volatility |
| `vol_20` | 20-day rolling volatility |
| `rsi_14` | RSI (14 periods) |

### Core Sentiment Features (4)
| Feature | Description |
|---------|-------------|
| `avg_sentiment_score` | Mean (P_pos - P_neg) per day |
| `positive_ratio` | Fraction of positive articles |
| `negative_ratio` | Fraction of negative articles |
| `news_count` | Number of articles that day |

---

## Walk-Forward Validation

The evaluation uses **time-series walk-forward cross-validation** to avoid lookahead bias:

| Fold | Train on years | Test on |
|------|---------------|---------|
| 2021 | 2015–2020     | 2021    |
| 2022 | 2015–2021     | 2022    |
| 2023 | 2015–2022     | 2023    |
| 2024 | 2015–2023     | 2024    |
| 2025 | 2015–2024     | 2025    |

Within each fold, the last 10% of the training data is used for validation.
All scalers are fit on training data only to prevent data leakage.

---

## Sentiment Pipeline

```
News collection (3 sources: Alpha Vantage, GDELT, yfinance)
       ↓
FinBERT+LoRA scoring (P_neg, P_neu, P_pos per article)
       ↓
Daily aggregation (avg_sentiment_score, positive_ratio, ...)
       ↓
Left-join with price data on date
       ↓
Zero-fill for no-news days
```

See `results/sentiment_finetuning_summary.md` for FinBERT training details.

---

## Author

Sara Hosseini Ghalehjough  
Master's Thesis in Computer Science / Data Science
