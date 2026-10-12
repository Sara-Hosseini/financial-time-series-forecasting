# A Hybrid Deep Learning Framework for Stock Price Prediction Integrating Quantile Regression and Sentiment Analysis

> M.Sc. Artificial Intelligence thesis (grade 1.8) · IU International University of Applied Sciences, Berlin · Sara Hosseini

**In one sentence:** forecasting a realistic *range* for future prices (10th, 50th and 90th percentiles) instead of a single number, testing whether financial news sentiment helps, and evaluating everything honestly against simple baselines.

| | |
|---|---|
| **Question** | Can a model give well-calibrated prediction intervals, and does news sentiment improve them? |
| **Data** | Daily prices for AAPL, MSFT and NVDA (Jan 2015 to Apr 2026, 2,844 trading days each) and about 1,200 to 1,350 news articles per stock (Alpha Vantage, GDELT, Yahoo Finance News) |
| **Methods** | Quantile LSTM · FinBERT fine-tuned with LoRA · XGBoost quantile baseline · naive random-walk baseline · 5-fold walk-forward validation · paired t-tests and Diebold-Mariano tests |
| **Tools** | Python, pandas, NumPy, SciPy, scikit-learn, XGBoost, TensorFlow/Keras, PyTorch, Hugging Face Transformers, PEFT |
| **Detailed results** | [sara-hosseini.github.io/project-thesis.html](https://sara-hosseini.github.io/project-thesis.html) |

### Why this matters beyond finance

The same approach applies to demand, revenue or capacity planning: decision-makers get a low/expected/high range, every model has to beat a simple baseline, and walk-forward validation shows how the forecast would really have performed at the time.

## Key Results

Evaluated with five-fold walk-forward validation (test years 2021 to 2025) on AAPL, MSFT and NVDA.

- Prediction intervals were close to the nominal 80% target at 1 day (78.8% to 83.0% coverage), versus 70.1% for the XGBoost baseline.
- At 10 days, coverage ranged from 70.9% (NVDA) to 80.1% (AAPL), versus 65.1% for the baseline.
- LoRA fine-tuning raised the sentiment model's macro F1 from 0.31 (zero-shot) to 0.80 on a held-out Financial PhraseBank subset.
- Sentiment did not consistently reduce price error (comparable to a naive random-walk baseline), but directional accuracy for AAPL at 10 days reached 72.2% on news-event days (58.2% overall).

---

## Research Objectives

This research addresses three main questions:

- Can financial sentiment improve stock price prediction?
- Does sentiment improve directional forecasting performance?
- Can probabilistic forecasting provide reliable confidence intervals for future prices?

---

## Main Contributions

✔ Hybrid Quantile LSTM architecture

✔ Financial sentiment integration using FinBERT

✔ LoRA fine-tuning for domain adaptation

✔ Walk-forward evaluation to prevent look-ahead bias

✔ Probabilistic forecasting using Quantile Regression

✔ Distribution shift analysis

✔ Statistical significance testing

✔ Event-based subgroup evaluation

✔ Comparison with XGBoost baseline

---

## Models Implemented

- Price-only LSTM
- Sentiment LSTM
- Sentiment Shift LSTM
- Quantile LSTM
- Transformer Quantile Model (experimental)
- XGBoost Baseline
- Event-enhanced Models

---

## Machine Learning Pipeline

```
Raw Market Data
        │
        ▼
 Feature Engineering
        │
        ▼
 Financial News Collection
        │
        ▼
 FinBERT + LoRA Fine-tuning
        │
        ▼
 Daily Sentiment Aggregation
        │
        ▼
 Event Feature Extraction
        │
        ▼
 Hybrid Deep Learning Models
        │
        ▼
 Walk-Forward Validation
        │
        ▼
 Statistical Evaluation
```

---

## Dataset

### Market Data

- Yahoo Finance
- Daily OHLCV prices
- Approximately 10 years of historical data
### Financial PhraseBank

The Financial PhraseBank dataset is publicly available from its original source and is therefore not redistributed in this repository.

Please download it directly from the original source before running the sentiment fine-tuning pipeline.

### Financial News

- News articles collected from Alpha Vantage, GDELT and Yahoo Finance News
- Financial PhraseBank (used to fine-tune the sentiment model)
- FinBERT + LoRA sentiment scores

---

## Evaluation Metrics

The models were evaluated using:

- MAPE
- Directional Accuracy
- Prediction Interval Coverage
- Interval Width
- Paired t-test
- Diebold-Mariano Test

---

## Where to Look in the Code

| What | File |
|---|---|
| Quantile LSTM (pinball loss, Q10/Q50/Q90) | [`src/models/lstm_quantile.py`](src/models/lstm_quantile.py) |
| Walk-forward evaluation | [`src/models/walk_forward_eval.py`](src/models/walk_forward_eval.py) |
| XGBoost quantile baseline | [`src/models/xgboost_baseline.py`](src/models/xgboost_baseline.py) |
| Naive random-walk baseline | [`naive_baseline.py`](naive_baseline.py) |
| FinBERT + LoRA fine-tuning | [`src/sentiment/train_finbert_phrasebank.py`](src/sentiment/train_finbert_phrasebank.py) |
| Paired t-test and Diebold-Mariano test | [`src/evaluation/statistical_tests.py`](src/evaluation/statistical_tests.py) |
| Event-day subgroup analysis | [`src/evaluation/subgroup_evaluation.py`](src/evaluation/subgroup_evaluation.py) |
| Exploratory analysis | [`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb), [`notebooks/03_price_sentiment_impact.ipynb`](notebooks/03_price_sentiment_impact.ipynb) |

---

## How to Run

```bash
git clone https://github.com/Sara-Hosseini/financial-time-series-forecasting.git
cd financial-time-series-forecasting
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py --skip-sentiment   # price-only pipeline (no sentiment step)
python run_pipeline.py                    # full pipeline; needs ALPHAVANTAGE_API_KEY in .env
                                          # and the Financial PhraseBank dataset (see Dataset)
```

Raw data is downloaded by the pipeline and is not committed to the repository.

---

## Repository Structure

```
src/
├── evaluation/
├── features/
├── models/
├── sentiment/
├── visualization/

results/
├── revised_experiments/

data/

notebooks/
```

---

## Technologies

- Python
- PyTorch
- TensorFlow / Keras
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- HuggingFace Transformers
- PEFT (LoRA)
- Matplotlib

---

## Key Features

- Multi-horizon forecasting (1, 5 and 10 days)
- Quantile prediction intervals
- Financial sentiment analysis
- Event-aware feature engineering
- Walk-forward validation
- Distribution shift analysis
- Statistical hypothesis testing
- Reproducible research pipeline

---

## Thesis

Master of Science in Artificial Intelligence

IU International University of Applied Sciences

Berlin, Germany

---

## Future Improvements

Potential future work includes:

- Knowledge Distillation
- Real-time prediction pipeline
- Portfolio optimization
- Multi-asset forecasting
- Alternative data integration
- Transformer-only architecture

---

## License

MIT License

---

## Author

**Sara Hosseini**

Berlin, Germany

LinkedIn:
https://www.linkedin.com/in/sara-hosseini-analyst/

GitHub:
https://github.com/Sara-Hosseini
