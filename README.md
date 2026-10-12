# A Hybrid Deep Learning Framework for Stock Price Prediction Integrating Quantile Regression and Sentiment Analysis

> Master's Thesis Project – MSc Artificial Intelligence  
> IU International University of Applied Sciences, Berlin

## Overview

This project presents a hybrid deep learning framework for multi-horizon stock price forecasting by combining historical market data with financial news sentiment.

Unlike traditional point forecasting approaches, the framework predicts **probabilistic price intervals (Q10, Q50, Q90)** to quantify forecast uncertainty while investigating whether financial sentiment improves predictive performance across different forecast horizons.

The project evaluates multiple deep learning and machine learning models using a rigorous walk-forward evaluation methodology on three large-cap US stocks:

- Apple (AAPL)
- Microsoft (MSFT)
- NVIDIA (NVDA)

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
- Transformer Quantile Model (TFT)
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

- Financial PhraseBank
- Financial News Articles
- FinBERT Sentiment Scores

---

## Evaluation Metrics

The models were evaluated using:

- MAE
- RMSE
- MAPE
- Directional Accuracy
- Prediction Interval Coverage
- Interval Width
- Paired t-test
- Diebold-Mariano Test

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
- Walk-forward backtesting
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
