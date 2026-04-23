# Stock Price Forecasting with LSTM

## Overview

In this project, I built a time-series forecasting model to predict future stock prices of Apple (AAPL).  
The main goal was to understand how well an LSTM model can learn patterns from historical price data and forecast future values.

Alongside the forecasting model, I also prepared a sentiment analysis pipeline using financial text data (FinBERT), which can be used later to improve predictions.

---

## Data

For price data, I used historical daily stock data from Yahoo Finance, including:

- Open
- High
- Low
- Close
- Volume

The dataset covers around 10 years of AAPL data.

For the NLP part, I used:

- `financial_phrasebank.csv`

which contains labeled financial text for sentiment analysis.

---

## Model Approach

The main model is an LSTM network trained on price-based features.

Instead of predicting the raw price directly, the model predicts:

log(C_{t+H} / C_t)

This helps stabilize the learning process and is common in financial modeling.

---

## Features

The model uses only price-derived features, such as:

- daily log returns
- short-term returns (5 and 10 days)
- moving averages
- high-low price range
- rolling volatility

I intentionally did **not** include the raw close price as an input feature in the final version to avoid leakage and unrealistic results.

---

## Training Setup

- Train / Validation / Test split based on time (no shuffling)
- Scaling fitted only on training data
- Sequence-based input (sliding window)
- LSTM with two layers
- Huber loss + Adam optimizer

Special care was taken to avoid data leakage, especially in how sequences are built for validation and test sets.

---

## Results

### 5-day forecast
- RMSE ≈ 8.7 – 9 USD  
- MAE ≈ 6.5 – 7 USD  
- MAPE ≈ 3%  
- Directional Accuracy ≈ 48% – 52%

### 10-day forecast
- RMSE ≈ 12 – 13 USD  
- MAE ≈ 9.5 – 10.5 USD  
- MAPE ≈ 4.5%  
- Directional Accuracy ≈ 45% – 49%

---

## Interpretation

The model is able to follow the general trend of the stock price, but it does not strongly outperform simple expectations.

This is expected, since stock markets are highly noisy and difficult to predict.

The directional accuracy being close to 50% suggests that predicting exact price direction remains challenging.

---

## Sentiment Analysis (FinBERT)

I also prepared a sentiment analysis workflow using financial text data.

This part is implemented in:

notebooks/finbert_finetune_kaggle.ipynb

The goal is to later combine sentiment signals with price-based features.

---

## How to Run

Activate your environment and run:

python src/lstm_price_model_keras.py

The script will:

- train the model
- evaluate it
- save plots, metrics, and predictions

---

## Notes

- The final stable model uses only price-based features
- Sentiment is prepared but not yet integrated into the final model
- Outputs are saved automatically for analysis and reporting

---

## Author

Sara Hosseini Ghalehjough