# Event Feature Summary
## Market-Moving Event/Fundamental Features for Thesis Revision

---

### 1. Motivation

The base sentiment models (sentiment_lstm, sentiment_shift_lstm) showed statistically significant
Q50 shifts in several ticker-horizon pairs, but the MAPE improvement over the price-only baseline
was modest across all trading days. This is expected: financial news does not move markets every day.

The key hypothesis is that **news and sentiment signals are more informative on information-rich days
(earnings announcements, regulatory decisions, analyst upgrades/downgrades)** than on quiet days.
To test this, we add structured event/fundamental features extracted from the existing article text,
and evaluate models separately on different subgroups.

---

### 2. Data Source

**File**: `data/processed/sentiment/news_raw.csv`
- 4,235 articles for AAPL (1,508), NVDA (1,392), MSFT (1,253) + filtered AMZN/GOOGL
- Date range: 2015-01-03 → 2026-05-29
- Article-level: each row is one news article with ticker, date, title, summary, text
- **`text` column** (title + summary, 0 nulls): used as the primary text for keyword matching

---

### 3. Extraction Method: Keyword / Rule-Based Detection

We apply 9 rule sets to the lowercased `text` column using Python `re` (regex) with word-boundary
matching (`\b`) for single-word tokens and substring matching for multi-word phrases.

| Feature | Keywords (sample) | Market impact weight |
|---------|------------------|---------------------|
| `earnings_event` | earnings, eps, quarterly results, guidance, beats estimates | 2.0 (high) |
| `regulatory_event` | SEC, FTC, DOJ, antitrust, fine, export ban, sanctions | 1.8 |
| `lawsuit_event` | lawsuit, class action, settlement, probe | 1.5 |
| `macro_event` | Federal Reserve, rate hike, inflation, FOMC, CPI | 1.5 |
| `analyst_upgrade` | upgrade, buy rating, price target raised, outperform | 1.2 |
| `analyst_downgrade` | downgrade, sell rating, price target cut, underperform | 1.2 |
| `management_change_event` | new CEO, CFO, resignation, leadership change | 1.0 |
| `partnership_event` | acquisition, merger, joint venture, collaboration | 0.8 |
| `product_launch` | launch, unveils, new product, AI chip, new iPhone | 0.8 |

---

### 4. Daily Aggregation

For each (ticker, date) pair:

- **Binary flags** (9 columns): `max()` across articles → 1 if any article that day matched
- **`event_count`**: sum of all binary flags (0–9), counts distinct event types detected
- **`event_intensity`**: weighted event score + 0.5 × log(1 + n_articles)
  - Weighted score = Σ (weight_i × flag_i) for each event type
  - Higher weight for earnings/regulatory/macro events vs. product/partnership

**Key property**: Trading days with no news are filled with zeros. This is methodologically correct —
absence of news is meaningful information for the model.

---

### 5. Coverage Statistics

Event features are available only for the ~27% of trading days covered by `news_raw.csv`.
The remaining ~73% of trading days have `event_count = 0` and `event_intensity = 0`.

This sparsity is a feature, not a bug: the subgroup analysis exploits it to compare
model behavior on information-rich vs. quiet days.

---

### 6. Model Feature Sets

| Model | Price | Sentiment | Event | Total |
|-------|-------|-----------|-------|-------|
| price_only_lstm | 9 | — | — | 9 |
| sentiment_lstm | 9 | 4 | — | 13 |
| sentiment_shift_lstm | 9 | 4 | — | 13 |
| sentiment_lstm_event | 9 | 4 | 11 | 24 |
| sentiment_shift_lstm_event | 9 | 4 | 11 | 24 |

For `sentiment_shift_lstm_event`, the 15 sentiment+event features (4+11) enter the MLP branch,
which is wider than the original 4-feature branch (32→16→8→1 units instead of 16→8→1).

---

### 7. Subgroup Evaluation Design

We evaluate all models separately for 6 observation subgroups:

| Subgroup | Filter condition | Thesis purpose |
|----------|-----------------|----------------|
| all_days | None | Global baseline |
| news_days | news_count > 0 | Does news help when news exists? |
| no_news_days | news_count == 0 | Stability without news |
| high_sentiment_days | abs(avg_sentiment_score) ≥ 0.5 | Strong sentiment signal days |
| event_days | event_count > 0 | Days with structured event signals |
| high_event_intensity | event_intensity ≥ 75th percentile | Highest-information days |

**Expected thesis finding**: Event-enhanced models should show a larger improvement
(lower MAPE, higher DirAcc) on `event_days` and `high_event_intensity` days than on `all_days`.
Even if the overall improvement is modest, a subgroup-specific improvement supports the claim
that structured event features carry predictive signal.

---

### 8. Why Keyword-Based vs. Fine-Tuned Event Classifier

A fine-tuned event classification model (e.g., BERT trained on labeled event categories)
would be more precise but requires:
1. Labeled training data for each event type (not available in the project timeline)
2. Significant compute for fine-tuning
3. Complex integration with the existing FinBERT+LoRA pipeline

Keyword-based detection is:
- Transparent and reproducible (no model black-box)
- Fast (no inference required)
- Easy to audit and extend
- Standard in financial NLP literature (e.g., Boudoukh et al. 2019)

Precision is sufficient for daily aggregation: even noisy event flags at the article level
aggregate to meaningful daily signals when `max()` is applied across multiple articles per day.

---

### 9. Limitations

1. **Recall gaps**: Niche or non-obvious event language may be missed (e.g., implied regulatory pressure).
2. **False positives**: Occasional keyword overlap (e.g., "probe" in non-investigation context).
3. **News coverage sparsity**: ~27% of trading days have any news. Event features are sparse.
4. **No severity quantification**: All matching articles are treated equally regardless of headline prominence.
5. **Retrospective bias check**: Keywords were designed before analysis — future work should validate on held-out dates.

---

### 10. Output Files

| File | Description |
|------|-------------|
| `event_features/event_features_daily.csv` | Daily event flags per ticker |
| `event_features/price_sentiment_event_dataset.csv` | Full 24-feature model input dataset |
| `event_features/subgroup_evaluation.csv` | Per-subgroup metrics for all models |
| `event_features/subgroup_statistical_tests.csv` | Paired t-tests and DM tests by subgroup |
| `event_features/xgboost_event_baseline_results.csv` | XGBoost event baseline metrics |
| `plots/event_features/` | Event coverage, frequency, subgroup MAPE, distribution shift |
