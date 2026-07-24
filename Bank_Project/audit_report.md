# Comprehensive Model Architecture and Validation Report
**Project:** Bank Branch Cash Forecasting System  
**Date:** July 2026  
**Pipeline Version:** 1.0.0

---

## 1. Executive Overview
This report details the end-to-end machine learning pipeline engineered to forecast daily cash inflows (deposits) and outflows (withdrawals) across 15 bank branches. Our objective is to predict exact cash requirements on a rolling basis to prevent branch shortages and manage surplus efficiently.

Instead of deploying generic algorithms, the pipeline dynamically evaluates multiple machine learning models using a strict, time-aware cross-validation protocol to guarantee robust generalization on unseen future dates. 

---

## 2. Models Evaluated and Selection Rationale
We evaluated a diverse suite of 6 machine learning algorithms, spanning linear models and advanced tree-based ensembles, to find the optimal balance between capturing linear trends and complex, non-linear cyclical relationships.

### Models Tested:
1. **Linear Regression:** Serves as a baseline to capture simple linear relationships between past cash flows and future needs.
2. **Ridge Regression:** Adds L2 regularization to prevent overfitting on highly correlated features (like moving averages).
3. **Lasso Regression:** Adds L1 regularization to automatically prune unhelpful features.
4. **Random Forest:** A bagging ensemble of decision trees, highly resilient to outliers and capable of capturing complex non-linear interactions without extensive tuning.
5. **Tuned XGBoost:** A powerful gradient boosting framework optimized for execution speed and model performance, capturing complex interactions iteratively.
6. **Tuned LightGBM:** A highly efficient gradient boosting framework that builds trees leaf-wise. It is exceptionally fast and often achieves the highest accuracy on structured financial time-series data.

### Why Tuned LightGBM Was Selected
Following rigorous evaluation, **Tuned LightGBM** was selected as the final production model for both `daily_withdrawals` and `daily_deposits`. 
- **Rationale:** Financial transaction data exhibits strong daily, weekly, and monthly cyclicality combined with sudden spikes (e.g., paydays). LightGBM's leaf-wise tree growth efficiently isolates these localized spikes and non-linear patterns much better than linear models or Random Forest, yielding the lowest Mean Absolute Error (MAE) and Weighted Absolute Percentage Error (WAPE).

---

## 3. Data Splits and Chronological Integrity
To ensure the model is evaluated exactly as it will perform in production, we abandoned traditional random K-Fold splitting. Instead, data was strictly partitioned chronologically to eliminate look-ahead bias (target leakage).

- **Total Features:** 48
- **Total Branches:** 15

### Split Ranges:
* **Training Set:** `2024-01-30` to `2025-05-31` (7,320 observations)
  * *Purpose:* Used to train the underlying model weights and learn historical cycles.
* **Validation Set:** `2025-06-01` to `2025-09-30` (1,830 observations)
  * *Purpose:* Used internally by our `TimeSeriesSplit` cross-validation to select the optimal model architecture and tune hyperparameters without bleeding into the final test evaluation.
* **Test Set (Out-of-Sample):** `2025-10-01` to `2026-03-31` (2,730 observations)
  * *Purpose:* A completely untouched 6-month holdout set used solely for final unbiased performance validation.

---

## 4. Final Model Hyperparameters (Tuned LightGBM)
The pipeline utilized a `RandomizedSearchCV` paired with `TimeSeriesSplit(n_splits=3)` to iteratively test hyperparameter combinations. The winning configuration used in production is:

* **n_estimators:** 100 (Number of boosted trees)
* **learning_rate:** 0.1 (Step size shrinking used to prevent overfitting)
* **max_depth:** -1 (No explicit limit, allowing leaf-wise growth to isolate complex patterns)
* **subsample:** 0.8 (Uses 80% of data per tree to prevent overfitting)
* **colsample_bytree:** 0.8 (Uses 80% of features per tree to increase robustness)
* **objective / scoring:** optimized for `neg_mean_absolute_error`

---

## 5. Feature Engineering Breakdown
A total of **48 features** were explicitly engineered for the daily forecasting horizon, carefully avoiding current-day leakage. 

* **Calendar & Cyclical (16 features):** `year`, `month`, `day`, `dayofweek`, `quarter`, trigonometric transformations (`month_sin`, `month_cos`, etc.), and boolean flags (`is_weekend`, `is_month_start`). These tell the model *when* the prediction is happening.
* **Exact Lags (8 features):** `1p_lag`, `7p_lag`, `14p_lag`, `28p_lag` for both deposits and withdrawals. Captures exact historical touchpoints (e.g., comparing today to exactly 4 weeks ago).
* **Rolling Windows (24 features):** 7-day, 14-day, and 30-day rolling `mean`, `std`, `min`, and `max`. These provide the model with a smoothed understanding of recent short-term and medium-term volume trends.

---

## 6. Final Performance Metrics (Test Set)
The deployed LightGBM models were tested against a standard **7-Day Rolling Average Baseline**, which simply predicts that today's cash flow will match the average of the last 7 days.

### Daily Withdrawals
* **Baseline MAE:** 10,296,561
* **LightGBM MAE:** 8,779,942
* **LightGBM RMSE:** 11,479,127
* **WAPE:** 24.21%
* **Improvement:** **+14.73%** reduction in forecast error versus baseline.

### Daily Deposits
* **Baseline MAE:** 9,756,600
* **LightGBM MAE:** 8,166,528
* **LightGBM RMSE:** 10,506,955
* **WAPE:** 23.39%
* **Improvement:** **+16.30%** reduction in forecast error versus baseline.

### Derivative Operational Metrics
The system does not predict *Net Cash* or *Cash Requirement* directly, as doing so leads to negative cash requirements and mathematical divergence. Instead, these are derived mathematically from the LightGBM models:
* `Predicted Net Cash = Predicted Deposits - Predicted Withdrawals`
* `Cash Requirement = max(Predicted Withdrawals - Predicted Deposits, 0)`

---

## 7. Conclusion
By leveraging exact chronological data splits, strict anti-leakage feature engineering, and time-aware hyperparameter tuning, the final architecture represents a robust, production-ready solution. The Tuned LightGBM engines have proven highly capable of deciphering complex temporal banking patterns, driving a consistent ~15% improvement in accuracy over traditional rolling heuristics.
