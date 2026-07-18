# Bank Branch Cash Forecasting System
## Software Specification (SPEC)

**Version:** 1.1  
**Project Type:** Data Science / Time Series Forecasting  
**Domain:** Banking  
**Prepared By:** Muhammad Usman  
**Status:** In Progress  
**Last Updated:** July 17, 2026

---

# 1. Project Overview

## 1.1 Purpose

The purpose of this project is to develop a machine learning-based cash forecasting system capable of predicting branch-level cash inflows, cash outflows, and net cash requirements at half-day intervals.

The solution will help banks optimize cash allocation, reduce idle cash, minimize shortages, and improve operational efficiency across their branch network.

---

# 2. Problem Statement

Banks maintain large amounts of physical cash to meet customer withdrawal demands.

Current cash planning methods rely heavily on manual estimation and historical averages, leading to:

- Excess idle cash
- Cash shortages
- High insurance costs
- Emergency cash replenishments
- Poor customer experience
- Inefficient capital utilization

The objective is to replace traditional forecasting methods with a data-driven forecasting system.

---

# 3. Business Objectives

The proposed solution should:

- Predict cash inflows
- Predict cash outflows
- Predict net cash requirements
- Reduce idle cash
- Reduce emergency replenishments
- Improve liquidity management
- Optimize cash distribution
- Support data-driven decision making

---

# 4. Project Scope

The system will forecast cash demand for each branch using:

- Historical transaction data
- Calendar information
- Salary cycles
- Public holidays
- Religious events
- Seasonal trends
- Historical customer behavior

Forecasts will be generated every **12 hours (Half-Day Forecasting).**

---

# 5. Functional Requirements

The system shall:

- Import historical banking data
- Clean and preprocess datasets
- Perform exploratory data analysis
- Generate forecasting features
- Train forecasting models
- Evaluate forecasting accuracy
- Predict future cash requirements
- Recommend cash replenishment amounts
- Generate business reports

---

# 6. Non-Functional Requirements

- High prediction accuracy
- Scalable for multiple branches
- Modular architecture
- Easy to maintain
- Reproducible experiments
- Fast inference time
- Secure data handling

---

# 7. Technology Stack

## Programming Language

- Python

## Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Prophet (Optional)
- Statsmodels
- TensorFlow / PyTorch (Optional)

## Development Tools

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

# 8. Project Phases

---

# Phase 1 — Data Understanding & Preparation

## Goal

Prepare a clean and analysis-ready dataset.

---

## Objectives

- Understand the banking dataset
- Explore transaction behavior
- Clean inconsistent records
- Generate business understanding

---

## Tasks

### 1. Data Collection ✅ COMPLETED

Collected historical data including:

- ✅ Branch ID (`tran_br_code`)
- ✅ Date (`start_date`)
- ✅ Time (`txn_hour`)
- ✅ Cash Debits (`TOTAL_DR`)
- ✅ Cash Credits (`TOTAL_CR`)

**Dataset:** 81,717 rows × 5 columns loaded from `Bank Cash Optimization.xlsx`

---

### 2. External Data

Integrate

- Public Holidays
- Religious Events
- Salary Days
- Weekend Indicator
- Month
- Quarter
- Weather (Optional)

---

### 3. Data Cleaning ✅ COMPLETED

Performed:

- ✅ Missing Value Handling (0 missing values found)
- ✅ Duplicate Removal (8 duplicates removed)
- ✅ Datetime Conversion (`start_date` → datetime64)
- ✅ Outlier Detection & Capping (IQR method on TOTAL_DR & TOTAL_CR)
- ✅ Data Validation (no negative balances, 0 remaining nulls)

**Cleaned Dataset:** 81,709 rows × 5 columns saved to `cleaned_bank_data.csv`

---

### 4. Exploratory Data Analysis ✅ COMPLETED

Generated comprehensive visualizations through multiple analysis scripts:

#### Phase 1 — Initial EDA (`phase1_eda.py`)
- **8 original plots** including cash flow trends, transaction distribution, peak hours, branch performance, correlation analysis, net cash analysis, hourly boxplots, and weekly patterns

#### Phase 2 — Cash Requirement & Deficit Analysis (`phase2_cash_analysis.py`)
- **5 additional plots** for half-day cash requirement, branch net cash position, branch-hour heatmap, daily net cash with rolling average, and branch deficit frequency/severity

#### Daily Merge & Time Series EDA (`daily_merge_and_timeseries_eda.py`)
- **10 time series plots** including daily cash flow trends, weekly seasonal patterns, monthly trends, cumulative cash flow, weekend vs weekday analysis, daily distributions, and time series decomposition

#### Combined EDA (`eda.py`) ✅ NEW
A unified script combining **11 selected graphs** from all three analysis modules, eliminating redundant visualizations:

| # | Plot | File | Source |
|---|------|------|--------|
| 1 | **Peak Hours Analysis** | `3_peak_hours_analysis.png` | Phase 1 EDA |
| 2 | **Branch Performance** | `4_branch_performance.png` | Phase 1 EDA |
| 3 | **Net Cash Analysis** | `6_net_cash_analysis.png` | Phase 1 EDA |
| 4 | **Hourly Boxplot** | `7_hourly_boxplot.png` | Phase 1 EDA |
| 5 | **Branch Net Cash Position** | `10_branch_net_cash_position.png` | Phase 2 Cash Analysis |
| 6 | **Branch × Hour Heatmap** | `11_branch_hour_heatmap.png` | Phase 2 Cash Analysis |
| 7 | **Weekly Seasonal Pattern** | `ts4_weekly_seasonal_pattern.png` | Time Series EDA |
| 8 | **Monthly Trend Analysis** | `ts5_monthly_trend_analysis.png` | Time Series EDA |
| 9 | **Cumulative Cash Flow** | `ts6_cumulative_cash_flow.png` | Time Series EDA |
| 10 | **Weekend vs Weekday** | `ts7_weekend_vs_weekday.png` | Time Series EDA |
| 11 | **Daily Distribution Analysis** | `ts8_daily_distribution_analysis.png` | Time Series EDA |

### Key EDA Findings

- **Total Net Cash Position:** Deficit of **-10,317.47M** (withdrawals exceed deposits)
- **Busiest Hour:** **12:00 PM** (8,980 transactions)
- **Busiest Day:** **Friday**
- **Highest Activity Branch:** **Branch 104** (46,090M in withdrawals)
- **Avg Transaction:** Withdrawal = 4.1M, Deposit = 3.98M
- **DR vs CR Correlation:** 0.4310 (moderate positive correlation)
- **Daily Avg Net Cash:** -15.45M (deficit trend)
- **Total Days Analyzed:** 668 (2024-01-02 to 2026-04-04)

---

## Deliverables

- ✅ Clean Dataset (`Bank DataSet/cleaned_bank_data.csv`)
- ✅ Daily Merged Dataset (`Bank DataSet/daily_merged_data.csv`)
- ✅ Data Dictionary (generated in `phase1_data_preparation.py`)
- ✅ **EDA Reports** (visualizations in `Bank DataSet/eda_plots/`)
- ✅ **Phase 1 EDA Script** (`Bank Project/phase1_eda.py`)
- ✅ **Phase 2 Cash Analysis Script** (`Bank Project/phase2_cash_analysis.py`)
- ✅ **Daily Merge & Time Series Script** (`Bank Project/daily_merge_and_timeseries_eda.py`)
- ✅ **Combined EDA Script** (`Bank Project/eda.py`) — Unified 11-graph analysis
- ⬜ Feature Summary

---

# Phase 2 — Feature Engineering & Forecast Model

## Goal

Develop forecasting models capable of predicting future cash demand.

---

## Objectives

Create meaningful features and compare forecasting algorithms.

---

## Tasks

### Feature Engineering

Generate

#### Time Features

- Year
- Month
- Week
- Day
- Hour
- Half-Day Slot

---

#### Lag Features

- Previous Withdrawal
- Previous Deposit
- Previous Net Cash

---

#### Rolling Statistics

- Moving Average
- Rolling Mean
- Rolling Sum
- Rolling Standard Deviation

---

#### Calendar Features

- Holiday
- Salary Day
- Ramadan
- Eid
- Weekend

---

### Model Development

Possible models

#### Machine Learning

- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

#### Time Series

- ARIMA
- SARIMA
- Prophet

#### Deep Learning (Optional)

- LSTM
- GRU

---

### Model Evaluation

Metrics

- MAE
- RMSE
- MAPE
- R² Score

---

### Model Selection

Compare

- Accuracy
- Speed
- Generalization
- Business Performance

---

## Deliverables

- Feature Engineered Dataset
- Trained Models
- Evaluation Report
- Best Forecasting Model

---

# Phase 3 — Forecasting & Decision Support

## Goal

Generate business forecasts and support cash management decisions.

---

## Objectives

Provide actionable recommendations for bank operations.

---

## Tasks

### Cash Forecasting

Predict

- Cash Inflow
- Cash Outflow
- Net Cash Requirement

Every

- Morning
- Afternoon

(Half-Day Forecast)

---

### Branch Risk Analysis

Detect

- High Withdrawal Branches
- Cash Shortage Risk
- Idle Cash Branches
- Sudden Transaction Spikes

---

### Cash Recommendation Engine

Recommend

- Required Cash
- Cash Replenishment Amount
- Redistribution Between Branches
- Emergency Refill Alerts

---

### Dashboard (Optional)

Visualize

- Forecast vs Actual
- Branch-wise Trends
- Cash Flow
- Risk Indicators
- Forecast Accuracy

---

## Deliverables

- Forecast Reports
- Business Recommendations
- Dashboard
- Final Documentation

---

# 9. Expected Inputs

Historical Banking Data

- Date
- Branch
- Deposits
- Withdrawals
- Opening Balance
- Closing Balance
- Cash Delivery
- ATM Transactions

External Features

- Holiday Calendar
- Salary Calendar
- Religious Events
- Weather (Optional)

---

# 10. Expected Outputs

The system should generate

- Predicted Cash Inflow
- Predicted Cash Outflow
- Predicted Net Cash
- Cash Requirement
- Branch Risk Status
- Replenishment Recommendation

---

# 11. Success Criteria

The project will be considered successful if it can:

- Forecast cash demand accurately
- Reduce idle cash
- Minimize shortages
- Improve branch liquidity
- Reduce operational cost
- Improve customer satisfaction

---

# 12. High-Level Workflow

```text
Historical Data
        │
        ▼
Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
EDA
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Cash Forecasting
        │
        ▼
Decision Support
        │
        ▼
Business Reports / Dashboard
```

---

# 13. Future Enhancements

- Real-time forecasting
- ATM-level forecasting
- Multi-branch optimization
- Weather integration
- Reinforcement Learning for cash allocation
- AI-powered recommendation engine
- Interactive web dashboard
- Automated report generation

---

# 14. Project Timeline

| Phase | Description | Deliverable | Status |
|--------|-------------|-------------|--------|
| Phase 1 | Data Understanding & Preparation | Clean Dataset + EDA Report + Visualizations | ✅ Completed |
| Phase 1b | Time Series & Cash Analysis | Daily Merge + 10 Time Series Plots + 5 Cash Analysis Plots | ✅ Completed |
| Phase 1c | Combined EDA Consolidation | Unified eda.py with 11 Selected Graphs | ✅ Completed |
| Phase 2 | Feature Engineering & Model Development | Feature-Engineered Dataset + Baseline Metrics | ✅ Completed |
| Phase 3 | Forecasting & Decision Support | Business Reports + Dashboard | ⬜ Not Started |

---

# 15. Conclusion

This project aims to modernize branch cash management by leveraging data science and machine learning techniques. By accurately forecasting branch-level cash inflows and outflows, the system will enable banks to optimize cash distribution, minimize operational costs, improve liquidity management, and enhance customer service through proactive cash planning.