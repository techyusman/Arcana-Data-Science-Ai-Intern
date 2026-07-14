# Bank Branch Cash Forecasting System — Meeting Preparation Guide

## Prepared for: Muhammad Usman
## Meeting Date: July 14, 2026 @ 12:00 PM
## Purpose: Progress Review & EDA Presentation

---

## 1. Project Overview & Problem Statement

The **Bank Branch Cash Forecasting System** is a machine learning-driven project designed to solve a critical operational challenge faced by banks: **optimal cash management across branch networks**. Banks are required to maintain sufficient physical cash at each branch to meet customer withdrawal demands, but current methods rely heavily on manual estimation and historical averages. This leads to several inefficiencies: **excess idle cash** that incurs high insurance costs, **cash shortages** that require expensive emergency replenishments, **poor capital utilization**, and ultimately a **poor customer experience** when cash is unavailable. The core objective is to replace traditional forecasting methods with a precise, data-driven system that can predict cash demand at the branch level with high accuracy.

The project forecasts three key metrics for each branch: **cash inflows (deposits)**, **cash outflows (withdrawals)**, and **net cash requirements**, all at **half-day intervals (morning and afternoon slots)**. This granular level of prediction enables bank operations teams to make proactive, informed decisions about cash allocation, replenishment scheduling, and inter-branch redistribution.

---

## 2. Business Objectives

The system is designed to deliver the following measurable business outcomes:

- **Predict cash inflows and outflows** accurately at the branch level
- **Predict net cash requirements** for each half-day period
- **Reduce idle cash** held at branches, lowering insurance and security costs
- **Minimize emergency cash replenishments**, reducing operational overhead
- **Improve liquidity management** across the entire branch network
- **Optimize cash distribution** between branches based on demand patterns
- **Support data-driven decision making** through automated forecasting and recommendations
- **Enhance customer satisfaction** by ensuring cash availability during peak hours

---

## 3. Data Collection & Understanding (Phase 1 — Completed)

### 3.1 Source Data

The project uses historical banking transaction data loaded from **`Bank Cash Optimization.xlsx`**. The dataset contains **81,717 rows × 5 columns** with the following attributes:

| Column | Description | Data Type |
|--------|-------------|-----------|
| `tran_br_code` | Branch Identifier (numeric code) | Object |
| `start_date` | Date of the transaction | Datetime |
| `txn_hour` | Hour of the transaction (0-23) | Numeric |
| `TOTAL_DR` | Total Debits — Cash Withdrawals (outflow) | Float |
| `TOTAL_CR` | Total Credits — Cash Deposits (inflow) | Float |

### 3.2 Key Observations from Raw Data

Upon loading and examining the raw data, the following was observed:

- **No missing values** were found in any of the 5 columns — the data collection process was clean
- The dataset spans multiple branches across a significant time period, with transactions recorded hourly from **hour 0 to hour 23**
- The `start_date` column was stored as an object type and required **conversion to datetime** for time-series analysis
- Multiple branches exist, allowing for **branch-wise performance comparisons** and **branch-specific forecasting**

---

## 4. Data Cleaning & Preprocessing (Phase 1 — Completed)

A comprehensive data cleaning pipeline was implemented in `phase1_data_preparation.py` with the following steps:

### 4.1 Datetime Conversion
The `start_date` column was successfully converted from object type to **datetime64** format, enabling time-based aggregations, trend analysis, and feature extraction (day of week, month, year, etc.).

### 4.2 Missing Value Handling
Although no missing values were present in the raw data, the pipeline was built with robust handling strategies:
- **Cash-related numerical columns** would be filled with 0 (representing no transaction)
- **Other numerical columns** would be filled with the median value
- **Categorical columns** would be filled with 'Unknown'
- Rows with missing critical data (branch code or date) would be dropped

### 4.3 Duplicate Removal
**8 duplicate rows** were identified and removed from the dataset using pandas `drop_duplicates()`, reducing the dataset from 81,717 to **81,709 rows**. These were likely data entry errors or system logging duplicates.

### 4.4 Outlier Detection & Capping (IQR Method)
Outliers in cash-related columns (`TOTAL_DR` and `TOTAL_CR`) were detected using the **Interquartile Range (IQR) method**:
- **Q1 (25th percentile)** and **Q3 (75th percentile)** were calculated for each column
- **IQR = Q3 - Q1** was computed
- **Lower bound = Q1 - 1.5×IQR** and **Upper bound = Q3 + 1.5×IQR** defined acceptable ranges
- Values outside these bounds were **capped (clipped)** at the boundary values rather than removed, preserving data volume while reducing extreme value influence

### 4.5 Data Validation
Final validation ensured:
- **No negative balance values** existed (any found would be clipped to 0)
- **Zero remaining missing values** after all preprocessing
- Final data types were appropriate for modeling

### 4.6 Summary of Data Preparation

| Metric | Value |
|--------|-------|
| Original Rows | 81,717 |
| Duplicates Removed | 8 |
| Final Rows | 81,709 |
| Final Columns | 5 |
| Missing Values | 0 (after cleaning) |
| Output File | `Bank DataSet/cleaned_bank_data.csv` |

The cleaned dataset was saved as a CSV file for the EDA phase.

---

## 5. Exploratory Data Analysis — Key Findings & Insights (Phase 1 — Completed)

The EDA phase, implemented in `phase1_eda.py`, generated **8 critical visualizations** and revealed deep business insights about branch cash flow patterns.

### 5.1 Cash Flow Trends (Plot 1 — Monthly Analysis)

**What we did:** We aggregated total withdrawals (TOTAL_DR) and deposits (TOTAL_CR) by month and plotted them as time-series trend lines.

**Key Findings:**
- **Withdrawals consistently exceed deposits** across all months, meaning the bank network operates at a net cash deficit
- Both withdrawals and deposits show **seasonal fluctuations** with certain months experiencing significantly higher transaction volumes
- The gap between withdrawals and deposits (net outflow) remains relatively stable but widens during high-activity months
- This persistent deficit means the bank must continuously inject cash into the branch network to meet withdrawal demands

**Business Implication:** The bank needs a robust cash replenishment strategy to cover the structural deficit. Forecasting must account for both the baseline deficit and seasonal variations.

### 5.2 Transaction Distribution (Plot 2 — Amount Patterns)

**What we did:** Histograms with Kernel Density Estimation (KDE) overlays were created for both withdrawal and deposit amounts to understand their statistical distributions.

**Key Findings:**
- **Average Withdrawal:** 4.1 Million per transaction
- **Average Deposit:** 3.98 Million per transaction
- Both distributions are **right-skewed (positively skewed)** — most transactions are relatively small amounts, with a long tail of larger transactions
- The mean is higher than the median for both, confirming the skewness
- Withdrawals have a slightly higher average than deposits, consistent with the net deficit position

**Business Implication:** The forecasting model should be robust to skewed distributions. Using median-based metrics for central tendency may be more representative than mean-based approaches.

### 5.3 Peak Hours Analysis (Plot 3 — Hourly Patterns)

**What we did:** We analyzed transaction frequency and average transaction amounts across each hour of the day (0-23).

**Key Findings:**
- **Busiest Hour: 12:00 PM (Noon)** — with **8,980 transactions**, this is the peak activity period
- Transaction frequency follows a **bell-shaped curve** during business hours:
  - **Low activity** from midnight to early morning (hours 0-7)
  - **Ramp up begins** around 8:00 AM as branches open
  - **Peaks sharply** around noon as customers visit during lunch breaks
  - **Gradual decline** through the afternoon
  - **Returns to low levels** by evening
- Average transaction amounts remain relatively **stable across hours**, suggesting that while frequency varies, the per-transaction size is consistent

**Business Implication:** Cash requirements peak around noon. Branches must be adequately stocked before this peak window. The half-day forecast split (morning 8AM-2PM, afternoon 2PM-8PM) aligns naturally with this pattern, with the morning slot covering the ramp-up to peak and the afternoon slot covering the decline.

### 5.4 Branch Performance Analysis (Plot 4 — Branch Comparison)

**What we did:** Branches were ranked by total transaction volume (sum of withdrawals + deposits) to identify the highest and lowest activity branches.

**Key Findings:**
- **Highest Activity Branch: Branch 104** — with **46,090 Million in withdrawals**, this is by far the busiest branch
- The **top 15 branches** handle a disproportionately large volume of transactions, suggesting a **Pareto-like distribution** (80/20 rule) where a small number of branches handle most of the cash flow
- The **bottom 15 branches** have significantly lower transaction volumes, which may raise questions about their operational viability or if they serve specific niche purposes
- There is a **wide disparity** between branch activity levels, meaning a one-size-fits-all cash allocation strategy would be highly inefficient

**Business Implication:** Cash allocation must be **branch-specific**. High-volume branches need frequent, large replenishments while low-volume branches may need only periodic attention. The forecasting model must learn branch-specific patterns.

### 5.5 Correlation Analysis (Plot 5 — DR vs CR Relationship)

**What we did:** A scatter plot of deposits (TOTAL_CR) vs withdrawals (TOTAL_DR) was created with a regression line, alongside a correlation matrix.

**Key Findings:**
- **DR vs CR Correlation Coefficient: 0.4310** — a **moderate positive correlation**
- This means branches with high withdrawals also tend to have high deposits, but the relationship is not extremely strong
- The regression line slope of approximately 0.3-0.4 indicates that for every unit increase in withdrawals, deposits increase by only 0.3-0.4 units, confirming the deficit structure
- Correlation with `txn_hour` is relatively weak, indicating that transaction amounts are not strongly tied to time of day (even though frequency is)

**Business Implication:** The moderate correlation means we cannot simply predict deposits from withdrawals (or vice versa). Both must be modeled as separate but related time series. This supports using multivariate forecasting approaches.

### 5.6 Net Cash Analysis (Plot 6 — Surplus/Deficit Position)

**What we did:** Net cash (TOTAL_CR - TOTAL_DR) was calculated for each transaction and aggregated by month. Positive values represent surplus, negative values represent deficit.

**Key Findings:**
- **Total Net Cash Position: Deficit of -10,317.47 Million** — a massive net cash outflow
- Almost all months show **negative net cash** (red bars), confirming a persistent structural deficit
- Some months show relatively smaller deficits, while others show extreme negative positions
- The deficit pattern is **seasonal** — certain months (potentially salary periods, festival seasons) show more severe outflows

**Business Implication:** The bank must maintain a **constant cash replenishment pipeline**. The system should not only forecast demand but also **flag months with historically severe deficits** for proactive management. This is where the **Cash Recommendation Engine** (Phase 3) becomes critical.

### 5.7 Hourly Transaction Distribution — Box Plots (Plot 7)

**What we did:** Box plots were created showing the distribution of transaction amounts for each hour, separately for withdrawals and deposits.

**Key Findings:**
- The **interquartile range (IQR) is relatively consistent across hours**, suggesting stable transaction sizes
- **Outliers exist across all hours** but were capped during preprocessing
- Both DR and CR distributions show similar patterns across hours, reinforcing that transaction size is less time-dependent than transaction frequency
- The median transaction amount is relatively **constant throughout the day**

**Business Implication:** Since transaction amounts are stable across hours, the key forecasting driver is **transaction frequency** (how many transactions will occur), not the amount per transaction. This simplifies the modeling approach — focus on volume prediction.

### 5.8 Weekly Pattern Analysis (Plot 8 — Day of Week)

**What we did:** Transaction counts and average amounts were analyzed across the seven days of the week.

**Key Findings:**
- **Busiest Day: Friday** — highest transaction count
- **Weekdays (Monday-Friday)** have significantly higher transaction volumes compared to **weekends (Saturday-Sunday)** which show reduced activity
- Average transaction amounts are **relatively consistent across days**, similar to the hourly pattern
- This suggests a **strong weekly seasonality** component that must be incorporated into the forecasting model

**Business Implication:** The forecasting model must include **day-of-week features**. Cash requirements for Fridays will be higher than Sundays. Additionally, if Friday is a holiday in the Islamic context (Jummah prayer timing considerations), this may affect branch operating hours and cash demand patterns. The weekend indicator feature (planned for Phase 2) will capture this.

---

## 6. Summary of Critical EDA Statistics

| Metric | Value |
|--------|-------|
| **Total Net Cash Position** | Deficit of **-10,317.47 Million** |
| **Total Withdrawals (DR)** | Significantly higher than deposits |
| **Total Deposits (CR)** | Insufficient to cover withdrawals |
| **Average Withdrawal per Transaction** | **4.1 Million** |
| **Average Deposit per Transaction** | **3.98 Million** |
| **Busiest Hour** | **12:00 PM** (8,980 transactions) |
| **Busiest Day** | **Friday** |
| **Highest Activity Branch** | **Branch 104** (46,090M in withdrawals) |
| **DR vs CR Correlation** | **0.4310** (moderate positive) |
| **Dataset Size** | **81,709 rows** after cleaning |
| **Number of Branches** | Multiple branches across the network |

---

## 7. Current Project Status

### ✅ Phase 1 — COMPLETED

| Deliverable | Status | Details |
|-------------|--------|---------|
| Data Collection | ✅ Done | 81,717 rows loaded from Excel |
| Data Cleaning | ✅ Done | 8 duplicates removed, outliers capped |
| Clean Dataset | ✅ Done | Saved to `Bank DataSet/cleaned_bank_data.csv` |
| EDA Visualizations | ✅ Done | 8 plots generated and saved |
| EDA Summary Report | ✅ Done | Key statistics and insights documented |
| Data Dictionary | ✅ Done | Column-level metadata generated |

### ⬜ Phase 2 — NOT STARTED

| Task | Status |
|------|--------|
| Feature Engineering (Time Features) | ⬜ Pending |
| Lag Features (Previous DR/CR/Net) | ⬜ Pending |
| Rolling Statistics (Moving Avg, Rolling Std) | ⬜ Pending |
| Calendar Features (Holidays, Salary Days, Ramadan, Eid) | ⬜ Pending |
| Model Development (Linear Regression, Random Forest, XGBoost, etc.) | ⬜ Pending |
| Model Evaluation (MAE, RMSE, MAPE, R²) | ⬜ Pending |
| Model Selection | ⬜ Pending |

### ⬜ Phase 3 — NOT STARTED

| Task | Status |
|------|--------|
| Cash Forecasting (Inflow/Outflow/Net) | ⬜ Pending |
| Branch Risk Analysis | ⬜ Pending |
| Cash Recommendation Engine | ⬜ Pending |
| Dashboard (Visualization) | ⬜ Pending |

---

## 8. Technology Stack & Tools Used

| Category | Tools |
|----------|-------|
| **Language** | Python |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **ML Models (Planned)** | Scikit-learn, XGBoost, LightGBM |
| **Time Series (Planned)** | Statsmodels, Prophet, ARIMA/SARIMA |
| **Deep Learning (Optional)** | TensorFlow/PyTorch (LSTM, GRU) |
| **Development** | VS Code, Jupyter Notebook, Git/GitHub |

---

## 9. Planned Feature Engineering (Phase 2)

Based on EDA insights, the following features will be generated:

### Time Features
- Year, Month, Week, Day, Hour
- **Half-Day Slot** (Morning: 8AM-2PM, Afternoon: 2PM-8PM) — aligns with the core forecasting requirement

### Lag Features (Autoregressive Components)
- Previous period's Withdrawal amount
- Previous period's Deposit amount
- Previous period's Net Cash position
- These capture the temporal dependency in cash flow patterns

### Rolling Statistics
- Moving Average (7-day, 30-day windows)
- Rolling Mean, Sum, and Standard Deviation
- These capture recent trends and volatility

### Calendar Features (External Data Integration)
- **Public Holidays** — reduced branch activity
- **Salary Days** — expected spike in withdrawals (salary disbursements)
- **Ramadan** — changes in business hours and customer behavior
- **Eid** — significant cash demand for celebrations
- **Weekend Indicator** — Saturday/Sunday reduced activity (confirmed by EDA)

### Target Variables
- `TOTAL_DR` (Withdrawals — Cash Outflow)
- `TOTAL_CR` (Deposits — Cash Inflow)
- `Net Cash` = TOTAL_CR - TOTAL_DR (Requirement)

---

## 10. Planned Modeling Strategy (Phase 2)

### Machine Learning Models
- **Linear Regression** — Baseline model for interpretability
- **Random Forest Regressor** — Captures non-linear patterns and feature interactions
- **Gradient Boosting** — Sequential ensemble for high accuracy
- **XGBoost** — Optimized gradient boosting, handles missing data well
- **LightGBM** — Faster training, leaf-wise tree growth

### Time Series Models
- **ARIMA** — Classical univariate time series
- **SARIMA** — Captures seasonality (weekly patterns confirmed in EDA)
- **Prophet (Facebook)** — Handles holidays and changepoints well

### Deep Learning (Optional)
- **LSTM** — Long Short-Term Memory networks for sequence prediction
- **GRU** — Gated Recurrent Units (simpler than LSTM)

### Evaluation Metrics
- **MAE (Mean Absolute Error)** — Interpretable in monetary terms
- **RMSE (Root Mean Square Error)** — Penalizes large errors more
- **MAPE (Mean Absolute Percentage Error)** — Percentage-based accuracy
- **R² Score** — Variance explained by the model

---

## 11. Key Talking Points for the Meeting

### What to Highlight:
1. **Business Context is Clear** — You understand the real-world banking problem (idle cash vs shortages) and can explain it in business terms
2. **Data Preparation was Thorough** — You handled duplicates, outliers, datetime conversion, and validation properly
3. **EDA Reveals Actionable Insights** — Not just plots, but you derived business implications from each visualization
4. **The Persistent Net Deficit is Critical** — 10,317 Million shortfall means the bank is continuously pumping cash into branches; accurate forecasting is essential to optimize this
5. **Seasonal Patterns are Clear** — Friday peak, noon peak, branch 104 dominance — these all inform feature engineering
6. **Phase 1 is Complete & Rigorous** — Clean data, 8 plots, summary report, data dictionary all delivered

### What to Acknowledge:
1. **Phase 2 is the Next Major Milestone** — Feature engineering and model development need to start
2. **External Data Integration** — Holidays, salary days, and religious events need to be sourced and integrated
3. **Calendar Feature Data** — Need to clarify where to source Pakistan-specific holiday, Ramadan, and Eid calendars
4. **Model Selection Will Be Data-Driven** — Multiple models will be compared objectively on MAE/RMSE/MAPE

### Questions to Be Prepared For:
- "How did you handle outliers in the cash data?" → IQR method with capping, not removal
- "What is the most important EDA finding?" → Net deficit of 10,317M and Friday/noon peaks
- "Why half-day forecasting?" → Matches natural transaction patterns (morning build-up, noon peak, afternoon decline)
- "How will you incorporate holidays?" → Calendar features in Phase 2 with external dataset
- "Which model do you expect to perform best?" → XGBoost or LightGBM for tabular data, with SARIMA as a time series baseline
- "How will you measure success?" → MAPE < 15%, R² > 0.85, and reduction in emergency replenishments
- "What branch-specific challenges exist?" → Branch 104 is dominant (46,090M), low-volume branches need different strategy
- "How does the DR-CR correlation (0.43) affect modeling?" → Confirms we need separate models or multivariate approach

---

## 12. Presentation Flow Recommendation

For a structured 15-20 minute presentation, follow this order:

1. **Problem Statement (2 min)** — Why cash forecasting matters for banks
2. **Project Objectives (1 min)** — What the system aims to achieve
3. **Data Overview (2 min)** — Source data, size, columns, quality
4. **Data Cleaning Approach (2 min)** — Duplicates, outliers, validation
5. **EDA Walkthrough (8 min)** — Go through the 8 plots, highlighting the key insight from each
   - Start with the **big picture** (Cash Flow Trend — Plot 1)
   - Move to **distributions** (Plot 2)
   - Then **temporal patterns** (Plots 3, 7, 8 — peak hours, hourly distribution, weekly)
   - Then **branch analysis** (Plot 4)
   - Then **relationships** (Plot 5 — correlation)
   - End with **net position** (Plot 6 — the most business-critical insight)
6. **Summary of Critical Insights (2 min)** — The key numbers that matter
7. **Next Steps & Timeline (2 min)** — Phase 2 feature engineering, modeling, evaluation
8. **Open for Questions** — Be prepared with answers from section 11 above

---

## 13. Quick Reference — EDA Plot Summary

| Plot | Key Question Answered | One-Line Insight |
|------|----------------------|-------------------|
| 1. Cash Flow Trend | Are withdrawals growing over time? | Persistent deficit with seasonal fluctuations |
| 2. Transaction Distribution | What do transaction amounts look like? | Right-skewed; avg withdrawal = 4.1M, avg deposit = 3.98M |
| 3. Peak Hours Analysis | When is the bank busiest? | 12:00 PM peak with 8,980 transactions |
| 4. Branch Performance | Which branches need most cash? | Branch 104 dominates with 46,090M in withdrawals |
| 5. Correlation Analysis | Do deposits predict withdrawals? | Moderate correlation (0.431) — both must be modeled |
| 6. Net Cash Analysis | Is the bank net positive or negative? | Massive deficit of -10,317.47M — constant cash injection needed |
| 7. Hourly Boxplot | How do amounts vary by hour? | Stable distribution — frequency drives demand, not amount |
| 8. Weekly Pattern | Which day is busiest? | Friday is peak, weekends are low |

---

## 14. Projects Overall Progress Summary

| Phase | Description | Status | Completion |
|-------|-------------|--------|------------|
| Phase 1 | Data Understanding & Preparation | ✅ **Completed** | **100%** |
| Phase 2 | Feature Engineering & Model Development | ⬜ **Not Started** | **0%** |
| Phase 3 | Forecasting & Decision Support | ⬜ **Not Started** | **0%** |

**Overall Project Progress: ~30% complete**

The foundation is solid — clean data, rich EDA insights, and clear understanding of the business problem. The next critical step is to execute Phase 2 (Feature Engineering + Model Development), which will transform insights into predictive power.

---

*End of Meeting Preparation Document*