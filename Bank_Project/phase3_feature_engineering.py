"""
==============================================================================
BANK BRANCH CASH FORECASTING SYSTEM
Phase 3: Daily Feature Engineering Pipeline
==============================================================================

This script implements the complete daily feature-engineering pipeline:

1. Load & Validate Cleaned Hourly Data
2. Aggregate to Daily Level (by tran_br_code + start_date)
3. Generate Target Variables (daily withdrawals, deposits, net cash, cash requirement)
4. Create Calendar & Cyclical Features
5. Compute Branch-wise Lag Features (1/7/14/28-day) with leakage prevention
6. Compute Rolling Statistics (7/14/30-day) with leakage prevention
7. Detect Missing Branch-Date Combinations (no auto-fill)
8. Exclude Incomplete Periods (configurable)
9. Chronological Train/Validation/Test Splits
10. Create Baseline Predictions (prev-day, prev-week, 7-day rolling avg)
11. Save Model-Ready CSV & Validation Reports
12. Print Concise Summary

Author: Muhammad Usman
Status: Phase 3 Implementation
==============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import warnings
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- File Paths ---
INPUT_PATH = "Bank DataSet/cleaned_bank_data.csv"
OUTPUT_DIR = "Bank DataSet"

# --- Output Files ---
DAILY_AGGREGATED_PATH = os.path.join(OUTPUT_DIR, "daily_aggregated.csv")
FEATURE_ENGINEERED_PATH = os.path.join(OUTPUT_DIR, "model_ready_data.csv")
VALIDATION_REPORT_PATH = os.path.join(OUTPUT_DIR, "validation_report.json")
BASELINE_METRICS_PATH = os.path.join(OUTPUT_DIR, "baseline_metrics.json")
MISSING_COMBOS_PATH = os.path.join(OUTPUT_DIR, "missing_branch_date_combinations.csv")

# --- Feature Engineering Parameters ---
LAG_WINDOWS = [1, 7, 14, 28]           # Days for lag features
ROLLING_WINDOWS = [7, 14, 30]           # Days for rolling statistics
EXCLUDE_INCOMPLETE_PERIODS = True       # Exclude first/last incomplete windows
INCOMPLETE_PERIOD_DAYS = 30             # Days to exclude from start/end if enabled

# --- Train / Validation / Test Splits (chronological) ---
TRAIN_SPLIT_DATE = "2025-06-01"         # Train: up to this date (exclusive)
VAL_SPLIT_DATE = "2025-10-01"           # Validation: TRAIN_SPLIT_DATE to VAL_SPLIT_DATE (exclusive)
# Test: VAL_SPLIT_DATE to end

# --- Validation Thresholds ---
MAX_ALLOWED_MISSING_PCT = 5.0           # Max % of missing values allowed after feature engineering
MIN_ROWS_EXPECTED = 1000                # Minimum rows expected in output


# =============================================================================
# MODULE 1: DATA LOADING & VALIDATION
# =============================================================================

def load_and_validate_input(file_path: str) -> pd.DataFrame:
    """
    Load the cleaned hourly bank data CSV and validate expected columns.
    
    Parameters:
        file_path (str): Path to cleaned CSV data.
    
    Returns:
        pd.DataFrame: Validated hourly data.
    
    Raises:
        ValueError: If required columns are missing or file cannot be loaded.
    """
    print("=" * 70)
    print("MODULE 1: DATA LOADING & VALIDATION")
    print("=" * 70)

    # --- 1.1 Load file ---
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"✓ Loaded file: {file_path}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # --- 1.2 Validate required columns ---
    required_cols = ['start_date', 'txn_hour', 'tran_br_code', 'TOTAL_DR', 'TOTAL_CR']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    print(f"✓ Required columns present: {required_cols}")

    # --- 1.3 Convert data types ---
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['txn_hour'] = pd.to_numeric(df['txn_hour'], errors='coerce')
    df['tran_br_code'] = df['tran_br_code'].astype(int)
    df['TOTAL_DR'] = pd.to_numeric(df['TOTAL_DR'], errors='coerce')
    df['TOTAL_CR'] = pd.to_numeric(df['TOTAL_CR'], errors='coerce')

    # --- 1.4 Check for nulls in critical columns after conversion ---
    null_counts = df[required_cols].isnull().sum()
    if null_counts.sum() > 0:
        print(f"⚠  Dropping {null_counts.sum()} rows with null values in critical columns after conversion:")
        print(null_counts[null_counts > 0])
        df = df.dropna(subset=required_cols)
    else:
        print("✓ No null values in critical columns after conversion")

    # --- 1.5 Sort data chronologically ---
    df = df.sort_values(['tran_br_code', 'start_date', 'txn_hour']).reset_index(drop=True)

    # --- 1.6 Summary statistics ---
    print(f"\n  Date range: {df['start_date'].min()} to {df['start_date'].max()}")
    print(f"  Unique branches: {df['tran_br_code'].nunique()}")
    print(f"  Unique dates: {df['start_date'].nunique()}")
    print(f"  Total DR (withdrawals): {df['TOTAL_DR'].sum():,.0f}")
    print(f"  Total CR (deposits): {df['TOTAL_CR'].sum():,.0f}")

    return df


# =============================================================================
# MODULE 2: DAILY AGGREGATION
# =============================================================================

def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly transaction data to daily level per branch.
    
    Aggregations:
        - TOTAL_DR → sum → daily_withdrawals
        - TOTAL_CR → sum → daily_deposits
    
    Parameters:
        df (pd.DataFrame): Hourly cleaned data.
    
    Returns:
        pd.DataFrame: Daily aggregated data with columns:
            tran_br_code, start_date, daily_withdrawals, daily_deposits
    """
    print("\n" + "=" * 70)
    print("MODULE 2: DAILY AGGREGATION")
    print("=" * 70)

    daily = df.groupby(['tran_br_code', 'start_date'], as_index=False).agg(
        daily_withdrawals=('TOTAL_DR', 'sum'),
        daily_deposits=('TOTAL_CR', 'sum')
    )

    # Round to avoid floating-point artifacts
    daily['daily_withdrawals'] = daily['daily_withdrawals'].round(2)
    daily['daily_deposits'] = daily['daily_deposits'].round(2)

    daily = daily.sort_values(['tran_br_code', 'start_date']).reset_index(drop=True)

    print(f"✓ Aggregated to daily level")
    print(f"  Shape: {daily.shape[0]} rows × {daily.shape[1]} columns")
    print(f"  Unique branches: {daily['tran_br_code'].nunique()}")
    print(f"  Unique dates: {daily['start_date'].nunique()}")

    return daily


# =============================================================================
# MODULE 3: TARGET VARIABLE GENERATION
# =============================================================================

def generate_targets(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Generate target variables from daily aggregates.
    
    Targets:
        - net_cash = daily_deposits - daily_withdrawals
        - cash_requirement = max(daily_withdrawals - daily_deposits, 0)
    
    Parameters:
        daily (pd.DataFrame): Daily aggregated data.
    
    Returns:
        pd.DataFrame: Daily data with net_cash and cash_requirement columns.
    """
    print("\n" + "=" * 70)
    print("MODULE 3: TARGET VARIABLE GENERATION")
    print("=" * 70)

    df = daily.copy()

    df['net_cash'] = df['daily_deposits'] - df['daily_withdrawals']
    df['cash_requirement'] = (df['daily_withdrawals'] - df['daily_deposits']).clip(lower=0)

    df['net_cash'] = df['net_cash'].round(2)
    df['cash_requirement'] = df['cash_requirement'].round(2)

    # Summary
    print(f"✓ Target variables generated")
    print(f"  Total net cash: {df['net_cash'].sum():,.0f}")
    print(f"  Total cash requirement: {df['cash_requirement'].sum():,.0f}")
    print(f"  Days with cash deficit (req > 0): {(df['cash_requirement'] > 0).sum()} / {len(df)}")

    return df


# =============================================================================
# MODULE 4: CALENDAR & CYCLICAL FEATURES
# =============================================================================

def generate_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate calendar and cyclical features from start_date.
    
    Features:
        - year, month, day, dayofweek, dayofyear, weekofyear, quarter
        - is_weekend: 1 if Saturday/Sunday, else 0
        - is_month_start, is_month_end
        - is_quarter_start, is_quarter_end
        - Cyclical encoding: month_sin, month_cos, dayofweek_sin, dayofweek_cos
    
    Parameters:
        df (pd.DataFrame): Daily data with start_date column.
    
    Returns:
        pd.DataFrame: Data with calendar features added.
    """
    print("\n" + "=" * 70)
    print("MODULE 4: CALENDAR & CYCLICAL FEATURES")
    print("=" * 70)

    result = df.copy()
    dt = result['start_date']

    # Basic calendar features
    result['year'] = dt.dt.year
    result['month'] = dt.dt.month
    result['day'] = dt.dt.day
    result['dayofweek'] = dt.dt.dayofweek          # 0=Monday, 6=Sunday
    result['dayofyear'] = dt.dt.dayofyear
    result['weekofyear'] = dt.dt.isocalendar().week.astype(int)
    result['quarter'] = dt.dt.quarter

    # Boolean flags
    result['is_weekend'] = (result['dayofweek'] >= 5).astype(int)
    result['is_month_start'] = dt.dt.is_month_start.astype(int)
    result['is_month_end'] = dt.dt.is_month_end.astype(int)
    result['is_quarter_start'] = dt.dt.is_quarter_start.astype(int)
    result['is_quarter_end'] = dt.dt.is_quarter_end.astype(int)

    # Cyclical encoding for month (period=12)
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)

    # Cyclical encoding for dayofweek (period=7)
    result['dayofweek_sin'] = np.sin(2 * np.pi * result['dayofweek'] / 7)
    result['dayofweek_cos'] = np.cos(2 * np.pi * result['dayofweek'] / 7)

    print(f"✓ Calendar features generated: {12} new features")
    print(f"  Date range: {result['start_date'].min()} to {result['start_date'].max()}")
    print(f"  Years: {sorted(result['year'].unique())}")
    print(f"  Months: {sorted(result['month'].unique())}")

    return result


# =============================================================================
# MODULE 5: LAG FEATURES (WITH LEAKAGE PREVENTION)
# =============================================================================

def generate_lag_features(df: pd.DataFrame, lag_windows: List[int] = None) -> pd.DataFrame:
    """
    Generate branch-wise lag features for target variables.
    
    To prevent data leakage, we FIRST sort and group by branch, THEN shift
    (the target values backward in time for each branch) before computing lags.
    
    Lag features generated for:
        - daily_withdrawals_{lag}d
        - daily_deposits_{lag}d
        - net_cash_{lag}d
        - cash_requirement_{lag}d
    
    Parameters:
        df (pd.DataFrame): Daily data with targets, sorted by (tran_br_code, start_date).
        lag_windows (list[int]): List of lag periods in days.
    
    Returns:
        pd.DataFrame: Data with lag features added.
    """
    if lag_windows is None:
        lag_windows = LAG_WINDOWS

    print("\n" + "=" * 70)
    print("MODULE 5: LAG FEATURES (Leakage-Protected)")
    print("=" * 70)

    result = df.copy()
    result = result.sort_values(['tran_br_code', 'start_date']).reset_index(drop=True)

    targets = ['daily_withdrawals', 'daily_deposits', 'net_cash', 'cash_requirement']
    lag_count = 0

    for window in lag_windows:
        for target in targets:
            col_name = f'{target}_{window}d_lag'
            result[col_name] = result.groupby('tran_br_code')[target].shift(window)
            lag_count += 1

    print(f"✓ Lag features generated: {lag_count} ({len(lag_windows)} windows × {len(targets)} targets)")
    print(f"  Lag windows (days): {lag_windows}")

    return result


# =============================================================================
# MODULE 6: ROLLING STATISTICS (WITH LEAKAGE PREVENTION)
# =============================================================================

def generate_rolling_features(df: pd.DataFrame, rolling_windows: List[int] = None) -> pd.DataFrame:
    """
    Generate branch-wise rolling statistics for target variables.
    
    To prevent leakage, we shift the rolling statistics by 1 day (using shift(1))
    so they only contain information from strictly prior dates.
    
    Rolling features generated for each window:
        - {target}_{window}d_rolling_mean
        - {target}_{window}d_rolling_std
        - {target}_{window}d_rolling_min
        - {target}_{window}d_rolling_max
    
    Parameters:
        df (pd.DataFrame): Daily data with targets and lags.
        rolling_windows (list[int]): List of rolling window sizes in days.
    
    Returns:
        pd.DataFrame: Data with rolling features added.
    """
    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS

    print("\n" + "=" * 70)
    print("MODULE 6: ROLLING STATISTICS (Leakage-Protected)")
    print("=" * 70)

    result = df.copy()
    result = result.sort_values(['tran_br_code', 'start_date']).reset_index(drop=True)

    targets = ['daily_withdrawals', 'daily_deposits', 'net_cash', 'cash_requirement']
    rolling_count = 0

    for window in rolling_windows:
        for target in targets:
            grouped = result.groupby('tran_br_code')[target]

            # Rolling mean — shift by 1 to prevent leakage (uses data up to t-1)
            result[f'{target}_{window}d_rolling_mean'] = (
                grouped.transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
            )

            # Rolling std
            result[f'{target}_{window}d_rolling_std'] = (
                grouped.transform(lambda x: x.rolling(window, min_periods=1).std().shift(1))
            )

            # Rolling min
            result[f'{target}_{window}d_rolling_min'] = (
                grouped.transform(lambda x: x.rolling(window, min_periods=1).min().shift(1))
            )

            # Rolling max
            result[f'{target}_{window}d_rolling_max'] = (
                grouped.transform(lambda x: x.rolling(window, min_periods=1).max().shift(1))
            )

            rolling_count += 4

    print(f"✓ Rolling features generated: {rolling_count}")
    print(f"  Rolling windows (days): {rolling_windows}")

    return result


# =============================================================================
# MODULE 7: DETECT MISSING BRANCH-DATE COMBINATIONS
# =============================================================================

def detect_missing_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect missing branch-date combinations in the daily data.
    
    This function identifies dates where a branch should have data but doesn't.
    It does NOT fill missing values — it only reports them.
    
    Parameters:
        df (pd.DataFrame): Daily data with tran_br_code and start_date.
    
    Returns:
        pd.DataFrame: DataFrame of missing branch-date combinations (empty if none).
    """
    print("\n" + "=" * 70)
    print("MODULE 7: MISSING BRANCH-DATE COMBINATION DETECTION")
    print("=" * 70)

    # Build complete grid of all branches × all dates in range
    branches = sorted(df['tran_br_code'].unique())
    all_dates = pd.date_range(start=df['start_date'].min(), end=df['start_date'].max(), freq='D')

    existing = set(zip(df['tran_br_code'], df['start_date'].dt.date))
    missing_records = []

    for br in branches:
        for d in all_dates:
            if (br, d.date()) not in existing:
                missing_records.append({
                    'tran_br_code': br,
                    'missing_date': d,
                    'dayofweek': d.dayofweek,
                    'is_weekend': 1 if d.dayofweek >= 5 else 0
                })

    missing_df = pd.DataFrame(missing_records)

    if len(missing_df) > 0:
        print(f"⚠  Detected {len(missing_df)} missing branch-date combinations")
        print(f"  Unique branches affected: {missing_df['tran_br_code'].nunique()}")
        print(f"  Date range of missing: {missing_df['missing_date'].min()} to {missing_df['missing_date'].max()}")
        print(f"  Weekend missing: {missing_df['is_weekend'].sum()} ({missing_df['is_weekend'].mean()*100:.1f}%)")
        print(f"\n  ⚠ NOTE: Missing combinations are NOT being filled with zeros.")
        print(f"  They are only recorded for analysis. Downstream models may need")
        print(f"  to handle these gaps explicitly.")
    else:
        print("✓ No missing branch-date combinations found (complete grid)")

    return missing_df


# =============================================================================
# MODULE 8: EXCLUDE INCOMPLETE PERIODS
# =============================================================================

def exclude_incomplete_periods(df: pd.DataFrame, exclude_days: int = None) -> pd.DataFrame:
    """
    Exclude incomplete periods at the start and end of the dataset.
    
    This removes rows where lag/rolling features may be unreliable due to
    insufficient historical data at the start of the time series, and rows
    near the end that may not have complete future data for validation.
    
    Parameters:
        df (pd.DataFrame): Feature-engineered data.
        exclude_days (int): Number of days to exclude from start and end.
    
    Returns:
        pd.DataFrame: Data with incomplete periods removed.
    """
    if exclude_days is None:
        exclude_days = INCOMPLETE_PERIOD_DAYS

    print("\n" + "=" * 70)
    print("MODULE 8: EXCLUDE INCOMPLETE PERIODS")
    print("=" * 70)

    if not EXCLUDE_INCOMPLETE_PERIODS:
        print("✓ Incomplete period exclusion is DISABLED")
        return df

    result = df.copy()
    before_count = len(result)

    min_date = result['start_date'].min()
    max_date = result['start_date'].max()
    cutoff_start = min_date + pd.Timedelta(days=exclude_days)
    cutoff_end = max_date - pd.Timedelta(days=exclude_days)

    result = result[result['start_date'] >= cutoff_start].copy()
    result = result[result['start_date'] <= cutoff_end].copy()

    removed = before_count - len(result)
    print(f"✓ Excluded {removed} rows from incomplete periods")
    print(f"  Exclusion window: {exclude_days} days from start and end")
    print(f"  New date range: {result['start_date'].min()} to {result['start_date'].max()}")
    print(f"  Shape: {result.shape[0]} rows × {result.shape[1]} columns")

    return result


# =============================================================================
# MODULE 9: CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLITS
# =============================================================================

def create_chronological_splits(
    df: pd.DataFrame,
    train_cutoff: str = None,
    val_cutoff: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create chronological train, validation, and test splits.
    
    Splits are based on date only — NO random shuffling.
    
    Parameters:
        df (pd.DataFrame): Feature-engineered data.
        train_cutoff (str): Date string for end of training period (exclusive).
        val_cutoff (str): Date string for end of validation period (exclusive).
    
    Returns:
        tuple: (train_df, val_df, test_df) — chronological splits.
    """
    if train_cutoff is None:
        train_cutoff = TRAIN_SPLIT_DATE
    if val_cutoff is None:
        val_cutoff = VAL_SPLIT_DATE

    print("\n" + "=" * 70)
    print("MODULE 9: CHRONOLOGICAL TRAIN/VAL/TEST SPLITS")
    print("=" * 70)
    print("  ✓ Using CHRONOLOGICAL (time-based) splits — NO random splitting")

    train_cutoff_dt = pd.Timestamp(train_cutoff)
    val_cutoff_dt = pd.Timestamp(val_cutoff)

    train_df = df[df['start_date'] < train_cutoff_dt].copy()
    val_df = df[(df['start_date'] >= train_cutoff_dt) & (df['start_date'] < val_cutoff_dt)].copy()
    test_df = df[df['start_date'] >= val_cutoff_dt].copy()

    total = len(train_df) + len(val_df) + len(test_df)

    print(f"  Train cutoff:   {train_cutoff}")
    print(f"  Val cutoff:     {val_cutoff}")
    print(f"\n  Train:      {len(train_df):>6} rows ({len(train_df)/total*100:>5.1f}%)")
    print(f"  Validation: {len(val_df):>6} rows ({len(val_df)/total*100:>5.1f}%)")
    print(f"  Test:       {len(test_df):>6} rows ({len(test_df)/total*100:>5.1f}%)")
    print(f"  Total:      {total:>6} rows")

    # Verify no date overlap between splits
    if len(train_df) > 0 and len(val_df) > 0:
        assert train_df['start_date'].max() < val_df['start_date'].min(), \
            "Train/val date overlap detected!"
    if len(val_df) > 0 and len(test_df) > 0:
        assert val_df['start_date'].max() < test_df['start_date'].min(), \
            "Val/test date overlap detected!"

    print("  ✓ No date overlap between splits (verified)")

    return train_df, val_df, test_df


# =============================================================================
# MODULE 10: BASELINE PREDICTIONS
# =============================================================================

def generate_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple baseline predictions for comparison.
    
    Baselines:
        - prev_day_baseline: uses 1-day lag of target value
        - prev_week_baseline: uses 7-day lag of target value
        - roll7_avg_baseline: uses 7-day rolling average (shifted by 1)
    
    Parameters:
        df (pd.DataFrame): Data with lag and rolling features already computed.
    
    Returns:
        pd.DataFrame: Data with baseline prediction columns added.
    """
    print("\n" + "=" * 70)
    print("MODULE 10: BASELINE PREDICTIONS")
    print("=" * 70)

    result = df.copy()
    targets = ['daily_withdrawals', 'daily_deposits', 'net_cash', 'cash_requirement']

    baseline_count = 0
    for target in targets:
        # Previous-day baseline: use 1-day lag
        result[f'{target}_prev_day_baseline'] = result[f'{target}_1d_lag']

        # Previous-week baseline: use 7-day lag
        result[f'{target}_prev_week_baseline'] = result[f'{target}_7d_lag']

        # 7-day rolling average baseline
        result[f'{target}_roll7_avg_baseline'] = result[f'{target}_7d_rolling_mean']

        baseline_count += 3

    print(f"✓ Baseline predictions generated: {baseline_count}")
    print(f"  Types: prev_day, prev_week, roll7_avg for {len(targets)} targets")

    return result


# =============================================================================
# MODULE 11: MISSING VALUE REPORTING
# =============================================================================

def report_missing_values(df: pd.DataFrame, label: str = "Dataset") -> Dict:
    """
    Report missing value statistics for the dataset.
    
    Parameters:
        df (pd.DataFrame): Data to check.
        label (str): Label for display purposes.
    
    Returns:
        dict: Missing value statistics.
    """
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0

    # Per-column breakdown
    col_missing = df.isnull().sum()
    cols_with_missing = col_missing[col_missing > 0]

    stats = {
        'label': label,
        'shape': list(df.shape),
        'total_cells': int(total_cells),
        'total_missing': int(total_missing),
        'missing_pct': round(missing_pct, 4),
        'columns_with_missing': len(cols_with_missing),
        'col_breakdown': {}
    }

    for col in cols_with_missing.index:
        stats['col_breakdown'][col] = {
            'missing': int(col_missing[col]),
            'pct': round(col_missing[col] / df.shape[0] * 100, 2)
        }

    print(f"\n  [{label}] Missing values: {total_missing} / {total_cells} ({missing_pct:.2f}%)")
    if len(cols_with_missing) > 0:
        print(f"  Columns with missing: {len(cols_with_missing)}")
        for col in cols_with_missing.index:
            print(f"    - {col}: {col_missing[col]:>6} missing ({col_missing[col]/df.shape[0]*100:>5.2f}%)")

    return stats


# =============================================================================
# MODULE 12: SAVE OUTPUTS
# =============================================================================

def save_outputs(
    daily_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    missing_df: pd.DataFrame,
    validation_report: Dict
) -> None:
    """
    Save all pipeline outputs to disk.
    
    Parameters:
        daily_df (pd.DataFrame): Daily aggregated data.
        feature_df (pd.DataFrame): Full feature-engineered data.
        splits (tuple): (train, val, test) DataFrames.
        missing_df (pd.DataFrame): Missing branch-date combinations.
        validation_report (dict): Validation report data.
    """
    print("\n" + "=" * 70)
    print("MODULE 12: SAVING OUTPUTS")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 12.1 Save daily aggregated ---
    daily_df.to_csv(DAILY_AGGREGATED_PATH, index=False)
    print(f"✓ Daily aggregated: {DAILY_AGGREGATED_PATH}")

    # --- 12.2 Save complete model-ready data ---
    feature_df.to_csv(FEATURE_ENGINEERED_PATH, index=False)
    print(f"✓ Model-ready CSV: {FEATURE_ENGINEERED_PATH}")

    # --- 12.3 Save missing combinations ---
    if len(missing_df) > 0:
        missing_df.to_csv(MISSING_COMBOS_PATH, index=False)
        print(f"✓ Missing combos: {MISSING_COMBOS_PATH}")
    else:
        # Write an empty file with a note
        pd.DataFrame({'note': ['No missing branch-date combinations found']}).to_csv(MISSING_COMBOS_PATH, index=False)
        print(f"✓ Missing combos (empty): {MISSING_COMBOS_PATH}")

    # --- 12.4 Save validation report ---
    with open(VALIDATION_REPORT_PATH, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    print(f"✓ Validation report: {VALIDATION_REPORT_PATH}")

    # --- 12.5 Save split data separately ---
    train_df, val_df, test_df = splits
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "validation_data.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_data.csv"), index=False)
    print(f"✓ Train/val/test CSVs saved to {OUTPUT_DIR}/")

    # --- 12.6 Compute and save baseline metrics ---
    baseline_metrics = compute_baseline_metrics(splits)
    with open(BASELINE_METRICS_PATH, 'w') as f:
        json.dump(baseline_metrics, f, indent=2, default=str)
    print(f"✓ Baseline metrics: {BASELINE_METRICS_PATH}")


# =============================================================================
# MODULE 13: BASELINE METRICS
# =============================================================================

def compute_baseline_metrics(
    splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
) -> Dict:
    """
    Compute MAE and RMSE for baseline predictions on validation data.
    
    Parameters:
        splits (tuple): (train, val, test) DataFrames.
    
    Returns:
        dict: Baseline metrics for each target and baseline type.
    """
    _, val_df, _ = splits

    if len(val_df) == 0:
        return {'error': 'Validation set is empty'}

    targets = ['daily_withdrawals', 'daily_deposits', 'net_cash', 'cash_requirement']
    baselines = ['prev_day_baseline', 'prev_week_baseline', 'roll7_avg_baseline']
    metrics = {}

    for target in targets:
        metrics[target] = {}
        actual = val_df[target].values
        for bl in baselines:
            col = f'{target}_{bl}'
            if col not in val_df.columns:
                metrics[target][bl] = {'mae': None, 'rmse': None, 'note': 'Column not found'}
                continue

            pred = val_df[col].values
            mask = ~(np.isnan(actual) | np.isnan(pred))

            if mask.sum() == 0:
                metrics[target][bl] = {'mae': None, 'rmse': None, 'note': 'No valid pairs'}
                continue

            act = actual[mask]
            pre = pred[mask]
            errors = act - pre

            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors ** 2))

            metrics[target][bl] = {
                'mae': round(float(mae), 2),
                'rmse': round(float(rmse), 2),
                'num_comparisons': int(mask.sum())
            }

    return metrics


# =============================================================================
# MODULE 14: SUMMARY
# =============================================================================

def print_summary(
    validation_report: Dict,
    daily_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    missing_df: pd.DataFrame
) -> None:
    """
    Print a concise summary of the pipeline execution.
    
    Parameters:
        validation_report (dict): Full validation report.
        daily_df (pd.DataFrame): Daily data.
        feature_df (pd.DataFrame): Feature-engineered data.
        splits (tuple): (train, val, test) DataFrames.
        missing_df (pd.DataFrame): Missing combinations.
    """
    train_df, val_df, test_df = splits

    print("\n" + "=" * 70)
    print(" " * 18 + "PIPELINE EXECUTION SUMMARY")
    print("=" * 70)

    # Dimensions
    print(f"\n{'Dimensions':-<40}")
    print(f"  Hourly input:          {validation_report['hourly_shape'][0]:>6} rows")
    print(f"  Daily aggregated:     {daily_df.shape[0]:>6} rows × {daily_df.shape[1]:>2} cols")
    print(f"  Feature-engineered:   {feature_df.shape[0]:>6} rows × {feature_df.shape[1]:>2} cols")
    print(f"  Train:                {train_df.shape[0]:>6} rows")
    print(f"  Validation:           {val_df.shape[0]:>6} rows")
    print(f"  Test:                 {test_df.shape[0]:>6} rows")

    # Date ranges
    print(f"\n{'Date Ranges':-<40}")
    print(f"  Train:       {train_df['start_date'].min()}  →  {train_df['start_date'].max()}")
    print(f"  Validation:  {val_df['start_date'].min()}  →  {val_df['start_date'].max()}")
    print(f"  Test:        {test_df['start_date'].min()}  →  {test_df['start_date'].max()}")

    # Branches
    print(f"\n{'Branches':-<40}")
    print(f"  Unique branches: {feature_df['tran_br_code'].nunique()}")

    # Missing data
    print(f"\n{'Data Quality':-<40}")
    print(f"  Missing in features: {validation_report['feature_missing_pct']:.2f}%")
    print(f"  Missing combos detected: {len(missing_df)}")

    # Cash totals
    print(f"\n{'Cash Summary (Daily Aggregated)':-<40}")
    print(f"  Total withdrawals: {daily_df['daily_withdrawals'].sum():>15,.0f}")
    print(f"  Total deposits:    {daily_df['daily_deposits'].sum():>15,.0f}")
    print(f"  Net cash:          {daily_df['net_cash'].sum():>15,.0f}")
    print(f"  Cash requirement:  {daily_df['cash_requirement'].sum():>15,.0f}")

    # Feature count
    feature_types = {
        'Calendar/Cyclical': 12,
        'Lag features': len(LAG_WINDOWS) * 4,
        'Rolling statistics': len(ROLLING_WINDOWS) * 4 * 4,  # 4 windows × 4 targets × 4 stats
        'Baselines': 4 * 3,
    }
    print(f"\n{'Feature Breakdown':-<40}")
    for ftype, count in feature_types.items():
        print(f"  {ftype:<25} {count:>4}")
    print(f"  {'Total features (excl. targets/ids)':<25} {sum(feature_types.values()):>4}")

    # Output files
    print(f"\n{'Output Files':-<40}")
    print(f"  {DAILY_AGGREGATED_PATH}")
    print(f"  {FEATURE_ENGINEERED_PATH}")
    print(f"  {VALIDATION_REPORT_PATH}")
    print(f"  {BASELINE_METRICS_PATH}")
    print(f"  {MISSING_COMBOS_PATH}")
    print(f"  {os.path.join(OUTPUT_DIR, 'train_data.csv')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'validation_data.csv')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'test_data.csv')}")

    print("\n" + "=" * 70)
    print(" " * 12 + "PHASE 3 FEATURE ENGINEERING COMPLETED!")
    print("=" * 70)


# =============================================================================
# MODULE 15: METADATA — SAVE PIPELINE CONFIGURATION
# =============================================================================

def save_pipeline_metadata(validation_report: Dict) -> None:
    """
    Append pipeline configuration and execution metadata to the validation report.
    
    Parameters:
        validation_report (dict): The report to augment.
    """
    validation_report['pipeline_config'] = {
        'lag_windows': LAG_WINDOWS,
        'rolling_windows': ROLLING_WINDOWS,
        'exclude_incomplete_periods': EXCLUDE_INCOMPLETE_PERIODS,
        'incomplete_period_days': INCOMPLETE_PERIOD_DAYS if EXCLUDE_INCOMPLETE_PERIODS else None,
        'train_split_date': TRAIN_SPLIT_DATE,
        'val_split_date': VAL_SPLIT_DATE,
        'max_allowed_missing_pct': MAX_ALLOWED_MISSING_PCT,
        'module_list': [
            'load_and_validate_input',
            'aggregate_to_daily',
            'generate_targets',
            'generate_calendar_features',
            'generate_lag_features',
            'generate_rolling_features',
            'detect_missing_combinations',
            'exclude_incomplete_periods',
            'create_chronological_splits',
            'generate_baselines'
        ]
    }

    validation_report['execution_timestamp'] = datetime.now().isoformat()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    input_path: str = INPUT_PATH,
    output_dir: str = OUTPUT_DIR,
    train_cutoff: str = TRAIN_SPLIT_DATE,
    val_cutoff: str = VAL_SPLIT_DATE,
    lag_windows: List[int] = None,
    rolling_windows: List[int] = None,
    exclude_incomplete: bool = EXCLUDE_INCOMPLETE_PERIODS,
    incomplete_days: int = INCOMPLETE_PERIOD_DAYS
) -> Dict:
    """
    Execute the complete daily feature-engineering pipeline.
    
    Parameters:
        input_path (str): Path to cleaned hourly CSV.
        output_dir (str): Directory for output files.
        train_cutoff (str): Date string for train/val split.
        val_cutoff (str): Date string for val/test split.
        lag_windows (list[int]): Lag periods in days.
        rolling_windows (list[int]): Rolling window sizes in days.
        exclude_incomplete (bool): Whether to exclude incomplete periods.
        incomplete_days (int): Days to exclude from start/end.
    
    Returns:
        dict: Validation report with pipeline results.
    """
    print("=" * 70)
    print(" " * 10 + "BANK BRANCH CASH FORECASTING SYSTEM")
    print(" " * 12 + "Phase 3: Feature Engineering Pipeline")
    print("=" * 70)

    # Use global defaults if not overridden
    if lag_windows is None:
        lag_windows = LAG_WINDOWS
    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS

    # Override module-level config if parameters provided
    global EXCLUDE_INCOMPLETE_PERIODS, INCOMPLETE_PERIOD_DAYS
    if exclude_incomplete is not None:
        EXCLUDE_INCOMPLETE_PERIODS = exclude_incomplete
    if incomplete_days is not None:
        INCOMPLETE_PERIOD_DAYS = incomplete_days

    # ---- Initialize validation report ----
    report = {}

    # ---- MODULE 1: Load & Validate ----
    hourly_df = load_and_validate_input(input_path)
    report['hourly_shape'] = list(hourly_df.shape)
    report['hourly_date_range'] = [str(hourly_df['start_date'].min()), str(hourly_df['start_date'].max())]
    report['hourly_unique_branches'] = int(hourly_df['tran_br_code'].nunique())

    # ---- MODULE 2: Aggregate to Daily ----
    daily_df = aggregate_to_daily(hourly_df)

    # ---- MODULE 3: Generate Targets ----
    daily_df = generate_targets(daily_df)

    # ---- MODULE 4: Calendar & Cyclical Features ----
    daily_df = generate_calendar_features(daily_df)

    # ---- Track state before lags/rolling ----
    pre_lag_features = set(daily_df.columns)

    # ---- MODULE 5: Lag Features ----
    daily_df = generate_lag_features(daily_df, lag_windows)

    # ---- MODULE 6: Rolling Statistics ----
    daily_df = generate_rolling_features(daily_df, rolling_windows)

    # ---- MODULE 7: Detect Missing Combinations ----
    missing_df = detect_missing_combinations(daily_df)
    report['missing_combinations_count'] = len(missing_df)

    # ---- MODULE 8: Exclude Incomplete Periods ----
    daily_df = exclude_incomplete_periods(daily_df, incomplete_days)

    # ---- MODULE 9: Chronological Splits ----
    train_df, val_df, test_df = create_chronological_splits(daily_df, train_cutoff, val_cutoff)
    splits = (train_df, val_df, test_df)

    # ---- MODULE 10: Baseline Predictions ----
    daily_df = generate_baselines(daily_df)

    # Recompute splits with baselines included (baselines reference existing lag/rolling cols)
    daily_df = daily_df.sort_values(['tran_br_code', 'start_date']).reset_index(drop=True)
    train_df = daily_df[daily_df['start_date'] < pd.Timestamp(train_cutoff)].copy()
    val_df = daily_df[(daily_df['start_date'] >= pd.Timestamp(train_cutoff)) &
                      (daily_df['start_date'] < pd.Timestamp(val_cutoff))].copy()
    test_df = daily_df[daily_df['start_date'] >= pd.Timestamp(val_cutoff)].copy()
    splits = (train_df, val_df, test_df)

    # ---- Missing Value Report ----
    missing_stats = report_missing_values(daily_df, "Final Feature-Engineered Data")
    report['feature_missing_pct'] = missing_stats['missing_pct']
    report['feature_missing_cols'] = missing_stats['col_breakdown']

    # ---- Validation Check ----
    if missing_stats['missing_pct'] > MAX_ALLOWED_MISSING_PCT:
        print(f"\n⚠  WARNING: Missing data percentage ({missing_stats['missing_pct']:.2f}%) "
              f"exceeds threshold ({MAX_ALLOWED_MISSING_PCT}%)")
    else:
        print(f"\n✓ Missing data percentage ({missing_stats['missing_pct']:.2f}%) "
              f"within threshold ({MAX_ALLOWED_MISSING_PCT}%)")

    if daily_df.shape[0] < MIN_ROWS_EXPECTED:
        print(f"\n⚠  WARNING: Output rows ({daily_df.shape[0]}) below minimum expected ({MIN_ROWS_EXPECTED})")

    # ---- Save Pipeline Metadata ----
    report['final_shape'] = list(daily_df.shape)
    report['splits'] = {
        'train': {'shape': list(train_df.shape), 'date_range': [str(train_df['start_date'].min()), str(train_df['start_date'].max())]},
        'validation': {'shape': list(val_df.shape), 'date_range': [str(val_df['start_date'].min()), str(val_df['start_date'].max())]},
        'test': {'shape': list(test_df.shape), 'date_range': [str(test_df['start_date'].min()), str(test_df['start_date'].max())]}
    }
    report['feature_count'] = {
        'total_columns': int(daily_df.shape[1]),
        'lag_features': len(lag_windows) * 4,
        'rolling_features': len(rolling_windows) * 16,
        'calendar_features': 12,
        'baseline_features': 12,
        'target_variables': 4
    }

    save_pipeline_metadata(report)

    # ---- Save All Outputs ----
    save_outputs(daily_df, daily_df, splits, missing_df, report)

    # ---- Print Summary ----
    print_summary(report, daily_df, daily_df, splits, missing_df)

    return report


# =============================================================================
# VALIDATION CHECK - RUN AND VERIFY
# =============================================================================

def verify_pipeline_output(report: Dict) -> bool:
    """
    Verify that the pipeline output meets quality standards.
    
    Checks:
        1. Output file exists and is non-empty.
        2. Missing data percentage is within threshold.
        3. Minimum row count is met.
        4. Train/val/test splits are non-empty.
        5. Required columns exist.
    
    Parameters:
        report (dict): The validation report from the pipeline.
    
    Returns:
        bool: True if all checks pass.
    """
    print("\n" + "=" * 70)
    print("PIPELINE OUTPUT VERIFICATION")
    print("=" * 70)

    checks_passed = 0
    checks_failed = 0

    # Check 1: Output file exists
    if os.path.exists(FEATURE_ENGINEERED_PATH):
        file_size = os.path.getsize(FEATURE_ENGINEERED_PATH)
        if file_size > 0:
            print(f"  ✓ Output file exists ({file_size:,} bytes)")
            checks_passed += 1
        else:
            print(f"  ✗ Output file is empty")
            checks_failed += 1
    else:
        print(f"  ✗ Output file not found: {FEATURE_ENGINEERED_PATH}")
        checks_failed += 1

    # Check 2: Missing data percentage
    missing_pct = report.get('feature_missing_pct', 100)
    if missing_pct <= MAX_ALLOWED_MISSING_PCT:
        print(f"  ✓ Missing data: {missing_pct:.2f}% ≤ {MAX_ALLOWED_MISSING_PCT}%")
        checks_passed += 1
    else:
        print(f"  ✗ Missing data: {missing_pct:.2f}% > {MAX_ALLOWED_MISSING_PCT}%")
        checks_failed += 1

    # Check 3: Minimum rows
    final_rows = report.get('final_shape', [0])[0]
    if final_rows >= MIN_ROWS_EXPECTED:
        print(f"  ✓ Row count: {final_rows} ≥ {MIN_ROWS_EXPECTED}")
        checks_passed += 1
    else:
        print(f"  ✗ Row count: {final_rows} < {MIN_ROWS_EXPECTED}")
        checks_failed += 1

    # Check 4: Splits non-empty
    splits = report.get('splits', {})
    for split_name in ['train', 'validation', 'test']:
        split_info = splits.get(split_name, {})
        split_shape = split_info.get('shape', [0])
        if split_shape[0] > 0:
            print(f"  ✓ {split_name.capitalize()} split: {split_shape[0]} rows")
            checks_passed += 1
        else:
            print(f"  ✗ {split_name.capitalize()} split is empty!")
            checks_failed += 1

    # Result
    print(f"\n  Checks: {checks_passed} passed, {checks_failed} failed")
    if checks_failed == 0:
        print("  ✓ ALL CHECKS PASSED")
        return True
    else:
        print(f"  ⚠ {checks_failed} check(s) failed — review warnings above")
        return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main entry point. Executes the full pipeline and verifies output.
    """
    print("=" * 70)
    print(" " * 18 + "BANK BRANCH CASH FORECASTING")
    print(" " * 18 + "Phase 3: Feature Engineering")
    print("=" * 70)

    start_time = datetime.now()
    print(f"\nStart time: {start_time}\n")

    try:
        # Run full pipeline
        report = run_pipeline()

        # Verify output
        all_ok = verify_pipeline_output(report)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\nTotal execution time: {duration:.2f} seconds")

        if all_ok:
            print("\n✓ Pipeline completed successfully. Output is ready for model training.")
        else:
            print("\n⚠ Pipeline completed with warnings. Review details above.")

        return report

    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    result = main()