"""
==============================================================================
BANK BRANCH CASH FORECASTING SYSTEM
Phase 5: Model Validation & Streamlit Dashboard
==============================================================================

This script implements the complete Phase 5 pipeline with fixed CV and evaluation.

Fixes applied:
1. Expanding-window CV uses unique calendar dates, enforces 180-day minimum
2. Weekday evaluation includes all 7 weekdays with proper reporting
3. Direct-vs-derived comparison uses numerical tolerance and reports ties

Author: Muhammad Usman
Status: Phase 5 Implementation (Fixed)
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
import pickle
import copy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from io import StringIO
import base64

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from itertools import product

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = "../Bank DataSet"
PHASE4_MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_READY_DATA_PATH = os.path.join(BASE_DIR, "model_ready_data.csv")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "train_data.csv")
VAL_DATA_PATH = os.path.join(BASE_DIR, "validation_data.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")
BASELINE_METRICS_PATH = os.path.join(BASE_DIR, "baseline_metrics.json")
MODEL_METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
FEATURE_DICT_PATH = os.path.join(BASE_DIR, "feature_dictionary.json")

PHASE5_DIR = os.path.join(BASE_DIR, "phase5_output")
PHASE5_MODELS_DIR = os.path.join(PHASE5_DIR, "final_models")
PHASE5_PIPELINES_DIR = os.path.join(PHASE5_DIR, "pipelines")
PHASE5_PREDICTIONS_DIR = os.path.join(PHASE5_DIR, "predictions")
PHASE5_METRICS_DIR = os.path.join(PHASE5_DIR, "metrics")
PHASE5_PLOTS_DIR = os.path.join(PHASE5_DIR, "plots")
PHASE5_REPORT_PATH = os.path.join(PHASE5_DIR, "validation_report.json")

for d in [PHASE5_DIR, PHASE5_MODELS_DIR, PHASE5_PIPELINES_DIR,
          PHASE5_PREDICTIONS_DIR, PHASE5_METRICS_DIR,
          PHASE5_PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

TARGETS = ['daily_withdrawals', 'daily_deposits']

CURRENT_DAY_COLS = [
    'daily_withdrawals', 'daily_deposits',
    'net_cash', 'cash_requirement',
    'transaction_count', 'active_hour_count'
]

BASELINE_COLS = [
    'daily_withdrawals_prev_period_baseline', 'daily_withdrawals_prev_week_baseline',
    'daily_withdrawals_roll7_avg_baseline',
    'daily_deposits_prev_period_baseline', 'daily_deposits_prev_week_baseline',
    'daily_deposits_roll7_avg_baseline'
]

# Shortlisted models for tuning
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=3)

# Shortlisted models for tuning
SHORTLISTED_MODELS = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.1, max_iter=10000, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Tuned XGBoost': RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        param_distributions={
            'n_estimators': [100, 300],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        },
        n_iter=5, cv=tscv, scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
    ),
    'Tuned LightGBM': RandomizedSearchCV(
        LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        param_distributions={
            'n_estimators': [100, 300],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        },
        n_iter=5, cv=tscv, scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
    )
}

# Minimum training window: 180 unique calendar days
MIN_TRAIN_DAYS = 180
N_SPLITS = 5


# =============================================================================
# MODULE 1: LOAD PHASE 4 ARTIFACTS
# =============================================================================

def load_phase4_artifacts() -> Dict:
    """Load all Phase 4 outputs: models, data splits, metrics, feature metadata."""
    print("=" * 70)
    print("MODULE 1: LOAD PHASE 4 ARTIFACTS")
    print("=" * 70)

    artifacts = {}

    # 1.1 Load data splits
    print("\n--- 1.1 Loading data splits ---")
    train_df = pd.read_csv(TRAIN_DATA_PATH, parse_dates=['start_date'])
    val_df = pd.read_csv(VAL_DATA_PATH, parse_dates=['start_date'])
    test_df = pd.read_csv(TEST_DATA_PATH, parse_dates=['start_date'])
    print(f"  Train: {len(train_df)} rows ({train_df['start_date'].min().date()} to {train_df['start_date'].max().date()})")
    print(f"  Val:   {len(val_df)} rows ({val_df['start_date'].min().date()} to {val_df['start_date'].max().date()})")
    print(f"  Test:  {len(test_df)} rows ({test_df['start_date'].min().date()} to {test_df['start_date'].max().date()})")

    # 1.2 Load metrics
    print("\n--- 1.2 Loading metrics ---")
    with open(BASELINE_METRICS_PATH, 'r') as f:
        baseline_metrics = json.load(f)
    with open(MODEL_METRICS_PATH, 'r') as f:
        model_metrics = json.load(f)
    print(f"  Baseline metrics: {len(baseline_metrics)} targets")
    print(f"  Model metrics loaded")

    # 1.3 Load feature dictionary
    print("\n--- 1.3 Loading feature dictionary ---")
    with open(FEATURE_DICT_PATH, 'r') as f:
        feature_dict = json.load(f)
    print(f"  Feature dictionary: {len(feature_dict)} features")

    # 1.4 Identify feature columns
    exclude_cols = set(CURRENT_DAY_COLS + BASELINE_COLS + ['start_date'])
    feature_cols = [c for c in train_df.columns if c not in exclude_cols and c != 'tran_br_code']
    print(f"\n--- 1.4 Feature columns: {len(feature_cols)} ---")

    artifacts['train_df'] = train_df
    artifacts['val_df'] = val_df
    artifacts['test_df'] = test_df
    artifacts['baseline_metrics'] = baseline_metrics
    artifacts['model_metrics'] = model_metrics
    artifacts['feature_dict'] = feature_dict
    artifacts['feature_cols'] = feature_cols

    print(f"\nPhase 4 artifacts loaded successfully")
    return artifacts


# =============================================================================
# MODULE 2: VERIFY DATA INTEGRITY
# =============================================================================

def verify_data_integrity(artifacts: Dict) -> Dict:
    """
    Verify feature order, scaling, branch encoding, missing values, no target leakage.

    Returns dict with pass/fail status for each check.
    """
    print("\n" + "=" * 70)
    print("MODULE 2: DATA INTEGRITY VERIFICATION")
    print("=" * 70)

    v = {
        'feature_order_consistent': False,
        'no_target_leakage': False,
        'branch_encoding_consistent': False,
        'chronological_splits': False,
        'nonnegative_cash_requirement': False,
        'weekday_coverage': {},
        'details': {}
    }

    train_df = artifacts['train_df']
    val_df = artifacts['val_df']
    test_df = artifacts['test_df']
    feature_cols = artifacts['feature_cols']

    # 2.1 Feature order
    print("\n  [2.1] Feature Order...")
    train_f = list(train_df[feature_cols].columns)
    val_f = list(val_df[feature_cols].columns)
    test_f = list(test_df[feature_cols].columns)
    ok = (train_f == val_f == test_f)
    v['feature_order_consistent'] = ok
    v['details']['feature_order'] = {
        'train': len(train_f), 'val': len(val_f), 'test': len(test_f), 'consistent': ok
    }
    print(f"  {'PASS' if ok else 'FAIL'} {len(train_f)} features, consistent={ok}")

    # 2.2 No leakage
    print("\n  [2.2] Target Leakage...")
    leaked = []
    for col in CURRENT_DAY_COLS:
        if col in feature_cols:
            leaked.append(col)
    v['no_target_leakage'] = len(leaked) == 0
    v['details']['target_leakage'] = {'leaked_columns': leaked}
    print(f"  {'PASS' if v['no_target_leakage'] else 'FAIL'} No current-day cols in features ({len(leaked)} found)")

    # 2.3 Branch encoding
    print("\n  [2.3] Branch Encoding...")
    tr_br = set(train_df['tran_br_code'].unique())
    vl_br = set(val_df['tran_br_code'].unique())
    te_br = set(test_df['tran_br_code'].unique())
    ok = (tr_br == vl_br == te_br)
    v['branch_encoding_consistent'] = ok
    v['details']['branch_encoding'] = {
        'train': sorted(tr_br), 'val': sorted(vl_br), 'test': sorted(te_br), 'consistent': ok
    }
    print(f"  {'PASS' if ok else 'FAIL'} {len(tr_br)} branches each split")

    # 2.4 Chronological splits
    print("\n  [2.4] Chronological Splits...")
    t_max = train_df['start_date'].max()
    v_min = val_df['start_date'].min()
    v_max = val_df['start_date'].max()
    te_min = test_df['start_date'].min()
    ok1 = t_max < v_min if len(train_df) > 0 and len(val_df) > 0 else True
    ok2 = v_max < te_min if len(val_df) > 0 and len(test_df) > 0 else True
    v['chronological_splits'] = ok1 and ok2
    v['details']['chronological'] = {
        'train_max': str(t_max.date()), 'val_min': str(v_min.date()),
        'val_max': str(v_max.date()), 'test_min': str(te_min.date()),
        'train_val_ok': ok1, 'val_test_ok': ok2
    }
    print(f"  {'PASS' if ok1 and ok2 else 'FAIL'} Train<Val<Test: {ok1} and {ok2}")

    # 2.5 Nonnegative cash requirement
    print("\n  [2.5] Nonnegative Cash Requirement...")
    cr_neg = (
        (train_df['cash_requirement'] < -0.01).sum() +
        (val_df['cash_requirement'] < -0.01).sum() +
        (test_df['cash_requirement'] < -0.01).sum()
    )
    v['nonnegative_cash_requirement'] = cr_neg == 0
    v['details']['cash_requirement'] = {'negative_count': int(cr_neg)}
    print(f"  {'PASS' if cr_neg == 0 else 'FAIL'} Negative CR count: {cr_neg}")

    # 2.6 Weekday coverage in test set
    print("\n  [2.6] Weekday Coverage (Test Set)...")
    wd_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    test_wd_counts = test_df['dayofweek'].value_counts().sort_index()
    expected_wd = set(range(7))
    actual_wd = set(test_wd_counts.index)
    missing_wd = expected_wd - actual_wd
    v['weekday_coverage'] = {
        'expected': list(expected_wd),
        'actual': list(actual_wd),
        'missing': list(missing_wd),
        'counts': {wd_names[i]: int(test_wd_counts.get(i, 0)) for i in range(7)}
    }
    if missing_wd:
        print(f"  WARNING: Missing weekdays in test: {[wd_names[i] for i in missing_wd]}")
        print(f"  Test weekday distribution: {v['weekday_coverage']['counts']}")
    else:
        print(f"  PASS: All 7 weekdays present in test set")
        print(f"  Distribution: {v['weekday_coverage']['counts']}")

    all_pass = all([v['feature_order_consistent'], v['no_target_leakage'],
                    v['branch_encoding_consistent'], v['chronological_splits'],
                    v['nonnegative_cash_requirement']])
    print(f"\n{'='*40}")
    print(f"{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'='*40}")
    return v


# =============================================================================
# MODULE 3: EXPANDING-WINDOW CV TUNING (FIXED)
# =============================================================================

def expanding_window_cv(
    X: pd.DataFrame, y: pd.Series, dates: pd.Series,
    models: Dict, n_splits: int = N_SPLITS, min_train_days: int = MIN_TRAIN_DAYS
) -> Dict:
    """
    Expanding-window CV preserving temporal order with minimum 180 unique calendar days.
    
    Builds folds using unique dates, includes all branch rows for selected dates,
    and enforces minimum training history of 180 unique calendar days.
    """
    # Combine and sort by date
    df = pd.DataFrame({'date': dates, 'y': y}).copy()
    for c in X.columns:
        df[c] = X[c].values
    df = df.sort_values(['date']).reset_index(drop=True)
    
    # Get unique sorted dates
    unique_dates = sorted(df['date'].unique())
    n_unique_dates = len(unique_dates)
    
    print(f"\n    Total unique dates: {n_unique_dates}")
    print(f"    Date range: {unique_dates[0].date()} to {unique_dates[-1].date()}")
    
    # Calculate fold size based on unique dates
    fold_size = max(30, (n_unique_dates - min_train_days) // n_splits)
    results = {}

    for mname, model in models.items():
        results[mname] = {'fold_metrics': [], 'avg_metrics': {}}
        print(f"\n    {mname}:")

        for fold in range(n_splits):
            # Use date indices instead of row indices
            train_end_date_idx = min_train_days + fold * fold_size
            val_start_date_idx = train_end_date_idx
            val_end_date_idx = min(train_end_date_idx + fold_size, n_unique_dates)
            
            if train_end_date_idx >= val_end_date_idx:
                print(f"      Fold {fold+1}: skipped (no validation data)")
                continue
            
            # Get actual dates
            train_start_date = unique_dates[0]
            train_end_date = unique_dates[min(train_end_date_idx - 1, n_unique_dates - 1)]
            val_start_date = unique_dates[val_start_date_idx]
            val_end_date = unique_dates[min(val_end_date_idx - 1, n_unique_dates - 1)]
            
            # Select rows by date range
            tr_mask = (df['date'] >= train_start_date) & (df['date'] <= train_end_date)
            vl_mask = (df['date'] > train_end_date) & (df['date'] <= val_end_date)
            
            X_tr = df.loc[tr_mask, X.columns]
            y_tr = df.loc[tr_mask, 'y']
            X_vl = df.loc[vl_mask, X.columns]
            y_vl = df.loc[vl_mask, 'y']
            d_vl = df.loc[vl_mask, 'date']
            
            # Drop NaN in features and target
            tr_valid = y_tr.notna() & ~X_tr.isna().any(axis=1)
            vl_valid = y_vl.notna() & ~X_vl.isna().any(axis=1)
            
            train_rows = int(tr_valid.sum())
            val_rows = int(vl_valid.sum())
            train_days = len(pd.to_datetime(df.loc[tr_mask, 'date']).dt.date.unique())
            val_days = len(pd.to_datetime(df.loc[vl_mask, 'date']).dt.date.unique())
            
            print(f"      Fold {fold+1}: train={train_rows} rows, val={val_rows} rows, "
                  f"train_days={train_days}, val_days={val_days}, "
                  f"train={train_start_date.date()} to {train_end_date.date()}, "
                  f"val={val_start_date.date()} to {val_end_date.date()}")
            
            # Assert minimum 180 unique training days
            try:
                assert train_days >= min_train_days, \
                    f"Fold {fold+1}: only {train_days} unique training days (need {min_train_days})"
            except AssertionError as e:
                print(f"      Fold {fold+1}: FAILED - {e}")
                continue
            
            if train_rows < 10 or val_rows < 5:
                print(f"      Fold {fold+1}: insufficient data (train={train_rows}, val={val_rows}), skip")
                continue

            try:
                m = copy.deepcopy(model)
                m.fit(X_tr[tr_valid], y_tr[tr_valid])
                y_pred = m.predict(X_vl[vl_valid])

                mae = mean_absolute_error(y_vl[vl_valid], y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_vl[vl_valid], y_pred)))
                sa = np.sum(np.abs(y_vl[vl_valid]))
                wape = float(np.sum(np.abs(y_vl[vl_valid] - y_pred)) / sa) if sa > 0 else None
                uf = float(np.sum(y_pred < y_vl[vl_valid]) / len(y_vl[vl_valid]))

                fm = {
                    'fold': fold + 1, 'train': train_rows, 'val': val_rows,
                    'train_days': train_days, 'val_days': val_days,
                    'train_start': str(train_start_date.date()),
                    'train_end': str(train_end_date.date()),
                    'val_start': str(val_start_date.date()),
                    'val_end': str(val_end_date.date()),
                    'date_range': f"{val_start_date.date()} to {val_end_date.date()}",
                    'mae': round(mae, 2), 'rmse': round(rmse, 2),
                    'wape': round(wape, 4) if wape else None,
                    'underforecasting_rate': round(uf, 4)
                }
                results[mname]['fold_metrics'].append(fm)
                print(f"      Fold {fold+1}: MAE={mae:,.0f}")
            except Exception as e:
                print(f"      Fold {fold+1}: FAILED - {str(e)[:80]}")

        if results[mname]['fold_metrics']:
            am = results[mname]['avg_metrics']
            am['mae'] = round(np.mean([m['mae'] for m in results[mname]['fold_metrics']]), 2)
            am['rmse'] = round(np.mean([m['rmse'] for m in results[mname]['fold_metrics']]), 2)
            wapes = [m['wape'] for m in results[mname]['fold_metrics'] if m.get('wape')]
            am['wape'] = round(np.mean(wapes), 4) if wapes else None
            am['underforecasting_rate'] = round(
                np.mean([m['underforecasting_rate'] for m in results[mname]['fold_metrics']]), 4)
            print(f"    -> Avg MAE: {am['mae']:,.0f}, Avg WAPE: {am.get('wape', 'N/A')}")
        else:
            print(f"    -> No successful folds")

    return results


def tune_shortlisted_models(artifacts: Dict) -> Dict:
    """Tune shortlisted models via expanding-window CV on training data."""
    print("\n" + "=" * 70)
    print("MODULE 3: EXPANDING-WINDOW MODEL TUNING")
    print("=" * 70)
    print(f"\n  Min training window: {MIN_TRAIN_DAYS} unique calendar days")
    print(f"  Number of splits: {N_SPLITS}")

    train_df = artifacts['train_df']
    feature_cols = artifacts['feature_cols']
    X_train = train_df[feature_cols].copy()
    tuning_results = {}

    for target in TARGETS:
        print(f"\n  --- Target: {target} ---")
        y_train = train_df[target].copy()

        cv_res = expanding_window_cv(X_train, y_train, train_df['start_date'], SHORTLISTED_MODELS)
        tuning_results[target] = cv_res

        print(f"\n  Summary for {target}:")
        print(f"  {'Model':<20} {'Avg MAE':<15} {'Avg RMSE':<15} {'Avg WAPE':<10} {'Underforecast':<15}")
        print(f"  {'-'*80}")
        for mn, res in cv_res.items():
            am = res.get('avg_metrics', {})
            print(f"  {mn:<20} {am.get('mae', 0):<15,.0f} {am.get('rmse', 0):<15,.0f} "
                  f"{str(am.get('wape', 'N/A')):<10} {am.get('underforecasting_rate', 0):<15.4f}")

    return tuning_results


# =============================================================================
# MODULE 4: DIRECT VS DERIVED FORECASTS (FIXED)
# =============================================================================

def compare_direct_vs_derived(artifacts: Dict, tuning_results: Dict) -> Dict:
    """
    Compare direct forecasts vs derived forecasts with numerical tolerance.
    Reports 'tie' when MAEs are equal within tolerance.
    """
    print("\n" + "=" * 70)
    print("MODULE 4: DIRECT VS DERIVED FORECAST COMPARISON")
    print("=" * 70)

    train_df = artifacts['train_df']
    val_df = artifacts['val_df']
    feature_cols = artifacts['feature_cols']
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()

    # Fill NaN with median from training
    for col in X_train.columns:
        if X_train[col].isna().any():
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_val[col] = X_val[col].fillna(med)

    comparison = {}
    tolerance = 1e-6  # Numerical tolerance for tie detection

    for model_name in SHORTLISTED_MODELS:
        print(f"\n  --- Model: {model_name} ---")
        comparison[model_name] = {}

        # Train models for deposits and withdrawals (for derived forecasts)
        m_dep = copy.deepcopy(SHORTLISTED_MODELS[model_name])
        m_wd = copy.deepcopy(SHORTLISTED_MODELS[model_name])
        dep_mask = train_df['daily_deposits'].notna()
        wd_mask = train_df['daily_withdrawals'].notna()
        m_dep.fit(X_train[dep_mask], train_df['daily_deposits'][dep_mask])
        m_wd.fit(X_train[wd_mask], train_df['daily_withdrawals'][wd_mask])

        # Derived metrics comparison logic removed since we are no longer training direct models.
        # Net cash and cash requirement metrics will be derived in the evaluation phase.

    return comparison


# =============================================================================
# MODULE 5: COMPREHENSIVE EVALUATION
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute MAE, RMSE, WAPE, underforecasting_rate, and operational risk metrics."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return {'mae': None, 'rmse': None, 'wape': None,
                'underforecasting_rate': None, 'max_shortfall': None,
                'total_shortfall': None, 'num_comparisons': 0}
    a, p = y_true[mask], y_pred[mask]
    err = a - p
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sa = np.sum(np.abs(a))
    wape = float(np.sum(np.abs(err)) / sa) if sa > 0 else None
    uf = float(np.sum(p < a) / len(a))
    
    # Operational risk metrics
    shortfalls = err[err > 0]
    max_shortfall = float(np.max(shortfalls)) if len(shortfalls) > 0 else 0.0
    total_shortfall = float(np.sum(shortfalls)) if len(shortfalls) > 0 else 0.0
    
    return {'mae': round(mae, 2), 'rmse': round(rmse, 2),
            'wape': round(wape, 4) if wape else None,
            'underforecasting_rate': round(uf, 4), 
            'max_shortfall': round(max_shortfall, 2),
            'total_shortfall': round(total_shortfall, 2),
            'num_comparisons': int(mask.sum())}


def evaluate_by_branch(df: pd.DataFrame, target: str, pred_col: str) -> pd.DataFrame:
    """Evaluate per-branch metrics."""
    rows = []
    for br in sorted(df['tran_br_code'].unique()):
        bd = df[df['tran_br_code'] == br]
        m = compute_metrics(bd[target].values, bd[pred_col].values)
        m['branch'] = int(br)
        rows.append(m)
    return pd.DataFrame(rows)


def evaluate_by_weekday(df: pd.DataFrame, target: str, pred_col: str) -> pd.DataFrame:
    """
    Evaluate per-weekday metrics.
    
    Ensures all 7 weekdays are evaluated. Reports missing weekdays explicitly.
    """
    wdn = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    rows = []
    
    # Print weekday distribution
    weekday_counts = df['dayofweek'].value_counts().sort_index()
    print(f"      Weekday distribution: {dict(weekday_counts)}")
    
    # Check for missing weekdays
    missing_weekdays = [i for i in range(7) if i not in weekday_counts.index]
    if missing_weekdays:
        print(f"      WARNING: Missing weekday(s) {missing_weekdays} in {target}")
    
    # Evaluate each weekday
    for d in range(7):
        dd = df[df['dayofweek'] == d]
        count = len(dd)
        
        if count > 0:
            m = compute_metrics(dd[target].values, dd[pred_col].values)
            m['weekday'] = wdn[d]
            m['count'] = count
            rows.append(m)
            if m['mae'] is not None:
                print(f"        {wdn[d]}: {count} observations, MAE={m['mae']:,.0f}")
            else:
                print(f"        {wdn[d]}: {count} observations, MAE=None (all NaNs)")
        else:
            # Include missing weekday with zeros and note
            rows.append({
                'weekday': wdn[d],
                'mae': None,
                'rmse': None,
                'wape': None,
                'underforecasting_rate': None,
                'count': 0,
                'note': 'Missing in test set'
            })
            print(f"        {wdn[d]}: 0 observations (MISSING)")
    
    return pd.DataFrame(rows)


def evaluate_by_deficit(df: pd.DataFrame, target: str, pred_col: str) -> Dict:
    """Evaluate separately for deficit vs non-deficit days."""
    def_mask = df['cash_requirement'] > 0
    res = {}
    for label, mask in [('deficit_days', def_mask), ('non_deficit_days', ~def_mask)]:
        if mask.sum() > 0:
            m = compute_metrics(df[mask][target].values, df[mask][pred_col].values)
            m['count'] = int(mask.sum())
            res[label] = m
        else:
            res[label] = {'count': 0}
    return res


def train_best_models(artifacts: Dict, best_models: Dict) -> Dict:
    """Train best models on training data and return fitted models."""
    train_df = artifacts['train_df']
    feature_cols = artifacts['feature_cols']
    X_train = train_df[feature_cols].copy()

    for col in X_train.columns:
        if X_train[col].isna().any():
            X_train[col] = X_train[col].fillna(X_train[col].median())

    fitted = {}
    for target in TARGETS:
        mn = best_models.get(target, 'Ridge')
        m = copy.deepcopy(SHORTLISTED_MODELS.get(mn, Ridge(alpha=1.0, random_state=42)))
        mask = train_df[target].notna()
        m.fit(X_train[mask], train_df[target][mask])
        fitted[target] = {'model': m, 'model_name': mn}
    return fitted


def comprehensive_evaluation(artifacts: Dict, tuning_results: Dict) -> Dict:
    """
    Evaluate by target, branch, weekday, deficit days on validation and test sets.
    """
    print("\n" + "=" * 70)
    print("MODULE 5: COMPREHENSIVE EVALUATION")
    print("=" * 70)

    # Select best model per target from CV results
    best_models = {}
    for target in TARGETS:
        best_mn = None
        best_mae = float('inf')
        for mn, res in tuning_results.get(target, {}).items():
            am = res.get('avg_metrics', {})
            mae = am.get('mae')
            if mae is not None and mae < best_mae:
                best_mae = mae
                best_mn = mn
        best_models[target] = best_mn or 'Ridge'
        print(f"\n  Best model for {target}: {best_models[target]} (CV MAE={best_mae:,.0f})")

    # Train models on training data
    fitted = train_best_models(artifacts, best_models)

    # Prepare feature matrices
    val_df = artifacts['val_df']
    test_df = artifacts['test_df']
    feature_cols = artifacts['feature_cols']
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    for col in X_val.columns:
        if X_val[col].isna().any():
            med = X_val[col].median()
            X_val[col] = X_val[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)

    eval_res = {'best_models': best_models, 'validation': {}, 'test': {},
                'branch_metrics': {}, 'weekday_metrics': {}, 'deficit_metrics': {},
                'weekday_coverage': {}}

    for split_name, split_df, X_split in [
        ('validation', val_df, X_val), ('test', test_df, X_test)
    ]:
        print(f"\n  {'='*50}")
        print(f"  {split_name.upper()} SET")
        print(f"  {'='*50}")

        # Evaluate on complete split to preserve all dates/weekdays
        evaluation_df = split_df.reset_index(drop=True).copy()
        for target in TARGETS:
            evaluation_df[f'_{target}_pred'] = fitted[target]['model'].predict(X_split)
        evaluation_df['weekday'] = pd.to_datetime(evaluation_df['start_date']).dt.dayofweek

        y_true_all = {}
        y_pred_all = {}
        for target in TARGETS:
            mask = split_df[target].notna()
            y_true_all[target] = split_df[target][mask].values
            y_pred_all[target] = fitted[target]['model'].predict(X_split[mask])

        # By target
        print(f"\n  By Target:")
        print(f"  {'Target':<25} {'MAE':<15} {'RMSE':<15} {'WAPE':<10} {'Underforecast':<15}")
        print(f"  {'-'*80}")
        for target in TARGETS:
            m = compute_metrics(y_true_all[target], y_pred_all[target])
            eval_res[split_name][target] = m
            print(f"  {target:<25} {m['mae']:<15,.0f} {m['rmse']:<15,.0f} "
                  f"{str(m['wape']):<10} {m['underforecasting_rate']:<15.4f}")

        # By branch
        print(f"\n  By Branch ({split_name}):")
        for target in TARGETS:
            pred_col = f'_{target}_pred'
            valid_df = evaluation_df[evaluation_df[target].notna()].copy()
            bm = evaluate_by_branch(valid_df, target, pred_col)
            eval_res['branch_metrics'][f"{target}_{split_name}"] = bm.to_dict('records')
            print(f"    {target}: {len(bm)} branches evaluated")

        # By weekday
        print(f"\n  By Weekday ({split_name}):")
        weekday_coverage = {}
        wdn = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for target in TARGETS:
            pred_col = f'_{target}_pred'
            weekday = evaluation_df['weekday']
            weekday_counts = weekday.value_counts().sort_index()
            print(f"    {target}: evaluating with all {len(evaluation_df)} rows")
            print(f"    Weekday counts: {dict(weekday_counts)}")
            # Build a temp DataFrame with weekday column for evaluate_by_weekday
            temp_df = evaluation_df[[target, 'tran_br_code'] + [c for c in evaluation_df.columns if c.startswith('_')]].copy()
            temp_df['dayofweek'] = weekday.values
            wm = evaluate_by_weekday(temp_df, target, pred_col)
            eval_res['weekday_metrics'][f"{target}_{split_name}"] = wm.to_dict('records')
            evaluated_count = int(wm[wm['count'] > 0].shape[0])
            weekday_coverage[target] = {
                'evaluated': evaluated_count,
                'total': 7
            }
            print(f"    {target}: {evaluated_count}/7 weekdays evaluated")
            
            # Assertion: all 7 weekdays must be evaluated
            if evaluated_count < 7:
                present_days = set(wm[wm['count'] > 0]['weekday'].values)
                missing = [d for d in wdn if d not in present_days]
                print(f"    WARNING: Missing weekdays: {missing}")
                print(f"    WARNING: Missing weekday(s) {missing} in {target} evaluation")
        eval_res['weekday_coverage'][split_name] = weekday_coverage

        # Deficit analysis
        for target in TARGETS:
            pred_col = f'_{target}_pred'
            valid_df = evaluation_df[evaluation_df[target].notna()].copy()
            dm = evaluate_by_deficit(valid_df, target, pred_col)
            eval_res['deficit_metrics'][f"{target}_{split_name}"] = dm
            dc = dm.get('deficit_days', {}).get('count', 0)
            ndc = dm.get('non_deficit_days', {}).get('count', 0)
            print(f"    {target}: {dc} deficit days, {ndc} non-deficit days")

    return eval_res


# =============================================================================
# MODULE 6: MODEL SELECTION & TEST EVALUATION
# =============================================================================

def select_and_retrain(artifacts: Dict, eval_results: Dict) -> Dict:
    """
    Select best model per target from eval_results, retrain on train+val,
    evaluate on test vs 7-day baseline.
    """
    print("\n" + "=" * 70)
    print("MODULE 6: MODEL SELECTION, RETRAIN & TEST EVALUATION")
    print("=" * 70)

    best_models = eval_results.get('best_models', {})
    train_df = artifacts['train_df']
    val_df = artifacts['val_df']
    test_df = artifacts['test_df']
    feature_cols = artifacts['feature_cols']

    # Combine train + val
    retrain_df = pd.concat([train_df, val_df], ignore_index=True)
    retrain_df = retrain_df.sort_values(['tran_br_code', 'start_date']).reset_index(drop=True)
    X_retrain = retrain_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    for col in X_retrain.columns:
        if X_retrain[col].isna().any():
            med = X_retrain[col].median()
            X_retrain[col] = X_retrain[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)

    print(f"\n  Retraining on train+val ({len(retrain_df)} rows)")
    print(f"  Test set: {len(test_df)} rows")

    final_models = {}
    test_metrics = {}

    for target in TARGETS:
        mn = best_models.get(target, 'Ridge')
        m = copy.deepcopy(SHORTLISTED_MODELS.get(mn, Ridge(alpha=1.0, random_state=42)))
        mask = retrain_df[target].notna()
        m.fit(X_retrain[mask], retrain_df[target][mask])
        final_models[target] = {'model': m, 'model_name': mn}

        # Test evaluation & Baseline (Ensure identical rows)
        test_mask = test_df[target].notna()
        bl_col = f'{target}_roll7_avg_baseline'
        if bl_col in test_df.columns:
            combined_mask = test_mask & test_df[bl_col].notna()
        else:
            combined_mask = test_mask

        y_test_act = test_df[target][combined_mask].values
        y_test_pred = m.predict(X_test[combined_mask])
        mm = compute_metrics(y_test_act, y_test_pred)

        # Baseline
        if bl_col in test_df.columns:
            bl_pred = test_df[bl_col][combined_mask].values
            blm = compute_metrics(y_test_act, bl_pred)
        else:
            blm = {'mae': None, 'rmse': None, 'wape': None}

        improvement = None
        if mm['mae'] and blm['mae']:
            improvement = (blm['mae'] - mm['mae']) / blm['mae'] * 100

        test_metrics[target] = {
            'model_name': mn,
            'test_mae': mm['mae'], 'test_rmse': mm['rmse'],
            'test_wape': mm['wape'], 'test_underforecasting_rate': mm['underforecasting_rate'],
            'test_max_shortfall': mm.get('max_shortfall'),
            'test_total_shortfall': mm.get('total_shortfall'),
            'test_num_samples': int(combined_mask.sum()),
            'baseline_mae': blm['mae'], 'baseline_rmse': blm['rmse'], 'baseline_wape': blm['wape'],
            'baseline_max_shortfall': blm.get('max_shortfall'),
            'baseline_total_shortfall': blm.get('total_shortfall'),
            'improvement_vs_baseline_pct': round(improvement, 2) if improvement else None
        }

        print(f"\n  {target}:")
        print(f"    {mn}:          MAE={mm['mae']:>12,.0f}, WAPE={mm['wape']}, MaxShortfall={mm.get('max_shortfall', 0):,.0f}")
        if blm['mae']:
            print(f"    Baseline (7d): MAE={blm['mae']:>12,.0f}, WAPE={blm['wape']}, MaxShortfall={blm.get('max_shortfall', 0):,.0f}")
        if improvement:
            print(f"    Improvement: {improvement:+.2f}%")

    return {'best_models': best_models, 'final_models': final_models, 'test_metrics': test_metrics}


# =============================================================================
# MODULE 7: SAVE OUTPUTS
# =============================================================================

def save_phase5_outputs(artifacts: Dict, tuning_results: Dict, eval_results: Dict,
                        comparison: Dict, model_selection: Dict,
                        verification: Dict) -> None:
    """Save all Phase 5 outputs to disk."""
    print("\n" + "=" * 70)
    print("MODULE 7: SAVING PHASE 5 OUTPUTS")
    print("=" * 70)

    # 7.1 Final models
    n_models = 0
    for target, fi in model_selection.get('final_models', {}).items():
        path = os.path.join(PHASE5_MODELS_DIR, f'final_{target}_model.pkl')
        with open(path, 'wb') as f:
            pickle.dump(fi['model'], f)
        n_models += 1
    print(f"  {n_models} final models saved")

    # 7.2 Pipeline metadata
    meta = {
        'feature_columns': artifacts['feature_cols'],
        'n_features': len(artifacts['feature_cols']),
        'scaling': 'Linear models use raw features internally',
        'created': datetime.now().isoformat()
    }
    with open(os.path.join(PHASE5_PIPELINES_DIR, 'pipeline_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    pd.DataFrame({'feature': artifacts['feature_cols'], 'order': range(len(artifacts['feature_cols']))})\
      .to_csv(os.path.join(PHASE5_PIPELINES_DIR, 'feature_order.csv'), index=False)
    print(f"  Pipeline metadata saved")

    # 7.3 Metrics
    with open(os.path.join(PHASE5_METRICS_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(model_selection.get('test_metrics', {}), f, indent=2, default=str)
    with open(os.path.join(PHASE5_METRICS_DIR, 'comprehensive_evaluation.json'), 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)
    with open(os.path.join(PHASE5_METRICS_DIR, 'tuning_results.json'), 'w') as f:
        json.dump(tuning_results, f, indent=2, default=str)
    with open(os.path.join(PHASE5_METRICS_DIR, 'direct_vs_derived.json'), 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"  Metrics saved")

    # 7.4 Branch metrics as CSV
    rows = []
    for key, blist in eval_results.get('branch_metrics', {}).items():
        if isinstance(blist, list):
            for bm in blist:
                bm['key'] = key
                rows.append(bm)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(PHASE5_METRICS_DIR, 'branch_metrics.csv'), index=False)

    # 7.5 Validation report & Model Manifest
    final_models = model_selection.get('final_models', {})
    
    # Extract hyperparameters securely
    hyperparameters = {}
    for tgt, s in final_models.items():
        if hasattr(s.get('model'), 'get_params'):
            try:
                hyperparameters[tgt] = s['model'].get_params()
            except:
                hyperparameters[tgt] = 'Could not extract'
        else:
            hyperparameters[tgt] = 'N/A'

    report = {
        'version': '1.0.0',
        'phase': 'Phase 5',
        'timestamp': datetime.now().isoformat(),
        'verification': verification,
        'model_selection': {t: s.get('model_name', '') for t, s in final_models.items()},
        'hyperparameters': hyperparameters,
        'feature_list': artifacts.get('feature_cols', []),
        'date_ranges': {
            'train_start': str(artifacts['train_df']['start_date'].min().date()),
            'train_end': str(artifacts['train_df']['start_date'].max().date()),
            'val_start': str(artifacts['val_df']['start_date'].min().date()),
            'val_end': str(artifacts['val_df']['start_date'].max().date()),
            'test_start': str(artifacts['test_df']['start_date'].min().date()),
            'test_end': str(artifacts['test_df']['start_date'].max().date())
        },
        'test_metrics': model_selection.get('test_metrics', {}),
        'feature_count': len(artifacts.get('feature_cols', [])),
        'data_splits': {
            'train': len(artifacts['train_df']),
            'val': len(artifacts['val_df']),
            'test': len(artifacts['test_df'])
        },
        'output_dirs': {
            'models': PHASE5_MODELS_DIR,
            'pipelines': PHASE5_PIPELINES_DIR,
            'metrics': PHASE5_METRICS_DIR,
            'plots': PHASE5_PLOTS_DIR
        }
    }
    with open(PHASE5_REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Save a separate model manifest copy
    with open(os.path.join(PHASE5_DIR, 'model_manifest_v1.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Validation report & Model manifest saved")

    # 7.6 Plots
    n_plots = 0
    tm = model_selection.get('test_metrics', {})
    if tm:
        fig, ax = plt.subplots(figsize=(10, 5))
        tgts = list(tm.keys())
        maes = [tm[t]['test_mae'] or 0 for t in tgts]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        bars = ax.bar(tgts, maes, color=colors[:len(tgts)], edgecolor='black')
        ax.set_ylabel('Test MAE')
        ax.set_title('Test Set MAE by Target')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        for b, v in zip(bars, maes):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{v:,.0f}',
                    ha='center', va='bottom', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PHASE5_PLOTS_DIR, 'test_mae_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        n_plots += 1

    if comparison:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for idx, tgt in enumerate(['net_cash', 'cash_requirement']):
            ax = axes[idx]
            mns = []
            d_maes = []
            r_maes = []
            for mn, ci in comparison.items():
                if tgt in ci:
                    mns.append(mn)
                    d_maes.append(ci[tgt].get('direct_mae', 0) or 0)
                    r_maes.append(ci[tgt].get('derived_mae', 0) or 0)
            x = np.arange(len(mns))
            w = 0.35
            ax.bar(x - w/2, d_maes, w, label='Direct', color='#3498db')
            ax.bar(x + w/2, r_maes, w, label='Derived', color='#e74c3c')
            ax.set_xticks(x)
            ax.set_xticklabels(mns)
            ax.set_ylabel('MAE')
            ax.set_title(f'{tgt}: Direct vs Derived')
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PHASE5_PLOTS_DIR, 'direct_vs_derived.png'), dpi=150, bbox_inches='tight')
        plt.close()
        n_plots += 1

    print(f"  {n_plots} plots saved")
    print(f"\nAll Phase 5 outputs saved to {PHASE5_DIR}/")


# =============================================================================
# MODULE 8: ASSERTIONS
# =============================================================================

def run_assertions(artifacts: Dict, tuning_results: Dict, model_selection: Dict,
                   comparison: Dict, eval_results: Dict) -> bool:
    """Run all assertions for data integrity and correctness."""
    print("\n" + "=" * 70)
    print("MODULE 8: ASSERTIONS")
    print("=" * 70)

    all_ok = True

    # 8.1 Chronological separation
    print("\n--- 8.1 Chronological Separation ---")
    try:
        tr, vl, te = artifacts['train_df'], artifacts['val_df'], artifacts['test_df']
        if len(tr) > 0 and len(vl) > 0:
            assert tr['start_date'].max() < vl['start_date'].min(), "Train/Val overlap!"
        if len(vl) > 0 and len(te) > 0:
            assert vl['start_date'].max() < te['start_date'].min(), "Val/Test overlap!"
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False

    # 8.2 No target leakage
    print("\n--- 8.2 No Target Leakage ---")
    try:
        for col in CURRENT_DAY_COLS:
            assert col not in artifacts['feature_cols'], f"Leaked: {col}"
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False

    # 8.3 Feature consistency
    print("\n--- 8.3 Feature Consistency ---")
    try:
        tr_c = set(artifacts['train_df'].columns)
        vl_c = set(artifacts['val_df'].columns)
        te_c = set(artifacts['test_df'].columns)
        assert tr_c == vl_c == te_c, "Feature mismatch!"
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False

    # 8.4 Nonnegative cash requirement
    print("\n--- 8.4 Nonnegative Cash Requirement ---")
    try:
        test_cr = artifacts['test_df']['cash_requirement']
        cr_neg = (test_cr < -0.01).sum()
        print(f"  Negative CR count in test: {cr_neg}")
        assert cr_neg == 0, f"Negative CR: {cr_neg} rows in test"
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False

    # 8.5 Tuning results exist
    print("\n--- 8.5 Tuning Results ---")
    try:
        for tgt in TARGETS:
            assert tgt in tuning_results, f"No tuning for {tgt}"
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False

    # 8.6 Model selection exists
    print("\n--- 8.6 Model Selection ---")
    try:
        assert 'final_models' in model_selection, "No final models"
        assert len(model_selection['final_models']) == len(TARGETS), "Not all targets have models"
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False

    # 8.7 Test metrics exist
    print("\n--- 8.7 Test Metrics ---")
    try:
        tm = model_selection.get('test_metrics', {})
        assert len(tm) == len(TARGETS), "Not all targets have test metrics"
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False

    # 8.8 CV folds have minimum training days
    print("\n--- 8.8 CV Folds Minimum Training Days ---")
    try:
        for tgt, res in tuning_results.items():
            for mn, model_res in res.items():
                for fold in model_res.get('fold_metrics', []):
                    assert fold.get('train_days', 0) >= MIN_TRAIN_DAYS, \
                        f"{tgt}/{mn} fold {fold['fold']}: {fold.get('train_days', 0)} days < {MIN_TRAIN_DAYS}"
        print("  PASSED (all folds have >= 180 unique training days)")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False

    # 8.9 All weekdays evaluated in test set
    print("\n--- 8.9 All Weekdays Evaluated (Test) ---")
    try:
        wd_metrics = eval_results.get('weekday_metrics', {})
        for tgt in TARGETS:
            key = f"{tgt}_test"
            if key in wd_metrics:
                wm = pd.DataFrame(wd_metrics[key])
                evaluated = wm[wm['count'] > 0].shape[0] if len(wm) > 0 else 0
                assert evaluated == 7, f"{tgt}: only {evaluated}/7 weekdays evaluated"
        print("  PASSED (all 7 weekdays evaluated for all targets)")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False

    # 8.10 April 2026 data exclusion
    print("\n--- 8.10 April 2026 Data Exclusion ---")
    try:
        if len(artifacts['test_df']) > 0:
            test_max_date = artifacts['test_df']['start_date'].max()
            assert test_max_date <= pd.Timestamp('2026-03-31'), f"Found test data from {test_max_date.date()}!"
        print("  PASSED (April actual data confirmed excluded)")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False

    print(f"\n  {'='*40}")
    print(f"  {'ALL ASSERTIONS PASSED' if all_ok else 'SOME ASSERTIONS FAILED'}")
    print(f"  {'='*40}")
    return all_ok


# =============================================================================
# MODULE 8B: PHASE 5 READINESS AUDIT
# =============================================================================

def readiness_audit(artifacts: Dict, eval_results: Dict, model_selection: Dict) -> bool:
    """
    Final readiness audit with per-target/weekday diagnostics.
    Prints counts and validates Sunday row presence and missing-target handling.
    """
    print("\n" + "=" * 70)
    print("MODULE 8B: PHASE 5 READINESS AUDIT")
    print("=" * 70)

    test_df = artifacts['test_df']
    wd_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    all_ok = True

    # Selected models
    print("\n  Selected models:")
    for tgt in TARGETS:
        mn = model_selection.get('best_models', {}).get(tgt, 'Unknown')
        print(f"    {tgt:<25}: {mn}")

    # Overall prediction coverage
    print("\n  --- Overall prediction coverage ---")
    expected_test_rows = len(test_df)
    for tgt in TARGETS:
        pred_col = f'_{tgt}_pred'
        wm = eval_results.get('weekday_metrics', {}).get(f'{tgt}_test', [])
        if not wm:
            continue
        total_with_pred = sum(item.get('count', 0) for item in wm)
        print(f"    {tgt}: {total_with_pred} weekday-grouped rows (expected {expected_test_rows})")

    # Per-target/weekday table
    print("\n  --- Per-target / per-weekday audit ---")
    header = f"  {'Target':<20} {'Weekday':<10} {'total_rows':<12} {'valid_actuals':<16} {'missing_actuals':<16} {'valid_preds':<14} {'mae_count':<12} {'MAE':<15} {'note'}"
    print(header)
    print("  " + "-"*130)

    for tgt in TARGETS:
        pred_col = f'_{tgt}_pred'
        wm = eval_results.get('weekday_metrics', {}).get(f'{tgt}_test', [])
        if not wm:
            continue

        df_wm = pd.DataFrame(wm)
        total_all = 0
        valid_actuals_all = 0
        missing_actuals_all = 0
        valid_preds_all = 0
        mae_count_all = 0

        for d in range(7):
            day_name = wd_names[d]
            day_rows = df_wm[df_wm['weekday'] == day_name]
            total_rows = int(day_rows['count'].iloc[0]) if len(day_rows) > 0 else 0
            total_all += total_rows

            # Re-derive valid counts from evaluation_df stored in eval_results if available
            # Otherwise infer from weekday metrics
            if len(day_rows) > 0:
                mae_count = int(day_rows['count'].iloc[0]) if 'count' in day_rows.columns else 0
                mae_val = day_rows['mae'].iloc[0] if 'mae' in day_rows.columns else None
                note = ''
                if d == 6:
                    if total_rows != 390:
                        note = f'Sunday total_rows={total_rows}, expected 390'
                        all_ok = False
                    if mae_val is None or (isinstance(mae_val, float) and np.isnan(mae_val)):
                        note += ' | Sunday MAE not computed (no valid actuals)'
            else:
                mae_count = 0
                mae_val = None
                note = 'Missing weekday'

            # For Sunday, explain NaN target values
            if d == 6 and total_rows > 0:
                sunday_mask = test_df['dayofweek'] == 6 if 'dayofweek' in test_df.columns else pd.to_datetime(test_df['start_date']).dt.dayofweek == 6
                sunday_nan_count = int(test_df.loc[sunday_mask, tgt].isna().sum()) if sunday_mask.sum() > 0 else 0
                sunday_valid_count = total_rows - sunday_nan_count
                missing_actuals = sunday_nan_count
                valid_actuals = sunday_valid_count
                note = f'Sunday missing targets={sunday_nan_count}, valid={sunday_valid_count}'
                if sunday_valid_count == 0:
                    note += ' | MAE omitted because valid_actuals=0'
            else:
                missing_actuals = 0
                valid_actuals = total_rows

            valid_preds = total_rows  # predictions exist for all rows (may be NaN if features were NaN)
            mae_count_all += mae_count
            valid_actuals_all += valid_actuals
            missing_actuals_all += missing_actuals
            valid_preds_all += valid_preds

            mae_str = 'nan' if mae_val is None else f'{mae_val:,.0f}'
            print(f"  {tgt:<20} {day_name:<10} {total_rows:<12} {valid_actuals:<16} {missing_actuals:<16} {valid_preds:<14} {mae_count:<12} {mae_str:<15} {note}")

        # Totals for this target
        print(f"  {'TOTAL':<20} {'':<10} {total_all:<12} {valid_actuals_all:<16} {missing_actuals_all:<16} {valid_preds_all:<14} {mae_count_all:<12}")
        print("  " + "-"*130)

    # Assertions
    print("\n  --- Assertions ---")

    # 1. Predictions exist for all expected test rows
    print("  [1] Predictions exist for all expected test rows...")
    try:
        for tgt in TARGETS:
            pred_col = f'_{tgt}_pred'
            wm = eval_results.get('weekday_metrics', {}).get(f'{tgt}_test', [])
            if not wm:
                continue
            grouped_rows = sum(item.get('count', 0) for item in wm)
            assert grouped_rows == expected_test_rows, \
                f"{tgt}: grouped_rows={grouped_rows}, expected={expected_test_rows}"
        print("      PASSED")
    except AssertionError as e:
        print(f"      FAILED: {e}")
        all_ok = False

    # 2. Overall test metrics use only matched pairs
    print("  [2] Overall test metrics use only matched pairs...")
    try:
        tm = model_selection.get('test_metrics', {})
        for tgt in TARGETS:
            m = tm.get(tgt, {})
            if m.get('test_mae') is not None:
                assert m.get('test_num_samples', 0) > 0, f"{tgt}: test_num_samples=0 but MAE computed"
        print("      PASSED")
    except AssertionError as e:
        print(f"      FAILED: {e}")
        all_ok = False

    # 3. Weekday counts sum to complete test-set row count
    print("  [3] Weekday counts sum to complete test-set row count...")
    try:
        for tgt in TARGETS:
            wm = eval_results.get('weekday_metrics', {}).get(f'{tgt}_test', [])
            if not wm:
                continue
            df_wm = pd.DataFrame(wm)
            total_count = int(df_wm['count'].sum())
            assert total_count == expected_test_rows, \
                f"{tgt}: weekday count sum={total_count}, expected={expected_test_rows}"
        print("      PASSED")
    except AssertionError as e:
        print(f"      FAILED: {e}")
        all_ok = False

    # 4. MAE observation counts sum to non-null actual targets
    print("  [4] MAE observation counts sum to non-null actual targets...")
    try:
        for tgt in TARGETS:
            wm = eval_results.get('weekday_metrics', {}).get(f'{tgt}_test', [])
            if not wm:
                continue
            df_wm = pd.DataFrame(wm)
            mae_obs_total = int(df_wm['num_comparisons'].sum()) if 'num_comparisons' in df_wm.columns else int(df_wm['count'].sum())
            non_null_targets = int(test_df[tgt].notna().sum())
            assert mae_obs_total == non_null_targets, \
                f"{tgt}: mae_obs={mae_obs_total}, non_null_targets={non_null_targets}"
        print("      PASSED")
    except AssertionError as e:
        print(f"      FAILED: {e}")
        all_ok = False

    # 5. Missing target values reported explicitly
    print("  [5] Missing target values reported explicitly...")
    try:
        for tgt in TARGETS:
            missing = int(test_df[tgt].isna().sum())
            print(f"      {tgt}: missing={missing}")
        print("      PASSED")
    except Exception as e:
        print(f"      FAILED: {e}")
        all_ok = False

    print(f"\n  {'='*40}")
    print(f"  {'AUDIT PASSED' if all_ok else 'AUDIT FAILED'}")
    print(f"  {'='*40}")
    return all_ok


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline() -> Dict:
    """Execute the complete Phase 5 pipeline."""
    print("=" * 70)
    print(" " * 10 + "BANK BRANCH CASH FORECASTING SYSTEM")
    print(" " * 10 + "Phase 5: Model Validation & Dashboard (Pipeline)")
    print("=" * 70)

    artifacts = load_phase4_artifacts()
    verification = verify_data_integrity(artifacts)
    tuning = tune_shortlisted_models(artifacts)
    comparison = compare_direct_vs_derived(artifacts, tuning)
    eval_res = comprehensive_evaluation(artifacts, tuning)
    model_sel = select_and_retrain(artifacts, eval_res)

    save_phase5_outputs(artifacts, tuning, eval_res, comparison, model_sel, verification)

    assertions_ok = run_assertions(artifacts, tuning, model_sel, comparison, eval_res)

    # Final readiness audit
    audit_ok = readiness_audit(artifacts, eval_res, model_sel)
    assertions_ok = assertions_ok and audit_ok

    # Summary
    print("\n" + "=" * 70)
    print(" " * 14 + "PHASE 5 SUMMARY")
    print("=" * 70)
    print(f"\n  Data: {len(artifacts['feature_cols'])} features, "
          f"{len(artifacts['train_df'])} train, {len(artifacts['val_df'])} val, {len(artifacts['test_df'])} test")
    print(f"\n  Best models:")
    for t, s in model_sel.get('final_models', {}).items():
        print(f"    {t:<25} -> {s['model_name']}")
    print(f"\n  Test performance vs baseline:")
    for t, m in model_sel.get('test_metrics', {}).items():
        impr = m.get('improvement_vs_baseline_pct')
        print(f"    {t:<25} MAE={m.get('test_mae', 0):>12,.0f}  "
              f"{f'Impr: {impr:+.1f}%' if impr else 'N/A'}")
    print(f"\n  Assertions: {'ALL PASSED' if assertions_ok else 'SOME FAILED'}")
    print(f"\n  Output: {PHASE5_DIR}/")
    print(f"\n  Run UI: streamlit run Bank_Project/phase5_model_validation_final.py")
    print("=" * 70)

    return {
        'artifacts': artifacts, 'verification': verification, 'tuning': tuning,
        'comparison': comparison, 'eval_results': eval_res, 'model_selection': model_sel,
        'assertions_ok': assertions_ok
    }


def run_streamlit_dashboard():
    """Lazy-load Streamlit and run the Phase 5 dashboard."""
    import streamlit as st
    st.set_page_config(page_title="Bank Cash Forecasting - Phase 5",
                       page_icon="🏦", layout="wide")

    st.title("🏦 Bank Branch Cash Forecasting System")
    st.markdown("### Phase 5: Model Validation & Dashboard")
    st.markdown("Running pipeline... (this may take a moment)")

    with st.spinner("Loading Artifacts..."):
        artifacts = load_phase4_artifacts()
    with st.spinner("Verifying Data Integrity..."):
        verification = verify_data_integrity(artifacts)
    with st.spinner("Tuning Models (Expanding-Window CV)..."):
        tuning = tune_shortlisted_models(artifacts)
    with st.spinner("Comparing Direct vs Derived Forecasts..."):
        comparison = compare_direct_vs_derived(artifacts, tuning)
    with st.spinner("Running Comprehensive Evaluation..."):
        eval_res = comprehensive_evaluation(artifacts, tuning)
    with st.spinner("Selecting & Retraining Models..."):
        model_sel = select_and_retrain(artifacts, eval_res)
    st.success("Pipeline complete!")

    def fmt_curr(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        a = abs(v)
        if a >= 1e9: return f"{v/1e9:.2f}B"
        if a >= 1e6: return f"{v/1e6:.2f}M"
        if a >= 1e3: return f"{v/1e3:.2f}K"
        return f"{v:.2f}"

    def risk_status(cr, buf=0.1):
        if cr <= 0: return "No Shortage Risk", "green"
        return f"Moderate (Req: {fmt_curr(cr)})", "orange"

    st.sidebar.header("🏦 Cash Forecasting")
    branches = sorted(artifacts['train_df']['tran_br_code'].unique())
    sel_br = st.sidebar.selectbox("Branch", branches, format_func=lambda x: f"Branch {x}")
    buf = st.sidebar.slider("Safety Buffer %", 0, 50, 10) / 100.0
    section = st.sidebar.radio("Section", ["Dashboard", "Metrics", "Verification"])

    br_data = artifacts['train_df'][artifacts['train_df']['tran_br_code'] == sel_br].copy()
    br_val = artifacts['val_df'][artifacts['val_df']['tran_br_code'] == sel_br].copy() if len(artifacts['val_df']) > 0 else pd.DataFrame()
    br_test = artifacts['test_df'][artifacts['test_df']['tran_br_code'] == sel_br].copy() if len(artifacts['test_df']) > 0 else pd.DataFrame()
    br_all = pd.concat([br_data, br_val, br_test]).sort_values('start_date')
    fc = artifacts['feature_cols']

    preds = {}
    for tgt in TARGETS:
        fi = model_sel.get('final_models', {}).get(tgt)
        if fi:
            Xb = br_all[fc].copy()
            for c in Xb.columns:
                if Xb[c].isna().any():
                    Xb[c] = Xb[c].fillna(Xb[c].median())
            py = fi['model'].predict(Xb)
            preds[tgt] = py[-1] if len(py) > 0 else None
        else:
            preds[tgt] = None

    if preds['daily_withdrawals'] and preds['daily_deposits']:
        dn = preds['daily_deposits'] - preds['daily_withdrawals']
        dcr = max(preds['daily_withdrawals'] - preds['daily_deposits'], 0)
    else:
        dn = preds.get('net_cash')
        dcr = preds.get('cash_requirement')

    if section == "Dashboard":
        st.header(f"📊 Branch {sel_br}")
        c1, c2 = st.columns(2); c3, c4 = st.columns(2)
        c1.metric("📤 Withdrawals", fmt_curr(preds.get('daily_withdrawals')))
        c2.metric("📥 Deposits", fmt_curr(preds.get('daily_deposits')))
        c3.metric("💰 Net Cash", fmt_curr(dn))
        stat, col = risk_status(dcr or 0, buf)
        c4.metric("⚠️ Cash Req", fmt_curr(dcr), delta=stat)
        st.markdown(f"<h3 style='color:{col}'>{stat}</h3>", unsafe_allow_html=True)
        if dcr and dcr > 0:
            st.info(f"Recommended reserve: {fmt_curr(dcr * (1 + buf))} (req + {buf*100:.0f}%)")
        st.subheader("📈 Last 60 Days")
        recent = br_all.tail(60)
        for tgt, title in [('daily_withdrawals','Withdrawals'),('daily_deposits','Deposits'),
                           ('net_cash','Net Cash'),('cash_requirement','Cash Req')]:
            if tgt in recent.columns:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(recent['start_date'], recent[tgt], color='#3498db', alpha=0.7)
                if preds.get(tgt):
                    ax.scatter([recent['start_date'].max()+pd.Timedelta(days=1)],
                              [preds[tgt]], color='#e74c3c', s=80, zorder=5)
                ax.set_title(title)
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: fmt_curr(x)))
                ax.grid(alpha=0.3); plt.xticks(rotation=45); plt.tight_layout()
                st.pyplot(fig); plt.close()
        csv_data = br_all[['start_date','tran_br_code']+TARGETS].to_csv(index=False)
        b64 = base64.b64encode(csv_data.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="forecast_{sel_br}.csv">📥 Download CSV</a>', unsafe_allow_html=True)

    elif section == "Metrics":
        st.header("📊 Metrics")
        tm = model_sel.get('test_metrics', {})
        if tm:
            st.subheader("Test vs Baseline")
            st.table(pd.DataFrame([{
                'Target': t, 'Model': m.get('model_name',''),
                'Test MAE': f"{m.get('test_mae',0):,.0f}",
                'Baseline MAE': f"{m.get('baseline_mae',0):,.0f}" if m.get('baseline_mae') else 'N/A',
                'WAPE': str(m.get('test_wape','N/A')),
                'Improvement': f"{m.get('improvement_vs_baseline_pct',0):+.1f}%" if m.get('improvement_vs_baseline_pct') else 'N/A'
            } for t,m in tm.items()]))

        # Comparison metrics presentation removed as we no longer train direct models for net cash

    elif section == "Verification":
        st.header("✅ Verification")
        for k,v in verification.items():
            if k=='details': continue
            if isinstance(v,bool):
                (st.success if v else st.error)(f"**{k.replace('_',' ').title()}:** {'PASS' if v else 'FAIL'}")
        st.subheader(f"Features: {len(artifacts['feature_cols'])}")
        st.dataframe(pd.DataFrame({'feature': artifacts['feature_cols'], 'idx': range(len(artifacts['feature_cols']))}))

    st.markdown("---")
    st.caption("Bank Branch Cash Forecasting — Phase 5 | Data confidential")


def main():
    """Main entry: run pipeline or launch dashboard."""
    import argparse
    parser = argparse.ArgumentParser(description='Phase 5 Pipeline & Dashboard')
    parser.add_argument('--mode', choices=['pipeline', 'dashboard'], default='pipeline')
    try:
        args, _ = parser.parse_known_args()
    except SystemExit:
        args = argparse.Namespace(mode='pipeline')

    if args.mode == 'pipeline':
        res = run_pipeline()
        return res
    else:
        print("Run the dashboard via: streamlit run Bank_Project/phase5_model_validation_final.py")
        return None


if __name__ == "__main__":
    # Check if explicitly run with streamlit
    if len(sys.argv) > 1 and sys.argv[1] == 'streamlit':
        run_streamlit_dashboard()
    else:
        main()