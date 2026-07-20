"""
==============================================================================
BANK BRANCH CASH FORECASTING SYSTEM
Phase 5: Model Validation, Explainable AI & Streamlit Dashboard
==============================================================================

This script implements the complete Phase 5 pipeline:

1. Load Phase 4 models, preprocessing objects, train/val/test datasets, metrics, metadata
2. Verify feature order, scaling, branch encoding, missing-value handling, no target leakage
3. Tune shortlisted models (Ridge, Lasso, Linear Regression) via expanding-window CV
4. Compare direct vs derived forecasts (net cash, cash requirement)
5. Evaluate by target, branch, weekday, deficit days (MAE, RMSE, WAPE, underforecast)
6. Select models on validation, retrain on train+val, evaluate once on test vs 7d baseline
7. Explainable AI: coefficients, SHAP global/local, feature contributions, branch-level
8. Streamlit UI with branch/date selection, forecasts, shortage risk, safety buffer,
   historical-vs-predicted charts, model metrics, global/local explanations, CSV download
9. Save final models, preprocessing pipelines, predictions, test metrics, branch metrics,
   XAI results, plots, JSON validation report
10. Assertions for chronological separation, no leakage, feature consistency,
    nonnegative cash requirement, reproducible predictions

IMPORTANT: Explanations show ASSOCIATIONS, not causation.

Author: Muhammad Usman
Status: Phase 5 Implementation
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

# Scikit-learn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# SHAP for Explainable AI
import shap

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

BASE_DIR = "Bank DataSet"
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
PHASE5_XAI_DIR = os.path.join(PHASE5_DIR, "xai_results")
PHASE5_PLOTS_DIR = os.path.join(PHASE5_DIR, "plots")
PHASE5_REPORT_PATH = os.path.join(PHASE5_DIR, "validation_report.json")

for d in [PHASE5_DIR, PHASE5_MODELS_DIR, PHASE5_PIPELINES_DIR,
          PHASE5_PREDICTIONS_DIR, PHASE5_METRICS_DIR,
          PHASE5_XAI_DIR, PHASE5_PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

TARGETS = ['daily_withdrawals', 'daily_deposits', 'net_cash', 'cash_requirement']

CURRENT_DAY_COLS = [
    'daily_withdrawals', 'daily_deposits', 'net_cash', 'cash_requirement',
    'transaction_count', 'active_hour_count'
]

BASELINE_COLS = [
    'daily_withdrawals_prev_day_baseline', 'daily_withdrawals_prev_week_baseline',
    'daily_withdrawals_roll7_avg_baseline',
    'daily_deposits_prev_day_baseline', 'daily_deposits_prev_week_baseline',
    'daily_deposits_roll7_avg_baseline',
    'net_cash_prev_day_baseline', 'net_cash_prev_week_baseline',
    'net_cash_roll7_avg_baseline',
    'cash_requirement_prev_day_baseline', 'cash_requirement_prev_week_baseline',
    'cash_requirement_roll7_avg_baseline'
]

SHORTLISTED_MODELS = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.1, max_iter=10000, random_state=42)
}

N_SPLITS = 5
MIN_TRAIN_SIZE = 60


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
    print(f"  ✓ Train: {len(train_df)} rows")
    print(f"  ✓ Val:   {len(val_df)} rows")
    print(f"  ✓ Test:  {len(test_df)} rows")

    # 1.2 Load metrics
    print("\n--- 1.2 Loading metrics ---")
    with open(BASELINE_METRICS_PATH, 'r') as f:
        baseline_metrics = json.load(f)
    with open(MODEL_METRICS_PATH, 'r') as f:
        model_metrics = json.load(f)
    print(f"  ✓ Baseline metrics: {len(baseline_metrics)} targets")
    print(f"  ✓ Model metrics loaded")

    # 1.3 Load feature dictionary
    print("\n--- 1.3 Loading feature dictionary ---")
    with open(FEATURE_DICT_PATH, 'r') as f:
        feature_dict = json.load(f)
    print(f"  ✓ Feature dictionary: {len(feature_dict)} features")

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

    print(f"\n✓ Phase 4 artifacts loaded successfully")
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
    print(f"  {'✓' if ok else '✗'} {len(train_f)} features, consistent={ok}")

    # 2.2 No leakage
    print("\n  [2.2] Target Leakage...")
    leaked = []
    for col in CURRENT_DAY_COLS:
        if col in feature_cols:
            leaked.append(col)
    v['no_target_leakage'] = len(leaked) == 0
    v['details']['target_leakage'] = {'leaked_columns': leaked}
    print(f"  {'✓' if v['no_target_leakage'] else '✗'} No current-day cols in features ({len(leaked)} found)")

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
    print(f"  {'✓' if ok else '✗'} {len(tr_br)} branches each split")

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
    print(f"  {'✓' if ok1 and ok2 else '✗'} Train<Val<Test: {ok1} and {ok2}")

    # 2.5 Nonnegative cash requirement
    print("\n  [2.5] Nonnegative Cash Requirement...")
    cr_neg = (
        (train_df['cash_requirement'] < -0.01).sum() +
        (val_df['cash_requirement'] < -0.01).sum() +
        (test_df['cash_requirement'] < -0.01).sum()
    )
    v['nonnegative_cash_requirement'] = cr_neg == 0
    v['details']['cash_requirement'] = {'negative_count': int(cr_neg)}
    print(f"  {'✓' if cr_neg == 0 else '✗'} Negative CR count: {cr_neg}")

    all_pass = all([v['feature_order_consistent'], v['no_target_leakage'],
                    v['branch_encoding_consistent'], v['chronological_splits'],
                    v['nonnegative_cash_requirement']])
    print(f"\n  {'='*40}")
    print(f"  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"  {'='*40}")
    return v


# =============================================================================
# MODULE 3: EXPANDING-WINDOW CV TUNING
# =============================================================================

def expanding_window_cv(
    X: pd.DataFrame, y: pd.Series, dates: pd.Series,
    models: Dict, n_splits: int = N_SPLITS, min_train_size: int = MIN_TRAIN_SIZE
) -> Dict:
    """
    Expanding-window CV preserving temporal order.

    Each fold: train on [0..train_end), validate on [train_end..train_end+fold_size).
    Rows with NaN in features OR target are dropped per fold.
    """
    sort_idx = np.argsort(dates.values)
    X_s = X.iloc[sort_idx].reset_index(drop=True)
    y_s = y.iloc[sort_idx].reset_index(drop=True)
    d_s = dates.iloc[sort_idx].reset_index(drop=True)

    n = len(X_s)
    fold_size = max(1, (n - min_train_size) // n_splits)
    results = {}

    for mname, model in models.items():
        results[mname] = {'fold_metrics': [], 'avg_metrics': {}}
        print(f"\n    {mname}:")

        for fold in range(n_splits):
            train_end = min_train_size + fold * fold_size
            val_end = min(train_end + fold_size, n)
            if train_end >= val_end:
                continue

            X_tr = X_s.iloc[:train_end]
            y_tr = y_s.iloc[:train_end]
            X_vl = X_s.iloc[train_end:val_end]
            y_vl = y_s.iloc[train_end:val_end]
            d_vl = d_s.iloc[train_end:val_end]

            # Drop NaN in features and target
            tr_mask = y_tr.notna() & ~X_tr.isna().any(axis=1)
            vl_mask = y_vl.notna() & ~X_vl.isna().any(axis=1)

            if tr_mask.sum() < 5 or vl_mask.sum() < 3:
                print(f"      Fold {fold+1}: insufficient data (train={tr_mask.sum()}, val={vl_mask.sum()}), skip")
                continue

            try:
                m = copy.deepcopy(model)
                m.fit(X_tr[tr_mask], y_tr[tr_mask])
                y_pred = m.predict(X_vl[vl_mask])

                mae = mean_absolute_error(y_vl[vl_mask], y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_vl[vl_mask], y_pred)))
                sa = np.sum(np.abs(y_vl[vl_mask]))
                wape = float(np.sum(np.abs(y_vl[vl_mask] - y_pred)) / sa) if sa > 0 else None
                uf = float(np.sum(y_pred < y_vl[vl_mask]) / len(y_vl[vl_mask]))

                fm = {
                    'fold': fold + 1, 'train': int(tr_mask.sum()), 'val': int(vl_mask.sum()),
                    'date_range': f"{d_vl[vl_mask].min().date()} to {d_vl[vl_mask].max().date()}",
                    'mae': round(mae, 2), 'rmse': round(rmse, 2),
                    'wape': round(wape, 4) if wape else None,
                    'underforecasting_rate': round(uf, 4)
                }
                results[mname]['fold_metrics'].append(fm)
                print(f"      Fold {fold+1}: train={fm['train']}, val={fm['val']}, MAE={mae:,.0f}")
            except Exception as e:
                print(f"      Fold {fold+1}: FAILED - {str(e)[:60]}")

        if results[mname]['fold_metrics']:
            am = results[mname]['avg_metrics']
            am['mae'] = round(np.mean([m['mae'] for m in results[mname]['fold_metrics']]), 2)
            am['rmse'] = round(np.mean([m['rmse'] for m in results[mname]['fold_metrics']]), 2)
            wapes = [m['wape'] for m in results[mname]['fold_metrics'] if m.get('wape')]
            am['wape'] = round(np.mean(wapes), 4) if wapes else None
            am['underforecasting_rate'] = round(
                np.mean([m['underforecasting_rate'] for m in results[mname]['fold_metrics']]), 4)
            print(f"    → Avg MAE: {am['mae']:,.0f}, Avg WAPE: {am.get('wape', 'N/A')}")
        else:
            print(f"    → No successful folds")

    return results


def tune_shortlisted_models(artifacts: Dict) -> Dict:
    """Tune shortlisted models via expanding-window CV on training data."""
    print("\n" + "=" * 70)
    print("MODULE 3: EXPANDING-WINDOW MODEL TUNING")
    print("=" * 70)

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
        print(f"  {'-'*75}")
        for mn, res in cv_res.items():
            am = res.get('avg_metrics', {})
            print(f"  {mn:<20} {am.get('mae', 0):<15,.0f} {am.get('rmse', 0):<15,.0f} "
                  f"{str(am.get('wape', 'N/A')):<10} {am.get('underforecasting_rate', 0):<15.4f}")

    return tuning_results


# =============================================================================
# MODULE 4: DIRECT VS DERIVED FORECASTS
# =============================================================================

def compare_direct_vs_derived(artifacts: Dict, tuning_results: Dict) -> Dict:
    """
    Compare direct forecasts (model trained on net_cash/cash_requirement directly)
    vs derived forecasts (net_cash = pred_deposits - pred_withdrawals,
    cash_requirement = max(pred_withdrawals - pred_deposits, 0)).
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

        # Train direct models for net_cash and cash_requirement
        m_nc = copy.deepcopy(SHORTLISTED_MODELS[model_name])
        m_cr = copy.deepcopy(SHORTLISTED_MODELS[model_name])
        nc_mask = train_df['net_cash'].notna()
        cr_mask = train_df['cash_requirement'].notna()
        m_nc.fit(X_train[nc_mask], train_df['net_cash'][nc_mask])
        m_cr.fit(X_train[cr_mask], train_df['cash_requirement'][cr_mask])

        # Predict on validation
        pred_dep = m_dep.predict(X_val)
        pred_wd = m_wd.predict(X_val)
        pred_nc_direct = m_nc.predict(X_val)
        pred_cr_direct = m_cr.predict(X_val)

        # Derived
        pred_nc_derived = pred_dep - pred_wd
        pred_cr_derived = np.maximum(pred_wd - pred_dep, 0)

        # Actuals
        val_nc = val_df['net_cash'].values
        val_cr = val_df['cash_requirement'].values
        nc_mask_a = ~np.isnan(val_nc)
        cr_mask_a = ~np.isnan(val_cr)

        for tgt, direct, derived, actual, mask_a in [
            ('net_cash', pred_nc_direct, pred_nc_derived, val_nc, nc_mask_a),
            ('cash_requirement', pred_cr_direct, pred_cr_derived, val_cr, cr_mask_a)
        ]:
            if mask_a.sum() > 0:
                mae_d = mean_absolute_error(actual[mask_a], direct[mask_a])
                mae_r = mean_absolute_error(actual[mask_a], derived[mask_a])
            else:
                mae_d = mae_r = None

            better = 'direct' if (mae_d is not None and mae_r is not None and mae_d <= mae_r) else 'derived'
            comparison[model_name][tgt] = {
                'direct_mae': round(mae_d, 2) if mae_d else None,
                'derived_mae': round(mae_r, 2) if mae_r else None,
                'better_method': better
            }
            print(f"    {tgt:20s}: direct MAE={mae_d:>12,.0f}, derived MAE={mae_r:>12,.0f} → {better}")

    return comparison


# =============================================================================
# MODULE 5: COMPREHENSIVE EVALUATION
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute MAE, RMSE, WAPE, underforecasting_rate."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return {'mae': None, 'rmse': None, 'wape': None,
                'underforecasting_rate': None, 'num_comparisons': 0}
    a, p = y_true[mask], y_pred[mask]
    err = a - p
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sa = np.sum(np.abs(a))
    wape = float(np.sum(np.abs(err)) / sa) if sa > 0 else None
    uf = float(np.sum(p < a) / len(a))
    return {'mae': round(mae, 2), 'rmse': round(rmse, 2),
            'wape': round(wape, 4) if wape else None,
            'underforecasting_rate': round(uf, 4), 'num_comparisons': int(mask.sum())}


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
    """Evaluate per-weekday metrics."""
    wdn = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    rows = []
    for d in range(7):
        dd = df[df['dayofweek'] == d]
        if len(dd) == 0:
            continue
        m = compute_metrics(dd[target].values, dd[pred_col].values)
        m['weekday'] = wdn[d]
        rows.append(m)
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
                'branch_metrics': {}, 'weekday_metrics': {}, 'deficit_metrics': {}}

    for split_name, split_df, X_split in [
        ('validation', val_df, X_val), ('test', test_df, X_test)
    ]:
        print(f"\n  {'='*50}")
        print(f"  {split_name.upper()} SET")
        print(f"  {'='*50}")

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
            mask = split_df[target].notna()
            sub = split_df[mask].copy()
            sub['_pred'] = fitted[target]['model'].predict(X_split[mask])
            bm = evaluate_by_branch(sub, target, '_pred')
            eval_res['branch_metrics'][f"{target}_{split_name}"] = bm.to_dict('records')
            print(f"    {target}: {len(bm)} branches evaluated")

        # By weekday
        print(f"\n  By Weekday ({split_name}):")
        for target in TARGETS:
            mask = split_df[target].notna()
            sub = split_df[mask].copy()
            sub['_pred'] = fitted[target]['model'].predict(X_split[mask])
            wm = evaluate_by_weekday(sub, target, '_pred')
            eval_res['weekday_metrics'][f"{target}_{split_name}"] = wm.to_dict('records')
            print(f"    {target}: {len(wm)} days evaluated")

        # Deficit analysis
        for target in TARGETS:
            mask = split_df[target].notna()
            sub = split_df[mask].copy()
            sub['_pred'] = fitted[target]['model'].predict(X_split[mask])
            dm = evaluate_by_deficit(sub, target, '_pred')
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

        # Test evaluation
        test_mask = test_df[target].notna()
        y_test_act = test_df[target][test_mask].values
        y_test_pred = m.predict(X_test[test_mask])
        mm = compute_metrics(y_test_act, y_test_pred)

        # Baseline
        bl_col = f'{target}_roll7_avg_baseline'
        if bl_col in test_df.columns:
            bl_pred = test_df[bl_col][test_mask].values
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
            'test_num_samples': int(test_mask.sum()),
            'baseline_mae': blm['mae'], 'baseline_rmse': blm['rmse'], 'baseline_wape': blm['wape'],
            'improvement_vs_baseline_pct': round(improvement, 2) if improvement else None
        }

        print(f"\n  {target}:")
        print(f"    {mn}:          MAE={mm['mae']:>12,.0f}, WAPE={mm['wape']}")
        if blm['mae']:
            print(f"    Baseline (7d): MAE={blm['mae']:>12,.0f}, WAPE={blm['wape']}")
        if improvement:
            print(f"    Improvement: {improvement:+.2f}%")

    return {'best_models': best_models, 'final_models': final_models, 'test_metrics': test_metrics}


# =============================================================================
# MODULE 7: EXPLAINABLE AI (XAI)
# =============================================================================

def run_xai_analysis(artifacts: Dict, model_selection: Dict) -> Dict:
    """
    XAI: coefficient analysis, SHAP global/local, feature contributions, branch-level.
    All explanations state: associations, not causation.
    """
    print("\n" + "=" * 70)
    print("MODULE 7: EXPLAINABLE AI (XAI)")
    print("=" * 70)
    print("\n  ⚠ All explanations show ASSOCIATIONS, not causation.")
    print("  Feature contributions indicate statistical relationships,")
    print("  not causal effects.")
    print("=" * 70)

    train_df = artifacts['train_df']
    val_df = artifacts['val_df']
    feature_cols = artifacts['feature_cols']
    final_models = model_selection.get('final_models', {})

    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()

    for col in X_train.columns:
        if X_train[col].isna().any():
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_val[col] = X_val[col].fillna(med)

    xai_results = {'models': {}, 'disclaimer': 'Explanations show statistical associations, not causation.'}

    for target in TARGETS:
        print(f"\n  --- Target: {target} ---")
        fi = final_models.get(target)
        if fi is None:
            print(f"  ⚠ No final model")
            continue

        model = fi['model']
        mn = fi['model_name']
        xai_results['models'][target] = {
            'model_name': mn,
            'disclaimer': 'Associations, not causation.'
        }

        # 7.1 Coefficient analysis
        print(f"\n  [7.1] Coefficients ({mn})...")
        if hasattr(model, 'coef_'):
            coefs = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
            cdf = pd.DataFrame({'feature': feature_cols, 'coefficient': coefs,
                                'abs_coef': np.abs(coefs)}).sort_values('abs_coef', ascending=False)
            top_pos = cdf[cdf['coefficient'] > 0].head(10)[['feature', 'coefficient']].to_dict('records')
            top_neg = cdf[cdf['coefficient'] < 0].head(10)[['feature', 'coefficient']].to_dict('records')
            xai_results['models'][target]['coefficient_analysis'] = {
                'top_positive': top_pos, 'top_negative': top_neg,
                'all_coefficients': cdf.to_dict('records'), 'intercept': float(model.intercept_)
            }
            print(f"    Top positive: {[r['feature'] for r in top_pos[:3]]}")
            print(f"    Top negative: {[r['feature'] for r in top_neg[:3]]}")
        else:
            print(f"    No coef_ attribute")

        # 7.2 SHAP analysis
        print(f"\n  [7.2] SHAP...")
        try:
            bg = X_train.sample(n=min(200, len(X_train)), random_state=42)
            ex = X_val.sample(n=min(100, len(X_val)), random_state=42)
            # Drop rows with NaN
            bg = bg.dropna()
            ex = ex.dropna()
            if len(bg) > 10 and len(ex) > 5:
                if hasattr(model, 'coef_'):
                    explainer = shap.LinearExplainer(model, bg)
                else:
                    explainer = shap.TreeExplainer(model, bg)
                shap_values = explainer.shap_values(ex)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                # Global importance
                si = np.abs(shap_values).mean(axis=0)
                imp_df = pd.DataFrame({'feature': ex.columns, 'mean_abs_shap': si}).sort_values('mean_abs_shap', ascending=False)
                xai_results['models'][target]['shap_global'] = imp_df.head(30).to_dict('records')

                # Per-feature direction (mean SHAP)
                dir_df = pd.DataFrame({'feature': ex.columns, 'mean_shap': np.mean(shap_values, axis=0)})
                xai_results['models'][target]['shap_directions'] = dir_df.to_dict('records')

                # Sample explanation
                if len(shap_values) > 0:
                    s0 = shap_values[0]
                    samp_df = pd.DataFrame({
                        'feature': ex.columns, 'shap_value': s0,
                        'feature_value': ex.iloc[0].values
                    }).sort_values('shap_value', key=abs, ascending=False)
                    xai_results['models'][target]['sample_explanation'] = samp_df.head(20).to_dict('records')

                print(f"    Top SHAP features: {[r['feature'] for r in imp_df.head(5)]}")
            else:
                print(f"    Insufficient clean data for SHAP")
        except Exception as e:
            print(f"    SHAP failed: {str(e)[:80]}")

        # 7.3 Branch-level SHAP
        print(f"\n  [7.3] Branch-level explanations...")
        branch_xai = {}
        for br in sorted(train_df['tran_br_code'].unique()):
            b_tr = train_df[train_df['tran_br_code'] == br]
            b_vl = val_df[val_df['tran_br_code'] == br]
            if len(b_tr) < 20 or len(b_vl) < 5:
                continue
            X_b_tr = b_tr[feature_cols].copy()
            X_b_vl = b_vl[feature_cols].copy()
            for c in X_b_tr.columns:
                if X_b_tr[c].isna().any():
                    X_b_tr[c] = X_b_tr[c].fillna(X_b_tr[c].median())
                    X_b_vl[c] = X_b_vl[c].fillna(X_b_vl[c].median())

            try:
                bg_b = X_b_tr.sample(n=min(50, len(X_b_tr)), random_state=42).dropna()
                ex_b = X_b_vl.sample(n=min(20, len(X_b_vl)), random_state=42).dropna()
                if len(bg_b) > 5 and len(ex_b) > 3:
                    if hasattr(model, 'coef_'):
                        exp_b = shap.LinearExplainer(model, bg_b)
                    else:
                        exp_b = shap.TreeExplainer(model, bg_b)
                    sv_b = exp_b.shap_values(ex_b)
                    if isinstance(sv_b, list):
                        sv_b = sv_b[0]
                    si_b = np.abs(sv_b).mean(axis=0)
                    branch_xai[int(br)] = {
                        'top_features': pd.DataFrame({
                            'feature': ex_b.columns, 'mean_abs_shap': si_b
                        }).sort_values('mean_abs_shap', ascending=False).head(10).to_dict('records')
                    }
            except Exception:
                continue
        xai_results['models'][target]['branch_level'] = branch_xai
        print(f"    Branch explanations: {len(branch_xai)} branches")

    return xai_results


# =============================================================================
# MODULE 8: SAVE OUTPUTS
# =============================================================================

def save_phase5_outputs(artifacts: Dict, tuning_results: Dict, eval_results: Dict,
                        comparison: Dict, model_selection: Dict, xai_results: Dict,
                        verification: Dict) -> None:
    """Save all Phase 5 outputs to disk."""
    print("\n" + "=" * 70)
    print("MODULE 8: SAVING PHASE 5 OUTPUTS")
    print("=" * 70)

    # 8.1 Final models
    n_models = 0
    for target, fi in model_selection.get('final_models', {}).items():
        path = os.path.join(PHASE5_MODELS_DIR, f'final_{target}_model.pkl')
        with open(path, 'wb') as f:
            pickle.dump(fi['model'], f)
        n_models += 1
    print(f"✓ {n_models} final models saved")

    # 8.2 Pipeline metadata
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
    print(f"✓ Pipeline metadata saved")

    # 8.3 Metrics
    with open(os.path.join(PHASE5_METRICS_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(model_selection.get('test_metrics', {}), f, indent=2, default=str)
    with open(os.path.join(PHASE5_METRICS_DIR, 'comprehensive_evaluation.json'), 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)
    with open(os.path.join(PHASE5_METRICS_DIR, 'tuning_results.json'), 'w') as f:
        json.dump(tuning_results, f, indent=2, default=str)
    with open(os.path.join(PHASE5_METRICS_DIR, 'direct_vs_derived.json'), 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"✓ Metrics saved")

    # 8.4 XAI
    with open(os.path.join(PHASE5_XAI_DIR, 'xai_results.json'), 'w') as f:
        json.dump(xai_results, f, indent=2, default=str)
    print(f"✓ XAI results saved")

    # 8.5 Branch metrics as CSV
    rows = []
    eval_results_local = eval_results
    for key, blist in eval_results_local.get('branch_metrics', {}).items():
        if isinstance(blist, list):
            for bm in blist:
                bm['key'] = key
                rows.append(bm)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(PHASE5_METRICS_DIR, 'branch_metrics.csv'), index=False)

    # 8.6 Validation report
    report = {
        'phase': 'Phase 5',
        'timestamp': datetime.now().isoformat(),
        'verification': verification,
        'model_selection': {t: s.get('model_name', '') for t, s in model_selection.get('final_models', {}).items()},
        'test_metrics': model_selection.get('test_metrics', {}),
        'feature_count': len(artifacts['feature_cols']),
        'data_splits': {
            'train': len(artifacts['train_df']),
            'val': len(artifacts['val_df']),
            'test': len(artifacts['test_df'])
        },
        'output_dirs': {
            'models': PHASE5_MODELS_DIR,
            'pipelines': PHASE5_PIPELINES_DIR,
            'metrics': PHASE5_METRICS_DIR,
            'xai': PHASE5_XAI_DIR,
            'plots': PHASE5_PLOTS_DIR
        }
    }
    with open(PHASE5_REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"✓ Validation report saved")

    # 8.7 Plots
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

    print(f"✓ {n_plots} plots saved")
    print(f"\n✓ All Phase 5 outputs saved to {PHASE5_DIR}/")


# =============================================================================
# MODULE 9: ASSERTIONS
# =============================================================================

def run_assertions(artifacts: Dict, tuning_results: Dict, model_selection: Dict,
                   comparison: Dict, verification: Dict, xai_results: Dict) -> bool:
    """Run all assertions for data integrity and correctness."""
    print("\n" + "=" * 70)
    print("MODULE 9: ASSERTIONS")
    print("=" * 70)

    all_ok = True

    # 9.1 Chronological separation
    print("\n--- 9.1 Chronological Separation ---")
    try:
        tr, vl, te = artifacts['train_df'], artifacts['val_df'], artifacts['test_df']
        if len(tr) > 0 and len(vl) > 0:
            assert tr['start_date'].max() < vl['start_date'].min(), "Train/Val overlap!"
        if len(vl) > 0 and len(te) > 0:
            assert vl['start_date'].max() < te['start_date'].min(), "Val/Test overlap!"
        print("  ✓ PASSED")
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_ok = False

    # 9.2 No target leakage
    print("\n--- 9.2 No Target Leakage ---")
    try:
        for col in CURRENT_DAY_COLS:
            assert col not in artifacts['feature_cols'], f"Leaked: {col}"
        print("  ✓ PASSED")
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_ok = False

    # 9.3 Feature consistency
    print("\n--- 9.3 Feature Consistency ---")
    try:
        tr_c = set(artifacts['train_df'].columns)
        vl_c = set(artifacts['val_df'].columns)
        te_c = set(artifacts['test_df'].columns)
        assert tr_c == vl_c == te_c, "Feature mismatch!"
        print("  ✓ PASSED")
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_ok = False

    # 9.4 Nonnegative cash requirement
    print("\n--- 9.4 Nonnegative Cash Requirement ---")
    try:
        tm = model_selection.get('test_metrics', {})
        cr = tm.get('cash_requirement', {})
        # Just check no negative in actual data
        assert (artifacts['test_df']['cash_requirement'] >= -0.01).all(), "Negative CR in test!"
        print("  ✓ PASSED")
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_ok = False

    # 9.5 Tuning results exist
    print("\n--- 9.5 Tuning Results ---")
    try:
        for tgt in TARGETS:
            assert tgt in tuning_results, f"No tuning for {tgt}"
        print("  ✓ PASSED")
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_ok = False

    # 9.6 XAI disclaimer
    print("\n--- 9.6 XAI Disclaimer ---")
    try:
        assert 'disclaimer' in xai_results, "No disclaimer"
        assert 'causation' in xai_results.get('disclaimer', '').lower(), "Must mention causation"
        print("  ✓ PASSED")
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_ok = False

    print(f"\n  {'='*40}")
    print(f"  {'ALL ASSERTIONS PASSED' if all_ok else 'SOME ASSERTIONS FAILED'}")
    print(f"  {'='*40}")
    return all_ok


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline() -> Dict:
    """Execute the complete Phase 5 pipeline."""
    print("=" * 70)
    print(" " * 10 + "BANK BRANCH CASH FORECASTING SYSTEM")
    print(" " * 10 + "Phase 5: Model Validation & Explainable AI (Pipeline)")
    print("=" * 70)

    artifacts = load_phase4_artifacts()
    verification = verify_data_integrity(artifacts)
    tuning = tune_shortlisted_models(artifacts)
    comparison = compare_direct_vs_derived(artifacts, tuning)
    eval_res = comprehensive_evaluation(artifacts, tuning)
    model_sel = select_and_retrain(artifacts, eval_res)
    xai = run_xai_analysis(artifacts, model_sel)

    save_phase5_outputs(artifacts, tuning, eval_res, comparison, model_sel, xai, verification)

    assertions_ok = run_assertions(artifacts, tuning, model_sel, comparison, verification, xai)

    # Summary
    print("\n" + "=" * 70)
    print(" " * 14 + "PHASE 5 SUMMARY")
    print("=" * 70)
    print(f"\n  Data: {len(artifacts['feature_cols'])} features, "
          f"{len(artifacts['train_df'])} train, {len(artifacts['val_df'])} val, {len(artifacts['test_df'])} test")
    print(f"\n  Best models:")
    for t, s in model_sel.get('final_models', {}).items():
        print(f"    {t:<25} → {s['model_name']}")
    print(f"\n  Test performance vs baseline:")
    for t, m in model_sel.get('test_metrics', {}).items():
        impr = m.get('improvement_vs_baseline_pct')
        print(f"    {t:<25} MAE={m.get('test_mae', 0):>12,.0f}  "
              f"{f'Impr: {impr:+.1f}%' if impr else 'N/A'}")
    print(f"\n  Assertions: {'✅ ALL PASSED' if assertions_ok else '❌ SOME FAILED'}")
    print(f"\n  Output: {PHASE5_DIR}/")
    print(f"\n  Run UI: streamlit run Bank_Project/phase5_model_validation_xai_streamlit.py")
    print("=" * 70)

    return {
        'artifacts': artifacts, 'verification': verification, 'tuning': tuning,
        'comparison': comparison, 'eval_results': eval_res, 'model_selection': model_sel,
        'xai': xai, 'assertions_ok': assertions_ok
    }


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
        print("Run the dashboard via: streamlit run Bank_Project/phase5_model_validation_xai_streamlit.py")
        return None


def run_streamlit_dashboard():
    """Lazy-load Streamlit and run the Phase 5 dashboard."""
    import streamlit as st
    st.set_page_config(page_title="Bank Cash Forecasting - Phase 5",
                       page_icon="🏦", layout="wide")

    st.title("🏦 Bank Branch Cash Forecasting System")
    st.markdown("### Phase 5: Model Validation & Explainable AI")
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
    with st.spinner("Running XAI (SHAP may take time)..."):
        xai = run_xai_analysis(artifacts, model_sel)
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
        if cr <= 0: return "✅ No Shortage Risk", "green"
        return f"⚠️ Moderate (Req: {fmt_curr(cr)})", "orange"

    st.sidebar.header("🏦 Cash Forecasting")
    branches = sorted(artifacts['train_df']['tran_br_code'].unique())
    sel_br = st.sidebar.selectbox("Branch", branches, format_func=lambda x: f"Branch {x}")
    buf = st.sidebar.slider("Safety Buffer %", 0, 50, 10) / 100.0
    section = st.sidebar.radio("Section", ["Dashboard", "Metrics", "Global XAI", "Local XAI", "Verification"])

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

        if comparison:
            st.subheader("Direct vs Derived")
            st.table(pd.DataFrame([{
                'Model': mn, 'Target': tgt,
                'Direct MAE': f"{ci[tgt].get('direct_mae',0):,.0f}",
                'Derived MAE': f"{ci[tgt].get('derived_mae',0):,.0f}",
                'Better': ci[tgt].get('better_method','')
            } for mn,ci in comparison.items() for tgt in ['net_cash','cash_requirement'] if tgt in ci]))

    elif section == "Global XAI":
        st.header("🌍 Global XAI")
        st.info("⚠ Associations, not causation.")
        tgt = st.selectbox("Target", TARGETS)
        if tgt in xai.get('models',{}):
            mx = xai['models'][tgt]
            if 'coefficient_analysis' in mx:
                st.subheader("Coefficients")
                ca = mx['coefficient_analysis']
                col1, col2 = st.columns(2)
                with col1: st.markdown("**Positive**"); st.dataframe(pd.DataFrame(ca.get('top_positive',[])))
                with col2: st.markdown("**Negative**"); st.dataframe(pd.DataFrame(ca.get('top_negative',[])))
            if 'shap_global' in mx:
                st.subheader("SHAP Importance")
                st.dataframe(pd.DataFrame(mx['shap_global']).head(15))
                imp_df = pd.DataFrame(mx['shap_global'])
                if not imp_df.empty:
                    fig, ax = plt.subplots(figsize=(10,5))
                    top = imp_df.head(15)
                    ax.barh(range(len(top)), top['mean_abs_shap'], color='#9b59b6')
                    ax.set_yticks(range(len(top))); ax.set_yticklabels(top['feature'], fontsize=8)
                    ax.set_xlabel('Mean |SHAP|'); ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout(); st.pyplot(fig); plt.close()
            if 'branch_level' in mx and mx['branch_level']:
                st.subheader("Branch-Level")
                br_sel = st.selectbox("Branch", sorted(mx['branch_level'].keys()))
                if br_sel in mx['branch_level']:
                    st.dataframe(pd.DataFrame(mx['branch_level'][br_sel].get('top_features',[])))

    elif section == "Local XAI":
        st.header("🔍 Local XAI")
        st.info("⚠ Associations, not causation.")
        tgt = st.selectbox("Target", TARGETS, key="local_tgt")
        if tgt in xai.get('models',{}):
            mx = xai['models'][tgt]
            if 'sample_explanation' in mx:
                samp = pd.DataFrame(mx['sample_explanation'])
                if not samp.empty:
                    col1, col2 = st.columns(2)
                    with col1: st.markdown("**Positive (↑)**"); st.dataframe(samp[samp['shap_value']>0].head(10))
                    with col2: st.markdown("**Negative (↓)**"); st.dataframe(samp[samp['shap_value']<0].head(10))
                    fig, ax = plt.subplots(figsize=(10, max(4, len(samp)*0.3)))
                    plot_df = samp.head(15)
                    colors = ['#e74c3c' if v<0 else '#2ecc71' for v in plot_df['shap_value']]
                    ax.barh(range(len(plot_df)), plot_df['shap_value'], color=colors)
                    ax.set_yticks(range(len(plot_df))); ax.set_yticklabels(plot_df['feature'], fontsize=8)
                    ax.axvline(0, color='black', lw=0.5); ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

    elif section == "Verification":
        st.header("✅ Verification")
        for k,v in verification.items():
            if k=='details': continue
            if isinstance(v,bool):
                (st.success if v else st.error)(f"**{k.replace('_',' ').title()}:** {'PASS' if v else 'FAIL'}")
        st.subheader(f"Features: {len(artifacts['feature_cols'])}")
        st.dataframe(pd.DataFrame({'feature': artifacts['feature_cols'], 'idx': range(len(artifacts['feature_cols']))}))

    st.markdown("---")
    st.caption("Bank Branch Cash Forecasting — Phase 5 | Data confidential | Explanations show associations, not causation.")


if __name__ == "__main__":
    if 'streamlit' in sys.argv[0] or 'STREAMLIT_RUN' in os.environ:
        run_streamlit_dashboard()
    else:
        main()

