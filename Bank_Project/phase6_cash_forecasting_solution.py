"""
==============================================================================
BANK BRANCH CASH FORECASTING SYSTEM
Phase 6: Cash Forecasting Solution & Decision Support Dashboard
==============================================================================

This script loads saved Phase 5 models and generates daily branch-level
cash forecasts without retraining.

Features:
- Daily forecasts for inflow, outflow, net cash, and cash requirement
- Replenishment guidance (shortage/surplus/balanced)
- Streamlit dashboard with interactive visualizations
- Model performance vs 7-day baseline
- Downloadable forecast CSV

Author: Muhammad Usman
Status: Phase 6 Implementation
==============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import json
import pickle
import warnings
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass  # Streamlit replaces stdout with a custom object that lacks reconfigure()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Resolve BASE_DIR relative to this script's location so it works
# regardless of the working directory (important for Streamlit)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "Bank DataSet")
PHASE5_DIR = os.path.join(BASE_DIR, "phase5_output")
PHASE6_DIR = os.path.join(BASE_DIR, "phase6_output")

# Paths
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "train_data.csv")
VAL_DATA_PATH = os.path.join(BASE_DIR, "validation_data.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")
MODEL_METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
BASELINE_METRICS_PATH = os.path.join(BASE_DIR, "baseline_metrics.json")

PHASE5_MODELS_DIR = os.path.join(PHASE5_DIR, "final_models")
PHASE5_PIPELINES_DIR = os.path.join(PHASE5_DIR, "pipelines")
PHASE5_METRICS_DIR = os.path.join(PHASE5_DIR, "metrics")

PHASE6_FORECASTS_DIR = os.path.join(PHASE6_DIR, "forecasts")
PHASE6_DASHBOARD_DIR = os.path.join(PHASE6_DIR, "dashboard_data")
PHASE6_REPORTS_DIR = os.path.join(PHASE6_DIR, "reports")

for d in [PHASE6_DIR, PHASE6_FORECASTS_DIR, PHASE6_DASHBOARD_DIR, PHASE6_REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

TARGETS = ['daily_withdrawals', 'daily_deposits']

# =============================================================================
# MODULE 1: LOAD PHASE 5 ARTIFACTS
# =============================================================================

def load_phase5_models_and_pipeline() -> Dict:
    """
    Load saved Phase 5 models and pipeline metadata.
    Does NOT retrain models.
    """
    print("=" * 70)
    print("MODULE 1: LOAD PHASE 5 ARTIFACTS")
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
    artifacts['train_df'] = train_df
    artifacts['val_df'] = val_df
    artifacts['test_df'] = test_df

    # 1.2 Load pipeline metadata
    print("\n--- 1.2 Loading pipeline metadata ---")
    pipeline_meta_path = os.path.join(PHASE5_PIPELINES_DIR, 'pipeline_metadata.json')
    if not os.path.exists(pipeline_meta_path):
        raise FileNotFoundError(f"Pipeline metadata not found at {pipeline_meta_path}")
    
    with open(pipeline_meta_path, 'r') as f:
        pipeline_meta = json.load(f)
    
    feature_cols = pipeline_meta['feature_columns']
    print(f"  Loaded {len(feature_cols)} features from pipeline metadata")
    artifacts['feature_cols'] = feature_cols
    artifacts['pipeline_meta'] = pipeline_meta

    # 1.3 Load final models
    print("\n--- 1.3 Loading final models ---")
    final_models = {}
    for target in TARGETS:
        model_path = os.path.join(PHASE5_MODELS_DIR, f'final_{target}_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        final_models[target] = model
        print(f"  Loaded {target} model")
    artifacts['final_models'] = final_models

    # 1.4 Load test metrics
    print("\n--- 1.4 Loading test metrics ---")
    test_metrics_path = os.path.join(PHASE5_METRICS_DIR, 'test_metrics.json')
    if os.path.exists(test_metrics_path):
        with open(test_metrics_path, 'r') as f:
            test_metrics = json.load(f)
        artifacts['test_metrics'] = test_metrics
        print(f"  Loaded test metrics for {len(test_metrics)} targets")
    else:
        artifacts['test_metrics'] = {}

    # 1.5 Load baseline metrics
    print("\n--- 1.5 Loading baseline metrics ---")
    if os.path.exists(BASELINE_METRICS_PATH):
        with open(BASELINE_METRICS_PATH, 'r') as f:
            baseline_metrics = json.load(f)
        artifacts['baseline_metrics'] = baseline_metrics
        print(f"  Loaded baseline metrics")
    else:
        artifacts['baseline_metrics'] = {}

    print("\nPhase 5 artifacts loaded successfully")
    return artifacts


# =============================================================================
# MODULE 2: FEATURE PREPARATION & VALIDATION
# =============================================================================

def prepare_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Prepare feature matrix ensuring correct column order and handling missing values.
    """
    # Ensure correct feature order
    X = df[feature_cols].copy()
    
    # Handle missing values with median (from training data if available)
    for col in X.columns:
        if X[col].isna().any():
            med = X[col].median()
            if pd.isna(med):
                med = 0
            X[col] = X[col].fillna(med)
            
    # Force all columns to be numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)
    
    return X


def validate_branch_and_date(branches: List, dates: List, valid_branches: set, 
                          min_date: datetime, max_date: datetime) -> List[str]:
    """Validate branch codes and forecast dates."""
    errors = []
    
    for br in branches:
        if br not in valid_branches:
            errors.append(f"Unknown branch: {br}")
    
    for dt in dates:
        if not isinstance(dt, (pd.Timestamp, datetime)):
            errors.append(f"Invalid date type: {dt}")
        elif dt < min_date:
            errors.append(f"Date {dt.date()} is before training data range")
    
    return errors


def _update_date_features(row: pd.Series, forecast_date: pd.Timestamp,
                      feature_cols: List[str]) -> pd.Series:
    """
    Update all calendar/time features in a row to match the actual forecast date.
    This ensures predictions change day-to-day instead of being flat.
    """
    row = row.copy()

    # Core calendar features
    if 'year' in feature_cols:
        row['year'] = forecast_date.year
    if 'month' in feature_cols:
        row['month'] = forecast_date.month
    if 'day' in feature_cols:
        row['day'] = forecast_date.day
    if 'dayofweek' in feature_cols:
        row['dayofweek'] = forecast_date.dayofweek
    if 'dayofyear' in feature_cols:
        row['dayofyear'] = forecast_date.dayofyear
    if 'weekofyear' in feature_cols:
        row['weekofyear'] = forecast_date.isocalendar()[1]
    if 'quarter' in feature_cols:
        row['quarter'] = forecast_date.quarter

    # Boolean calendar features
    if 'is_weekend' in feature_cols:
        row['is_weekend'] = int(forecast_date.dayofweek >= 5)
    if 'is_month_start' in feature_cols:
        row['is_month_start'] = int(forecast_date.day == 1)
    if 'is_month_end' in feature_cols:
        row['is_month_end'] = int(forecast_date.day == forecast_date.days_in_month)
    if 'is_quarter_start' in feature_cols:
        row['is_quarter_start'] = int(forecast_date.month in [1, 4, 7, 10] and forecast_date.day == 1)
    if 'is_quarter_end' in feature_cols:
        row['is_quarter_end'] = int(
            (forecast_date.month in [3, 6, 9, 12]) and
            (forecast_date.day == forecast_date.days_in_month)
        )

    # Cyclical features
    if 'month_sin' in feature_cols:
        row['month_sin'] = math.sin(2 * math.pi * forecast_date.month / 12)
    if 'month_cos' in feature_cols:
        row['month_cos'] = math.cos(2 * math.pi * forecast_date.month / 12)
    if 'dayofweek_sin' in feature_cols:
        row['dayofweek_sin'] = math.sin(2 * math.pi * forecast_date.dayofweek / 7)
    if 'dayofweek_cos' in feature_cols:
        row['dayofweek_cos'] = math.cos(2 * math.pi * forecast_date.dayofweek / 7)

    return row


def _build_feature_row(base_row: pd.Series, forecast_date: pd.Timestamp,
                       feature_cols: list, prediction_cache: dict, 
                       branch_hist: pd.DataFrame) -> pd.Series:
    """
    """
    row = base_row.copy()
    
    # 1. Calendar features
    row = _update_date_features(row, forecast_date, feature_cols)
    
    def _val(target: str, ref_date: pd.Timestamp) -> float:
        ref_date = ref_date.normalize()
        if ref_date in prediction_cache:
            return prediction_cache[ref_date].get(target, np.nan)
        match = branch_hist[branch_hist['start_date'] == ref_date]
        if len(match) > 0 and target in match.columns and pd.notna(match.iloc[0][target]):
            return float(match.iloc[0][target])
        return np.nan

    def get_past_period(dt: pd.Timestamp, k_periods: int) -> pd.Timestamp:
        return dt - pd.Timedelta(days=k_periods)

    # 2. Exact-period lag features
    for lag_periods, suffix in [(1, '1p_lag'), (7, '7p_lag'), (14, '14p_lag'), (28, '28p_lag')]:
        ref_date = get_past_period(forecast_date, lag_periods)
        for target in TARGETS:
            col = f'{target}_{suffix}'
            if col in feature_cols:
                v = _val(target, ref_date)
                if not pd.isna(v):
                    row[col] = v
                    
    # 3. Rolling window features
    for window, suffix in [(7, '7p_rolling'), (14, '14p_rolling'), (30, '30p_rolling')]:
        vals = []
        for p in range(1, window + 1):
            ref_date = get_past_period(forecast_date, p)
            for target in TARGETS:
                v = _val(target, ref_date)
                if not pd.isna(v):
                    vals.append((target, v))
        for target in TARGETS:
            t_vals = [v for t, v in vals if t == target]
            if t_vals:
                mean_col = f'{target}_{suffix}_mean'
                std_col  = f'{target}_{suffix}_std'
                min_col  = f'{target}_{suffix}_min'
                max_col  = f'{target}_{suffix}_max'
                if mean_col in feature_cols: row[mean_col] = np.mean(t_vals)
                if std_col in feature_cols:  row[std_col] = np.std(t_vals, ddof=1) if len(t_vals) > 1 else 0.0
                if min_col in feature_cols:  row[min_col] = np.min(t_vals)
                if max_col in feature_cols:  row[max_col] = np.max(t_vals)
                
    # 4. Baseline feature
    for target in TARGETS:
        bl_col = f'{target}_roll7_avg_baseline'
        if bl_col in feature_cols:
            bl_vals = []
            for p in range(1, 8):
                ref_date = get_past_period(forecast_date, p)
                v = _val(target, ref_date)
                if not pd.isna(v):
                    bl_vals.append(v)
            row[bl_col] = np.mean(bl_vals) if bl_vals else row.get(bl_col, 0)

    return row

def generate_forecast_explanation(row: pd.Series, pred_withdrawals: float, pred_deposits: float) -> str:
    """Generate a rule-based explanation based on date features and rolling averages."""
    reasons = []
    
    # 1. Calendar events
    if row.get('is_salary_day', 0) == 1:
        reasons.append("Salary day spike anticipated.")
    elif row.get('is_month_start', 0) == 1:
        reasons.append("Start of month typically sees higher transaction volume.")
    elif row.get('is_month_end', 0) == 1:
        reasons.append("End of month patterns indicate elevated cash movement.")
        
    # 2. Day of week patterns
    if row.get('is_weekend', 0) == 1:
        reasons.append("Weekend trend detected; typically lower activity.")
    elif row.get('dayofweek', 0) == 4: # Friday
        reasons.append("Friday peak activity expected.")
        
    # 3. Rolling averages comparison
    bl_wdr = row.get('daily_withdrawals_roll7_avg_baseline', 0)
    bl_dep = row.get('daily_deposits_roll7_avg_baseline', 0)
    
    if bl_wdr > 0 and pred_withdrawals > bl_wdr * 1.2:
        reasons.append("Withdrawals projected 20%+ above recent average.")
    elif bl_dep > 0 and pred_deposits > bl_dep * 1.2:
        reasons.append("Deposits projected 20%+ above recent average.")
        
    if not reasons:
        reasons.append("Aligns with recent historical trends and rolling averages.")
        
    return " ".join(reasons)


# =============================================================================
# MODULE 3: FORECAST GENERATION
# =============================================================================

def generate_daily_forecasts(artifacts: Dict, forecast_dates: List[datetime],
                              branches: Optional[List] = None) -> pd.DataFrame:
    """
    Generate half-daily branch-level forecasts for cash inflow, outflow, net cash,
    and cash requirement.
    """
    print("\n" + "=" * 70)
    print("MODULE 3: GENERATE HALF-DAILY FORECASTS")
    print("=" * 70)
    
    train_df = artifacts['train_df']
    test_df = artifacts['test_df']
    feature_cols = artifacts['feature_cols']
    final_models = artifacts['final_models']
    
    all_branches = sorted(train_df['tran_br_code'].unique())
    if branches is None:
        branches = all_branches
    else:
        branches = [b for b in branches if b in all_branches]
    
    if not branches:
        raise ValueError("No valid branches specified")
    
    min_date = train_df['start_date'].min()
    max_date = test_df['start_date'].max() if len(test_df) > 0 else train_df['start_date'].max()
    
    forecast_dates = sorted([pd.Timestamp(d).normalize() for d in forecast_dates])
    
    errors = validate_branch_and_date(branches, forecast_dates, set(all_branches), min_date, max_date)
    if errors:
        for err in errors[:5]:
            print(f"  WARNING: {err}")
    
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    all_data = all_data.sort_values(['tran_br_code', 'start_date']).reset_index(drop=True)
    
    forecasts = []
    
    for branch in branches:
        branch_data = all_data[all_data['tran_br_code'] == branch].copy()
        prediction_cache: Dict[pd.Timestamp, Dict[str, float]] = {}
        
        if len(forecast_dates) > 0:
            forecast_origin = forecast_dates[0]
            hist_only = branch_data[branch_data['start_date'] < forecast_origin]
        else:
            hist_only = branch_data
        
        for forecast_date in forecast_dates:
            try:
                if len(hist_only) == 0:
                    print(f"  WARNING: No historical data for branch {branch} before {forecast_date.date()}")
                    continue
                
                base_row = hist_only.iloc[-1].copy()
                base_row['start_date'] = forecast_date
                
                # Build fully correct feature row for this period
                updated_row = _build_feature_row(
                    base_row, forecast_date, feature_cols, prediction_cache, hist_only
                )
                
                X = prepare_features(updated_row.to_frame().T, feature_cols)
                
                pred_deposits = final_models['daily_deposits'].predict(X)[0]
                pred_withdrawals = final_models['daily_withdrawals'].predict(X)[0]
                pred_net_cash = pred_deposits - pred_withdrawals
                pred_cash_requirement = max(pred_withdrawals - pred_deposits, 0)
                
                assert pred_cash_requirement >= 0, f"Negative cash requirement: {pred_cash_requirement}"
                
                prediction_cache[(forecast_date)] = {
                    'daily_deposits': pred_deposits,
                    'daily_withdrawals': pred_withdrawals,
                    'net_cash': pred_net_cash,
                    'cash_requirement': pred_cash_requirement,
                }
                
                if pred_cash_requirement > 0:
                    status = "Shortage"
                elif pred_deposits > pred_withdrawals * 1.1:
                    status = "Surplus"
                else:
                    status = "Balanced"
                    
                explanation = generate_forecast_explanation(updated_row, pred_withdrawals, pred_deposits)
                
                forecasts.append({
                    'branch_code': branch,
                    'forecast_date': forecast_date,
                    
                    'predicted_deposits': pred_deposits,
                    'predicted_withdrawals': pred_withdrawals,
                    'predicted_net_cash': pred_net_cash,
                    'predicted_cash_requirement': pred_cash_requirement,
                    'status': status,
                    'explanation': explanation
                })
                
            except Exception as e:
                print(f"  ERROR forecasting branch {branch} on {forecast_date.date()}: {str(e)[:100]}")
                continue
    
    forecast_df = pd.DataFrame(forecasts)
    
    if len(forecast_df) > 0:
        print(f"\n  Generated {len(forecast_df)} half-daily forecasts")
        print(f"  Branches: {forecast_df['branch_code'].nunique()}")
        print(f"  Dates: {forecast_df['forecast_date'].nunique()}")
        print(f"  Date range: {forecast_df['forecast_date'].min().date()} to {forecast_df['forecast_date'].max().date()}")
        print(f"  Total predicted inflow: {forecast_df['predicted_deposits'].sum():,.0f}")
        print(f"  Total predicted outflow: {forecast_df['predicted_withdrawals'].sum():,.0f}")
        print(f"  Total predicted cash requirement: {forecast_df['predicted_cash_requirement'].sum():,.0f}")
        print(f"  Status distribution:")
        for status, count in forecast_df['status'].value_counts().items():
            print(f"    {status}: {count}")
    else:
        print("  WARNING: No forecasts generated")
    
    return forecast_df


# =============================================================================
# MODULE 4: REPLENISHMENT GUIDANCE
# =============================================================================

def generate_replenishment_guidance(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate simple replenishment guidance based on forecast results.
    """
    print("\n" + "=" * 70)
    print("MODULE 4: REPLENISHMENT GUIDANCE")
    print("=" * 70)
    
    if len(forecast_df) == 0:
        print("  No forecast data available")
        return pd.DataFrame()
    
    # Add guidance
    forecast_df['replenishment_guidance'] = forecast_df['status'].apply(lambda x: {
        'Shortage': 'Replenishment may be required',
        'Surplus': 'Possible excess idle cash',
        'Balanced': 'No major cash gap expected'
    }.get(x, 'Unknown'))
    
    # Add decision support notes
    guidance_rows = []
    for branch in forecast_df['branch_code'].unique():
        branch_data = forecast_df[forecast_df['branch_code'] == branch]
        
        total_cr = branch_data['predicted_cash_requirement'].sum()
        total_net = branch_data['predicted_net_cash'].sum()
        avg_deposits = branch_data['predicted_deposits'].mean()
        avg_withdrawals = branch_data['predicted_withdrawals'].mean()
        
        if total_cr > 0:
            recommendation = "Monitor closely and prepare for potential replenishment"
        elif total_net > avg_deposits * 0.2:
            recommendation = "Consider reallocating excess cash to branches with shortage"
        else:
            recommendation = "Maintain current cash levels"
        
        guidance_rows.append({
            'branch_code': branch,
            'forecast_days': len(branch_data),
            'total_predicted_inflow': branch_data['predicted_deposits'].sum(),
            'total_predicted_outflow': branch_data['predicted_withdrawals'].sum(),
            'total_predicted_net_cash': total_net,
            'total_predicted_cash_requirement': total_cr,
            'shortage_days': int((branch_data['status'] == 'Shortage').sum()),
            'surplus_days': int((branch_data['status'] == 'Surplus').sum()),
            'balanced_days': int((branch_data['status'] == 'Balanced').sum()),
            'primary_status': branch_data['status'].mode()[0] if len(branch_data) > 0 else 'Unknown',
            'recommendation': recommendation
        })
    
    guidance_df = pd.DataFrame(guidance_rows)
    
    print(f"\n  Generated guidance for {len(guidance_df)} branches")
    print(f"  Shortage branches: {(guidance_df['primary_status'] == 'Shortage').sum()}")
    print(f"  Surplus branches: {(guidance_df['primary_status'] == 'Surplus').sum()}")
    print(f"  Balanced branches: {(guidance_df['primary_status'] == 'Balanced').sum()}")
    
    return guidance_df


# =============================================================================
# MODULE 5: MODEL PERFORMANCE VALIDATION
# =============================================================================

def validate_model_performance(artifacts: Dict, forecast_df: pd.DataFrame) -> Dict:
    """
    Validate model performance against 7-day baseline using test data.
    """
    print("\n" + "=" * 70)
    print("MODULE 5: MODEL PERFORMANCE VALIDATION")
    print("=" * 70)
    
    test_df = artifacts['test_df']
    test_metrics = artifacts.get('test_metrics', {})
    baseline_metrics = artifacts.get('baseline_metrics', {})
    final_models = artifacts['final_models']
    feature_cols = artifacts['feature_cols']
    
    validation = {}
    
    # 5.1 Test set predictions
    print("\n--- 5.1 Test Set Performance ---")
    X_test = prepare_features(test_df, feature_cols)
    
    test_predictions = test_df.copy()
    for target in TARGETS:
        pred_col = f'pred_{target}'
        test_predictions[pred_col] = final_models[target].predict(X_test)
    
    # Compute metrics on test set
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    for target in TARGETS:
        mask = test_df[target].notna()
        if mask.sum() == 0:
            continue
        
        bl_col = f'{target}_roll7_avg_baseline'
        if bl_col in test_df.columns:
            combined_mask = mask & test_df[bl_col].notna()
        else:
            combined_mask = mask

        if combined_mask.sum() == 0:
            continue
            
        y_true = test_df[target][combined_mask].values
        y_pred = test_predictions[f'pred_{target}'][combined_mask].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Operational Risk Metrics
        err = y_true - y_pred
        shortfalls = err[err > 0]
        max_shortfall = float(np.max(shortfalls)) if len(shortfalls) > 0 else 0.0
        total_shortfall = float(np.sum(shortfalls)) if len(shortfalls) > 0 else 0.0
        
        # Baseline (7-day rolling average)
        if bl_col in test_df.columns:
            y_bl = test_df[bl_col][combined_mask].values
            bl_mae = mean_absolute_error(y_true, y_bl)
            improvement = (bl_mae - mae) / bl_mae * 100 if bl_mae > 0 else 0
            
            bl_err = y_true - y_bl
            bl_shortfalls = bl_err[bl_err > 0]
            bl_max_shortfall = float(np.max(bl_shortfalls)) if len(bl_shortfalls) > 0 else 0.0
            bl_total_shortfall = float(np.sum(bl_shortfalls)) if len(bl_shortfalls) > 0 else 0.0
        else:
            bl_mae = None
            improvement = None
            bl_max_shortfall = None
            bl_total_shortfall = None
        
        validation[target] = {
            'model_mae': round(mae, 2),
            'model_rmse': round(rmse, 2),
            'max_shortfall': round(max_shortfall, 2),
            'total_shortfall': round(total_shortfall, 2),
            'baseline_mae': round(bl_mae, 2) if bl_mae else None,
            'baseline_max_shortfall': round(bl_max_shortfall, 2) if bl_max_shortfall else None,
            'baseline_total_shortfall': round(bl_total_shortfall, 2) if bl_total_shortfall else None,
            'improvement_pct': round(improvement, 2) if improvement else None,
            'test_samples': int(combined_mask.sum())
        }
        
        print(f"\n  {target}:")
        print(f"    Model MAE:    {mae:>12,.0f} | Max Shortfall: {max_shortfall:>12,.0f}")
        if bl_mae:
            print(f"    Baseline MAE: {bl_mae:>12,.0f} | Max Shortfall: {bl_max_shortfall:>12,.0f}")
            print(f"    Improvement:  {improvement:>11.1f}%")
    
    # 5.2 Forecast validation
    print("\n--- 5.2 Forecast Validation ---")
    if len(forecast_df) > 0:
        print(f"  Forecast records: {len(forecast_df)}")
        print(f"  Branches covered: {forecast_df['branch_code'].nunique()}")
        print(f"  Date range: {forecast_df['forecast_date'].min().date()} to {forecast_df['forecast_date'].max().date()}")
        
        # Check for negative values
        neg_cr = (forecast_df['predicted_cash_requirement'] < 0).sum()
        neg_net = (forecast_df['predicted_net_cash'] < 0).sum()
        
        validation['forecast_checks'] = {
            'negative_cash_requirement': int(neg_cr),
            'negative_net_cash': int(neg_net),
            'all_nonnegative_cr': neg_cr == 0,
            'all_nonnegative_net': neg_net >= 0  # Net cash can be negative
        }
        
        print(f"  Negative cash requirements: {neg_cr}")
        print(f"  Negative net cash values: {neg_net}")
    
    return validation


# =============================================================================
# MODULE 6: ASSERTIONS
# =============================================================================

def run_assertions(artifacts: Dict, forecast_df: pd.DataFrame, validation: Dict) -> bool:
    """
    Run all assertions for correctness and data integrity.
    """
    print("\n" + "=" * 70)
    print("MODULE 6: ASSERTIONS")
    print("=" * 70)
    
    all_ok = True
    
    # 6.1 Correct feature order
    print("\n--- 6.1 Feature Order ---")
    try:
        feature_cols = artifacts['feature_cols']
        final_models = artifacts['final_models']
        assert len(feature_cols) > 0, "No features loaded"
        # Verify feature count matches what models expect
        for target, model in final_models.items():
            if hasattr(model, 'n_features_in_'):
                assert model.n_features_in_ == len(feature_cols), \
                    f"Feature count mismatch for {target}: model expects {model.n_features_in_}, got {len(feature_cols)}"
        print(f"  PASSED ({len(feature_cols)} features validated)")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False
    
    # 6.2 Reproducible predictions
    print("\n--- 6.2 Reproducible Predictions ---")
    try:
        test_df = artifacts['test_df']
        X_test = prepare_features(test_df, artifacts['feature_cols'])
        
        # Predict twice and compare
        pred1 = artifacts['final_models']['daily_deposits'].predict(X_test)
        pred2 = artifacts['final_models']['daily_deposits'].predict(X_test)
        
        assert np.allclose(pred1, pred2), "Predictions not reproducible"
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False
    
    # 6.3 Nonnegative cash requirement
    print("\n--- 6.3 Nonnegative Cash Requirement ---")
    try:
        if len(forecast_df) > 0:
            min_cr = forecast_df['predicted_cash_requirement'].min()
            assert min_cr >= 0, f"Negative cash requirement found: {min_cr}"
            print(f"  PASSED (min={min_cr:,.0f})")
        else:
            print("  SKIPPED (no forecasts)")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False
    
    # 6.4 Correct net-cash formula
    print("\n--- 6.4 Net Cash Formula ---")
    try:
        if len(forecast_df) > 0:
            calc_net = forecast_df['predicted_deposits'] - forecast_df['predicted_withdrawals']
            assert np.allclose(calc_net, forecast_df['predicted_net_cash']), "Net cash formula incorrect"
            print("  PASSED")
        else:
            print("  SKIPPED (no forecasts)")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False
    
    # 6.5 Valid branch and date values
    print("\n--- 6.5 Valid Branch and Date Values ---")
    try:
        if len(forecast_df) > 0:
            valid_branches = set(artifacts['train_df']['tran_br_code'].unique())
            invalid_branches = set(forecast_df['branch_code']) - valid_branches
            assert len(invalid_branches) == 0, f"Invalid branches: {invalid_branches}"
            
            # Check dates are valid (allow future dates for forecasting)
            min_date = artifacts['train_df']['start_date'].min()
            forecast_dates = pd.to_datetime(forecast_df['forecast_date'])
            assert (forecast_dates >= min_date).all(), \
            f"Some forecast dates are before training data range"
            print("  PASSED")
        else:
            print("  SKIPPED (no forecasts)")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False
    
    # 6.6 No current-day target leakage
    print("\n--- 6.6 No Current-Day Target Leakage ---")
    try:
        current_day_cols = ['daily_withdrawals', 'daily_deposits', 'net_cash', 'cash_requirement',
                       'transaction_count', 'active_hour_count']
        leaked = [col for col in current_day_cols if col in artifacts['feature_cols']]
        assert len(leaked) == 0, f"Current-day targets in features: {leaked}"
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False
    
    # 6.7 No confidential raw transaction data displayed
    print("\n--- 6.7 No Confidential Data Exposure ---")
    try:
        # Check that forecast output doesn't contain raw transaction data
        if len(forecast_df) > 0:
            forbidden_cols = ['TOTAL_DR', 'TOTAL_CR', 'txn_hour', 'cleaned_data']
            exposed = [col for col in forecast_df.columns if col in forbidden_cols]
            assert len(exposed) == 0, f"Confidential data exposed: {exposed}"
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_ok = False
    
    print(f"\n{'='*40}")
    print(f"{'ALL ASSERTIONS PASSED' if all_ok else 'SOME ASSERTIONS FAILED'}")
    print(f"{'='*40}")
    return all_ok


# =============================================================================
# MODULE 7: SAVE OUTPUTS
# =============================================================================

def save_phase6_outputs(forecast_df: pd.DataFrame, guidance_df: pd.DataFrame,
                     validation: Dict, assertions_ok: bool) -> None:
    """
    Save all Phase 6 outputs to disk.
    """
    print("\n" + "=" * 70)
    print("MODULE 7: SAVE PHASE 6 OUTPUTS")
    print("=" * 70)
    
    # 7.1 Forecasts CSV
    forecast_path = os.path.join(PHASE6_FORECASTS_DIR, 'branch_cash_forecasts.csv')
    forecast_df.to_csv(forecast_path, index=False)
    print(f"\n  Saved: {forecast_path}")
    
    # 7.2 Guidance CSV
    if len(guidance_df) > 0:
        guidance_path = os.path.join(PHASE6_REPORTS_DIR, 'replenishment_guidance.csv')
        guidance_df.to_csv(guidance_path, index=False)
        print(f"  Saved: {guidance_path}")
    
    # 7.3 Model validation JSON
    validation_path = os.path.join(PHASE6_REPORTS_DIR, 'model_validation.json')
    with open(validation_path, 'w') as f:
        json.dump(validation, f, indent=2, default=str)
    print(f"  Saved: {validation_path}")
    
    # 7.4 Dashboard data (aggregated for quick loading)
    if len(forecast_df) > 0:
        dashboard_path = os.path.join(PHASE6_DASHBOARD_DIR, 'dashboard_data.csv')
        forecast_df.to_csv(dashboard_path, index=False)
        print(f"  Saved: {dashboard_path}")
    
    # 7.5 Validation checks JSON
    checks_path = os.path.join(PHASE6_REPORTS_DIR, 'validation_checks.json')
    checks = {
        'timestamp': datetime.now().isoformat(),
        'assertions_passed': assertions_ok,
        'forecast_records': len(forecast_df),
        'branches_covered': int(forecast_df['branch_code'].nunique()) if len(forecast_df) > 0 else 0,
        'dates_covered': int(forecast_df['forecast_date'].nunique()) if len(forecast_df) > 0 else 0,
        'validation': validation
    }
    with open(checks_path, 'w') as f:
        json.dump(checks, f, indent=2, default=str)
    print(f"  Saved: {checks_path}")
    
    print(f"\nAll Phase 6 outputs saved to {PHASE6_DIR}/")


# =============================================================================
# MODULE 8: SUMMARY REPORT
# =============================================================================

def print_summary(forecast_df: pd.DataFrame, guidance_df: pd.DataFrame,
               validation: Dict, assertions_ok: bool) -> None:
    """
    Print concise summary of Phase 6 results.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "PHASE 6 SUMMARY")
    print("=" * 70)
    
    if len(forecast_df) == 0:
        print("\n  No forecasts generated")
        return
    
    n_branches = forecast_df['branch_code'].nunique()
    n_dates = forecast_df['forecast_date'].nunique()
    total_inflow = forecast_df['predicted_deposits'].sum()
    total_outflow = forecast_df['predicted_withdrawals'].sum()
    total_cr = forecast_df['predicted_cash_requirement'].sum()
    
    n_shortage = (forecast_df['status'] == 'Shortage').sum()
    n_surplus = (forecast_df['status'] == 'Surplus').sum()
    n_balanced = (forecast_df['status'] == 'Balanced').sum()
    
    print(f"\n  Forecast Coverage:")
    print(f"    Branches: {n_branches}")
    print(f"    Forecast dates: {n_dates}")
    print(f"    Date range: {forecast_df['forecast_date'].min().date()} to {forecast_df['forecast_date'].max().date()}")
    
    print(f"\n  Predicted Cash Flows:")
    print(f"    Total inflow:  {total_inflow:>15,.0f}")
    print(f"    Total outflow: {total_outflow:>15,.0f}")
    print(f"    Net cash:      {total_inflow - total_outflow:>15,.0f}")
    print(f"    Total CR:      {total_cr:>15,.0f}")
    
    print(f"\n  Status Distribution:")
    print(f"    Shortage: {n_shortage}")
    print(f"    Surplus:  {n_surplus}")
    print(f"    Balanced: {n_balanced}")
    
    print(f"\n  Model Performance (Test Set):")
    for target in TARGETS:
        if target in validation:
            v = validation[target]
            print(f"    {target:<25} MAE={v.get('model_mae', 0):>10,.0f}", end="")
            if v.get('improvement_pct') is not None:
                print(f"  (vs baseline: {v['improvement_pct']:>+6.1f}%)")
            else:
                print()
    
    print(f"\n  Assertions: {'ALL PASSED' if assertions_ok else 'SOME FAILED'}")
    
    print(f"\n  Saved Outputs:")
    print(f"    Forecasts:  {PHASE6_FORECASTS_DIR}/")
    print(f"    Guidance:   {PHASE6_REPORTS_DIR}/")
    print(f"    Dashboard:  {PHASE6_DASHBOARD_DIR}/")
    
    print(f"\n  Dashboard: streamlit run {os.path.join('Bank_Project', 'phase6_cash_forecasting_solution.py')}")
    print("=" * 70)


# =============================================================================
# MODULE 9: STREAMLIT DASHBOARD (User-Friendly Redesign)
# =============================================================================

# ---- Formatting Helpers ----

def _fmt_pkr(value: float, short: bool = True) -> str:
    """Format a number as PKR with readable abbreviation.
    
    Examples:
        53_625_180  -> 'PKR 53.6M'
        1_200_000   -> 'PKR 1.2M'
        850_000     -> 'PKR 850K'
        -4_350_000  -> '-PKR 4.4M'
    """
    if pd.isna(value):
        return "—"
    sign = "-" if value < 0 else ""
    av = abs(value)
    if not short:
        return f"{sign}PKR {av:,.0f}"
    if av >= 1_000_000_000:
        return f"{sign}PKR {av / 1_000_000_000:.1f}B"
    if av >= 1_000_000:
        return f"{sign}PKR {av / 1_000_000:.1f}M"
    if av >= 1_000:
        return f"{sign}PKR {av / 1_000:.0f}K"
    return f"{sign}PKR {av:,.0f}"


def _fmt_pkr_num(value: float) -> str:
    """Format number for table display (shorter, no 'PKR' prefix)."""
    if pd.isna(value):
        return "—"
    sign = "-" if value < 0 else ""
    av = abs(value)
    if av >= 1_000_000_000:
        return f"{sign}{av / 1_000_000_000:.1f}B"
    if av >= 1_000_000:
        return f"{sign}{av / 1_000_000:.1f}M"
    if av >= 1_000:
        return f"{sign}{av / 1_000:.0f}K"
    return f"{sign}{av:,.0f}"


def _status_badge(status: str) -> str:
    """Return an emoji + styled label for branch status."""
    mapping = {
        'Shortage': '🔴 Shortage',
        'Surplus': '🟢 Surplus',
        'Balanced': '🟡 Balanced',
    }
    return mapping.get(status, status)


def _status_color(status: str) -> str:
    """CSS color for each status."""
    return {
        'Shortage': '#ef4444',
        'Surplus': '#22c55e',
        'Balanced': '#eab308',
    }.get(status, '#6b7280')


def _friendly_target_name(target: str) -> str:
    """Human-readable label for internal target names."""
    return {
        'daily_withdrawals': 'Withdrawals',
        'daily_deposits': 'Deposits',
        'net_cash': 'Cash Balance',
        'cash_requirement': 'Cash Shortfall',
    }.get(target, target)


# ---- Dashboard ----

def run_streamlit_dashboard():
    """Launch the redesigned Streamlit dashboard for cash forecasting."""
    import streamlit as st

    # ── Page config ──
    st.set_page_config(
        page_title="Cash Forecasting Dashboard",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS for premium look ──
    st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* KPI card styling */
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.12);
        margin-bottom: 8px;
    }
    .kpi-card .label {
        font-size: 13px;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-card .value {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .kpi-card .sub {
        font-size: 12px;
        color: #94a3b8;
    }

    /* Status pill */
    .status-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
    }
    .status-shortage { background: #fecaca; color: #991b1b; }
    .status-surplus  { background: #bbf7d0; color: #166534; }
    .status-balanced { background: #fef08a; color: #854d0e; }

    /* Alert cards */
    .alert-card {
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        font-size: 14px;
    }
    .alert-shortage {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border-left: 4px solid #ef4444;
    }
    .alert-surplus {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-left: 4px solid #22c55e;
    }
    .alert-balanced {
        background: linear-gradient(135deg, #fefce8, #fef9c3);
        border-left: 4px solid #eab308;
    }

    /* Model health bar */
    .health-bar {
        background: #e2e8f0;
        border-radius: 8px;
        height: 24px;
        overflow: hidden;
        margin: 8px 0;
    }
    .health-bar-fill {
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 600;
        color: white;
    }

    /* Section headers */
    .section-header {
        font-size: 14px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown("# 🏦 Cash Forecasting Dashboard")
    st.markdown(
        "<p style='color:#64748b; margin-top:-10px; font-size:15px;'>"
        "Predict branch cash needs &bull; Spot shortages early &bull; Optimize cash allocation"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Load artifacts ──
    with st.spinner("Loading models and data..."):
        artifacts = load_phase5_models_and_pipeline()

    # ── Sidebar controls ──
    st.sidebar.markdown("## ⚙️ Forecast Settings")
    st.sidebar.markdown("---")

    # Date selection — force a 7-day forecast
    test_dates = artifacts['test_df']['start_date']
    data_min_date = test_dates.min().date() if len(test_dates) > 0 else datetime.now().date()
    data_max_date = test_dates.max().date() if len(test_dates) > 0 else datetime.now().date()
    # The max start date should be at most 6 days before data_max_date to allow 7 full days
    max_start_date = data_max_date - timedelta(days=6)
    if max_start_date < data_min_date:
        max_start_date = data_min_date
    default_start = max_start_date

    st.sidebar.markdown("**📅 7-Day Forecast Period**")
    st.sidebar.caption("Pick the start date for your 7-day forecast")
    forecast_start = st.sidebar.date_input(
        "Start Date", default_start,
        min_value=data_min_date, max_value=data_max_date,
    )
    
    # Calculate 7-day range
    forecast_end = forecast_start + timedelta(days=6)
    if forecast_end > data_max_date:
        forecast_end = data_max_date
        
    st.sidebar.caption(f"Forecasting to: **{forecast_end}**")

    forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='D')

    # Branch selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🏢 Branch Search**")
    st.sidebar.caption("Search and select a specific branch")
    all_branches = sorted(artifacts['train_df']['tran_br_code'].unique())
    
    selected_branch = st.sidebar.selectbox(
        "Search Branch", all_branches,
        index=0,
        label_visibility="collapsed",
    )
    selected_branches = [selected_branch] if selected_branch else []

    if not selected_branches:
        st.sidebar.warning("Please select at least one branch")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Data covers **{data_min_date}** to **{data_max_date}** "
        f"({len(test_dates)} test records across {len(all_branches)} branches)"
    )

    # ── Generate forecasts ──
    with st.spinner("Generating forecasts..."):
        forecast_df = generate_daily_forecasts(
            artifacts, forecast_dates.tolist(), selected_branches
        )

    if len(forecast_df) == 0:
        st.error("❌ No forecasts could be generated. Please check your date range and branch selections.")
        st.stop()

    # Generate guidance
    guidance_df = generate_replenishment_guidance(forecast_df)

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview",
        "🏢 Branch Analysis",
        "📈 Forecast Trends",
        "🎯 Model Health",
        "📥 Export Data",
    ])

    # ==================================================================
    #  TAB 1: OVERVIEW
    # ==================================================================
    with tab1:
        # ---- Big KPI Cards ----
        total_deposits = forecast_df['predicted_deposits'].sum()
        total_withdrawals = forecast_df['predicted_withdrawals'].sum()
        total_net = forecast_df['predicted_net_cash'].sum()
        total_cr = forecast_df['predicted_cash_requirement'].sum()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kpi-card">
            <div class="label">Expected Deposits</div>
            <div class="value" style="color:#4ade80">{_fmt_pkr(total_deposits)}</div>
            <div class="sub">Total money coming in</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card">
            <div class="value" style="color:#f87171">{_fmt_pkr(total_withdrawals)}</div>
            <div class="label">Expected Withdrawals</div>
            <div class="sub">Total money going out</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            net_color = "#4ade80" if total_net >= 0 else "#f87171"
            st.markdown(f"""
            <div class="kpi-card">
            <div class="label">Cash Balance</div>
            <div class="value" style="color:{net_color}">{_fmt_pkr(total_net)}</div>
            <div class="sub">Deposits minus withdrawals</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            cr_color = "#f87171" if total_cr > 0 else "#4ade80"
            st.markdown(f"""
            <div class="kpi-card">
            <div class="label">Cash Shortfall</div>
            <div class="value" style="color:{cr_color}">{_fmt_pkr(total_cr)}</div>
            <div class="sub">Extra cash needed to cover withdrawals</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Risk Summary ----
        st.markdown('<div class="section-header">Risk Summary</div>', unsafe_allow_html=True)

        n_shortage = int((forecast_df['status'] == 'Shortage').sum())
        n_surplus = int((forecast_df['status'] == 'Surplus').sum())
        n_balanced = int((forecast_df['status'] == 'Balanced').sum())
        total_fc = len(forecast_df)

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            pct = n_shortage / total_fc * 100 if total_fc > 0 else 0
            st.markdown(f"""
            <div class="alert-card alert-shortage">
            <strong>🔴 {n_shortage} Shortage Days</strong> ({pct:.0f}% of forecasts)<br>
            <span style="font-size:12px; color:#991b1b">
            Branches where withdrawals exceed deposits — may need cash replenishment
            </span>
            </div>""", unsafe_allow_html=True)
        with rc2:
            pct = n_balanced / total_fc * 100 if total_fc > 0 else 0
            st.markdown(f"""
            <div class="alert-card alert-balanced">
            <strong>🟡 {n_balanced} Balanced Days</strong> ({pct:.0f}% of forecasts)<br>
            <span style="font-size:12px; color:#854d0e">
            Cash levels are adequate — no immediate action needed
            </span>
            </div>""", unsafe_allow_html=True)
        with rc3:
            pct = n_surplus / total_fc * 100 if total_fc > 0 else 0
            st.markdown(f"""
            <div class="alert-card alert-surplus">
            <strong>🟢 {n_surplus} Surplus Days</strong> ({pct:.0f}% of forecasts)<br>
            <span style="font-size:12px; color:#166534">
            Excess idle cash — consider reallocating to branches with shortages
            </span>
            </div>""", unsafe_allow_html=True)

        # ---- Top Alerts (branches needing attention) ----
        if len(guidance_df) > 0:
            st.markdown('<div class="section-header">Branches Needing Attention</div>',
                    unsafe_allow_html=True)
            shortage_branches = guidance_df[
            guidance_df['primary_status'] == 'Shortage'
            ].sort_values('total_predicted_cash_requirement', ascending=False)

            if len(shortage_branches) > 0:
                for _, row in shortage_branches.head(5).iterrows():
                    st.markdown(f"""
                    <div class="alert-card alert-shortage">
                        <strong>Branch {int(row['branch_code'])}</strong> — 
                        needs <strong>{_fmt_pkr(row['total_predicted_cash_requirement'])}</strong> 
                        over {int(row['forecast_days'])} days 
                        ({int(row['shortage_days'])} shortage days)<br>
                        <span style="font-size:12px">💡 {row['recommendation']}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("✅ No branches are predicted to have cash shortages in this period!")

    # ==================================================================
    #  TAB 2: BRANCH ANALYSIS
    # ==================================================================
    with tab2:
        st.markdown("### 🏢 Branch-by-Branch Comparison")
        st.caption("Compare all selected branches side by side. Click a column header to sort.")

        if len(guidance_df) > 0:
            display_guidance = guidance_df.copy()
            display_guidance['Status'] = display_guidance['primary_status'].apply(_status_badge)
            display_guidance['Avg Daily Deposits'] = display_guidance['total_predicted_inflow'] / display_guidance['forecast_days']
            display_guidance['Avg Daily Withdrawals'] = display_guidance['total_predicted_outflow'] / display_guidance['forecast_days']

            # Build display table
            table_data = []
            for _, row in display_guidance.iterrows():
                table_data.append({
                    'Branch': int(row['branch_code']),
                    'Status': row['Status'],
                    'Days': int(row['forecast_days']),
                    'Avg Deposits/Day': _fmt_pkr(row['Avg Daily Deposits']),
                    'Avg Withdrawals/Day': _fmt_pkr(row['Avg Daily Withdrawals']),
                    'Total Shortfall': _fmt_pkr(row['total_predicted_cash_requirement']),
                    'Shortage Days': int(row['shortage_days']),
                    'Recommendation': row['recommendation'],
                })
            st.dataframe(
                pd.DataFrame(table_data),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")

        # Per-branch detail cards
        st.markdown("### 📋 Detailed Branch Cards")
        st.caption("Expand a branch to see its daily forecast breakdown")

        for branch in selected_branches:
            branch_data = forecast_df[forecast_df['branch_code'] == branch]
            if len(branch_data) == 0:
                continue

            avg_dep = branch_data['predicted_deposits'].mean()
            avg_wdr = branch_data['predicted_withdrawals'].mean()
            total_cr_br = branch_data['predicted_cash_requirement'].sum()
            mode_status = branch_data['status'].mode()[0]
            badge = _status_badge(mode_status)

            with st.expander(f"Branch {branch}  —  {badge}  |  Avg Deposits: {_fmt_pkr(avg_dep)}/day  |  Shortfall: {_fmt_pkr(total_cr_br)}", expanded=True):
                # Daily breakdown table
                daily_table = branch_data[['forecast_date', 'predicted_deposits',
                                           'predicted_withdrawals', 'predicted_net_cash',
                                           'predicted_cash_requirement', 'status', 'explanation']].copy()
                daily_table.columns = ['Date', 'Deposits', 'Withdrawals', 'Balance', 'Shortfall', 'Status', 'Explanation']
                daily_table['Date'] = pd.to_datetime(daily_table['Date']).dt.strftime('%A, %b %d')
                daily_table['Deposits'] = daily_table['Deposits'].apply(_fmt_pkr)
                daily_table['Withdrawals'] = daily_table['Withdrawals'].apply(_fmt_pkr)
                daily_table['Balance'] = daily_table['Balance'].apply(_fmt_pkr)
                daily_table['Shortfall'] = daily_table['Shortfall'].apply(_fmt_pkr)
                daily_table['Status'] = daily_table['Status'].apply(_status_badge)

                st.dataframe(daily_table, use_container_width=True, hide_index=True)

    # ==================================================================
    #  TAB 3: FORECAST TRENDS
    # ==================================================================
    with tab3:
        st.markdown("### 📈 Forecast Trend Charts")
        st.caption(
            "These charts show the model's predicted cash flows for each branch. "
            "A rising line means the model expects increasing activity."
        )

        # Use Streamlit native charts for interactivity
        chart_branches = selected_branches[:5]  # Limit for readability

        # --- Deposits chart ---
        st.markdown('<div class="section-header">Expected Deposits Over Time</div>',
                unsafe_allow_html=True)
        st.caption("💡 **Deposits** = money customers put into their accounts")

        dep_chart_data = pd.DataFrame()
        for branch in chart_branches:
            bd = forecast_df[forecast_df['branch_code'] == branch].copy()
            bd = bd.set_index('forecast_date')
            dep_chart_data[f'Branch {branch}'] = bd['predicted_deposits'] / 1_000_000
        if not dep_chart_data.empty:
            dep_chart_data.index.name = 'Date'
            st.line_chart(dep_chart_data, use_container_width=True)
            st.caption("Values shown in millions (PKR)")

        # --- Withdrawals chart ---
        st.markdown('<div class="section-header">Expected Withdrawals Over Time</div>',
                unsafe_allow_html=True)
        st.caption("💡 **Withdrawals** = money customers take out of their accounts")

        wdr_chart_data = pd.DataFrame()
        for branch in chart_branches:
            bd = forecast_df[forecast_df['branch_code'] == branch].copy()
            bd = bd.set_index('forecast_date')
            wdr_chart_data[f'Branch {branch}'] = bd['predicted_withdrawals'] / 1_000_000
        if not wdr_chart_data.empty:
            wdr_chart_data.index.name = 'Date'
            st.line_chart(wdr_chart_data, use_container_width=True)
            st.caption("Values shown in millions (PKR)")

        # --- Cash Shortfall chart ---
        st.markdown('<div class="section-header">Cash Shortfall (Extra Cash Needed)</div>',
                unsafe_allow_html=True)
        st.caption(
            "💡 **Cash Shortfall** = how much extra cash a branch needs when "
            "withdrawals exceed deposits. Zero means the branch is self-sufficient."
        )

        cr_chart_data = pd.DataFrame()
        for branch in chart_branches:
            bd = forecast_df[forecast_df['branch_code'] == branch].copy()
            bd = bd.set_index('forecast_date')
            cr_chart_data[f'Branch {branch}'] = bd['predicted_cash_requirement'] / 1_000_000
        if not cr_chart_data.empty:
            cr_chart_data.index.name = 'Date'
            st.bar_chart(cr_chart_data, use_container_width=True)
            st.caption("Values shown in millions (PKR)")

    # ==================================================================
    #  TAB 4: MODEL HEALTH
    # ==================================================================
    with tab4:
        st.markdown("### 🎯 How Accurate Is This Forecast?")
        st.caption(
            "We compare our ML model against a simple baseline (7-day average). "
            "A higher improvement % means the model is significantly smarter than just using last week's average."
        )

        test_metrics = artifacts.get('test_metrics', {})

        if test_metrics:
            for target in TARGETS:
                if target not in test_metrics:
                    continue
                m = test_metrics[target]
                friendly = _friendly_target_name(target)
                model_mae = m.get('test_mae', 0)
                baseline_mae = m.get('baseline_mae', 0)
                improvement = m.get('improvement_vs_baseline_pct', 0)
                model_name = m.get('model_name', 'ML Model')

                st.markdown(f"#### {friendly}")

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric(
                        "Model's Average Error",
                        _fmt_pkr(model_mae),
                        help="On average, the model's prediction is off by this much"
                    )
                with mc2:
                    st.metric(
                        "Simple Average Error",
                        _fmt_pkr(baseline_mae) if baseline_mae else "N/A",
                        help="Error if we just used last week's average instead"
                    )
                with mc3:
                    delta_str = f"{improvement:+.1f}%" if improvement else "N/A"
                    st.metric(
                        "Improvement",
                        delta_str,
                        help="How much better the model is compared to a simple 7-day average"
                    )

            # Visual health bar
            if baseline_mae and baseline_mae > 0:
                pct_fill = max(0, min(100, improvement))
                bar_color = "#22c55e" if pct_fill > 10 else "#eab308" if pct_fill > 0 else "#ef4444"
                st.markdown(f"""
                <div class="health-bar">
                    <div class="health-bar-fill" style="width:{max(pct_fill, 5)}%; background:{bar_color}">
                        {pct_fill:.1f}% better
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(
                f"<p style='font-size:12px; color:#94a3b8'>"
                f"Model: {model_name} &bull; Tested on {m.get('test_num_samples', '?')} records"
                f"</p>",
                unsafe_allow_html=True,
            )
            st.markdown("---")

            # Overall summary
            avg_improvement = np.mean([
                test_metrics[t].get('improvement_vs_baseline_pct', 0)
                for t in TARGETS if t in test_metrics
            ])
            if avg_improvement > 10:
                st.success(
                    f"✅ **Overall: The model is {avg_improvement:.0f}% more accurate** than a "
                    f"simple 7-day average. Your forecasts are data-driven and reliable."
                )
            elif avg_improvement > 0:
                st.info(
                    f"ℹ️ **Overall: The model is {avg_improvement:.0f}% more accurate** than a "
                    f"simple average. Decent, but predictions should be used as guidance."
                )
            else:
                st.warning(
                    "⚠️ The model is performing similarly to a simple average. "
                    "Treat forecasts as rough estimates."
                )
        else:
            st.info("Model performance metrics are not available yet. Run the forecast pipeline first.")

    # ==================================================================
    #  TAB 5: EXPORT DATA
    # ==================================================================
    with tab5:
        st.markdown("### 📥 Download Your Data")
        st.caption("Export the forecast data as CSV files for further analysis or reporting.")

        # Preview
        st.markdown("**Forecast Preview**")
        preview_df = forecast_df.copy()
        preview_df['forecast_date'] = preview_df['forecast_date'].dt.strftime('%Y-%m-%d')
        preview_cols = {
            'branch_code': 'Branch',
            'forecast_date': 'Date',
            'predicted_deposits': 'Expected Deposits (PKR)',
            'predicted_withdrawals': 'Expected Withdrawals (PKR)',
            'predicted_net_cash': 'Cash Balance (PKR)',
            'predicted_cash_requirement': 'Cash Shortfall (PKR)',
            'status': 'Status',
        }
        preview_df = preview_df[list(preview_cols.keys())].rename(columns=preview_cols)
        st.dataframe(preview_df.head(20), use_container_width=True, hide_index=True)

        if len(preview_df) > 20:
            st.caption(f"Showing first 20 of {len(preview_df)} rows")

        dc1, dc2 = st.columns(2)
        with dc1:
            csv = forecast_df.to_csv(index=False)
            st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f'cash_forecast_{forecast_start}_{forecast_end}.csv',
            mime='text/csv',
            use_container_width=True,
            )
        with dc2:
            if len(guidance_df) > 0:
                guidance_csv = guidance_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Branch Guidance CSV",
                    data=guidance_csv,
                    file_name='branch_guidance.csv',
                    mime='text/csv',
                    use_container_width=True,
                )

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#94a3b8; font-size:12px;'>"
        "🏦 Bank Branch Cash Forecasting System &bull; Phase 6 &bull; "
        "Decision Support Only — not financial advice"
        "</p>",
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point for Phase 6."""
    import argparse
    parser = argparse.ArgumentParser(description='Phase 6: Cash Forecasting Solution')
    parser.add_argument('--mode', choices=['forecast', 'dashboard'], default='forecast')
    parser.add_argument('--start-date', type=str, help='Start date for forecast (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for forecast (YYYY-MM-DD)')
    parser.add_argument('--branches', type=int, nargs='+', help='Branch codes to forecast')
    
    try:
        args, _ = parser.parse_known_args()
    except SystemExit:
        args = argparse.Namespace(mode='forecast', start_date=None, end_date=None, branches=None)
    
    if args.mode == 'dashboard':
        print("Launching dashboard...")
        print("Run this command: streamlit run Bank_Project/phase6_cash_forecasting_solution.py")
        run_streamlit_dashboard()
        return
    
    # Forecast mode
    # Load artifacts
    artifacts = load_phase5_models_and_pipeline()
    
    # Determine forecast dates — default to last 7 days of test set
    if args.start_date and args.end_date:
        forecast_dates = pd.date_range(start=args.start_date, end=args.end_date, freq='D')
    else:
        last_date = artifacts['test_df']['start_date'].max() if len(artifacts['test_df']) > 0 else datetime.now()
        forecast_dates = pd.date_range(start=last_date - timedelta(days=6),
                                   end=last_date, freq='D')
    
    # Generate forecasts
    forecast_df = generate_daily_forecasts(artifacts, forecast_dates.tolist(), 
                                        branches=args.branches)
    
    # Generate guidance
    guidance_df = generate_replenishment_guidance(forecast_df) if len(forecast_df) > 0 else pd.DataFrame()
    
    # Validate model performance
    validation = validate_model_performance(artifacts, forecast_df) if len(forecast_df) > 0 else {}
    
    # Run assertions
    assertions_ok = run_assertions(artifacts, forecast_df, validation) if len(forecast_df) > 0 else False
    
    # Save outputs
    save_phase6_outputs(forecast_df, guidance_df, validation, assertions_ok)
    
    # Print summary
    print_summary(forecast_df, guidance_df, validation, assertions_ok)
    
    return {
        'forecasts': forecast_df,
        'guidance': guidance_df,
        'validation': validation,
        'assertions_passed': assertions_ok
    }


if __name__ == "__main__":
    # Auto-detect whether we're running under Streamlit
    if 'streamlit' in sys.modules:
        run_streamlit_dashboard()
    else:
        main()