"""
==============================================================================
BANK BRANCH CASH FORECASTING SYSTEM
Phase 4: Model Development & Evaluation
==============================================================================

This script implements the model development pipeline:

1. Load model-ready data and review baseline results
2. Prepare feature matrix (exclude current-day predictors)
3. Train multiple models: Linear Regression, Random Forest, XGBoost, LightGBM
4. Evaluate against best baseline (7-day rolling average)
5. Save model artifacts, evaluation metrics, and predictions

Best Baseline (from phase3_corrected_feature_engineering.py):
  - Withdrawals:    7d rolling avg (MAE=10,528,539, RMSE=13,881,532, WAPE=0.2948)
  - Deposits:       7d rolling avg (MAE=10,289,307, RMSE=13,366,581, WAPE=0.2930)
  - Net Cash:       7d rolling avg (MAE=7,770,228, RMSE=10,229,240, WAPE=1.0324)
  - Cash Requirement: 7d rolling avg (MAE=4,579,655, RMSE=6,422,798, WAPE=1.1277)

ML models must outperform these baselines.

Author: Muhammad Usman
Status: Phase 4 Implementation
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
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# Scikit-learn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone

# XGBoost & LightGBM
import xgboost as xgb
import lightgbm as lgb

# Prophet
from prophet import Prophet

# Visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- File Paths ---
INPUT_PATH = "../Bank DataSet/model_ready_data.csv"
OUTPUT_DIR = "../Bank DataSet"
BASELINE_METRICS_PATH = os.path.join(OUTPUT_DIR, "baseline_metrics.json")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "model_plots")

# --- Output Files ---
MODEL_METRICS_PATH = os.path.join(OUTPUT_DIR, "model_metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "feature_importance.json")
MODEL_REPORT_PATH = os.path.join(OUTPUT_DIR, "model_report.json")

# --- Target Variables ---
TARGETS = ['daily_withdrawals', 'daily_deposits']

# --- Models to Train ---
MODELS = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1, max_iter=10000),
    'RandomForest': RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        n_jobs=-1, random_state=42, verbose=0
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        learning_rate=0.1, random_state=42
    ),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, n_jobs=-1
    ),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1, n_jobs=-1
    ),
    'Tuned XGBoost': RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
        param_distributions={
            'n_estimators': [100, 300, 500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.7, 0.9]
        },
        n_iter=10, cv=3, scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1, verbose=0
    ),
    'Tuned LightGBM': RandomizedSearchCV(
        lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1),
        param_distributions={
            'n_estimators': [100, 300, 500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.7, 0.9]
        },
        n_iter=10, cv=3, scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1, verbose=0
    ),
    'Prophet': 'Prophet'
}

# --- Columns to Exclude from Predictors ---
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

# --- Evaluation Metrics ---
METRICS_NAMES = ['mae', 'rmse', 'wape', 'underforecasting_rate']


# =============================================================================
# MODULE 1: BASELINE REVIEW
# =============================================================================

def review_baselines(baseline_path: str = BASELINE_METRICS_PATH) -> Dict:
    """
    Load and review baseline metrics to identify best baseline per target.

    Parameters:
        baseline_path (str): Path to baseline_metrics.json.

    Returns:
        dict: Best baseline per target with metrics.
    """
    print("=" * 70)
    print("MODULE 1: BASELINE REVIEW")
    print("=" * 70)

    with open(baseline_path, 'r') as f:
        baselines = json.load(f)

    best_baselines = {}
    print(f"\n{'Target':<25} {'Best Baseline':<25} {'MAE':<15} {'RMSE':<15} {'WAPE':<10}")
    print(f"{'-'*90}")

    for target in TARGETS:
        if target not in baselines:
            continue

        target_baselines = baselines[target]
        best_bl = None
        best_mae = float('inf')

        for bl_name, bl_metrics in target_baselines.items():
            mae = bl_metrics.get('mae')
            if mae is not None and mae < best_mae:
                best_mae = mae
                best_bl = bl_name

        if best_bl:
            best_info = target_baselines[best_bl]
            bl_display = best_bl.replace('_baseline', '').replace('_', ' ')
            print(f"  {target:<25} {bl_display:<25} "
                  f"{best_info['mae']:<15,.0f} {best_info['rmse']:<15,.0f} "
                  f"{best_info['wape']:<10.4f}")

            best_baselines[target] = {
                'baseline_name': best_bl,
                'baseline_display': bl_display,
                'mae': best_info['mae'],
                'rmse': best_info['rmse'],
                'wape': best_info['wape'],
                'underforecasting_rate': best_info.get('underforecasting_rate'),
                'num_comparisons': best_info.get('num_comparisons')
            }

    print(f"\n{'='*90}")
    print("  Best baseline across ALL targets: 7-day Rolling Average (roll7_avg_baseline)")
    print("  ML models must outperform these MAE/RMSE/WAPE values.")
    print(f"{'='*90}")

    return best_baselines


# =============================================================================
# MODULE 2: LOAD & PREPARE FEATURE MATRIX
# =============================================================================

def load_and_prepare_data(file_path: str = INPUT_PATH) -> Dict:
    """
    Load model-ready data and prepare feature matrix.

    Excludes:
        - Current-day predictors (targets + transaction info)
        - Baseline columns (not used as features for ML)
        - start_date (kept separately for evaluation/plotting)

    Parameters:
        file_path (str): Path to model-ready CSV.

    Returns:
        dict: Contains train/val/test DataFrames with features, targets, dates.
    """
    print("\n" + "=" * 70)
    print("MODULE 2: LOAD & PREPARE FEATURE MATRIX")
    print("=" * 70)

    # Load full data
    df = pd.read_csv(file_path, parse_dates=['start_date'])
    print(f"✓ Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")

    # Identify feature columns (exclude current-day, baselines, identifiers)
    exclude_cols = set(CURRENT_DAY_COLS + BASELINE_COLS + ['start_date'])
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"\n  Feature columns: {len(feature_cols)}")
    print(f"  Excluded current-day: {len(CURRENT_DAY_COLS)}")
    print(f"  Excluded baselines: {len(BASELINE_COLS)}")

    # Create splits
    train_mask = df['start_date'] < pd.Timestamp('2025-06-01')
    val_mask = (df['start_date'] >= pd.Timestamp('2025-06-01')) & \
               (df['start_date'] < pd.Timestamp('2025-10-01'))
    test_mask = df['start_date'] >= pd.Timestamp('2025-10-01')

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    print(f"\n  Train: {len(train_df):>6} rows ({train_df['start_date'].min().date()} → {train_df['start_date'].max().date()})")
    print(f"  Val:   {len(val_df):>6} rows ({val_df['start_date'].min().date()} → {val_df['start_date'].max().date()})")
    print(f"  Test:  {len(test_df):>6} rows ({test_df['start_date'].min().date()} → {test_df['start_date'].max().date()})")

    # Prepare feature matrices
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    # Handle NaN in features (fill with median from training)
    for col in X_train.columns:
        if X_train[col].isna().any():
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_val[col] = X_val[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

    # Prepare target matrices
    y_train = {}
    y_val = {}
    y_test = {}
    for target in TARGETS:
        y_train[target] = train_df[target].values
        y_val[target] = val_df[target].values
        y_test[target] = test_df[target].values

    # Prepare baseline predictions for comparison
    baseline_train = {}
    baseline_val = {}
    baseline_test = {}
    for target in TARGETS:
        bl_col = f'{target}_roll7_avg_baseline'
        if bl_col in train_df.columns:
            baseline_train[target] = train_df[bl_col].values
            baseline_val[target] = val_df[bl_col].values
            baseline_test[target] = test_df[bl_col].values

    result = {
        'feature_cols': feature_cols,
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'baseline_train': baseline_train,
        'baseline_val': baseline_val,
        'baseline_test': baseline_test,
        'dates_train': train_df['start_date'].values,
        'dates_val': val_df['start_date'].values,
        'dates_test': test_df['start_date'].values,
        'branches_train': train_df['tran_br_code'].values,
        'branches_val': val_df['tran_br_code'].values,
        'branches_test': test_df['tran_br_code'].values,
        'train_df': train_df, 'val_df': val_df, 'test_df': test_df
    }

    print(f"\n✓ Feature matrix prepared: {X_train.shape[1]} features")
    print(f"  Missing values handled: filled with training median")

    return result


# =============================================================================
# MODULE 3: TRAIN MODELS
# =============================================================================

def train_models(data: Dict, models: Dict = None) -> Dict:
    """
    Train all models for each target variable.

    Parameters:
        data (dict): Prepared data from load_and_prepare_data().
        models (dict): Dictionary of model name → model instance.

    Returns:
        dict: Trained models organized by target and model name.
    """
    if models is None:
        models = MODELS

    print("\n" + "=" * 70)
    print("MODULE 3: MODEL TRAINING")
    print("=" * 70)

    X_train = data['X_train']
    X_val = data['X_val']

    trained_models = {}
    total_combos = len(TARGETS) * len(models)
    combo_count = 0

    for target in TARGETS:
        trained_models[target] = {}
        y_train = data['y_train'][target]
        y_val = data['y_val'][target]

        # Remove NaN rows for this target
        train_mask = ~np.isnan(y_train)
        val_mask = ~np.isnan(y_val)

        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask]
        X_val_clean = X_val[val_mask]
        y_val_clean = y_val[val_mask]

        print(f"\n  --- Target: {target} ---")
        print(f"  Train samples: {len(y_train_clean)}, Val samples: {len(y_val_clean)}")

        for model_name, model in models.items():
            combo_count += 1
            print(f"  [{combo_count}/{total_combos}] Training {model_name}...", end=' ')

            try:
                if model_name == 'Prophet':
                    df_train_prophet = pd.DataFrame({'ds': data['dates_train'][train_mask], 'y': y_train_clean})
                    df_val_prophet = pd.DataFrame({'ds': data['dates_val'][val_mask]})
                    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                    m.fit(df_train_prophet)
                    y_pred_train = m.predict(df_train_prophet)['yhat'].values
                    y_pred_val = m.predict(df_val_prophet)['yhat'].values
                    model_to_save = m
                else:
                    # Train
                    model_copy = clone(model)
                    model_copy.fit(X_train_clean, y_train_clean)

                    # Predict
                    y_pred_train = model_copy.predict(X_train_clean)
                    y_pred_val = model_copy.predict(X_val_clean)
                    model_to_save = model_copy.best_estimator_ if hasattr(model_copy, 'best_estimator_') else model_copy

                trained_models[target][model_name] = {
                    'model': model_to_save,
                    'val_mae': mean_absolute_error(y_val_clean, y_pred_val),
                    'val_rmse': np.sqrt(mean_squared_error(y_val_clean, y_pred_val)),
                    'y_pred_train': y_pred_train,
                    'y_pred_val': y_pred_val,
                    'y_true_train': y_train_clean,
                    'y_true_val': y_val_clean
                }
                print(f"✓ (val MAE: {trained_models[target][model_name]['val_mae']:,.0f})")

            except Exception as e:
                print(f"✗ FAILED: {str(e)[:80]}")
                trained_models[target][model_name] = None

    print(f"\n✓ Training complete: {combo_count} model-target combinations attempted")

    return trained_models


# =============================================================================
# MODULE 4: EVALUATE MODELS
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute comprehensive evaluation metrics.

    Parameters:
        y_true (np.ndarray): Actual values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        dict: Metrics including MAE, RMSE, WAPE, underforecasting_rate.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return {'mae': None, 'rmse': None, 'wape': None, 'underforecasting_rate': None,
                'num_comparisons': 0}

    act = y_true[mask]
    pre = y_pred[mask]
    errors = act - pre

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    sum_abs_actual = np.sum(np.abs(act))
    wape = float(np.sum(np.abs(errors)) / sum_abs_actual) if sum_abs_actual > 0 else None
    underforecasting_rate = float(np.sum(pre < act) / len(act))

    return {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'wape': round(wape, 4) if wape is not None else None,
        'underforecasting_rate': round(underforecasting_rate, 4),
        'num_comparisons': int(mask.sum())
    }


def evaluate_models(
    data: Dict,
    trained_models: Dict,
    best_baselines: Dict
) -> Dict:
    """
    Evaluate all trained models and compare against baselines.

    Parameters:
        data (dict): Prepared data.
        trained_models (dict): Trained models from train_models().
        best_baselines (dict): Best baseline per target.

    Returns:
        dict: Comprehensive evaluation results.
    """
    print("\n" + "=" * 70)
    print("MODULE 4: MODEL EVALUATION")
    print("=" * 70)

    results = {}

    for target in TARGETS:
        print(f"\n  {'='*60}")
        print(f"  TARGET: {target}")
        print(f"  {'='*60}")

        results[target] = {}
        y_val = data['y_val'][target]

        # Baseline metrics
        bl_info = best_baselines.get(target, {})
        results[target]['baseline'] = {
            'name': bl_info.get('baseline_display', 'N/A'),
            'mae': bl_info.get('mae'),
            'rmse': bl_info.get('rmse'),
            'wape': bl_info.get('wape'),
            'underforecasting_rate': bl_info.get('underforecasting_rate')
        }

        print(f"\n  {'Model':<25} {'MAE':<15} {'RMSE':<15} {'WAPE':<10} {'Underforecast':<15} {'vs Baseline':<15}")
        print(f"  {'-'*95}")

        # Print baseline first
        print(f"  {'BASELINE (7d roll avg)':<25} "
              f"{bl_info.get('mae', 0):<15,.0f} "
              f"{bl_info.get('rmse', 0):<15,.0f} "
              f"{bl_info.get('wape', 0):<10.4f} "
              f"{bl_info.get('underforecasting_rate', 0):<15.4f} "
              f"{'---':<15}")

        # Evaluate each model
        for model_name, model_info in trained_models.get(target, {}).items():
            if model_info is None:
                results[target][model_name] = {'error': 'Training failed'}
                print(f"  {model_name:<25} {'TRAINING FAILED':<60}")
                continue

            y_pred_val = model_info['y_pred_val']
            y_true_val = model_info['y_true_val']

            metrics = compute_metrics(y_true_val, y_pred_val)
            results[target][model_name] = metrics

            # Compare to baseline
            baseline_mae = bl_info.get('mae')
            if baseline_mae and metrics['mae']:
                improvement = (baseline_mae - metrics['mae']) / baseline_mae * 100
                vs_baseline = f"{improvement:+.1f}%"
            else:
                vs_baseline = "N/A"

            print(f"  {model_name:<25} "
                  f"{metrics['mae']:<15,.0f} "
                  f"{metrics['rmse']:<15,.0f} "
                  f"{metrics['wape']:<10.4f} "
                  f"{metrics['underforecasting_rate']:<15.4f} "
                  f"{vs_baseline:<15}")

    return results


# =============================================================================
# MODULE 5: FEATURE IMPORTANCE
# =============================================================================

def analyze_feature_importance(
    data: Dict,
    trained_models: Dict,
    top_n: int = 20
) -> Dict:
    """
    Extract and save feature importance from tree-based models.

    Parameters:
        data (dict): Prepared data.
        trained_models (dict): Trained models.
        top_n (int): Number of top features to report.

    Returns:
        dict: Feature importance for each target and model.
    """
    print("\n" + "=" * 70)
    print("MODULE 5: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    feature_cols = data['feature_cols']
    importance_results = {}

    for target in TARGETS:
        importance_results[target] = {}

        for model_name in ['RandomForest', 'XGBoost', 'LightGBM', 'GradientBoosting', 'Tuned XGBoost', 'Tuned LightGBM']:
            model_info = trained_models.get(target, {}).get(model_name)
            if model_info is None:
                continue

            model = model_info['model']

            # Extract feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                continue

            # Create importance DataFrame
            imp_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)

            importance_results[target][model_name] = {
                'top_features': imp_df.head(top_n).to_dict('records'),
                'all_features': imp_df.to_dict('records')
            }

            print(f"\n  Top 5 features for {target} — {model_name}:")
            for i, row in enumerate(imp_df.head(5).itertuples()):
                print(f"    {i+1}. {row.feature}: {row.importance:.4f}")

    return importance_results


# =============================================================================
# MODULE 6: SAVE OUTPUTS
# =============================================================================

def save_outputs(
    data: Dict,
    trained_models: Dict,
    evaluation_results: Dict,
    importance_results: Dict,
    best_baselines: Dict
) -> None:
    """
    Save all model outputs to disk.

    Parameters:
        data (dict): Prepared data.
        trained_models (dict): Trained models.
        evaluation_results (dict): Evaluation metrics.
        importance_results (dict): Feature importance.
        best_baselines (dict): Best baseline per target.
    """
    print("\n" + "=" * 70)
    print("MODULE 6: SAVING OUTPUTS")
    print("=" * 70)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- 6.1 Save trained models ---
    model_count = 0
    for target in TARGETS:
        for model_name, model_info in trained_models.get(target, {}).items():
            if model_info is None:
                continue
            model_path = os.path.join(MODELS_DIR, f"{target}_{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model_info['model'], f)
            model_count += 1
    print(f"✓ Saved {model_count} trained models to {MODELS_DIR}/")

    # --- 6.2 Save predictions ---
    for split_name, split_key in [('train', 'train'), ('val', 'val'), ('test', 'test')]:
        pred_records = []
        for target in TARGETS:
            for model_name, model_info in trained_models.get(target, {}).items():
                if model_info is None:
                    continue
                y_pred_key = f'y_pred_{split_name}'
                y_true_key = f'y_true_{split_name}'
                if y_pred_key in model_info:
                    for i in range(len(model_info[y_pred_key])):
                        pred_records.append({
                            'target': target,
                            'model': model_name,
                            'split': split_name,
                            'actual': model_info[y_true_key][i],
                            'predicted': model_info[y_pred_key][i]
                        })

        if pred_records:
            pred_df = pd.DataFrame(pred_records)
            pred_path = os.path.join(PREDICTIONS_DIR, f"predictions_{split_name}.csv")
            pred_df.to_csv(pred_path, index=False)
            print(f"✓ Saved {split_name} predictions: {pred_path}")

    # --- 6.3 Save evaluation metrics ---
    with open(MODEL_METRICS_PATH, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    print(f"✓ Saved model metrics: {MODEL_METRICS_PATH}")

    # --- 6.4 Save feature importance ---
    with open(FEATURE_IMPORTANCE_PATH, 'w') as f:
        json.dump(importance_results, f, indent=2, default=str)
    print(f"✓ Saved feature importance: {FEATURE_IMPORTANCE_PATH}")

    # --- 6.5 Save model report ---
    report = {
        'execution_timestamp': datetime.now().isoformat(),
        'best_baseline': '7-day Rolling Average (roll7_avg_baseline)',
        'baseline_metrics': best_baselines,
        'models_trained': list(MODELS.keys()),
        'targets': TARGETS,
        'feature_count': len(data['feature_cols']),
        'evaluation_summary': {}
    }

    for target in TARGETS:
        if target in evaluation_results:
            report['evaluation_summary'][target] = {}
            baseline = evaluation_results[target].get('baseline', {})
            report['evaluation_summary'][target]['baseline_mae'] = baseline.get('mae')

            best_model = None
            best_mae = float('inf')
            for model_name, metrics in evaluation_results[target].items():
                if model_name == 'baseline':
                    continue
                if isinstance(metrics, dict) and metrics.get('mae'):
                    if metrics['mae'] < best_mae:
                        best_mae = metrics['mae']
                        best_model = model_name

            report['evaluation_summary'][target]['best_model'] = best_model
            report['evaluation_summary'][target]['best_model_mae'] = best_mae
            if baseline.get('mae') and best_mae < float('inf'):
                improvement = (baseline['mae'] - best_mae) / baseline['mae'] * 100
                report['evaluation_summary'][target]['improvement_vs_baseline_pct'] = round(improvement, 2)

    with open(MODEL_REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"✓ Saved model report: {MODEL_REPORT_PATH}")


# =============================================================================
# MODULE 7: VISUALIZATIONS
# =============================================================================

def create_visualizations(
    data: Dict,
    trained_models: Dict,
    evaluation_results: Dict
) -> None:
    """
    Create model evaluation plots.

    Parameters:
        data (dict): Prepared data.
        trained_models (dict): Trained models.
        evaluation_results (dict): Evaluation metrics.
    """
    print("\n" + "=" * 70)
    print("MODULE 7: VISUALIZATIONS")
    print("=" * 70)

    # --- Plot 1: Model Comparison Bar Chart ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        target_results = evaluation_results.get(target, {})

        models_list = []
        mae_values = []
        colors = []

        # Add baseline
        baseline = target_results.get('baseline', {})
        if baseline.get('mae'):
            models_list.append('Baseline\n(7d roll avg)')
            mae_values.append(baseline['mae'])
            colors.append('#95a5a6')

        # Add models
        model_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6',
                        '#f39c12', '#1abc9c', '#34495e']
        for i, (model_name, metrics) in enumerate(target_results.items()):
            if model_name == 'baseline':
                continue
            if isinstance(metrics, dict) and metrics.get('mae'):
                models_list.append(model_name)
                mae_values.append(metrics['mae'])
                colors.append(model_colors[i % len(model_colors)])

        bars = ax.bar(range(len(mae_values)), mae_values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(mae_values)))
        ax.set_xticklabels(models_list, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title(f'{target} — Model Comparison (Validation)', fontsize=14, fontweight='bold')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, mae_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.01,
                    f'{val:,.0f}', ha='center', va='bottom', fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison_mae.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved model comparison plot")

    # --- Plot 2: Best Model Predictions vs Actual (Validation) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes = np.array(axes).flatten()  # Ensure it's a flat array even if 1D

    for idx, target in enumerate(TARGETS):
        ax = axes[idx]

        # Find best model for this target
        target_results = evaluation_results.get(target, {})
        best_model_name = None
        best_mae = float('inf')
        for model_name, metrics in target_results.items():
            if model_name == 'baseline':
                continue
            if isinstance(metrics, dict) and metrics.get('mae') and metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                best_model_name = model_name

        if best_model_name is None:
            ax.text(0.5, 0.5, 'No model available', ha='center', va='center', transform=ax.transAxes)
            continue

        # Get predictions
        model_info = trained_models.get(target, {}).get(best_model_name)
        if model_info is None:
            ax.text(0.5, 0.5, 'Model info not found', ha='center', va='center', transform=ax.transAxes)
            continue

        y_true = model_info['y_true_val']
        y_pred = model_info['y_pred_val']

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, c='#3498db', edgecolors='none')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Actual', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title(f'{target} — {best_model_name} (Val)', fontsize=14, fontweight='bold')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax.grid(alpha=0.3)

        # Add metrics text
        metrics = target_results.get(best_model_name, {})
        text = f"MAE: {metrics.get('mae', 0):,.0f}\nRMSE: {metrics.get('rmse', 0):,.0f}\nWAPE: {metrics.get('wape', 0):.4f}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'predictions_vs_actual.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved predictions vs actual plot")

    # --- Plot 3: Feature Importance Heatmap (Top 10) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, target in enumerate(TARGETS):
        ax = axes[idx]

        # Get XGBoost feature importance
        model_info = trained_models.get(target, {}).get('XGBoost')
        if model_info is None:
            ax.text(0.5, 0.5, 'XGBoost not available', ha='center', va='center', transform=ax.transAxes)
            continue

        model = model_info['model']
        if not hasattr(model, 'feature_importances_'):
            ax.text(0.5, 0.5, 'No feature importance', ha='center', va='center', transform=ax.transAxes)
            continue

        importances = model.feature_importances_
        feature_cols = data['feature_cols']
        imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
        imp_df = imp_df.sort_values('importance', ascending=True).tail(10)

        ax.barh(range(len(imp_df)), imp_df['importance'].values, color='#2ecc71', edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(imp_df)))
        ax.set_yticklabels(imp_df['feature'].values, fontsize=9)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'{target} — Top 10 Features (XGBoost)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved feature importance plot")

    print(f"\n  All plots saved to: {PLOTS_DIR}/")


# =============================================================================
# MODULE 8: SUMMARY
# =============================================================================

def print_summary(
    evaluation_results: Dict,
    best_baselines: Dict,
    data: Dict
) -> None:
    """
    Print a comprehensive summary of model development results.

    Parameters:
        evaluation_results (dict): Evaluation metrics.
        best_baselines (dict): Best baseline per target.
        data (dict): Prepared data.
    """
    print("\n" + "=" * 70)
    print(" " * 14 + "MODEL DEVELOPMENT SUMMARY")
    print("=" * 70)

    print(f"\n{'Data Dimensions':-<50}")
    print(f"  Training features:   {data['X_train'].shape[1]}")
    print(f"  Training samples:    {data['X_train'].shape[0]}")
    print(f"  Validation samples:  {data['X_val'].shape[0]}")
    print(f"  Test samples:        {data['X_test'].shape[0]}")

    print(f"\n{'Best Baseline (7-day Rolling Average)':-<50}")
    for target in TARGETS:
        bl = best_baselines.get(target, {})
        print(f"  {target:<25} MAE={bl.get('mae', 0):>12,.0f}  WAPE={bl.get('wape', 0):.4f}")

    print(f"\n{'Best Model Performance (Validation)':-<50}")
    print(f"  {'Target':<25} {'Best Model':<20} {'MAE':<15} {'vs Baseline':<15}")
    print(f"  {'-'*75}")

    for target in TARGETS:
        target_results = evaluation_results.get(target, {})
        baseline_mae = target_results.get('baseline', {}).get('mae')

        best_model = None
        best_mae = float('inf')
        for model_name, metrics in target_results.items():
            if model_name == 'baseline':
                continue
            if isinstance(metrics, dict) and metrics.get('mae') and metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                best_model = model_name

        if best_model and baseline_mae:
            improvement = (baseline_mae - best_mae) / baseline_mae * 100
            print(f"  {target:<25} {best_model:<20} {best_mae:<15,.0f} {improvement:+.1f}%")
        else:
            print(f"  {target:<25} {'N/A':<20} {'N/A':<15} {'N/A':<15}")

    print(f"\n{'Output Files':-<50}")
    print(f"  Models:       {MODELS_DIR}/")
    print(f"  Predictions:  {PREDICTIONS_DIR}/")
    print(f"  Plots:        {PLOTS_DIR}/")
    print(f"  Metrics:      {MODEL_METRICS_PATH}")
    print(f"  Importance:   {FEATURE_IMPORTANCE_PATH}")
    print(f"  Report:       {MODEL_REPORT_PATH}")

    print("\n" + "=" * 70)
    print(" " * 12 + "PHASE 4 MODEL DEVELOPMENT COMPLETED!")
    print("=" * 70)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline() -> Dict:
    """
    Execute the complete model development pipeline.

    Returns:
        dict: Summary of results.
    """
    print("=" * 70)
    print(" " * 10 + "BANK BRANCH CASH FORECASTING SYSTEM")
    print(" " * 12 + "Phase 4: Model Development")
    print("=" * 70)

    # ---- MODULE 1: Review Baselines ----
    best_baselines = review_baselines()

    # ---- MODULE 2: Load & Prepare Data ----
    data = load_and_prepare_data()

    # ---- MODULE 3: Train Models ----
    trained_models = train_models(data)

    # ---- MODULE 4: Evaluate Models ----
    evaluation_results = evaluate_models(data, trained_models, best_baselines)

    # ---- MODULE 5: Feature Importance ----
    importance_results = analyze_feature_importance(data, trained_models)

    # ---- MODULE 6: Save Outputs ----
    save_outputs(data, trained_models, evaluation_results, importance_results, best_baselines)

    # ---- MODULE 7: Visualizations ----
    create_visualizations(data, trained_models, evaluation_results)

    # ---- MODULE 8: Summary ----
    print_summary(evaluation_results, best_baselines, data)

    return {
        'best_baselines': best_baselines,
        'evaluation_results': evaluation_results,
        'importance_results': importance_results,
        'feature_count': len(data['feature_cols']),
        'trained_model_count': sum(
            1 for t in trained_models for m in trained_models[t] if trained_models[t][m] is not None
        )
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main entry point. Executes the full model development pipeline.
    """
    print("=" * 70)
    print(" " * 14 + "BANK BRANCH CASH FORECASTING")
    print(" " * 14 + "Phase 4: Model Development")
    print("=" * 70)

    start_time = datetime.now()
    print(f"\nStart time: {start_time}\n")

    try:
        results = run_pipeline()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\nTotal execution time: {duration:.2f} seconds")
        print(f"\n✓ Phase 4 completed successfully.")

        return results

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