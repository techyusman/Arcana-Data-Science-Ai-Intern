"""
===============================================================================
BANK BRANCH CASH FORECASTING SYSTEM
Daily Merge & Time Series EDA
===============================================================================

This script performs:
1. Daily merge/aggregation of transaction data
2. Time series visualizations for daily data
3. Save plots to Bank Project/eda_plots/

Author: Muhammad Usman
Status: Phase 1 - Time Series EDA
===============================================================================
"""

# =============================================================================
# STEP 1: IMPORTS & CONFIGURATION
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import warnings
import os
import sys

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "Bank DataSet/cleaned_bank_data.csv"
OUTPUT_DIR = "Bank Project/eda_plots"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STEP 2: LOAD AND PREPARE DATA
# =============================================================================

def load_and_prepare_data(file_path):
    """Load cleaned data and prepare for daily aggregation."""
    print("=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(file_path)
    df['start_date'] = pd.to_datetime(df['start_date'])
    
    print(f"[OK] Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Date Range: {df['start_date'].min().date()} to {df['start_date'].max().date()}")
    print(f"  Branches: {df['tran_br_code'].nunique()}")
    
    return df


def perform_daily_merge(df):
    """
    Perform daily merge/aggregation of transaction data.
    aggregates all transactions for each day across all branches.
    """
    print("\n" + "=" * 70)
    print("PERFORMING DAILY MERGE/AGGREGATION")
    print("=" * 70)
    
    # Extract date only (remove time component)
    df['date'] = df['start_date'].dt.date
    
    # Aggregate by date - sum all transactions for each day
    daily_df = df.groupby('date').agg({
        'TOTAL_DR': 'sum',
        'TOTAL_CR': 'sum',
        'tran_br_code': 'nunique',  # Count unique branches active
        'txn_hour': 'count'  # Count total transactions
    }).reset_index()
    
    # Rename columns for clarity
    daily_df.columns = ['date', 'daily_withdrawals', 'daily_deposits', 
                        'active_branches', 'total_transactions']
    
    # Calculate net cash position
    daily_df['net_cash'] = daily_df['daily_deposits'] - daily_df['daily_withdrawals']
    
    # Convert date back to datetime for time series analysis
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # Sort by date
    daily_df = daily_df.sort_values('date').reset_index(drop=True)
    
    # Add time-based features
    daily_df['year'] = daily_df['date'].dt.year
    daily_df['month'] = daily_df['date'].dt.month
    daily_df['day'] = daily_df['date'].dt.day
    daily_df['dayofweek'] = daily_df['date'].dt.dayofweek  # Monday=0, Sunday=6
    daily_df['day_name'] = daily_df['date'].dt.day_name()
    daily_df['is_weekend'] = daily_df['dayofweek'].isin([5, 6]).astype(int)
    daily_df['quarter'] = daily_df['date'].dt.quarter
    
    print(f"[OK] Daily aggregation complete:")
    print(f"  Total Days: {len(daily_df)}")
    print(f"  Date Range: {daily_df['date'].min().date()} to {daily_df['date'].max().date()}")
    print(f"  Avg Daily Withdrawals: {daily_df['daily_withdrawals'].mean()/1e6:.2f}M")
    print(f"  Avg Daily Deposits: {daily_df['daily_deposits'].mean()/1e6:.2f}M")
    print(f"  Avg Daily Net Cash: {daily_df['net_cash'].mean()/1e6:.2f}M")
    
    # Save daily merged data
    output_csv = "Bank DataSet/daily_merged_data.csv"
    daily_df.to_csv(output_csv, index=False)
    print(f"\n[OK] Daily merged data saved to: {output_csv}")
    
    return daily_df


# =============================================================================
# STEP 3: TIME SERIES VISUALIZATIONS
# =============================================================================

def plot_ts1_daily_cash_flow_trend(daily_df):
    """
    Time Series Plot 1: Daily Cash Flow Trend
    Line plot showing daily withdrawals vs deposits over time.
    """
    print("\n[TS-1] Daily Cash Flow Trend...")
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Plot withdrawals and deposits
    ax.plot(daily_df['date'], daily_df['daily_withdrawals'] / 1e6, 
            linewidth=2, label='Daily Withdrawals (DR)', color='#e74c3c', alpha=0.8)
    ax.plot(daily_df['date'], daily_df['daily_deposits'] / 1e6, 
            linewidth=2, label='Daily Deposits (CR)', color='#2ecc71', alpha=0.8)
    
    # Add 7-day rolling average for smoothing
    daily_df['dr_rolling'] = daily_df['daily_withdrawals'].rolling(window=7, min_periods=1).mean()
    daily_df['cr_rolling'] = daily_df['daily_deposits'].rolling(window=7, min_periods=1).mean()
    
    ax.plot(daily_df['date'], daily_df['dr_rolling'] / 1e6, 
            linewidth=2.5, label='DR - 7 Day Rolling Avg', color='#c0392b', linestyle='--')
    ax.plot(daily_df['date'], daily_df['cr_rolling'] / 1e6, 
            linewidth=2.5, label='CR - 7 Day Rolling Avg', color='#27ae60', linestyle='--')
    
    ax.set_title('Daily Cash Flow Trend (Withdrawals vs Deposits)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Amount (Millions)', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ts1_daily_cash_flow_trend.png", dpi=150)
    plt.close()
    print("  [OK] Saved: ts1_daily_cash_flow_trend.png")


def plot_ts2_daily_net_cash_position(daily_df):
    """
    Time Series Plot 2: Daily Net Cash Position
    Bar chart showing daily net cash with color coding for surplus/deficit.
    """
    print("[TS-2] Daily Net Cash Position...")
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Color bars based on positive/negative values
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in daily_df['net_cash']]
    
    ax.bar(daily_df['date'], daily_df['net_cash'] / 1e6, 
           color=colors, edgecolor='white', alpha=0.8, width=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    ax.set_title('Daily Net Cash Position (Deposits - Withdrawals)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Net Cash (Millions)', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x labels if many dates
    if len(daily_df) > 30:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ts2_daily_net_cash_position.png", dpi=150)
    plt.close()
    print("  [OK] Saved: ts2_daily_net_cash_position.png")


def plot_ts3_daily_transaction_volume(daily_df):
    """
    Time Series Plot 3: Daily Transaction Volume
    Dual axis plot showing transaction count and active branches.
    """
    print("[TS-3] Daily Transaction Volume...")
    
    fig, ax1 = plt.subplots(figsize=(16, 7))
    
    # Primary axis: Transaction count
    color1 = '#3498db'
    ax1.bar(daily_df['date'], daily_df['total_transactions'], 
            color=color1, alpha=0.6, label='Total Transactions')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Total Transactions', color=color1, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Secondary axis: Active branches
    ax2 = ax1.twinx()
    color2 = '#e67e22'
    ax2.plot(daily_df['date'], daily_df['active_branches'], 
             color=color2, linewidth=2, marker='o', markersize=3, label='Active Branches')
    ax2.set_ylabel('Active Branches', color=color2, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title and grid
    plt.title('Daily Transaction Volume & Active Branches', 
              fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
    
    if len(daily_df) > 30:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ts3_daily_transaction_volume.png", dpi=150)
    plt.close()
    print("  [OK] Saved: ts3_daily_transaction_volume.png")


def plot_ts4_weekly_seasonal_pattern(daily_df):
    """
    Time Series Plot 4: Weekly Seasonal Pattern
    Shows average daily patterns by day of week.
    """
    print("[TS-4] Weekly Seasonal Pattern...")
    
    # Aggregate by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_stats = daily_df.groupby('day_name').agg({
        'daily_withdrawals': 'mean',
        'daily_deposits': 'mean',
        'net_cash': 'mean',
        'total_transactions': 'mean',
        'active_branches': 'mean'
    }).reindex(day_order).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Average Withdrawals and Deposits by Day
    ax1 = axes[0, 0]
    x_pos = np.arange(len(day_order))
    width = 0.35
    ax1.bar(x_pos - width/2, weekly_stats['daily_withdrawals'] / 1e6, 
            width, label='Withdrawals', color='#e74c3c', alpha=0.8)
    ax1.bar(x_pos + width/2, weekly_stats['daily_deposits'] / 1e6, 
            width, label='Deposits', color='#2ecc71', alpha=0.8)
    ax1.set_title('Avg Daily Amount by Day of Week', fontweight='bold')
    ax1.set_xlabel('Day of Week')
    ax1.set_ylabel('Amount (Millions)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(day_order, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Net Cash by Day
    ax2 = axes[0, 1]
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in weekly_stats['net_cash']]
    ax2.bar(day_order, weekly_stats['net_cash'] / 1e6, color=colors, alpha=0.8, edgecolor='white')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_title('Avg Net Cash by Day of Week', fontweight='bold')
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Net Cash (Millions)')
    ax2.set_xticklabels(day_order, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Transaction Count by Day
    ax3 = axes[1, 0]
    ax3.bar(day_order, weekly_stats['total_transactions'], color='#3498db', alpha=0.8, edgecolor='white')
    ax3.set_title('Avg Transaction Count by Day of Week', fontweight='bold')
    ax3.set_xlabel('Day of Week')
    ax3.set_ylabel('Transaction Count')
    ax3.set_xticklabels(day_order, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Active Branches by Day
    ax4 = axes[1, 1]
    ax4.plot(day_order, weekly_stats['active_branches'], marker='o', 
             linewidth=2, markersize=8, color='#9b59b6')
    ax4.fill_between(range(len(day_order)), weekly_stats['active_branches'], 
                     alpha=0.3, color='#9b59b6')
    ax4.set_title('Avg Active Branches by Day of Week', fontweight='bold')
    ax4.set_xlabel('Day of Week')
    ax4.set_ylabel('Active Branches')
    ax4.set_xticklabels(day_order, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Weekly Seasonal Patterns', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ts4_weekly_seasonal_pattern.png", dpi=150)
    plt.close()
    print("  [OK] Saved: ts4_weekly_seasonal_pattern.png")


def plot_ts5_monthly_trend_analysis(daily_df):
    """
    Time Series Plot 5: Monthly Trend Analysis
    Shows monthly aggregated trends over time.
    """
    print("[TS-5] Monthly Trend Analysis...")
    
    # Aggregate by month
    daily_df['year_month'] = daily_df['date'].dt.to_period('M')
    monthly_df = daily_df.groupby('year_month').agg({
        'daily_withdrawals': 'sum',
        'daily_deposits': 'sum',
        'net_cash': 'sum',
        'total_transactions': 'sum',
        'active_branches': 'mean'
    }).reset_index()
    monthly_df['year_month'] = monthly_df['year_month'].astype(str)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Monthly Withdrawals and Deposits
    ax1 = axes[0]
    ax1.plot(monthly_df['year_month'], monthly_df['daily_withdrawals'] / 1e6, 
             marker='o', linewidth=2, label='Withdrawals', color='#e74c3c')
    ax1.plot(monthly_df['year_month'], monthly_df['daily_deposits'] / 1e6, 
             marker='s', linewidth=2, label='Deposits', color='#2ecc71')
    ax1.set_title('Monthly Cash Flow Trends', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Amount (Millions)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if len(monthly_df) > 12:
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Monthly Net Cash
    ax2 = axes[1]
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in monthly_df['net_cash']]
    ax2.bar(monthly_df['year_month'], monthly_df['net_cash'] / 1e6, 
            color=colors, edgecolor='white', alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_title('Monthly Net Cash Position', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Net Cash (Millions)')
    ax2.grid(axis='y', alpha=0.3)
    
    if len(monthly_df) > 12:
        ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Monthly Trend Analysis', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ts5_monthly_trend_analysis.png", dpi=150)
    plt.close()
    print("  [OK] Saved: ts5_monthly_trend_analysis.png")


def plot_ts6_cumulative_cash_flow(daily_df):
    """
    Time Series Plot 6: Cumulative Cash Flow
    Shows cumulative withdrawals and deposits over time.
    """
    print("[TS-6] Cumulative Cash Flow...")
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Calculate cumulative sums
    daily_df['cumulative_dr'] = daily_df['daily_withdrawals'].cumsum()
    daily_df['cumulative_cr'] = daily_df['daily_deposits'].cumsum()
    daily_df['cumulative_net'] = daily_df['net_cash'].cumsum()
    
    ax.plot(daily_df['date'], daily_df['cumulative_dr'] / 1e6, 
            linewidth=2, label='Cumulative Withdrawals', color='#e74c3c')
    ax.plot(daily_df['date'], daily_df['cumulative_cr'] / 1e6, 
            linewidth=2, label='Cumulative Deposits', color='#2ecc71')
    ax.plot(daily_df['date'], daily_df['cumulative_net'] / 1e6, 
            linewidth=2.5, label='Cumulative Net Cash', color='#9b59b6', linestyle='--')
    
    ax.fill_between(daily_df['date'], daily_df['cumulative_dr'] / 1e6, 
                    daily_df['cumulative_cr'] / 1e6, 
                    where=(daily_df['cumulative_cr'] > daily_df['cumulative_dr']),
                    alpha=0.2, color='#2ecc71', label='Surplus Zone')
    ax.fill_between(daily_df['date'], daily_df['cumulative_dr'] / 1e6, 
                    daily_df['cumulative_cr'] / 1e6, 
                    where=(daily_df['cumulative_cr'] <= daily_df['cumulative_dr']),
                    alpha=0.2, color='#e74c3c', label='Deficit Zone')
    
    ax.set_title('Cumulative Cash Flow Analysis', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Cumulative Amount (Millions)', fontsize=11)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if len(daily_df) > 30:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ts6_cumulative_cash_flow.png", dpi=150)
    plt.close()
    print("  [OK] Saved: ts6_cumulative_cash_flow.png")


def plot_ts7_weekend_vs_weekday(daily_df):
    """
    Time Series Plot 7: Weekend vs Weekday Analysis
    Compares transaction patterns between weekends and weekdays.
    """
    print("[TS-7] Weekend vs Weekday Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Group by weekend status
    weekend_stats = daily_df.groupby('is_weekend').agg({
        'daily_withdrawals': 'mean',
        'daily_deposits': 'mean',
        'net_cash': 'mean',
        'total_transactions': 'mean',
        'active_branches': 'mean'
    }).reset_index()
    
    labels = ['Weekday', 'Weekend']
    
    # Plot 1: Average Transaction Amount
    ax1 = axes[0, 0]
    x = np.arange(len(labels))
    width = 0.35
    ax1.bar(x - width/2, weekend_stats['daily_withdrawals'] / 1e6, 
            width, label='Withdrawals', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width/2, weekend_stats['daily_deposits'] / 1e6, 
            width, label='Deposits', color='#2ecc71', alpha=0.8)
    ax1.set_title('Avg Daily Amount', fontweight='bold')
    ax1.set_ylabel('Amount (Millions)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Net Cash
    ax2 = axes[0, 1]
    colors = ['#3498db', '#e67e22']
    ax2.bar(labels, weekend_stats['net_cash'] / 1e6, color=colors, alpha=0.8, edgecolor='white')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_title('Avg Net Cash', fontweight='bold')
    ax2.set_ylabel('Net Cash (Millions)')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Transaction Count
    ax3 = axes[1, 0]
    ax3.bar(labels, weekend_stats['total_transactions'], color='#9b59b6', alpha=0.8, edgecolor='white')
    ax3.set_title('Avg Transaction Count', fontweight='bold')
    ax3.set_ylabel('Transactions')
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Active Branches
    ax4 = axes[1, 1]
    ax4.bar(labels, weekend_stats['active_branches'], color='#1abc9c', alpha=0.8, edgecolor='white')
    ax4.set_title('Avg Active Branches', fontweight='bold')
    ax4.set_ylabel('Branch Count')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Weekend vs Weekday Comparison', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ts7_weekend_vs_weekday.png", dpi=150)
    plt.close()
    print("  [OK] Saved: ts7_weekend_vs_weekday.png")


def plot_ts8_daily_distribution_analysis(daily_df):
    """
    Time Series Plot 8: Daily Distribution Analysis
    Histograms and KDE plots for daily aggregated values.
    """
    print("[TS-8] Daily Distribution Analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Daily Withdrawals Distribution
    ax1 = axes[0, 0]
    sns.histplot(daily_df['daily_withdrawals'] / 1e6, bins=30, kde=True, 
                 color='#e74c3c', ax=ax1)
    ax1.set_title('Daily Withdrawals', fontweight='bold')
    ax1.set_xlabel('Amount (Millions)')
    ax1.set_ylabel('Frequency')
    
    # Plot 2: Daily Deposits Distribution
    ax2 = axes[0, 1]
    sns.histplot(daily_df['daily_deposits'] / 1e6, bins=30, kde=True, 
                 color='#2ecc71', ax=ax2)
    ax2.set_title('Daily Deposits', fontweight='bold')
    ax2.set_xlabel('Amount (Millions)')
    ax2.set_ylabel('Frequency')
    
    # Plot 3: Net Cash Distribution
    ax3 = axes[0, 2]
    sns.histplot(daily_df['net_cash'] / 1e6, bins=30, kde=True, 
                 color='#3498db', ax=ax3)
    ax3.set_title('Daily Net Cash', fontweight='bold')
    ax3.set_xlabel('Net Cash (Millions)')
    ax3.set_ylabel('Frequency')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    # Plot 4: Transaction Count Distribution
    ax4 = axes[1, 0]
    sns.histplot(daily_df['total_transactions'], bins=30, kde=True, 
                 color='#9b59b6', ax=ax4)
    ax4.set_title('Daily Transaction Count', fontweight='bold')
    ax4.set_xlabel('Transaction Count')
    ax4.set_ylabel('Frequency')
    
    # Plot 5: Active Branches Distribution
    ax5 = axes[1, 1]
    sns.histplot(daily_df['active_branches'], bins=20, kde=True, 
                 color='#1abc9c', ax=ax5)
    ax5.set_title('Active Branches per Day', fontweight='bold')
    ax5.set_xlabel('Branch Count')
    ax5.set_ylabel('Frequency')
    
    # Plot 6: Box plot of Net Cash
    ax6 = axes[1, 2]
    ax6.boxplot(daily_df['net_cash'] / 1e6, vert=True, patch_artist=True)
    ax6.set_title('Net Cash Box Plot', fontweight='bold')
    ax6.set_ylabel('Net Cash (Millions)')
    ax6.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Daily Distribution Analysis', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ts8_daily_distribution_analysis.png", dpi=150)
    plt.close()
    print("  [OK] Saved: ts8_daily_distribution_analysis.png")


def plot_ts9_correlation_matrix_daily(daily_df):
    """
    Time Series Plot 9: Daily Correlation Matrix
    Correlation heatmap of daily aggregated features.
    """
    print("[TS-9] Daily Correlation Matrix...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Select numerical columns for correlation
    corr_cols = ['daily_withdrawals', 'daily_deposits', 'net_cash', 
                 'total_transactions', 'active_branches', 'is_weekend']
    
    # Correlation matrix
    ax1 = axes[0]
    corr_matrix = daily_df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax1,
                cbar_kws={'shrink': 0.8})
    ax1.set_title('Daily Features Correlation Matrix', fontsize=12, fontweight='bold')
    
    # Scatter: Withdrawals vs Deposits
    ax2 = axes[1]
    ax2.scatter(daily_df['daily_withdrawals'] / 1e6, daily_df['daily_deposits'] / 1e6,
                alpha=0.5, s=50, c=daily_df['net_cash'], cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    # Add regression line
    z = np.polyfit(daily_df['daily_withdrawals'] / 1e6, daily_df['daily_deposits'] / 1e6, 1)
    p = np.poly1d(z)
    x_line = np.linspace(daily_df['daily_withdrawals'].min() / 1e6, 
                         daily_df['daily_withdrawals'].max() / 1e6, 100)
    ax2.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7, 
             label=f'Trend (slope={z[0]:.3f})')
    
    ax2.set_title('Withdrawals vs Deposits (Daily)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Withdrawals (Millions)')
    ax2.set_ylabel('Deposits (Millions)')
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    plt.suptitle('Daily Correlation Analysis', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ts9_daily_correlation_matrix.png", dpi=150)
    plt.close()
    print("  [OK] Saved: ts9_daily_correlation_matrix.png")


def plot_ts10_time_series_decomposition(daily_df):
    """
    Time Series Plot 10: Time Series Decomposition
    Shows trend, seasonal, and residual components.
    """
    print("[TS-10] Time Series Decomposition...")
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # Use net cash for decomposition
    ts_data = daily_df.set_index('date')['net_cash']
    
    # Decompose with period=7 (weekly seasonality)
    decomposition = seasonal_decompose(ts_data, model='additive', period=7, extrapolate_trend='freq')
    
    # Observed
    axes[0].plot(ts_data.index, ts_data.values, color='#3498db', linewidth=1.5)
    axes[0].set_title('Observed', fontweight='bold')
    axes[0].set_ylabel('Net Cash')
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    axes[1].plot(decomposition.trend.index, decomposition.trend.values, 
                 color='#e74c3c', linewidth=2)
    axes[1].set_title('Trend', fontweight='bold')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal
    axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, 
                 color='#2ecc71', linewidth=1.5)
    axes[2].set_title('Seasonal (Weekly)', fontweight='bold')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    axes[3].plot(decomposition.resid.index, decomposition.resid.values, 
                 color='#9b59b6', linewidth=1)
    axes[3].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[3].set_title('Residual', fontweight='bold')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle('Time Series Decomposition (Net Cash)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ts10_time_series_decomposition.png", dpi=150)
    plt.close()
    print("  [OK] Saved: ts10_time_series_decomposition.png")


def generate_timeseries_summary_report(daily_df):
    """Generate comprehensive time series summary report."""
    print("\n" + "=" * 70)
    print("TIME SERIES EDA SUMMARY REPORT")
    print("=" * 70)
    
    print(f"\n[DAILY AGGREGATED DATASET]")
    print(f"   - Total Days: {len(daily_df)}")
    print(f"   - Date Range: {daily_df['date'].min().date()} to {daily_df['date'].max().date()}")
    
    print(f"\n[CASH FLOW SUMMARY]")
    print(f"   - Total Withdrawals: {daily_df['daily_withdrawals'].sum()/1e6:.2f}M")
    print(f"   - Total Deposits: {daily_df['daily_deposits'].sum()/1e6:.2f}M")
    print(f"   - Total Net Cash: {daily_df['net_cash'].sum()/1e6:.2f}M")
    print(f"   - Avg Daily Net Cash: {daily_df['net_cash'].mean()/1e6:.2f}M")
    
    print(f"\n[TRANSACTION PATTERNS]")
    print(f"   - Avg Daily Transactions: {daily_df['total_transactions'].mean():.0f}")
    print(f"   - Avg Active Branches: {daily_df['active_branches'].mean():.1f}")
    print(f"   - Max Daily Transactions: {daily_df['total_transactions'].max():,}")
    
    print(f"\n[WEEKDAY vs WEEKEND]")
    weekday_avg = daily_df[daily_df['is_weekend'] == 0]['net_cash'].mean()
    weekend_avg = daily_df[daily_df['is_weekend'] == 1]['net_cash'].mean()
    print(f"   - Avg Weekday Net Cash: {weekday_avg/1e6:.2f}M")
    print(f"   - Avg Weekend Net Cash: {weekend_avg/1e6:.2f}M")
    diff = weekend_avg - weekday_avg
    print(f"   - Difference: {diff/1e6:.2f}M ({'Higher' if diff > 0 else 'Lower'} on weekends)")
    
    print(f"\n[VOLATILITY ANALYSIS]")
    print(f"   - Net Cash Std Dev: {daily_df['net_cash'].std()/1e6:.2f}M")
    print(f"   - Max Daily Deficit: {daily_df['net_cash'].min()/1e6:.2f}M")
    print(f"   - Max Daily Surplus: {daily_df['net_cash'].max()/1e6:.2f}M")
    
    print(f"\n[MONTHLY PATTERNS]")
    monthly_avg = daily_df.groupby(daily_df['date'].dt.month)['net_cash'].mean()
    best_month = monthly_avg.idxmax()
    worst_month = monthly_avg.idxmin()
    print(f"   - Best Month (Avg Net Cash): Month {best_month} ({monthly_avg[best_month]/1e6:.2f}M)")
    print(f"   - Worst Month (Avg Net Cash): Month {worst_month} ({monthly_avg[worst_month]/1e6:.2f}M)")
    
    print("\n" + "=" * 70)
    print(f"[OK] All time series plots saved to: {OUTPUT_DIR}")
    print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run daily merge and time series EDA."""
    print("=" * 70)
    print(" " * 10 + "BANK BRANCH CASH FORECASTING SYSTEM")
    print(" " * 10 + "Daily Merge & Time Series EDA")
    print("=" * 70)
    
    # Load and prepare data
    df = load_and_prepare_data(DATA_PATH)
    
    # Perform daily merge
    daily_df = perform_daily_merge(df)
    
    # Generate all time series visualizations
    print("\n" + "=" * 70)
    print("GENERATING TIME SERIES VISUALIZATIONS")
    print("=" * 70)
    
    plot_ts1_daily_cash_flow_trend(daily_df)
    plot_ts2_daily_net_cash_position(daily_df)
    plot_ts3_daily_transaction_volume(daily_df)
    plot_ts4_weekly_seasonal_pattern(daily_df)
    plot_ts5_monthly_trend_analysis(daily_df)
    plot_ts6_cumulative_cash_flow(daily_df)
    plot_ts7_weekend_vs_weekday(daily_df)
    plot_ts8_daily_distribution_analysis(daily_df)
    plot_ts9_correlation_matrix_daily(daily_df)
    plot_ts10_time_series_decomposition(daily_df)
    
    # Generate summary report
    generate_timeseries_summary_report(daily_df)
    
    print("\n" + "=" * 70)
    print(" " * 15 + "TIME SERIES EDA COMPLETED!")
    print("=" * 70)
    print(f"\n[FOLDER] All plots saved to: '{OUTPUT_DIR}/'")
    print("\nGenerated time series plots:")
    print("  1. ts1_daily_cash_flow_trend.png      - Daily withdrawals vs deposits with rolling avg")
    print("  2. ts2_daily_net_cash_position.png    - Daily net cash position")
    print("  3. ts3_daily_transaction_volume.png   - Daily transaction count & active branches")
    print("  4. ts4_weekly_seasonal_pattern.png    - Weekly patterns across days")
    print("  5. ts5_monthly_trend_analysis.png     - Monthly aggregated trends")
    print("  6. ts6_cumulative_cash_flow.png       - Cumulative cash flow analysis")
    print("  7. ts7_weekend_vs_weekday.png         - Weekend vs weekday comparison")
    print("  8. ts8_daily_distribution_analysis.png - Distribution of daily metrics")
    print("  9. ts9_daily_correlation_matrix.png   - Correlation analysis of daily features")
    print(" 10. ts10_time_series_decomposition.png - Trend, seasonal, residual decomposition")
    print("\nOutput Files:")
    print("  - Bank DataSet/daily_merged_data.csv  (Daily aggregated dataset)")
    print("  - Bank Project/eda_plots/             (All visualizations)")


if __name__ == "__main__":
    main()