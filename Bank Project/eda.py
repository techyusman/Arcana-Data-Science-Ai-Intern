"""
===============================================================================
BANK BRANCH CASH FORECASTING SYSTEM
Combined EDA: Peak Hours, Branch Performance, Net Cash, Time Series Analysis
===============================================================================

This script combines selected visualizations from three analysis modules:
1. Phase 1 EDA - Peak hours, branch performance, net cash, hourly boxplots
2. Phase 2 Cash Analysis - Branch net cash position, branch-hour heatmap
3. Time Series EDA - Weekly patterns, monthly trends, cumulative cash flow,
                     weekend vs weekday, daily distribution

Author: Muhammad Usman
===============================================================================
"""

# =============================================================================
# IMPORTS & CONFIGURATION
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
OUTPUT_DIR = "Bank DataSet/eda_plots"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

def load_data(file_path):
    """Load cleaned bank transaction data."""
    print("=" * 70)
    print("LOADING CLEANED DATASET")
    print("=" * 70)

    df = pd.read_csv(file_path)
    df['start_date'] = pd.to_datetime(df['start_date'])

    print(f"[OK] Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Date Range: {df['start_date'].min().date()} to {df['start_date'].max().date()}")
    print(f"  Branches: {df['tran_br_code'].nunique()}")
    print(f"  Hours Range: {df['txn_hour'].min()} - {df['txn_hour'].max()}")

    return df


def perform_daily_merge(df):
    """
    Perform daily merge/aggregation of transaction data.
    Aggregates all transactions for each day across all branches.
    """
    print("\n" + "=" * 70)
    print("PERFORMING DAILY MERGE/AGGREGATION")
    print("=" * 70)

    # Extract date only
    df['date'] = df['start_date'].dt.date

    # Aggregate by date
    daily_df = df.groupby('date').agg({
        'TOTAL_DR': 'sum',
        'TOTAL_CR': 'sum',
        'tran_br_code': 'nunique',
        'txn_hour': 'count'
    }).reset_index()

    # Rename columns
    daily_df.columns = ['date', 'daily_withdrawals', 'daily_deposits',
                        'active_branches', 'total_transactions']

    # Calculate net cash position
    daily_df['net_cash'] = daily_df['daily_deposits'] - daily_df['daily_withdrawals']

    # Convert date back to datetime
    daily_df['date'] = pd.to_datetime(daily_df['date'])

    # Sort by date
    daily_df = daily_df.sort_values('date').reset_index(drop=True)

    # Add time-based features
    daily_df['year'] = daily_df['date'].dt.year
    daily_df['month'] = daily_df['date'].dt.month
    daily_df['day'] = daily_df['date'].dt.day
    daily_df['dayofweek'] = daily_df['date'].dt.dayofweek
    daily_df['day_name'] = daily_df['date'].dt.day_name()
    daily_df['is_weekend'] = daily_df['dayofweek'].isin([5, 6]).astype(int)
    daily_df['quarter'] = daily_df['date'].dt.quarter

    print(f"[OK] Daily aggregation complete:")
    print(f"  Total Days: {len(daily_df)}")
    print(f"  Date Range: {daily_df['date'].min().date()} to {daily_df['date'].max().date()}")
    print(f"  Avg Daily Withdrawals: {daily_df['daily_withdrawals'].mean()/1e6:.2f}M")
    print(f"  Avg Daily Deposits: {daily_df['daily_deposits'].mean()/1e6:.2f}M")
    print(f"  Avg Daily Net Cash: {daily_df['net_cash'].mean()/1e6:.2f}M")

    return daily_df


# =============================================================================
# PLOT 3: PEAK HOURS ANALYSIS (from Phase 1 EDA)
# =============================================================================

def plot_3_peak_hours_analysis(df):
    """
    Plot 3: Peak Hours Analysis
    Average transaction amounts and frequencies by hour of day.
    """
    print("\n[3/11] Peak Hours Analysis...")

    hour_stats = df.groupby('txn_hour').agg({
        'TOTAL_DR': ['mean', 'sum', 'count'],
        'TOTAL_CR': ['mean', 'sum']
    }).round(2)
    hour_stats.columns = ['_'.join(col).strip() for col in hour_stats.columns.values]
    hour_stats = hour_stats.reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Transaction frequency by hour
    ax1 = axes[0]
    ax1.bar(hour_stats['txn_hour'], hour_stats['TOTAL_DR_count'],
            color='#3498db', edgecolor='white', alpha=0.8)
    ax1.set_title('Transaction Frequency by Hour', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Number of Transactions')
    ax1.set_xticks(range(0, 24))
    ax1.grid(axis='y', alpha=0.3)

    # Right: Average amounts by hour
    ax2 = axes[1]
    ax2.plot(hour_stats['txn_hour'], hour_stats['TOTAL_DR_mean'] / 1e6,
             marker='o', linewidth=2, label='Avg Withdrawal', color='#e74c3c')
    ax2.plot(hour_stats['txn_hour'], hour_stats['TOTAL_CR_mean'] / 1e6,
             marker='s', linewidth=2, label='Avg Deposit', color='#2ecc71')
    ax2.set_title('Average Transaction Amount by Hour', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Average Amount (Millions)')
    ax2.set_xticks(range(0, 24))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_peak_hours_analysis.png", dpi=150)
    plt.close()
    print("  [OK] Saved: 3_peak_hours_analysis.png")


# =============================================================================
# PLOT 4: BRANCH PERFORMANCE (from Phase 1 EDA)
# =============================================================================

def plot_4_branch_performance(df):
    """
    Plot 4: Branch Performance
    Top and bottom branches by total transaction volume.
    """
    print("[4/11] Branch Performance Analysis...")

    branch_stats = df.groupby('tran_br_code').agg({
        'TOTAL_DR': 'sum',
        'TOTAL_CR': 'sum',
        'start_date': 'count'
    }).rename(columns={'start_date': 'txn_count'}).reset_index()

    branch_stats['total_volume'] = branch_stats['TOTAL_DR'] + branch_stats['TOTAL_CR']
    branch_stats = branch_stats.sort_values('total_volume', ascending=False)

    # Top 15 and Bottom 15 branches
    top15 = branch_stats.head(15)
    bottom15 = branch_stats.tail(15)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top 15
    ax1 = axes[0]
    bars1 = ax1.barh(range(len(top15)), top15['total_volume'] / 1e6, color='#2ecc71', edgecolor='white')
    ax1.set_yticks(range(len(top15)))
    ax1.set_yticklabels([str(int(x)) for x in top15['tran_br_code']])
    ax1.set_title('Top 15 Branches by Total Transaction Volume', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Total Volume (Millions)')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars1, top15['total_volume'] / 1e6):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}M', va='center', fontsize=8)

    # Bottom 15
    ax2 = axes[1]
    bars2 = ax2.barh(range(len(bottom15)), bottom15['total_volume'] / 1e6, color='#e74c3c', edgecolor='white')
    ax2.set_yticks(range(len(bottom15)))
    ax2.set_yticklabels([str(int(x)) for x in bottom15['tran_br_code']])
    ax2.set_title('Bottom 15 Branches by Total Transaction Volume', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Total Volume (Millions)')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars2, bottom15['total_volume'] / 1e6):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}M', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_branch_performance.png", dpi=150)
    plt.close()
    print("  [OK] Saved: 4_branch_performance.png")


# =============================================================================
# PLOT 6: NET CASH ANALYSIS (from Phase 1 EDA)
# =============================================================================

def plot_6_net_cash_analysis(df):
    """
    Plot 6: Net Cash Analysis
    Net = TOTAL_CR - TOTAL_DR (positive = surplus, negative = deficit).
    Shows net cash position over time.
    """
    print("[6/11] Net Cash Analysis...")

    df_net = df.copy()
    df_net['net_cash'] = df_net['TOTAL_CR'] - df_net['TOTAL_DR']
    df_net['year_month'] = df_net['start_date'].dt.to_period('M')

    monthly_net = df_net.groupby('year_month')['net_cash'].sum().reset_index()
    monthly_net['year_month'] = monthly_net['year_month'].astype(str)

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in monthly_net['net_cash']]
    ax.bar(monthly_net['year_month'], monthly_net['net_cash'] / 1e6,
           color=colors, edgecolor='white', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title('Monthly Net Cash Position (Deposits - Withdrawals)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Net Cash (Millions)')

    if len(monthly_net) > 12:
        plt.xticks(rotation=45, ha='right')

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/6_net_cash_analysis.png", dpi=150)
    plt.close()
    print("  [OK] Saved: 6_net_cash_analysis.png")


# =============================================================================
# PLOT 7: HOURLY BOXPLOT (from Phase 1 EDA)
# =============================================================================

def plot_7_hourly_boxplot(df):
    """
    Plot 7: Transaction Amount Distribution by Hour (Box Plots)
    Shows how transaction amounts vary across different hours.
    """
    print("[7/11] Hourly Transaction Distribution (Box Plots)...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    hour_order = sorted(df['txn_hour'].unique())

    # Box plot for TOTAL_DR by hour
    ax1 = axes[0]
    data_dr = [df[df['txn_hour'] == h]['TOTAL_DR'] / 1e6 for h in hour_order]
    bp1 = ax1.boxplot(data_dr, labels=hour_order, patch_artist=True,
                       showfliers=False, widths=0.6)
    for patch in bp1['boxes']:
        patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.6)
    ax1.set_title('Withdrawals (DR) Distribution by Hour', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Amount (Millions)')
    ax1.grid(axis='y', alpha=0.3)

    # Box plot for TOTAL_CR by hour
    ax2 = axes[1]
    data_cr = [df[df['txn_hour'] == h]['TOTAL_CR'] / 1e6 for h in hour_order]
    bp2 = ax2.boxplot(data_cr, labels=hour_order, patch_artist=True,
                       showfliers=False, widths=0.6)
    for patch in bp2['boxes']:
        patch.set_facecolor('#2ecc71')
        patch.set_alpha(0.6)
    ax2.set_title('Deposits (CR) Distribution by Hour', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Amount (Millions)')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/7_hourly_boxplot.png", dpi=150)
    plt.close()
    print("  [OK] Saved: 7_hourly_boxplot.png")


# =============================================================================
# PLOT 10: BRANCH NET CASH POSITION (from Phase 2 Cash Analysis)
# =============================================================================

def plot_10_branch_net_cash_position(df):
    """
    Plot 10: Branch-Wise Net Cash Position
    Shows net cash position per branch: surplus (green) or deficit (red).
    """
    print("[10/11] Branch-Wise Net Cash Position...")

    df['net_cash'] = df['TOTAL_CR'] - df['TOTAL_DR']

    branch_net = df.groupby('tran_br_code').agg({
        'net_cash': 'sum',
        'TOTAL_DR': 'sum',
        'TOTAL_CR': 'sum',
        'start_date': 'count'
    }).rename(columns={'start_date': 'txn_count'}).reset_index()

    branch_net['net_cash_m'] = branch_net['net_cash'] / 1e6
    branch_net = branch_net.sort_values('net_cash', ascending=True).reset_index(drop=True)
    branch_net['rank'] = range(1, len(branch_net) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    # Top: All branches net cash bar chart
    ax1 = axes[0]
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in branch_net['net_cash']]
    bars = ax1.bar(range(len(branch_net)), branch_net['net_cash_m'],
                   color=colors, edgecolor='white', alpha=0.8, width=0.8)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_title('Branch-Wise Net Cash Position (All Branches)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Branch Code (sorted by net cash)')
    ax1.set_ylabel('Net Cash (Millions)')
    ax1.set_xticks(range(0, len(branch_net), max(1, len(branch_net) // 20)))
    ax1.set_xticklabels(branch_net['tran_br_code'].iloc[::max(1, len(branch_net) // 20)].astype(int))
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=1.2)

    surplus_branches = (branch_net['net_cash'] >= 0).sum()
    deficit_branches = (branch_net['net_cash'] < 0).sum()
    ax1.text(0.02, 0.95, f'Surplus: {surplus_branches} | Deficit: {deficit_branches}',
             transform=ax1.transAxes, fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top')

    # Bottom: Top 10 deficit and surplus branches
    ax2 = axes[1]
    top_deficit = branch_net.head(10)
    top_surplus = branch_net.tail(10).iloc[::-1]

    plot_data = pd.concat([top_deficit, top_surplus])
    plot_colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in plot_data['net_cash']]

    bars2 = ax2.barh(range(len(plot_data)), plot_data['net_cash_m'],
                     color=plot_colors, edgecolor='white', alpha=0.85, height=0.7)

    ax2.set_yticks(range(len(plot_data)))
    ax2.set_yticklabels([f"Branch {int(b)}" for b in plot_data['tran_br_code']])
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_title('Top 10 Deficit vs Top 10 Surplus Branches', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Net Cash (Millions)')
    ax2.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars2, plot_data['net_cash_m']):
        label_x = bar.get_width() - (0.5 if val > 0 else -0.5)
        ax2.text(label_x, bar.get_y() + bar.get_height() / 2.,
                 f'{val:+,.2f}M', va='center',
                 ha='left' if val > 0 else 'right', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_branch_net_cash_position.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: 10_branch_net_cash_position.png")


# =============================================================================
# PLOT 11: BRANCH HOUR HEATMAP (from Phase 2 Cash Analysis)
# =============================================================================

def plot_11_branch_hour_heatmap(df):
    """
    Plot 11: Branch × Hour Net-Cash Heatmap
    Heatmap showing average net cash position for each branch at each hour.
    """
    print("[11/11] Branch × Hour Net-Cash Heatmap...")

    df['net_cash'] = df['TOTAL_CR'] - df['TOTAL_DR']

    pivot = df.pivot_table(
        values='net_cash',
        index='tran_br_code',
        columns='txn_hour',
        aggfunc='mean'
    ) / 1e6

    fig, ax = plt.subplots(figsize=(18, max(10, pivot.shape[0] * 0.3)))

    cmap = sns.diverging_palette(10, 130, s=80, l=55, center='light', as_cmap=True)
    sns.heatmap(pivot, cmap=cmap, center=0, annot=False,
                fmt='.2f', linewidths=0.3, linecolor='gray',
                cbar_kws={'label': 'Avg Net Cash (Millions)', 'shrink': 0.8},
                ax=ax)

    ax.set_title('Branch × Hour Net-Cash Heatmap\n(Average Net Cash in Millions — Green=Surplus, Red=Deficit)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Branch Code', fontsize=12)
    ax.axvline(x=11.5, color='white', linestyle='--', linewidth=2, alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/11_branch_hour_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: 11_branch_hour_heatmap.png")


# =============================================================================
# TS4: WEEKLY SEASONAL PATTERN (from Time Series EDA)
# =============================================================================

def plot_ts4_weekly_seasonal_pattern(daily_df):
    """
    Time Series Plot 4: Weekly Seasonal Pattern
    Shows average daily patterns by day of week.
    """
    print("[TS4/11] Weekly Seasonal Pattern...")

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


# =============================================================================
# TS5: MONTHLY TREND ANALYSIS (from Time Series EDA)
# =============================================================================

def plot_ts5_monthly_trend_analysis(daily_df):
    """
    Time Series Plot 5: Monthly Trend Analysis
    Shows monthly aggregated trends over time.
    """
    print("[TS5/11] Monthly Trend Analysis...")

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


# =============================================================================
# TS6: CUMULATIVE CASH FLOW (from Time Series EDA)
# =============================================================================

def plot_ts6_cumulative_cash_flow(daily_df):
    """
    Time Series Plot 6: Cumulative Cash Flow
    Shows cumulative withdrawals and deposits over time.
    """
    print("[TS6/11] Cumulative Cash Flow...")

    fig, ax = plt.subplots(figsize=(16, 7))

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


# =============================================================================
# TS7: WEEKEND VS WEEKDAY (from Time Series EDA)
# =============================================================================

def plot_ts7_weekend_vs_weekday(daily_df):
    """
    Time Series Plot 7: Weekend vs Weekday Analysis
    Compares transaction patterns between weekends and weekdays.
    """
    print("[TS7/11] Weekend vs Weekday Analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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


# =============================================================================
# TS8: DAILY DISTRIBUTION ANALYSIS (from Time Series EDA)
# =============================================================================

def plot_ts8_daily_distribution_analysis(daily_df):
    """
    Time Series Plot 8: Daily Distribution Analysis
    Histograms and KDE plots for daily aggregated values.
    """
    print("[TS8/11] Daily Distribution Analysis...")

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


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all selected EDA visualizations."""
    print("=" * 70)
    print("         BANK BRANCH CASH FORECASTING SYSTEM")
    print("               Combined EDA Analysis")
    print("=" * 70)

    # Load raw transaction data
    df = load_data(DATA_PATH)

    # Perform daily merge for time series plots
    daily_df = perform_daily_merge(df)

    # Generate all selected visualizations
    print("\n" + "=" * 70)
    print("GENERATING SELECTED VISUALIZATIONS")
    print("=" * 70)

    # Phase 1 EDA plots (raw data)
    plot_3_peak_hours_analysis(df)
    plot_4_branch_performance(df)
    plot_6_net_cash_analysis(df)
    plot_7_hourly_boxplot(df)

    # Phase 2 Cash Analysis plots (raw data)
    plot_10_branch_net_cash_position(df)
    plot_11_branch_hour_heatmap(df)

    # Time Series plots (daily aggregated data)
    plot_ts4_weekly_seasonal_pattern(daily_df)
    plot_ts5_monthly_trend_analysis(daily_df)
    plot_ts6_cumulative_cash_flow(daily_df)
    plot_ts7_weekend_vs_weekday(daily_df)
    plot_ts8_daily_distribution_analysis(daily_df)

    print("\n" + "=" * 70)
    print("         COMBINED EDA COMPLETED!")
    print("=" * 70)
    print(f"\n[FOLDER] All plots saved to: '{OUTPUT_DIR}/'")
    print("\nGenerated plots (11 selected):")
    print("   3. peak_hours_analysis.png       - Transaction frequency & avg by hour")
    print("   4. branch_performance.png         - Top & bottom branches by volume")
    print("   6. net_cash_analysis.png          - Monthly net cash position")
    print("   7. hourly_boxplot.png             - Transaction distribution by hour")
    print("  10. branch_net_cash_position.png   - Branch-wise surplus/deficit breakdown")
    print("  11. branch_hour_heatmap.png        - Branch × hour net-cash heatmap")
    print("  ts4. weekly_seasonal_pattern.png   - Weekly patterns across days")
    print("  ts5. monthly_trend_analysis.png    - Monthly aggregated trends")
    print("  ts6. cumulative_cash_flow.png      - Cumulative cash flow analysis")
    print("  ts7. weekend_vs_weekday.png        - Weekend vs weekday comparison")
    print("  ts8. daily_distribution_analysis.png - Distribution of daily metrics")


if __name__ == "__main__":
    main()