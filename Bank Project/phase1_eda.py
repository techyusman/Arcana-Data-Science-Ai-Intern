"""
===============================================================================
BANK BRANCH CASH FORECASTING SYSTEM
Phase 1: Exploratory Data Analysis (EDA)
===============================================================================

This script performs exploratory data analysis on the cleaned bank dataset:
1. Load cleaned data
2. Generate essential visualizations for insights
3. Save plots for reporting

Author: Muhammad Usman
Status: Phase 1 - EDA
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
# STEP 2: LOAD DATA
# =============================================================================

def load_cleaned_data(file_path):
    """Load the cleaned dataset."""
    print("=" * 70)
    print("LOADING CLEANED DATASET")
    print("=" * 70)

    df = pd.read_csv(file_path)
    df['start_date'] = pd.to_datetime(df['start_date'])

    print(f"[OK] Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Date Range: {df['start_date'].min()} to {df['start_date'].max()}")
    print(f"  Branches: {df['tran_br_code'].nunique()}")
    print(f"  Hours Range: {df['txn_hour'].min()} - {df['txn_hour'].max()}")

    return df


# =============================================================================
# STEP 3: EDA - ALL VISUALIZATIONS
# =============================================================================

def plot_1_cash_flow_over_time(df):
    """
    Plot 1: Cash Flow Over Time
    Shows monthly trends of Total Debits (withdrawals) and Total Credits (deposits).
    """
    print("\n[1/8] Cash Flow Over Time...")

    # Aggregate by month
    df_monthly = df.copy()
    df_monthly['year_month'] = df_monthly['start_date'].dt.to_period('M')
    monthly_agg = df_monthly.groupby('year_month').agg({
        'TOTAL_DR': 'sum',
        'TOTAL_CR': 'sum'
    }).reset_index()
    monthly_agg['year_month'] = monthly_agg['year_month'].astype(str)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(monthly_agg['year_month'], monthly_agg['TOTAL_DR'] / 1e6,
            marker='o', linewidth=2, label='Total Withdrawals (DR)', color='#e74c3c')
    ax.plot(monthly_agg['year_month'], monthly_agg['TOTAL_CR'] / 1e6,
            marker='s', linewidth=2, label='Total Deposits (CR)', color='#2ecc71')

    # Rotate x labels if many months
    if len(monthly_agg) > 12:
        plt.xticks(rotation=45, ha='right')

    ax.set_title('Monthly Cash Flow Trends (Withdrawals vs Deposits)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Amount (Millions)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_cash_flow_trend.png", dpi=150)
    plt.close()
    print("  [OK] Saved")


def plot_2_transaction_distribution(df):
    """
    Plot 2: Distribution of Transaction Amounts
    Histograms with KDE for both Debits and Credits.
    """
    print("[2/8] Transaction Amount Distribution...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # TOTAL_DR distribution
    ax1 = axes[0]
    sns.histplot(df['TOTAL_DR'] / 1e6, bins=50, kde=True, color='#e74c3c', ax=ax1)
    ax1.set_title('Distribution of Withdrawals (TOTAL_DR)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Amount (Millions)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(df['TOTAL_DR'].mean() / 1e6, color='darkred', linestyle='--', linewidth=2, label=f'Mean: {df["TOTAL_DR"].mean()/1e6:.2f}M')
    ax1.legend()

    # TOTAL_CR distribution
    ax2 = axes[1]
    sns.histplot(df['TOTAL_CR'] / 1e6, bins=50, kde=True, color='#2ecc71', ax=ax2)
    ax2.set_title('Distribution of Deposits (TOTAL_CR)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Amount (Millions)')
    ax2.set_ylabel('Frequency')
    ax2.axvline(df['TOTAL_CR'].mean() / 1e6, color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {df["TOTAL_CR"].mean()/1e6:.2f}M')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_transaction_distribution.png", dpi=150)
    plt.close()
    print("  [OK] Saved")


def plot_3_peak_hours_analysis(df):
    """
    Plot 3: Peak Hours Analysis
    Average transaction amounts and frequencies by hour of day.
    """
    print("[3/8] Peak Hours Analysis...")

    # Group by hour
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
    print("  [OK] Saved")


def plot_4_branch_performance(df):
    """
    Plot 4: Branch Performance
    Top and bottom branches by total transaction volume.
    """
    print("[4/8] Branch Performance Analysis...")

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
    # Add value labels
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
    print("  [OK] Saved")


def plot_5_correlation_analysis(df):
    """
    Plot 5: Correlation Analysis
    Scatter plot of Deposits vs Withdrawals with heatmap of correlations.
    """
    print("[5/8] Correlation Analysis...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Scatter: TOTAL_CR vs TOTAL_DR
    ax1 = axes[0]
    # Sample for performance
    sample = df.sample(min(5000, len(df)), random_state=42)
    ax1.scatter(sample['TOTAL_DR'] / 1e6, sample['TOTAL_CR'] / 1e6,
                alpha=0.3, s=10, c='#3498db', edgecolors='none')
    ax1.set_title('Deposits vs Withdrawals (Sample: 5,000 transactions)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Withdrawals (TOTAL_DR) - Millions')
    ax1.set_ylabel('Deposits (TOTAL_CR) - Millions')

    # Add regression line
    z = np.polyfit(sample['TOTAL_DR'] / 1e6, sample['TOTAL_CR'] / 1e6, 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample['TOTAL_DR'].min() / 1e6, sample['TOTAL_DR'].max() / 1e6, 100)
    ax1.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7,
             label=f'Trend (slope={z[0]:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # Correlation heatmap
    ax2 = axes[1]
    corr_cols = ['txn_hour', 'TOTAL_DR', 'TOTAL_CR']
    # Also add derived features for richer correlation
    df_corr = df[corr_cols].copy()
    corr_matrix = df_corr.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax2,
                cbar_kws={'shrink': 0.8})
    ax2.set_title('Correlation Matrix', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_correlation_analysis.png", dpi=150)
    plt.close()
    print("  [OK] Saved")


def plot_6_net_cash_analysis(df):
    """
    Plot 6: Net Cash Analysis
    Net = TOTAL_CR - TOTAL_DR (positive = surplus, negative = deficit).
    Shows net cash position over time.
    """
    print("[6/8] Net Cash Analysis...")

    df_net = df.copy()
    df_net['net_cash'] = df_net['TOTAL_CR'] - df_net['TOTAL_DR']
    df_net['year_month'] = df_net['start_date'].dt.to_period('M')

    monthly_net = df_net.groupby('year_month')['net_cash'].sum().reset_index()
    monthly_net['year_month'] = monthly_net['year_month'].astype(str)

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in monthly_net['net_cash']]
    ax.bar(monthly_net['year_month'], monthly_net['net_cash'] / 1e6, color=colors, edgecolor='white', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title('Monthly Net Cash Position (Deposits - Withdrawals)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Net Cash (Millions)')

    # Rotate labels if many months
    if len(monthly_net) > 12:
        plt.xticks(rotation=45, ha='right')

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/6_net_cash_analysis.png", dpi=150)
    plt.close()
    print("  [OK] Saved")


def plot_7_hourly_boxplot(df):
    """
    Plot 7: Transaction Amount Distribution by Hour (Box Plots)
    Shows how transaction amounts vary across different hours.
    """
    print("[7/8] Hourly Transaction Distribution (Box Plots)...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot for TOTAL_DR by hour
    ax1 = axes[0]
    hour_order = sorted(df['txn_hour'].unique())
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
    print("  [OK] Saved")


def plot_8_weekly_pattern(df):
    """
    Plot 8: Weekly / Day-of-Week Pattern
    Shows transaction patterns across days of the week.
    """
    print("[8/8] Weekly Pattern Analysis...")

    df_wk = df.copy()
    df_wk['day_of_week'] = df_wk['start_date'].dt.dayofweek  # Monday=0, Sunday=6
    df_wk['day_name'] = df_wk['start_date'].dt.day_name()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Use simpler aggregation - count by day_name directly
    day_counts = df_wk.groupby('day_name').size().reset_index(name='txn_count')

    # Averages by day_name
    day_avgs = df_wk.groupby('day_name').agg({
        'TOTAL_DR': 'mean',
        'TOTAL_CR': 'mean'
    }).round(2).reset_index()

    # Merge and ensure day order
    weekday_stats = day_counts.merge(day_avgs, on='day_name')
    weekday_stats['day_name'] = pd.Categorical(weekday_stats['day_name'], categories=day_order, ordered=True)
    weekday_stats = weekday_stats.sort_values('day_name').reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Top: Transaction count by day
    ax1 = axes[0]
    colors_wk = ['#3498db'] * 7
    # Highlight weekend
    colors_wk[5] = '#e67e22'  # Saturday
    colors_wk[6] = '#e67e22'  # Sunday
    bars = ax1.bar(weekday_stats['day_name'], weekday_stats['txn_count'],
                   color=colors_wk, edgecolor='white', alpha=0.8)
    ax1.set_title('Transaction Count by Day of Week', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Day of Week')
    ax1.set_ylabel('Number of Transactions')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height):,}', ha='center', va='bottom', fontsize=9)

    # Bottom: Average amounts by day
    ax2 = axes[1]
    ax2.plot(weekday_stats['day_name'], weekday_stats['TOTAL_DR'] / 1e6,
             marker='o', linewidth=2, markersize=8, label='Avg Withdrawal', color='#e74c3c')
    ax2.plot(weekday_stats['day_name'], weekday_stats['TOTAL_CR'] / 1e6,
             marker='s', linewidth=2, markersize=8, label='Avg Deposit', color='#2ecc71')
    ax2.set_title('Average Transaction Amount by Day of Week', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Average Amount (Millions)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/8_weekly_pattern.png", dpi=150)
    plt.close()
    print("  [OK] Saved")


# =============================================================================
# STEP 4: GENERATE EDA SUMMARY REPORT
# =============================================================================

def generate_summary_report(df):
    """Print a comprehensive EDA summary to console."""
    print("\n" + "=" * 70)
    print("EDA SUMMARY REPORT")
    print("=" * 70)

    # Basic stats
    print("\n[DATASET OVERVIEW]")
    print(f"   - Transactions: {df.shape[0]:,}")
    print(f"   - Branches:     {df['tran_br_code'].nunique():,}")
    print(f"   - Date Range:   {df['start_date'].min().date()} to {df['start_date'].max().date()}")
    print(f"   - Hours:        {int(df['txn_hour'].min())}:00 - {int(df['txn_hour'].max())}:00")

    # Cash flow totals
    total_dr = df['TOTAL_DR'].sum()
    total_cr = df['TOTAL_CR'].sum()
    net = total_cr - total_dr
    print(f"\n[CASH FLOW SUMMARY]")
    print(f"   - Total Withdrawals (DR): {total_dr:,.2f} ({total_dr/1e6:.2f}M)")
    print(f"   - Total Deposits (CR):    {total_cr:,.2f} ({total_cr/1e6:.2f}M)")
    net_status = "Surplus" if net >= 0 else "Deficit"
    print(f"   - Net Cash Position:      {net:,.2f} ({net/1e6:.2f}M) [{net_status}]")

    # Per-transaction averages
    print(f"\n[PER-TRANSACTION AVERAGES]")
    print(f"   - Avg Withdrawal: {df['TOTAL_DR'].mean():,.2f}")
    print(f"   - Avg Deposit:    {df['TOTAL_CR'].mean():,.2f}")
    print(f"   - Median Withdrawal: {df['TOTAL_DR'].median():,.2f}")
    print(f"   - Median Deposit:    {df['TOTAL_CR'].median():,.2f}")

    # Busiest hours
    peak_hour = df.groupby('txn_hour').size().idxmax()
    peak_hour_count = df.groupby('txn_hour').size().max()
    print(f"\n[PEAK HOURS]")
    print(f"   - Busiest Hour:  {int(peak_hour)}:00 ({peak_hour_count:,} transactions)")

    # Busiest day
    df_copy = df.copy()
    df_copy['day_name'] = df_copy['start_date'].dt.day_name()
    busiest_day = df_copy.groupby('day_name').size().idxmax()
    print(f"   - Busiest Day:   {busiest_day}")

    # Top branch
    branch_vol = df.groupby('tran_br_code')['TOTAL_DR'].sum()
    top_branch = branch_vol.idxmax()
    top_branch_val = branch_vol.max()
    print(f"\n[BRANCH INSIGHTS]")
    print(f"   - Highest Withdrawal Branch: {int(top_branch)} ({top_branch_val:,.2f})")

    # Correlation
    corr = df['TOTAL_DR'].corr(df['TOTAL_CR'])
    print(f"\n[CORRELATION]")
    print(f"   - DR vs CR Correlation: {corr:.4f}")

    print("\n" + "=" * 70)
    print("[OK] EDA Summary Complete - All plots saved to:", OUTPUT_DIR)
    print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all EDA steps."""
    print("=" * 70)
    print("         BANK BRANCH CASH FORECASTING SYSTEM")
    print("             Phase 1: Exploratory Data Analysis")
    print("=" * 70)

    # Load data
    df = load_cleaned_data(DATA_PATH)

    # Run visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    print()

    plot_1_cash_flow_over_time(df)
    plot_2_transaction_distribution(df)
    plot_3_peak_hours_analysis(df)
    plot_4_branch_performance(df)
    plot_5_correlation_analysis(df)
    plot_6_net_cash_analysis(df)
    plot_7_hourly_boxplot(df)
    plot_8_weekly_pattern(df)

    # Summary report
    generate_summary_report(df)

    print("\n" + "=" * 70)
    print("         PHASE 1 - EDA COMPLETED!")
    print("=" * 70)
    print(f"\n[FOLDER] All plots saved to: '{OUTPUT_DIR}/'")
    print("\nGenerated plots:")
    print("  1. cash_flow_trend.png      - Monthly withdrawals vs deposits trend")
    print("  2. transaction_distribution.png - Distribution of transaction amounts")
    print("  3. peak_hours_analysis.png   - Transaction frequency & avg by hour")
    print("  4. branch_performance.png    - Top & bottom branches by volume")
    print("  5. correlation_analysis.png  - DR vs CR scatter + correlation matrix")
    print("  6. net_cash_analysis.png     - Monthly net cash position")
    print("  7. hourly_boxplot.png        - Transaction distribution by hour")
    print("  8. weekly_pattern.png        - Day-of-week transaction patterns")


if __name__ == "__main__":
    main()