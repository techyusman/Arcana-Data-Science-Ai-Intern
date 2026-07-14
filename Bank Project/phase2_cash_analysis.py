"""
===============================================================================
BANK BRANCH CASH FORECASTING SYSTEM
Phase 2: Half-Day Cash Requirement & Deficit Analysis
===============================================================================

Generates 5 key analytical plots for cash forecasting:
  1. Half-Day Cash Requirement Analysis
  2. Branch-Wise Net Cash Position
  3. Branch × Hour Net-Cash Heatmap
  4. Daily Net Cash with 7-Day Rolling Average
  5. Branch Deficit Frequency and Severity

Author: Muhammad Usman
Status: Phase 2 Implementation
===============================================================================
"""

# =============================================================================
# STEP 1: IMPORTS & CONFIGURATION
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# STEP 2: LOAD & PREPARE DATA
# =============================================================================

def load_and_prepare_data(file_path):
    """
    Load cleaned data and create derived columns:
      - net_cash = TOTAL_CR - TOTAL_DR (positive = surplus, negative = deficit)
      - half_day_label: 'Morning' (hours 0-11) or 'Afternoon' (hours 12-23)
      - date-only column for daily aggregation
    """
    print("=" * 70)
    print("LOADING & PREPARING DATA")
    print("=" * 70)

    df = pd.read_csv(file_path)
    df['start_date'] = pd.to_datetime(df['start_date'])

    # Net cash: positive = surplus (deposits > withdrawals), negative = deficit
    df['net_cash'] = df['TOTAL_CR'] - df['TOTAL_DR']

    # Half-day definition
    # Morning: 0-11, Afternoon: 12-23
    def get_half_day(hour):
        if 0 <= hour <= 11:
            return 'Morning'
        else:
            return 'Afternoon'

    df['half_day'] = df['txn_hour'].apply(get_half_day)

    # Date-only column (no time component)
    df['date'] = df['start_date'].dt.date

    # Year-month for aggregation
    df['year_month'] = df['start_date'].dt.to_period('M').astype(str)

    print(f"[OK] Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"     Date Range: {df['start_date'].min().date()} to {df['start_date'].max().date()}")
    print(f"     Branches:   {df['tran_br_code'].nunique():,}")
    print(f"     Net Cash:   {df['net_cash'].sum() / 1e6:+,.2f} M")
    print(f"     Half-Days:  {df['half_day'].value_counts().to_dict()}")

    return df


# =============================================================================
# PLOT 1: HALF-DAY CASH REQUIREMENT ANALYSIS
# =============================================================================

def plot_1_half_day_cash_requirement(df):
    """
    Plot 1: Half-Day Cash Requirement Analysis
    Aggregates total cash requirement (net_cash) per half-day slot over time.
    Shows which half-day periods have the highest cash demand.
    """
    print("\n[1/5] Half-Day Cash Requirement Analysis...")

    # Aggregate net cash by date and half-day
    halfday_agg = df.groupby(['date', 'half_day']).agg({
        'net_cash': 'sum',
        'TOTAL_DR': 'sum',
        'TOTAL_CR': 'sum'
    }).reset_index()
    halfday_agg['date'] = pd.to_datetime(halfday_agg['date'])

    # Cash requirement = -net_cash (positive = cash needed, negative = surplus)
    halfday_agg['cash_required'] = -halfday_agg['net_cash']

    # Sort by date
    halfday_agg = halfday_agg.sort_values('date')

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # --- Top: Half-Day Net Cash Over Time ---
    ax1 = axes[0]
    for half_day in ['Morning', 'Afternoon']:
        subset = halfday_agg[halfday_agg['half_day'] == half_day]
        color = '#e67e22' if half_day == 'Morning' else '#2980b9'
        ax1.plot(subset['date'], subset['net_cash'] / 1e6,
                 marker='.', linestyle='-', linewidth=1.2, markersize=4,
                 label=f'{half_day}', color=color, alpha=0.8)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('Half-Day Net Cash Position Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Net Cash (Millions)')
    ax1.legend(title='Half-Day Slot', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # --- Bottom: Total Cash Requirement by Half-Day (aggregated) ---
    ax2 = axes[1]
    halfday_totals = df.groupby('half_day').agg({
        'net_cash': 'sum',
        'TOTAL_DR': 'sum',
        'TOTAL_CR': 'sum'
    })
    # Cash requirement (positive = need cash)
    halfday_totals['cash_required'] = -halfday_totals['net_cash']

    bars = ax2.bar(halfday_totals.index, halfday_totals['cash_required'] / 1e6,
                   color=['#e67e22', '#2980b9'], edgecolor='white', alpha=0.85, width=0.5)
    ax2.set_title('Total Cash Requirement by Half-Day Slot', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Total Cash Required (Millions)')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, halfday_totals['cash_required'] / 1e6):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{val:+,.2f}M', ha='center', va='bottom' if val >= 0 else 'top',
                 fontsize=11, fontweight='bold')

    # Add DR/CR breakdown annotation
    for i, half_day in enumerate(halfday_totals.index):
        dr_val = halfday_totals.loc[half_day, 'TOTAL_DR'] / 1e6
        cr_val = halfday_totals.loc[half_day, 'TOTAL_CR'] / 1e6
        ax2.annotate(f'DR: {dr_val:,.2f}M | CR: {cr_val:,.2f}M',
                     xy=(i, 0), xytext=(i, -halfday_totals['cash_required'].max() / 1e6 * 0.12),
                     ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/9_half_day_cash_requirement.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved -> 9_half_day_cash_requirement.png")


# =============================================================================
# PLOT 2: BRANCH-WISE NET CASH POSITION
# =============================================================================

def plot_2_branch_net_cash_position(df):
    """
    Plot 2: Branch-Wise Net Cash Position
    Shows net cash position per branch: surplus (green) or deficit (red).
    """
    print("[2/5] Branch-Wise Net Cash Position...")

    # Aggregate net cash by branch
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

    # --- Top: All branches net cash bar chart ---
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

    # Color the y=0 line
    ax1.axhline(y=0, color='black', linewidth=1.2)

    # Add summary stats on the chart
    surplus_branches = (branch_net['net_cash'] >= 0).sum()
    deficit_branches = (branch_net['net_cash'] < 0).sum()
    ax1.text(0.02, 0.95, f'Surplus: {surplus_branches} | Deficit: {deficit_branches}',
             transform=ax1.transAxes, fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top')

    # --- Bottom: Top 10 deficit and surplus branches ---
    ax2 = axes[1]

    # Top 10 most negative (worst deficit)
    top_deficit = branch_net.head(10)
    # Top 10 most positive (largest surplus)
    top_surplus = branch_net.tail(10).iloc[::-1]

    # Combine for a focused view
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

    # Add value labels
    for bar, val in zip(bars2, plot_data['net_cash_m']):
        label_x = bar.get_width() - (0.5 if val > 0 else -0.5)
        ax2.text(label_x, bar.get_y() + bar.get_height() / 2.,
                 f'{val:+,.2f}M', va='center',
                 ha='left' if val > 0 else 'right', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_branch_net_cash_position.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved -> 10_branch_net_cash_position.png")


# =============================================================================
# PLOT 3: BRANCH × HOUR NET-CASH HEATMAP
# =============================================================================

def plot_3_branch_hour_heatmap(df):
    """
    Plot 3: Branch × Hour Net-Cash Heatmap
    Heatmap showing average net cash position for each branch at each hour,
    revealing patterns of when specific branches have surplus/deficit.
    """
    print("[3/5] Branch × Hour Net-Cash Heatmap...")

    # Pivot table: branches vs hours, average net cash
    pivot = df.pivot_table(
        values='net_cash',
        index='tran_br_code',
        columns='txn_hour',
        aggfunc='mean'
    ) / 1e6  # Convert to millions

    fig, ax = plt.subplots(figsize=(18, max(10, pivot.shape[0] * 0.3)))

    # Create heatmap
    cmap = sns.diverging_palette(10, 130, s=80, l=55, center='light', as_cmap=True)
    sns.heatmap(pivot, cmap=cmap, center=0, annot=False,
                fmt='.2f', linewidths=0.3, linecolor='gray',
                cbar_kws={'label': 'Avg Net Cash (Millions)', 'shrink': 0.8},
                ax=ax)

    ax.set_title('Branch × Hour Net-Cash Heatmap\n(Average Net Cash in Millions — Green=Surplus, Red=Deficit)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Branch Code', fontsize=12)

    # Highlight potential half-day split
    ax.axvline(x=11.5, color='white', linestyle='--', linewidth=2, alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/11_branch_hour_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved -> 11_branch_hour_heatmap.png")


# =============================================================================
# PLOT 4: DAILY NET CASH WITH 7-DAY ROLLING AVERAGE
# =============================================================================

def plot_4_daily_net_cash_rolling_avg(df):
    """
    Plot 4: Daily Net Cash with 7-Day Rolling Average
    Shows daily net cash position with a smoothed rolling average trendline.
    """
    print("[4/5] Daily Net Cash with 7-Day Rolling Average...")

    # Aggregate by date
    daily = df.groupby('date').agg({
        'net_cash': 'sum',
        'TOTAL_DR': 'sum',
        'TOTAL_CR': 'sum'
    }).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date')

    # 7-day rolling average of net cash
    daily['rolling_7d'] = daily['net_cash'].rolling(window=7, min_periods=3).mean()

    # Calculate cumulative net cash
    daily['cumulative_net'] = daily['net_cash'].cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # --- Top: Daily Net Cash + 7-Day Rolling Average ---
    ax1 = axes[0]

    # Color bars by surplus/deficit
    colors_daily = ['#2ecc71' if x >= 0 else '#e74c3c' for x in daily['net_cash']]
    ax1.bar(daily['date'], daily['net_cash'] / 1e6, color=colors_daily,
            alpha=0.5, width=0.8, label='Daily Net Cash', edgecolor='none')

    # Rolling average line
    ax1.plot(daily['date'], daily['rolling_7d'] / 1e6,
             color='#2c3e50', linewidth=2.5, marker='', linestyle='-',
             label='7-Day Rolling Avg', zorder=5)

    # Highlight the trend region
    ax1.fill_between(daily['date'], daily['rolling_7d'] / 1e6, 0,
                     where=(daily['rolling_7d'] >= 0),
                     alpha=0.1, color='#2ecc71')
    ax1.fill_between(daily['date'], daily['rolling_7d'] / 1e6, 0,
                     where=(daily['rolling_7d'] < 0),
                     alpha=0.1, color='#e74c3c')

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_title('Daily Net Cash Position with 7-Day Rolling Average', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Net Cash (Millions)')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Format x-axis dates
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Annotate the overall trend direction
    recent_trend = daily['rolling_7d'].dropna()
    if len(recent_trend) > 0:
        last_val = recent_trend.iloc[-1]
        # Determine trend direction
        if len(recent_trend) >= 7:
            previous_week_avg = recent_trend.iloc[-7:-1].mean()
            trend_dir = "Improving" if last_val > previous_week_avg else "Declining"
        else:
            trend_dir = "Stable"
        ax1.text(0.02, 0.95, f'Trend: {trend_dir} | Latest 7D Avg: {last_val / 1e6:+,.2f}M',
                 transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- Cumulative Net Cash overlay on same axes ---
    ax2 = axes[0].twinx()
    ax2.plot(daily['date'], daily['cumulative_net'] / 1e6,
             color='purple', linewidth=2, linestyle='--', alpha=0.7,
             label='Cumulative Net Cash')
    ax2.set_ylabel('Cumulative Net Cash (Millions)', color='purple', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.legend(loc='lower left', fontsize=10)

    # --- Bottom subplot: 7-day rolling components (DR and CR separately) ---
    ax3 = axes[1]
    daily['rolling_dr_7d'] = daily['TOTAL_DR'].rolling(window=7, min_periods=3).mean()
    daily['rolling_cr_7d'] = daily['TOTAL_CR'].rolling(window=7, min_periods=3).mean()

    ax3.plot(daily['date'], daily['rolling_dr_7d'] / 1e6,
             color='#e74c3c', linewidth=2, label='7D Avg Withdrawals (DR)')
    ax3.plot(daily['date'], daily['rolling_cr_7d'] / 1e6,
             color='#2ecc71', linewidth=2, label='7D Avg Deposits (CR)')

    # Fill between DR and CR to show gap
    ax3.fill_between(daily['date'],
                     daily['rolling_dr_7d'] / 1e6,
                     daily['rolling_cr_7d'] / 1e6,
                     alpha=0.15, color='#3498db',
                     label='Deposit/Withdrawal Gap')

    ax3.set_title('7-Day Rolling Average: Withdrawals vs Deposits', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Amount (Millions)')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Annotate the average gap
    avg_gap = (daily['rolling_cr_7d'] - daily['rolling_dr_7d']).mean()
    ax3.text(0.02, 0.95, f'Avg Daily Gap (CR - DR): {avg_gap / 1e6:+,.2f}M',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/12_daily_net_cash_rolling_avg.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved -> 12_daily_net_cash_rolling_avg.png")


# =============================================================================
# PLOT 5: BRANCH DEFICIT FREQUENCY AND SEVERITY
# =============================================================================

def plot_5_branch_deficit_analysis(df):
    """
    Plot 5: Branch Deficit Frequency and Severity
    For each branch, calculate:
      - Deficit Frequency: % of transactions where withdrawals > deposits
      - Deficit Severity: Average deficit amount when deficit occurs
    """
    print("[5/5] Branch Deficit Frequency and Severity...")

    # Identify deficit transactions
    df['is_deficit'] = df['net_cash'] < 0
    df['deficit_amount'] = df['net_cash'].clip(upper=0)  # Negative values only

    # Aggregate by branch
    branch_deficit = df.groupby('tran_br_code').agg(
        total_txns=('net_cash', 'count'),
        deficit_txns=('is_deficit', 'sum'),
        total_deficit_amount=('deficit_amount', 'sum'),
        avg_net_cash=('net_cash', 'mean'),
        total_dr=('TOTAL_DR', 'sum'),
        total_cr=('TOTAL_CR', 'sum')
    ).reset_index()

    # Calculate metrics
    branch_deficit['deficit_freq_pct'] = (branch_deficit['deficit_txns'] / branch_deficit['total_txns']) * 100
    # Average deficit severity (only when deficit occurs)
    branch_deficit['avg_deficit_severity'] = (
        branch_deficit['total_deficit_amount'] / branch_deficit['deficit_txns']
    ) / 1e6  # Millions

    # Classify branches
    def classify_branch(row):
        freq = row['deficit_freq_pct']
        sev = row['avg_deficit_severity']
        if freq > 60 and sev < -2:
            return 'High Risk'
        elif freq > 40 and sev < -1:
            return 'Moderate Risk'
        elif freq > 20:
            return 'Low Risk'
        else:
            return 'Stable'

    branch_deficit['risk_category'] = branch_deficit.apply(classify_branch, axis=1)

    # Sort by deficit frequency descending
    branch_deficit = branch_deficit.sort_values('deficit_freq_pct', ascending=False).reset_index(drop=True)
    branch_deficit['rank'] = range(1, len(branch_deficit) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # --- Top-Left: Deficit Frequency (Bar Chart - Top 20) ---
    ax1 = axes[0, 0]
    top20_freq = branch_deficit.head(20)

    colors_freq = ['#e74c3c' if x > 50 else '#e67e22' if x > 30 else '#f1c40f' if x > 15 else '#2ecc71'
                   for x in top20_freq['deficit_freq_pct']]

    bars1 = ax1.barh(range(len(top20_freq)), top20_freq['deficit_freq_pct'],
                     color=colors_freq, edgecolor='white', alpha=0.85, height=0.7)
    ax1.set_yticks(range(len(top20_freq)))
    ax1.set_yticklabels([f"Branch {int(b)}" for b in top20_freq['tran_br_code']])
    ax1.set_title('Top 20 Branches by Deficit Frequency', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Deficit Frequency (% of Transactions)')
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.7, label='50% Threshold')
    ax1.axvline(x=30, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='30% Threshold')
    ax1.legend(fontsize=8)

    # Add value labels
    for bar, val in zip(bars1, top20_freq['deficit_freq_pct']):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2.,
                 f'{val:.1f}%', va='center', fontsize=8)

    # --- Top-Right: Deficit Severity vs Frequency (Scatter) ---
    ax2 = axes[0, 1]
    scatter = ax2.scatter(branch_deficit['deficit_freq_pct'],
                          branch_deficit['avg_deficit_severity'],
                          c=branch_deficit['deficit_freq_pct'],
                          cmap='RdYlGn_r', s=80, alpha=0.7,
                          edgecolors='black', linewidth=0.5)

    # Label notable points (top 5 most severe)
    top_severe = branch_deficit.nsmallest(5, 'avg_deficit_severity')
    for _, row in top_severe.iterrows():
        ax2.annotate(f"{int(row['tran_br_code'])}",
                     (row['deficit_freq_pct'], row['avg_deficit_severity']),
                     xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')

    # Quadrant lines
    ax2.axvline(x=50, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=-2, color='gray', linestyle=':', alpha=0.5)
    ax2.text(55, -1.5, 'High Frequency', fontsize=8, color='gray', alpha=0.6)
    ax2.text(55, -5, 'High Risk Zone', fontsize=9, color='red', alpha=0.4, fontweight='bold')

    ax2.set_title('Deficit Frequency vs Severity\n(Each dot = one branch)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Deficit Frequency (%)')
    ax2.set_ylabel('Avg Deficit Severity (Millions)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Deficit Frequency (%)', shrink=0.8)

    # --- Bottom-Left: Risk Category Distribution ---
    ax3 = axes[1, 0]
    risk_counts = branch_deficit['risk_category'].value_counts()
    risk_order = ['High Risk', 'Moderate Risk', 'Low Risk', 'Stable']
    risk_counts = risk_counts.reindex(risk_order).fillna(0)

    risk_colors = {'High Risk': '#e74c3c', 'Moderate Risk': '#e67e22',
                   'Low Risk': '#f1c40f', 'Stable': '#2ecc71'}
    bar_colors = [risk_colors.get(x, '#95a5a6') for x in risk_counts.index]

    wedges, texts, autotexts = ax3.pie(
        risk_counts.values,
        labels=risk_counts.index,
        autopct='%1.1f%%',
        colors=bar_colors,
        startangle=90,
        explode=[0.05 if x == 'High Risk' else 0 for x in risk_counts.index],
        shadow=True
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    ax3.set_title('Branch Risk Classification', fontsize=13, fontweight='bold')

    # --- Bottom-Right: Combined Deficit Impact (Frequency × Severity) ---
    ax4 = axes[1, 1]

    # Calculate combined risk score
    branch_deficit['risk_score'] = (
        branch_deficit['deficit_freq_pct'] * abs(branch_deficit['avg_deficit_severity'])
    )
    top_risk = branch_deficit.nlargest(15, 'risk_score')

    risk_colors_bar = ['#e74c3c' if x > 200 else '#e67e22' if x > 100 else '#f1c40f'
                       for x in top_risk['risk_score']]

    bars4 = ax4.barh(range(len(top_risk)), top_risk['risk_score'],
                     color=risk_colors_bar, edgecolor='white', alpha=0.85, height=0.7)
    ax4.set_yticks(range(len(top_risk)))
    ax4.set_yticklabels([f"Branch {int(b)}" for b in top_risk['tran_br_code']])
    ax4.set_title('Top 15 Branches by Combined Risk Score\n(Frequency × Severity)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Risk Score (Deficit Freq% × |Avg Severity|)')
    ax4.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars4, top_risk['risk_score']):
        ax4.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2.,
                 f'{val:.0f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/13_branch_deficit_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved -> 13_branch_deficit_analysis.png")

    # Print deficit summary table
    print("\n--- Deficit Summary ---")
    print(f"  Total Branches Analyzed: {len(branch_deficit)}")
    print(f"  High Risk Branches:     {(branch_deficit['risk_category'] == 'High Risk').sum()}")
    print(f"  Moderate Risk Branches: {(branch_deficit['risk_category'] == 'Moderate Risk').sum()}")
    print(f"  Low Risk Branches:      {(branch_deficit['risk_category'] == 'Low Risk').sum()}")
    print(f"  Stable Branches:        {(branch_deficit['risk_category'] == 'Stable').sum()}")
    print(f"  Avg Deficit Frequency:  {branch_deficit['deficit_freq_pct'].mean():.2f}%")
    print(f"  Avg Deficit Severity:   {branch_deficit['avg_deficit_severity'].mean():+,.2f}M")

    return branch_deficit


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_analysis_summary(df):
    """Print a comprehensive summary of the analysis."""
    print("\n" + "=" * 70)
    print("PHASE 2 — CASH REQUIREMENT ANALYSIS SUMMARY")
    print("=" * 70)

    # Overall metrics
    total_dr = df['TOTAL_DR'].sum()
    total_cr = df['TOTAL_CR'].sum()
    net_cash = total_cr - total_dr
    deficit_txns = (df['net_cash'] < 0).sum()
    surplus_txns = (df['net_cash'] >= 0).sum()
    total_txns = len(df)

    print(f"\n[CASH FLOW OVERVIEW]")
    print(f"  Total Withdrawals (DR):  {total_dr / 1e6:,.2f} M")
    print(f"  Total Deposits (CR):     {total_cr / 1e6:,.2f} M")
    print(f"  Net Cash Position:       {net_cash / 1e6:+,.2f} M  ({'Surplus' if net_cash >= 0 else 'Deficit'})")

    print(f"\n[TRANSACTION ANALYSIS]")
    print(f"  Total Transactions:      {total_txns:,}")
    print(f"  Deficit Transactions:    {deficit_txns:,} ({deficit_txns / total_txns * 100:.1f}%)")
    print(f"  Surplus Transactions:    {surplus_txns:,} ({surplus_txns / total_txns * 100:.1f}%)")

    # Half-day breakdown
    print(f"\n[HALF-DAY BREAKDOWN]")
    for half_day in ['Morning', 'Afternoon']:
        subset = df[df['half_day'] == half_day]
        hd_net = subset['net_cash'].sum()
        hd_dr = subset['TOTAL_DR'].sum()
        hd_cr = subset['TOTAL_CR'].sum()
        hd_txns = len(subset)
        print(f"  {half_day}:")
        print(f"    Transactions: {hd_txns:,}")
        print(f"    Withdrawals:  {hd_dr / 1e6:,.2f}M")
        print(f"    Deposits:     {hd_cr / 1e6:,.2f}M")
        print(f"    Net Cash:     {hd_net / 1e6:+,.2f}M")

    # Branch summary
    print(f"\n[BRANCH SUMMARY]")
    print(f"  Total Branches:          {df['tran_br_code'].nunique()}")
    branch_net = df.groupby('tran_br_code')['net_cash'].sum()
    print(f"  Branches in Deficit:     {(branch_net < 0).sum()}")
    print(f"  Branches in Surplus:     {(branch_net >= 0).sum()}")

    # Daily summary
    daily_net = df.groupby('date')['net_cash'].sum()
    print(f"\n[DAILY SUMMARY]")
    print(f"  Total Days:              {len(daily_net)}")
    print(f"  Avg Daily Net Cash:      {daily_net.mean() / 1e6:+,.2f}M")
    print(f"  Best Day:                {daily_net.max() / 1e6:+,.2f}M")
    print(f"  Worst Day:               {daily_net.min() / 1e6:+,.2f}M")

    print("\n" + "=" * 70)
    print("[OK] Phase 2 Analysis Complete — 5 new plots added")
    print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all 5 analyses and generate plots."""
    print("=" * 70)
    print("         BANK BRANCH CASH FORECASTING SYSTEM")
    print("         Phase 2: Cash Requirement & Deficit Analysis")
    print("=" * 70)

    # Load data
    df = load_and_prepare_data(DATA_PATH)

    # Generate the 5 plots
    print("\n" + "=" * 70)
    print("GENERATING ANALYTICAL PLOTS")
    print("=" * 70)

    plot_1_half_day_cash_requirement(df)
    plot_2_branch_net_cash_position(df)
    plot_3_branch_hour_heatmap(df)
    plot_4_daily_net_cash_rolling_avg(df)
    branch_deficit = plot_5_branch_deficit_analysis(df)

    # Summary
    print_analysis_summary(df)

    print("\n" + "=" * 70)
    print("         PHASE 2 — CASH ANALYSIS COMPLETED!")
    print("=" * 70)
    print(f"\n[FOLDER] All plots saved to: '{OUTPUT_DIR}/'")
    print("\nGenerated plots (5 new):")
    print("   9. half_day_cash_requirement.png   — Half-day net cash over time + totals")
    print("  10. branch_net_cash_position.png    — Branch-wise surplus/deficit breakdown")
    print("  11. branch_hour_heatmap.png         — Branch × hour net-cash heatmap")
    print("  12. daily_net_cash_rolling_avg.png  — Daily net cash + 7-day rolling average")
    print("  13. branch_deficit_analysis.png     — Deficit frequency, severity, risk scoring")

    return df


if __name__ == "__main__":
    df = main()