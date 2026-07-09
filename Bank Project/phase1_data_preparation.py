"""
================================================================================
BANK BRANCH CASH FORECASTING SYSTEM
Phase 1: Data Understanding & Preparation
================================================================================

This script handles Phase 1 of the project:
1. Load and examine the banking dataset
2. Clean and preprocess the data
3. Perform Exploratory Data Analysis (EDA)
4. Integrate external features (holidays, weekends, etc.)
5. Generate clean, analysis-ready dataset

Author: Muhammad Usman
Status: Phase 1 Implementation
================================================================================
"""

# =============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# =============================================================================
# Libraries are like toolboxes - each provides different tools for data analysis

import pandas as pd  # For data manipulation and analysis (like Excel for Python)
import numpy as np  # For numerical calculations and arrays
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (saves plots without opening windows)
import matplotlib.pyplot as plt  # For creating visual charts and graphs
import seaborn as sns  # For making beautiful statistical plots
from datetime import datetime, timedelta  # For working with dates and times
import warnings  # To handle warning messages
import sys  # For system-level settings
warnings.filterwarnings('ignore')  # Hide unimportant warning messages
sys.stdout.reconfigure(encoding='utf-8')  # Fix Unicode display on Windows

# =============================================================================
# STEP 2: CONFIGURATION SETTINGS
# =============================================================================
# These settings help us customize our analysis

# Set plotting style for better-looking charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define file paths (where our data is stored)
DATASET_PATH = "../Bank DataSet/Bank Cash Optimization.xlsx"  # Input data file
OUTPUT_PATH = "../Bank DataSet/cleaned_bank_data.csv"  # Where we'll save clean data

# =============================================================================
# STEP 3: LOAD THE DATASET
# =============================================================================

def load_dataset(file_path):
    """
    This function loads the Excel file into a pandas DataFrame.
    DataFrame is like a table in Python - it has rows and columns.
    
    Parameters:
        file_path (str): Path to the Excel file
        
    Returns:
        DataFrame: The loaded data
    """
    print("="*70)
    print("STEP 1: LOADING DATASET")
    print("="*70)
    
    try:
        # Read Excel file - we'll try different sheets if needed
        df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
        
        # If multiple sheets exist, combine them or use the first one
        if isinstance(df, dict):
            print(f"Found {len(df)} sheets: {list(df.keys())}")
            # Use the first sheet for now
            sheet_name = list(df.keys())[0]
            df = df[sheet_name]
            print(f"Using sheet: '{sheet_name}'")
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
    
    except FileNotFoundError:
        print(f"✗ Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading file: {str(e)}")
        return None

# =============================================================================
# STEP 4: UNDERSTAND THE DATA
# =============================================================================

def explore_data_structure(df):
    """
    This function explores the basic structure of our dataset.
    It tells us what columns we have, their data types, and gives first few rows.
    
    Parameters:
        df (DataFrame): The dataset to explore
    """
    print("\n" + "="*70)
    print("STEP 2: EXPLORING DATA STRUCTURE")
    print("="*70)
    
    # 1. Show column names and their data types
    print("\n📋 DATA TYPES:")
    print("-" * 70)
    print(df.dtypes)
    
    # 2. Show first 5 rows to see what the data looks like
    print("\n📊 FIRST 5 ROWS OF DATA:")
    print("-" * 70)
    print(df.head())
    
    # 3. Show last 5 rows
    print("\n📊 LAST 5 ROWS OF DATA:")
    print("-" * 70)
    print(df.tail())
    
    # 4. Get basic statistics for numerical columns
    print("\n📈 NUMERICAL COLUMNS STATISTICS:")
    print("-" * 70)
    print(df.describe())
    
    # 5. Check for missing values
    print("\n❓ MISSING VALUES CHECK:")
    print("-" * 70)
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    # Only show columns with missing values
    missing_info = pd.DataFrame({
        'Missing Count': missing_data,
        'Percentage': missing_percentage
    })
    missing_info = missing_info[missing_info['Missing Count'] > 0]
    
    if len(missing_info) > 0:
        print(missing_info)
    else:
        print("✓ No missing values found!")

# =============================================================================
# STEP 5: UNDERSTAND COLUMNS MEANING
# =============================================================================

def create_data_dictionary(df):
    """
    This function creates a data dictionary explaining what each column means.
    A data dictionary is like a user manual for your dataset.
    
    Parameters:
        df (DataFrame): The dataset
        
    Returns:
        DataFrame: Data dictionary with column information
    """
    print("\n" + "="*70)
    print("STEP 3: CREATING DATA DICTIONARY")
    print("="*70)
    
    # Create a dictionary explaining each column
    column_info = {}
    
    for column in df.columns:
        column_info[column] = {
            'Data Type': str(df[column].dtype),
            'Unique Values': df[column].nunique(),
            'Missing Values': df[column].isnull().sum(),
            'Sample Values': str(df[column].dropna().head(3).tolist())
        }
    
    # Convert to DataFrame for better display
    data_dict = pd.DataFrame(column_info).T
    print("\n📚 DATA DICTIONARY:")
    print(data_dict)
    
    return data_dict

# =============================================================================
# STEP 6: CLEAN DATE/TIME COLUMNS
# =============================================================================

def clean_datetime_columns(df):
    """
    This function identifies and converts date/time columns to proper datetime format.
    Proper datetime format allows us to do time-based analysis.
    
    Parameters:
        df (DataFrame): The dataset
        
    Returns:
        DataFrame: Dataset with cleaned datetime columns
    """
    print("\n" + "="*70)
    print("STEP 4: CLEANING DATE/TIME COLUMNS")
    print("="*70)
    
    # List of common date column names
    date_columns = ['date', 'datetime', 'timestamp', 'time', 'Date', 'Date Time', 
                    'Transaction Date', 'Timestamp']
    
    found_date_cols = []
    
    # Find columns that might be date columns
    for col in df.columns:
        # Check if column name suggests it's a date column
        if any(date_keyword in col.lower() for date_keyword in date_columns):
            try:
                # Try to convert to datetime
                df[col] = pd.to_datetime(df[col])
                found_date_cols.append(col)
                print(f"✓ Converted '{col}' to datetime format")
            except:
                print(f"✗ Could not convert '{col}' to datetime")
    
    if len(found_date_cols) == 0:
        print("⚠ No standard date columns found. Looking for date-like patterns...")
        
        # Try to find columns with date-like values
        for col in df.columns:
            try:
                sample = df[col].dropna().head(5)
                if sample.dtype == 'object':
                    # Try parsing
                    test_parse = pd.to_datetime(sample, errors='coerce')
                    if test_parse.notna().sum() > 3:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        found_date_cols.append(col)
                        print(f"✓ Converted '{col}' to datetime format")
            except:
                pass
    
    print(f"\n✓ Found {len(found_date_cols)} date/time column(s)")
    return df

# =============================================================================
# STEP 7: HANDLE MISSING VALUES
# =============================================================================

def handle_missing_values(df):
    """
    This function handles missing values in the dataset.
    We can: remove them, fill them with zeros, or fill with average values.
    
    Parameters:
        df (DataFrame): The dataset
        
    Returns:
        DataFrame: Dataset with handled missing values
    """
    print("\n" + "="*70)
    print("STEP 5: HANDLING MISSING VALUES")
    print("="*70)
    
    initial_rows = len(df)
    
    # Strategy 1: For numerical columns, fill with 0 or mean
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            # If it's cash/amount column, fill with 0 (no transaction occurred)
            if any(keyword in col.lower() for keyword in ['cash', 'deposit', 'withdrawal', 'balance', 'amount']):
                df[col].fillna(0, inplace=True)
                print(f"✓ Filled missing values in '{col}' with 0")
            else:
                # For other numerical columns, use median
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"✓ Filled missing values in '{col}' with median: {median_val:.2f}")
    
    # Strategy 2: For categorical columns, fill with 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna('Unknown', inplace=True)
            print(f"✓ Filled missing values in '{col}' with 'Unknown'")
    
    # Strategy 3: Remove rows where critical columns are still missing
    critical_cols = [col for col in df.columns if 'branch' in col.lower() or 'date' in col.lower()]
    
    if critical_cols:
        df.dropna(subset=critical_cols, inplace=True)
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            print(f"✓ Removed {rows_removed} rows with missing critical data")
    
    print(f"\n✓ Missing value handling complete!")
    print(f"  - Final dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df

# =============================================================================
# STEP 8: REMOVE DUPLICATES
# =============================================================================

def remove_duplicates(df):
    """
    This function removes duplicate rows from the dataset.
    Duplicates can skew our analysis.
    
    Parameters:
        df (DataFrame): The dataset
        
    Returns:
        DataFrame: Dataset with duplicates removed
    """
    print("\n" + "="*70)
    print("STEP 6: REMOVING DUPLICATES")
    print("="*70)
    
    initial_rows = len(df)
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Found {duplicates} duplicate rows")
    
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        rows_removed = initial_rows - len(df)
        print(f"✓ Removed {rows_removed} duplicate rows")
    
    print(f"✓ Final dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df

# =============================================================================
# STEP 9: GENERATE EXTERNAL FEATURES
# =============================================================================

def generate_external_features(df):
    """
    This function adds useful features from dates, like:
    - Day of week (Monday, Tuesday, etc.)
    - Weekend indicator (Yes/No)
    - Month, Quarter, Year
    - Holiday information
    
    Parameters:
        df (DataFrame): The dataset with date column
        
    Returns:
        DataFrame: Dataset with new features
    """
    print("\n" + "="*70)
    print("STEP 7: GENERATING EXTERNAL FEATURES")
    print("="*70)
    
    # Find the date column (usually the first datetime column)
    date_cols = df.select_dtypes(include=['datetime64']).columns
    
    if len(date_cols) == 0:
        print("⚠ No date column found. Skipping feature generation.")
        return df
    
    date_col = date_cols[0]  # Use the first date column
    print(f"Using date column: '{date_col}'")
    
    # Extract basic date features
    print("\n📅 Generating date features...")
    
    df['Year'] = df[date_col].dt.year  # Year (2024, 2025, etc.)
    df['Month'] = df[date_col].dt.month  # Month number (1-12)
    df['Day'] = df[date_col].dt.day  # Day of month (1-31)
    df['Week'] = df[date_col].dt.isocalendar().week  # Week number (1-52)
    df['Quarter'] = df[date_col].dt.quarter  # Quarter (1-4)
    
    # Day of week: Monday=0, Sunday=6
    df['DayOfWeek'] = df[date_col].dt.dayofweek
    
    # Weekend indicator: 1 if weekend (Saturday/Sunday), 0 otherwise
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Half-day indicator (morning/afternoon)
    # Check for existing hour/time columns first (e.g., txn_hour)
    hour_cols = [col for col in df.columns if 'hour' in col.lower() or 'time' in col.lower()]
    
    if hour_cols and df[hour_cols[0]].dtype in ['int64', 'float64']:
        # Use existing hour column (like txn_hour)
        df['Hour'] = df[hour_cols[0]]
        print(f"  Using existing hour column: '{hour_cols[0]}'")
    elif 'Hour' not in df.columns:
        # Try to extract hour from datetime column
        try:
            df['Hour'] = df[date_col].dt.hour
        except:
            df['Hour'] = 12  # Default to noon if no time info
    
    # Morning (6-14) vs Afternoon (14-22) vs Night
    df['HalfDay'] = df['Hour'].apply(
        lambda x: 'Morning' if 6 <= x < 14 else ('Afternoon' if 14 <= x < 22 else 'Night')
    )
    
    # Day name
    df['DayName'] = df[date_col].dt.day_name()
    
    # Date features generated
    print("✓ Year")
    print("✓ Month")
    print("✓ Day")
    print("✓ Week")
    print("✓ Quarter")
    print("✓ Day of Week")
    print("✓ Weekend Indicator")
    print("✓ Half-Day Slot")
    print("✓ Day Name")
    
    # Add Pakistani holidays (simplified version)
    print("\n🎉 Adding holiday features...")
    
    # Common Pakistani public holidays (dates)
    pakistan_holidays = [
        # New Year
        '01-01',
        # Pakistan Day (March 23)
        '03-23',
        # Labour Day (May 1)
        '05-01',
        # Independence Day (August 14)
        '08-14',
        # Defence Day (September 6)
        '09-06',
        # Iqbal Day (November 9)
        '11-09',
        # Quaid-e-Azam Day (December 25)
        '12-25',
        # Eid-ul-Fitr (approximate - varies by moon sighting)
        '03-30', '03-31', '04-01',  # Example dates for 2024
        # Eid-ul-Adha (approximate)
        '06-16', '06-17', '06-18',  # Example dates
    ]
    
    # Mark holidays
    df['IsHoliday'] = df[date_col].apply(
        lambda x: 1 if x.strftime('%m-%d') in pakistan_holidays else 0
    )
    
    print("✓ Holiday indicators added")
    
    # Salary cycle approximation (first week of month typically salary)
    df['IsSalaryWeek'] = df['Day'].apply(lambda x: 1 if x <= 7 else 0)
    print("✓ Salary week indicator added")
    
    new_features = [c for c in df.columns if c not in [date_col, 'txn_hour', 'tran_br_code', 'TOTAL_DR', 'TOTAL_CR']]
    print(f"\n✓ Generated {len(new_features)} new features: {new_features}")
    
    return df

# =============================================================================
# STEP 10: HANDLE OUTLIERS
# =============================================================================

def handle_outliers(df):
    """
    This function detects and handles outliers in numerical columns.
    Outliers are values that are too high or too low compared to normal values.
    
    Parameters:
        df (DataFrame): The dataset
        
    Returns:
        DataFrame: Dataset with outliers handled
    """
    print("\n" + "="*70)
    print("STEP 8: HANDLING OUTLIERS")
    print("="*70)
    
    # Find numerical columns that might have outliers
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Focus on cash-related columns
    cash_cols = [col for col in numerical_cols if any(
        keyword in col.lower() for keyword in ['cash', 'deposit', 'withdrawal', 'balance', 'amount',
                                                'total_dr', 'total_cr', 'dr', 'cr', 'debit', 'credit']
    )]
    
    outlier_counts = {}
    
    for col in cash_cols:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[col].quantile(0.25)  # 25% of data is below this
        Q3 = df[col].quantile(0.75)  # 75% of data is below this
        
        # Calculate Interquartile Range (IQR)
        IQR = Q3 - Q1
        
        # Define outlier boundaries
        lower_bound = Q1 - 1.5 * IQR  # Too low outliers
        upper_bound = Q3 + 1.5 * IQR  # Too high outliers
        
        # Find outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if len(outliers) > 0:
            outlier_counts[col] = len(outliers)
            print(f"⚠ Column '{col}': {len(outliers)} outliers detected")
            
            # Cap outliers at boundaries instead of removing them
            # This preserves data while reducing extreme effects
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    if len(outlier_counts) > 0:
        print("\n✓ Outliers capped at reasonable boundaries")
    else:
        print("✓ No significant outliers found")
    
    return df

# =============================================================================
# STEP 11: DATA VALIDATION
# =============================================================================

def validate_data(df):
    """
    This function validates the data to ensure it makes business sense.
    Example: Closing balance = Opening balance + Deposits - Withdrawals
    
    Parameters:
        df (DataFrame): The dataset
        
    Returns:
        DataFrame: Validated dataset
    """
    print("\n" + "="*70)
    print("STEP 9: DATA VALIDATION")
    print("="*70)
    
    # Try to find relevant columns
    balance_cols = [col for col in df.columns if 'balance' in col.lower()]
    deposit_cols = [col for col in df.columns if 'deposit' in col.lower()]
    withdrawal_cols = [col for col in df.columns if 'withdrawal' in col.lower()]
    
    if balance_cols and deposit_cols and withdrawal_cols:
        print("✓ Found balance, deposit, and withdrawal columns")
        
        # Basic validation: ensure no negative balances
        for col in balance_cols:
            if (df[col] < 0).sum() > 0:
                print(f"⚠ {col} has {(df[col] < 0).sum()} negative values (clipped to 0)")
                df[col] = df[col].clip(lower=0)
    
    # Check for logical errors
    print("\n✓ Data validates successfully")
    
    return df

# =============================================================================
# STEP 12: GENERATE SUMMARY STATISTICS
# =============================================================================

def generate_summary_report(df, original_df):
    """
    This function generates a comprehensive summary report of the data.
    
    Parameters:
        df (DataFrame): Cleaned dataset
        original_df (DataFrame): Original dataset (before cleaning)
    """
    print("\n" + "="*70)
    print("STEP 10: SUMMARY REPORT")
    print("="*70)
    
    print("\n📊 DATASET COMPARISON:")
    print("-" * 70)
    print(f"Original dataset: {original_df.shape[0]} rows × {original_df.shape[1]} columns")
    print(f"Cleaned dataset:  {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Rows removed:     {original_df.shape[0] - df.shape[0]}")
    print(f"Features added:   {df.shape[1] - original_df.shape[1]}")
    
    print("\n📊 NEW FEATURES SUMMARY:")
    print("-" * 70)
    
    # Show value counts for categorical features
    if 'IsWeekend' in df.columns:
        print("\nWeekend vs Weekday distribution:")
        print(df['IsWeekend'].value_counts())
    
    if 'HalfDay' in df.columns:
        print("\nHalf-Day distribution:")
        print(df['HalfDay'].value_counts())
    
    if 'IsHoliday' in df.columns:
        print("\nHoliday distribution:")
        print(df['IsHoliday'].value_counts())
    
    if 'Month' in df.columns:
        print("\nMonthly distribution:")
        print(df['Month'].value_counts().sort_index())

# =============================================================================
# STEP 13: VISUALIZATION (EDA)
# =============================================================================

def create_visualizations(df):
    """
    This function creates visual charts to understand the data better.
    
    Parameters:
        df (DataFrame): The cleaned dataset
    """
    print("\n" + "="*70)
    print("STEP 11: CREATING VISUALIZATIONS")
    print("="*70)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Bank Cash Data - Exploratory Data Analysis', fontsize=16, fontweight='bold')
    
    # Find withdrawal/debit column
    dr_col = next((c for c in df.columns if c in ['Withdrawal', 'Cash Withdrawal', 'TOTAL_DR']), None)
    # Find deposit/credit column  
    cr_col = next((c for c in df.columns if c in ['Deposit', 'Cash Deposit', 'TOTAL_CR']), None)
    
    # Plot 1: Cash Withdrawals/Debits Distribution
    if dr_col:
        axes[0, 0].hist(df[dr_col], bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title(f'{dr_col} Distribution')
        axes[0, 0].set_xlabel('Amount')
        axes[0, 0].set_ylabel('Frequency')
    
    # Plot 2: Cash Deposits/Credits Distribution
    if cr_col:
        axes[0, 1].hist(df[cr_col], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title(f'{cr_col} Distribution')
        axes[0, 1].set_xlabel('Amount')
        axes[0, 1].set_ylabel('Frequency')
    
    # Plot 3: Weekend vs Weekday
    if 'IsWeekend' in df.columns:
        weekend_counts = df['IsWeekend'].value_counts()
        axes[0, 2].pie(weekend_counts.values, labels=['Weekday', 'Weekend'], 
                       autopct='%1.1f%%', colors=['lightblue', 'orange'])
        axes[0, 2].set_title('Weekend vs Weekday')
    
    # Plot 4: Monthly Trends
    if 'Month' in df.columns:
        monthly_data = df.groupby('Month').size()
        axes[1, 0].bar(monthly_data.index, monthly_data.values, color='teal')
        axes[1, 0].set_title('Transactions by Month')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Count')
    
    # Plot 5: Half-Day Distribution
    if 'HalfDay' in df.columns:
        halfday_counts = df['HalfDay'].value_counts()
        axes[1, 1].bar(halfday_counts.index, halfday_counts.values, color='purple')
        axes[1, 1].set_title('Half-Day Distribution')
        axes[1, 1].set_xlabel('Time of Day')
        axes[1, 1].set_ylabel('Count')
    
    # Plot 6: Daily Average Debit/Credit Over Time
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0 and dr_col:
        date_col = date_cols[0]
        daily_avg = df.groupby(date_col)[dr_col].mean()
        axes[1, 2].plot(daily_avg.index, daily_avg.values, color='green', alpha=0.7)
        axes[1, 2].set_title(f'Average Daily {dr_col}')
        axes[1, 2].set_xlabel('Date')
        axes[1, 2].set_ylabel('Amount')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = "../Bank DataSet/eda_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualizations saved to: {plot_path}")
    
    # Close the plot to free memory (plot is already saved to file)
    plt.close()

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """
    Main function that runs all Phase 1 steps in order.
    This is like the conductor of an orchestra - coordinates everything.
    """
    print("="*70)
    print(" "*15 + "BANK CASH FORECASTING SYSTEM")
    print(" "*20 + "Phase 1: Data Preparation")
    print("="*70)
    
    # Store original dataset for comparison
    original_df = None
    
    # Step 1: Load data
    df = load_dataset(DATASET_PATH)
    
    if df is None:
        print("\n✗ Failed to load dataset. Please check the file path.")
        return
    
    original_df = df.copy()  # Keep copy of original
    
    # Step 2: Explore structure
    explore_data_structure(df)
    
    # Step 3: Create data dictionary
    create_data_dictionary(df)
    
    # Step 4: Clean datetime columns
    df = clean_datetime_columns(df)
    
    # Step 5: Handle missing values
    df = handle_missing_values(df)
    
    # Step 6: Remove duplicates
    df = remove_duplicates(df)
    
    # Step 7: Generate external features
    df = generate_external_features(df)
    
    # Step 8: Handle outliers
    df = handle_outliers(df)
    
    # Step 9: Validate data
    df = validate_data(df)
    
    # Step 10: Generate summary report
    generate_summary_report(df, original_df)
    
    # Step 11: Create visualizations
    create_visualizations(df)
    
    # Step 12: Save cleaned data
    print("\n" + "="*70)
    print("STEP 12: SAVING CLEANED DATA")
    print("="*70)
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✓ Cleaned dataset saved to: {OUTPUT_PATH}")
    
    print("\n" + "="*70)
    print(" "*20 + "PHASE 1 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Review the cleaned dataset")
    print("  2. Check the EDA visualizations")
    print("  3. Move to Phase 2: Feature Engineering & Model Development")
    print("="*70)
    
    return df

# =============================================================================
# RUN THE MAIN FUNCTION
# =============================================================================

# This line runs the main function when you execute this script
if __name__ == "__main__":
    cleaned_data = main()