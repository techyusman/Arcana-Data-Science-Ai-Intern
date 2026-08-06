"""
================================================================================
BANK BRANCH CASH FORECASTING SYSTEM
Phase 1: Data Understanding & Preparation
================================================================================

This script handles Phase 1 of the project:
1. Load and examine the banking dataset
2. Clean and preprocess the data

Author: Muhammad Usman
Status: Phase 1 Implementation
================================================================================
"""

# =============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# =============================================================================

import pandas as pd    # For data manipulation (DataFrames, Series, reading files)
import numpy as np     # For numerical operations (arrays, math)
import warnings        # To suppress unimportant warnings
import sys             # For system-level settings

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')


# =============================================================================
# STEP 2: CONFIGURATION
# =============================================================================

# File paths
DATASET_PATH = "Bank DataSet/Bank Cash Optimization.xlsx"
OUTPUT_PATH = "Bank DataSet/cleaned_bank_data.csv"


# =============================================================================
# TASK 1: LOAD AND EXAMINE THE BANKING DATASET
# =============================================================================

# ---- 1.1 Load the Dataset ----

def load_dataset(file_path):
    """
    Load the Excel file into a pandas DataFrame.
    Reads all sheets and uses the first one.

    Parameters:
        file_path (str): Path to the Excel file

    Returns:
        DataFrame or None: The loaded data, or None if loading fails
    """
    print("=" * 70)
    print("TASK 1.1: LOADING DATASET")
    print("=" * 70)

    try:
        # Read all sheets from the Excel file
        df = pd.read_excel(file_path, sheet_name=None)

        # If multiple sheets exist, show them and use the first one
        if isinstance(df, dict):
            print(f"Found {len(df)} sheet(s): {list(df.keys())}")
            sheet_name = list(df.keys())[0]
            df = df[sheet_name]
            print(f"Using sheet: '{sheet_name}'")

        # Show basic info about the loaded data
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Rows: {df.shape[0]}")
        print(f"  Columns: {df.shape[1]}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        return df

    except FileNotFoundError:
        print(f"✗ Error: File not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"✗ Error loading file: {str(e)}")
        return None


# ---- 1.2 Examine the Data Structure ----

def examine_data(df):
    """
    Explore the structure of the dataset using:
      - df.dtypes       (data types of each column)
      - df.head()        (first 5 rows)
      - df.tail()        (last 5 rows)
      - df.describe()    (statistics for numerical columns)
      - df.shape         (rows, columns)
      - df.isnull().sum() (missing values per column)

    Parameters:
        df (DataFrame): The dataset to examine
    """
    print("\n" + "=" * 70)
    print("TASK 1.2: EXAMINING DATA STRUCTURE")
    print("=" * 70)

    # Shape of the DataFrame
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Data types of each column
    print("\n--- Data Types ---")
    print(df.dtypes)

    # First 5 rows — see what the data looks like
    print("\n--- First 5 Rows (head) ---")
    print(df.head())

    # Last 5 rows — check the end of the data
    print("\n--- Last 5 Rows (tail) ---")
    print(df.tail())

    # Statistical summary of numerical columns
    print("\n--- Numerical Statistics (describe) ---")
    print(df.describe())

    # Missing values check
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    missing_info = pd.DataFrame({
        'Missing Count': missing,
        'Percentage (%)': missing_pct.round(2)
    })
    # Show only columns that have missing values
    has_missing = missing_info[missing_info['Missing Count'] > 0]

    if len(has_missing) > 0:
        print(has_missing)
    else:
        print("✓ No missing values found!")


# ---- 1.3 Create Data Dictionary ----

def create_data_dictionary(df):
    """
    Create a data dictionary explaining each column.
    Uses: df.dtypes, df[col].nunique(), df[col].isnull().sum(), df[col].head()

    Parameters:
        df (DataFrame): The dataset

    Returns:
        DataFrame: Data dictionary with column-level information
    """
    print("\n" + "=" * 70)
    print("TASK 1.3: DATA DICTIONARY")
    print("=" * 70)

    column_info = {}

    for col in df.columns:
        column_info[col] = {
            'Data Type': str(df[col].dtype),
            'Unique Values': df[col].nunique(),
            'Missing': df[col].isnull().sum(),
            'Sample Values': str(df[col].dropna().head(3).tolist())
        }

    data_dict = pd.DataFrame(column_info).T
    print("\n", data_dict)

    return data_dict


# =============================================================================
# TASK 2: CLEAN AND PREPROCESS THE DATA
# =============================================================================

# ---- 2.1 Convert Date/Time Columns ----

def clean_datetime_columns(df):
    """
    Find and convert date/time columns to proper datetime format.
    Uses: pd.to_datetime()

    Parameters:
        df (DataFrame): The dataset

    Returns:
        DataFrame: Dataset with converted datetime columns
    """
    print("\n" + "=" * 70)
    print("TASK 2.1: CLEANING DATE/TIME COLUMNS")
    print("=" * 70)

    # Common keywords that indicate a date column
    date_keywords = ['date', 'datetime', 'timestamp', 'time']
    converted = []

    # Strategy 1: Check column names for date-like keywords
    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            try:
                df[col] = pd.to_datetime(df[col], format='mixed', dayfirst=True)
                converted.append(col)
                print(f"✓ Converted '{col}' to datetime")
            except Exception:
                print(f"✗ Could not convert '{col}' to datetime")

    # Strategy 2: If no standard date columns found, look for date-like values
    if len(converted) == 0:
        print("Looking for date-like patterns in object columns...")
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    sample = df[col].dropna().head(5)
                    test = pd.to_datetime(sample, format='mixed', dayfirst=True, errors='coerce')
                    if test.notna().sum() > 3:
                        df[col] = pd.to_datetime(df[col], format='mixed', dayfirst=True, errors='coerce')
                        converted.append(col)
                        print(f"✓ Converted '{col}' to datetime")
                except Exception:
                    pass

    print(f"\nTotal datetime columns converted: {len(converted)}")
    return df


# ---- 2.2 Handle Missing Values ----

def handle_missing_values(df):
    """
    Handle missing values using appropriate strategies:
      - Numerical cash/amount columns → fill with 0 (no transaction)
      - Other numerical columns → fill with median
      - Categorical columns → fill with 'Unknown'
      - Drop rows where critical columns (branch, date) are still null

    Uses: df.select_dtypes(), df[col].fillna(), df[col].median(), df.dropna()

    Parameters:
        df (DataFrame): The dataset

    Returns:
        DataFrame: Dataset with missing values handled
    """
    print("\n" + "=" * 70)
    print("TASK 2.2: HANDLING MISSING VALUES")
    print("=" * 70)

    initial_rows = len(df)

    # --- Numerical columns ---
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            # Cash-related columns: fill with 0
            cash_keywords = ['cash', 'deposit', 'withdrawal', 'balance', 'amount',
                             'total_dr', 'total_cr', 'dr', 'cr']
            if any(kw in col.lower() for kw in cash_keywords):
                df[col] = df[col].fillna(0)
                print(f"✓ '{col}' — filled with 0")
            else:
                # Other numerical: fill with median
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"✓ '{col}' — filled with median ({median_val:.2f})")

    # --- Categorical columns ---
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna('Unknown')
            print(f"✓ '{col}' — filled with 'Unknown'")

    # --- Drop rows where critical columns are null ---
    critical_cols = [col for col in df.columns
                     if 'branch' in col.lower() or 'date' in col.lower()]

    if critical_cols:
        df = df.dropna(subset=critical_cols)
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            print(f"✓ Removed {rows_removed} rows with missing critical data")

    print(f"\nFinal shape after missing value handling: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


# ---- 2.3 Remove Duplicates ----

def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset.
    Uses: df.duplicated().sum(), df.drop_duplicates()

    Parameters:
        df (DataFrame): The dataset

    Returns:
        DataFrame: Dataset without duplicate rows
    """
    print("\n" + "=" * 70)
    print("TASK 2.3: REMOVING DUPLICATES")
    print("=" * 70)

    initial_rows = len(df)
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates}")

    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"✓ Removed {initial_rows - len(df)} duplicate rows")

    print(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


# ---- 2.4 Handle Outliers ----

def handle_outliers(df):
    """
    Detect and cap outliers in cash-related numerical columns
    using the IQR (Interquartile Range) method.
      - Q1 = 25th percentile
      - Q3 = 75th percentile
      - IQR = Q3 - Q1
      - Lower bound = Q1 - 1.5 * IQR
      - Upper bound = Q3 + 1.5 * IQR

    Uses: df[col].quantile(), df[col].clip()

    Parameters:
        df (DataFrame): The dataset

    Returns:
        DataFrame: Dataset with outliers capped
    """
    print("\n" + "=" * 70)
    print("TASK 2.4: HANDLING OUTLIERS (IQR Method)")
    print("=" * 70)

    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    # Focus on cash-related columns for outlier handling
    cash_keywords = ['cash', 'deposit', 'withdrawal', 'balance', 'amount',
                     'total_dr', 'total_cr', 'dr', 'cr', 'debit', 'credit']
    cash_cols = [col for col in numerical_cols
                 if any(kw in col.lower() for kw in cash_keywords)]

    outlier_summary = {}

    for col in cash_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers before capping
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        if outlier_count > 0:
            outlier_summary[col] = outlier_count
            # Cap outliers at the boundaries
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"⚠ '{col}': {outlier_count} outliers → capped at [{lower_bound:.2f}, {upper_bound:.2f}]")

    if len(outlier_summary) == 0:
        print("✓ No significant outliers detected")
    else:
        print(f"\n✓ Outliers handled in {len(outlier_summary)} column(s)")

    return df


# ---- 2.5 Validate Data ----

def validate_data(df):
    """
    Validate the data for business-logic correctness.
    Example: Ensure no negative balance values exist.

    Uses: df[col].clip()

    Parameters:
        df (DataFrame): The dataset

    Returns:
        DataFrame: Validated dataset
    """
    print("\n" + "=" * 70)
    print("TASK 2.5: DATA VALIDATION")
    print("=" * 70)

    # Find balance-related columns and ensure no negative values
    balance_cols = [col for col in df.columns if 'balance' in col.lower()]

    for col in balance_cols:
        negatives = (df[col] < 0).sum()
        if negatives > 0:
            print(f"⚠ '{col}': {negatives} negative values → clipped to 0")
            df[col] = df[col].clip(lower=0)

    # Final missing values check after all preprocessing
    remaining_missing = df.isnull().sum().sum()
    print(f"\nRemaining missing values: {remaining_missing}")

    # Final data types summary
    print(f"\n--- Final Data Types ---")
    print(df.dtypes)

    print("\n✓ Data validation complete")

    return df


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary(df, original_df):
    """
    Print a comparison between original and cleaned datasets.

    Parameters:
        df (DataFrame): Cleaned dataset
        original_df (DataFrame): Original dataset before cleaning
    """
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    print(f"\nOriginal dataset:  {original_df.shape[0]} rows × {original_df.shape[1]} columns")
    print(f"Cleaned dataset:   {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Rows removed:      {original_df.shape[0] - df.shape[0]}")
    print(f"Columns unchanged: {df.shape[1]}")

    # Show unique values for key columns using value_counts()
    print("\n--- Key Column Distributions ---")
    for col in df.columns:
        if df[col].nunique() <= 20 and df[col].dtype == 'object':
            print(f"\n{col}:")
            print(df[col].value_counts())


# =============================================================================
# SAVE CLEANED DATA
# =============================================================================

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.

    Parameters:
        df (DataFrame): Cleaned dataset
        output_path (str): File path for the output CSV
    """
    print("\n" + "=" * 70)
    print("SAVING CLEANED DATA")
    print("=" * 70)

    df.to_csv(output_path, index=False)
    print(f"✓ Cleaned dataset saved to: {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function — runs all Phase 1 steps in order.
    Task 1: Load and examine the banking dataset
    Task 2: Clean and preprocess the data
    """
    print("=" * 70)
    print(" " * 10 + "BANK BRANCH CASH FORECASTING SYSTEM")
    print(" " * 15 + "Phase 1: Data Preparation")
    print(" " * 10 + "Task 1: Load & Examine | Task 2: Clean & Preprocess")
    print("=" * 70)

    # ---- TASK 1: Load and Examine ----
    df = load_dataset(DATASET_PATH)

    if df is None:
        print("\n✗ Failed to load dataset. Please check the file path.")
        return

    # Keep a copy of original data for comparison
    original_df = df.copy()

    examine_data(df)
    create_data_dictionary(df)

    # ---- TASK 2: Clean and Preprocess ----
    df = clean_datetime_columns(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = validate_data(df)

    # ---- Summary & Save ----
    print_summary(df, original_df)
    save_cleaned_data(df, OUTPUT_PATH)

    print("\n" + "=" * 70)
    print(" " * 15 + "PHASE 1 (Tasks 1 & 2) COMPLETED!")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Review the cleaned dataset")
    print("  2. Move to EDA and external feature integration")
    print("  3. Proceed to Phase 2: Feature Engineering & Model Development")
    print("=" * 70)

    return df


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    cleaned_data = main()