import pandas as pd

# Descriptive Statistics on a Pandas Series
# A Series is like a single column of data.

data_series = pd.Series([10, 20, 30, 40, 50, 60, 20, 30, 15])

print("--- Descriptive Statistics for Series ---")
# describe() generates descriptive statistics that summarize the central tendency, 
# dispersion, and shape of a dataset's distribution.
print(data_series.describe())
print("\n" + "="*50 + "\n")


# Descriptive Statistics on a Pandas DataFrame
# A DataFrame is tabular data with rows and columns.

data = {
    'Age': [25, 30, 35, 40, 45, 50, None],
    'Salary': [50000, 60000, 55000, 70000, 80000, 95000, 60000],
    'Department': ['IT', 'HR', 'IT', 'Sales', 'Sales', 'HR', 'IT']
}
df = pd.DataFrame(data)

# By default, describe() only analyzes numeric columns
print("--- Numeric Columns Describe ---")
print(df.describe())
print()


# Individual Common Descriptive Functions
# You can calculate individual statistics separately if you need specific metrics.

print("--- Individual Descriptive Functions ---")
print("Mean of Age:             ",df['Age'].mean())             # Average
print("Median of Salary:        ",df['Salary'].median())        # Middle value
print("Mode of Department:      ",df['Department'].mode()[0])       # Most frequent value
print("Standard Deviation:      ",df['Salary'].std())           # Measure of spread
print("Variance:                ",df['Salary'].var())           # Variance
print("Minimum Age:             ",df['Age'].min())              # Smallest value
print("Maximum Salary:        ", df['Salary'].max())           # Largest value
print("Total Count (Age):   " , df['Age'].count())                # Number of non-null observations
