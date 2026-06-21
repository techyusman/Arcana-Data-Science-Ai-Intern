import pandas as pd

# ==========================================
# CREATING DATA STRUCTURES (Series & DataFrame)
# ==========================================
print("--- Creating DataStructures ---")
# 1. Series (1D Data - like a column in an Excel sheet)
names_series = pd.Series(["Alice", "Bob", "Charlie", "David"])
print("Series:\n", names_series, "\n")

# 2. DataFrame (2D Data - like an entire Excel sheet or SQL table)
data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [25, 30, 35, 40],
    "City": ["New York", "London", "Paris", "Tokyo"]
}
df = pd.DataFrame(data)
print("DataFrame:\n", df, "\n")


# ==========================================
# INSPECTING DATA
# ==========================================
print("--- Inspecting Data ---")
# Get the first n rows (default is 5)
print("df.head(2):\n", df.head(2), "\n")

# Get info about the DataFrame (columns, data types, non-null counts)
print("df.info():")
df.info()
print("\n")

# Get statistical summary of numerical columns
print("df.describe():\n", df.describe(), "\n")

# Get the shape (rows, columns)
print("Shape:", df.shape)


# ==========================================
# SELECTING AND FILTERING
# ==========================================
print("\n--- Selecting & Filtering ---")
# Selecting a single column (returns a Series)
print("Selecting 'Age' column:\n", df["Age"], "\n")

# Selecting multiple columns (returns a DataFrame)
print("Selecting 'Name' and 'City':\n", df[["Name", "City"]], "\n")

# Filtering rows based on a condition
older_than_30 = df[df["Age"] > 30]
print("People older than 30:\n", older_than_30, "\n")


# ==========================================
# ADDING AND MODIFYING DATA
# ==========================================
print("--- Adding & Modifying Data ---")
# Adding a new column
df["Salary"] = [50000, 60000, 70000, 80000]
print("After adding Salary:\n", df, "\n")

# Applying a function to a column
# E.g., giving a 10% raise
df["Salary"] = df["Salary"] * 1.10
print("After 10% raise:\n", df, "\n")


# ==========================================
# HANDLING MISSING DATA (Example)
# ==========================================
print("--- Handling Missing Data ---")
# Creating a DataFrame with a missing value (NaN)
df_missing = pd.DataFrame({
    "A": [1, 2, None, 4],
    "B": [5, None, None, 8]
})
print("DataFrame with NaNs:\n", df_missing, "\n")

# Fill missing values with a specific number
df_filled = df_missing.fillna(0)
print("Filled NaNs with 0:\n", df_filled)

# Drop rows with any missing values
df_dropped = df_missing.dropna()
print("\nDropped rows with NaNs:\n", df_dropped)
