import numpy as np

# ==========================================
# CREATING ARRAYS
# ==========================================
print("--- Creating Arrays ---")
# From a Python list
arr_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:\n", arr_1d)

# 2D Array (Matrix)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:\n", arr_2d)

# Array of zeros and ones
zeros = np.zeros((2, 3))  # 2 rows, 3 columns of zeros
ones = np.ones((3, 2))    # 3 rows, 2 columns of ones
print("Zeros:\n", zeros)

# Array with a range of values (start, stop, step)
range_arr = np.arange(0, 10, 2)
print("Arange (0 to 10, step 2):\n", range_arr)

# Array with evenly spaced values
linspace_arr = np.linspace(0, 1, 5) # 5 numbers evenly spaced between 0 and 1
print("Linspace:\n", linspace_arr)


# ==========================================
# INSPECTING ARRAYS (Attributes)
# ==========================================
print("\n--- Array Attributes ---")
print("Shape of arr_2d:", arr_2d.shape) # Output: (2, 3)
print("Number of dimensions:", arr_2d.ndim) # Output: 2
print("Data type:", arr_2d.dtype)       # Output: int32 or int64
print("Total elements:", arr_2d.size)   # Output: 6



# ==========================================
# MATH OPERATIONS & BROADCASTING
# ==========================================
print("\n--- Math Operations ---")
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

# Element-wise operations (No for-loops needed!)
print("Addition:", a + b)       # [11, 22, 33]
print("Multiplication:", a * b) # [10, 40, 90]

# Broadcasting: operations between arrays of different shapes or arrays and scalars
print("Multiply by scalar (a * 5):", a * 5) # [5, 10, 15]


# ==========================================
# 5. AGGREGATION & STATISTICS
# ==========================================
print("\n--- Aggregation ---")
data = np.array([[1, 2], [3, 4], [5, 6]])

print("Sum of all elements:", data.sum())
print("Sum of each column:", data.sum(axis=0)) # Column-wise sum
print("Sum of each row:", data.sum(axis=1))    # Row-wise sum
print("Mean (Average):", data.mean())
print("Maximum value:", data.max())

