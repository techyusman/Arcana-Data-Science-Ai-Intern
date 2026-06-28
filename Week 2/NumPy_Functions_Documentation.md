# NumPy Functions Documentation

> **Source Notebook:** `Numpy.ipynb`
> **Total Functions Extracted:** 50+

---

## Table of Contents

1. [Array Creation Functions](#1-array-creation-functions)
2. [Array Manipulation Functions](#2-array-manipulation-functions)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Statistical Operations](#4-statistical-operations)
5. [Random Number Generation](#5-random-number-generation)
6. [Linear Algebra Functions](#6-linear-algebra-functions)
7. [Array Properties & Attributes](#7-array-properties--attributes)
8. [Utility / Helper Functions](#8-utility--helper-functions)

---

## 1. Array Creation Functions

### 1.1 `np.array()`

**Description:** Creates a NumPy array from a Python list, tuple, or other array-like object. This is the most fundamental way to create arrays in NumPy.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `object` | array_like | The input data — a list, tuple, nested sequences, or another array to convert into an ndarray. |
| `dtype` | data-type, optional | The desired data type for the array elements (e.g., `int`, `float`, `complex`). If not specified, NumPy infers it from the input data. |

**Usage in notebook:**
```python
arr = np.array([1, 2, 3, 4])
matrix = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
```

---

### 1.2 `np.zeros()`

**Description:** Creates an array filled entirely with zeros. Commonly used for initializing arrays before populating them with computed values.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `shape` | int or tuple of ints | The dimensions of the output array (e.g., `(2, 3)` for a 2×3 matrix). |
| `dtype` | data-type, optional | The desired data type for the elements. Default is `float64`. |

**Usage in notebook:**
```python
zeros_array = np.zeros((2, 3))
```

---

### 1.3 `np.ones()`

**Description:** Creates an array filled entirely with ones. Useful for initializing weight matrices or creating mask arrays.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `shape` | int or tuple of ints | The dimensions of the output array. |
| `dtype` | data-type, optional | The desired data type for the elements. Default is `float64`. |

**Usage in notebook:**
```python
ones_array = np.ones((2, 3))
```

---

### 1.4 `np.empty()`

**Description:** Creates an uninitialized array. The array elements contain whatever data was already in that memory location. Faster than `np.zeros()` or `np.ones()` since it skips initialization.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `shape` | int or tuple of ints | The dimensions of the output array. |
| `dtype` | data-type, optional | The desired data type for the elements. Default is `float64`. |

**Usage in notebook:**
```python
empty_array = np.empty((2, 3))
```

> **Warning:** Values in an empty array are *not* guaranteed to be zero — they are whatever was previously in memory.

---

### 1.5 `np.arange()`

**Description:** Creates an array with evenly spaced values within a given interval (similar to Python's `range()` but returns an ndarray). Often used for generating index sequences.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `start` | number, optional | The start of the interval. Default is `0`. |
| `stop` | number | The end of the interval (exclusive — this value is **not** included). |
| `step` | number, optional | The spacing between consecutive values. Default is `1`. |
| `dtype` | data-type, optional | The data type of the output array. |

**Usage in notebook:**
```python
range_array = np.arange(5)        # → [0, 1, 2, 3, 4]
arr = np.arange(6)                # → [0, 1, 2, 3, 4, 5]
```

---

### 1.6 `np.linspace()`

**Description:** Creates an array of evenly spaced values over a specified interval. Unlike `arange()`, you specify the **number of points** rather than the step size, and both endpoints are included by default.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `start` | float | The starting value of the sequence. |
| `stop` | float | The ending value of the sequence (included by default). |
| `num` | int, optional | The number of evenly spaced samples to generate. Default is `50`. |
| `endpoint` | bool, optional | If `True` (default), `stop` is the last sample. If `False`, it is excluded. |

**Usage in notebook:**
```python
linspace_array = np.linspace(0, 1, 5)  # → [0.0, 0.25, 0.5, 0.75, 1.0]
```

---

### 1.7 `np.logspace()`

**Description:** Creates an array of values spaced evenly on a logarithmic (log10) scale. Useful for generating parameters that span several orders of magnitude.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `start` | float | The starting exponent (`base**start` is the first value). |
| `stop` | float | The ending exponent (`base**stop` is the last value). |
| `num` | int, optional | The number of samples to generate. Default is `50`. |
| `base` | float, optional | The base of the logarithm. Default is `10.0`. |

**Usage in notebook:**
```python
logspace_array = np.logspace(0, 2, 3)  # → [1., 10., 100.]
```

---

### 1.8 `np.meshgrid()`

**Description:** Creates coordinate matrices from 1-D coordinate vectors. Essential for evaluating functions on 2D grids and creating surface plots.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `*xi` | array_like | One or more 1-D arrays representing the coordinates of a grid. |
| `indexing` | str, optional | Cartesian (`'xy'`, default) or matrix (`'ij'`) indexing of the output. |

**Usage in notebook:**
```python
x = np.array([1, 2, 3])
y = np.array([4, 5])
X, Y = np.meshgrid(x, y)
```

---

### 1.9 `np.eye()`

**Description:** Creates a 2-D identity matrix — a square matrix with ones on the main diagonal and zeros elsewhere.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `N` | int | The number of rows (and columns for a square matrix) in the output. |
| `M` | int, optional | The number of columns. If not specified, defaults to `N` (producing a square matrix). |
| `k` | int, optional | Index of the diagonal. `0` (default) = main diagonal, positive = above, negative = below. |
| `dtype` | data-type, optional | The data type of the returned array. Default is `float64`. |

**Usage in notebook:**
```python
identity_matrix = np.eye(3)
```

---

### 1.10 `np.diag()`

**Description:** Extracts a diagonal from a matrix, or constructs a diagonal matrix from a 1-D array.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `v` | array_like | If `v` is a 1-D array, returns a 2-D array with `v` on the diagonal. If `v` is a 2-D array, extracts and returns the diagonal. |
| `k` | int, optional | The diagonal index. `0` (default) = main diagonal, positive = above, negative = below. |

**Usage in notebook:**
```python
diag_matrix = np.diag([1, 2, 3])
```

---

## 2. Array Manipulation Functions

### 2.1 `ndarray.reshape()`

**Description:** Returns a new array with the same data but a different shape. The total number of elements must remain unchanged.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `newshape` | int or tuple of ints | The desired new shape. One dimension can be `-1`, in which case NumPy automatically computes it. |
| `order` | `'C'`, `'F'`, or `'A'`, optional | Read/write order of elements: `'C'` (row-major, default), `'F'` (column-major), `'A'` (original order). |

**Usage in notebook:**
```python
reshaped_array = arr.reshape((2, 3))
a2.reshape(2, 1)
```

---

### 2.2 `ndarray.ravel()`

**Description:** Returns a contiguous flattened (1-D) array. Equivalent to `reshape(-1)` but may return a view instead of a copy when possible, making it more memory-efficient.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `order` | `'C'`, `'F'`, `'A'`, `'K'`, optional | The order of flattening. `'C'` (row-major, default) reads elements row by row; `'F'` (Fortran) reads column by column. |

**Usage in notebook:**
```python
flattened_array = reshaped_array.ravel()
X.ravel()
Y.ravel()
```

---

### 2.3 `ndarray.T` (Transpose)

**Description:** Returns the transposed array (attribute, not a function call). For a 2-D array, this swaps rows and columns. For a 1-D array, it returns the same array unchanged.

**Usage in notebook:**
```python
transposed_array = reshaped_array.T
transposed_matrix = matrix.T
coords = np.array([X.ravel(), Y.ravel()]).T
```

---

### 2.4 `np.swapaxes()`

**Description:** Interchanges two axes of an array. A generalization of transpose for multi-dimensional arrays.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `a` | array_like | The input array whose axes will be swapped. |
| `axis1` | int | The first axis to swap. |
| `axis2` | int | The second axis to swap. |

**Usage in notebook:**
```python
swapped_axes_array = np.swapaxes(reshaped_array, 0, 1)
```

---

### 2.5 `np.concatenate()`

**Description:** Joins a sequence of arrays along an existing axis. All arrays must have the same shape except along the join axis.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `arrays` | sequence of array_like | The arrays to concatenate. Must be passed as a **tuple or list**. |
| `axis` | int, optional | The axis along which to join. Default is `0` (first axis). Use `None` to flatten all arrays first. |

**Usage in notebook:**
```python
concatenated_array = np.concatenate((arr1, arr2))
```

---

### 2.6 `np.split()`

**Description:** Splits an array into multiple sub-arrays of equal size along the given axis.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `ary` | ndarray | The array to be split. |
| `indices_or_sections` | int or 1-D array | If an integer `N`, split into `N` equal parts. If a 1-D array, the entries indicate the split points. |
| `axis` | int, optional | The axis along which to split. Default is `0`. |

**Usage in notebook:**
```python
split_array = np.split(concatenated_array, 2)
```

---

### 2.7 `np.hstack()`

**Description:** Stacks arrays horizontally (column-wise). For 1-D arrays, this is equivalent to concatenation along the first axis. For 2-D arrays, it concatenates along the second axis (columns).

| Parameter | Type | Purpose |
|-----------|------|---------|
| `tup` | sequence of ndarray | The arrays to stack, passed as a **tuple**. All arrays must have the same number of rows (for 2-D). |

**Usage in notebook:**
```python
hstack_array = np.hstack((arr1, arr2))
horizontal_stacking = np.hstack((a1, a2.reshape(2, 1)))
```

---

### 2.8 `np.vstack()`

**Description:** Stacks arrays vertically (row-wise). For 1-D arrays, each array is treated as a row. For 2-D arrays, it concatenates along the first axis (rows).

| Parameter | Type | Purpose |
|-----------|------|---------|
| `tup` | sequence of ndarray | The arrays to stack, passed as a **tuple**. All arrays must have the same number of columns (for 2-D). |

**Usage in notebook:**
```python
vstack_array = np.vstack((arr1, arr2))
```

---

### 2.9 `np.repeat()`

**Description:** Repeats each element of an array a specified number of times.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `a` | array_like | The input array whose elements will be repeated. |
| `repeats` | int or array of ints | The number of repetitions for each element. If a single int, all elements are repeated that many times. |
| `axis` | int, optional | The axis along which to repeat. If `None` (default), the array is flattened first. |

**Usage in notebook:**
```python
repeated_array = np.repeat(arr1, 2)  # → [1, 1, 2, 2, 3, 3]
```

---

### 2.10 `np.tile()`

**Description:** Constructs a new array by repeating the input array a given number of times along each axis. Think of it as "tiling a floor" with copies of the array.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `A` | array_like | The input array to tile/repeat. |
| `reps` | int or tuple of ints | The number of repetitions along each axis. `(2, 3)` means repeat 2 times vertically and 3 times horizontally. |

**Usage in notebook:**
```python
tiled_array_2d = np.tile([[1, 2], [3, 4]], (2, 3))
```

---

## 3. Mathematical Operations

### 3.1 `np.add()`

**Description:** Adds two arrays element-wise. Equivalent to the `+` operator but available as a function.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `x1` | array_like | The first input array (addend). |
| `x2` | array_like | The second input array (addend). Must be broadcastable to the same shape as `x1`. |

**Usage in notebook:**
```python
arr_sum = np.add([1, 2, 3], [4, 5, 6])  # → [5, 7, 9]
```

---

### 3.2 `np.subtract()`

**Description:** Subtracts the second array from the first, element-wise. Equivalent to the `-` operator.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `x1` | array_like | The minuend (the array to subtract from). |
| `x2` | array_like | The subtrahend (the array to subtract). Must be broadcastable to the same shape as `x1`. |

**Usage in notebook:**
```python
arr_diff = np.subtract([4, 5, 6], [1, 2, 3])  # → [3, 3, 3]
```

---

### 3.3 `np.multiply()`

**Description:** Multiplies two arrays element-wise. This is NOT matrix multiplication — it multiplies corresponding elements.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `x1` | array_like | The first input array (multiplicand). |
| `x2` | array_like | The second input array (multiplier). Must be broadcastable to the same shape as `x1`. |

**Usage in notebook:**
```python
arr_prod = np.multiply([1, 2, 3], [4, 5, 6])  # → [4, 10, 18]
```

---

### 3.4 `np.divide()`

**Description:** Divides the first array by the second, element-wise. Returns float results.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `x1` | array_like | The dividend (numerator array). |
| `x2` | array_like | The divisor (denominator array). Must be broadcastable to the same shape as `x1`. |

**Usage in notebook:**
```python
arr_div = np.divide([4, 5, 6], [1, 2, 3])  # → [4.0, 2.5, 2.0]
```

---

### 3.5 `np.power()`

**Description:** Raises each element of the first array to the power specified by the corresponding element in the second array.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `x1` | array_like | The base array. |
| `x2` | array_like | The exponent array. Can be a scalar (applied to all elements) or an array of the same shape. |

**Usage in notebook:**
```python
arr_pow = np.power([1, 2, 3], 2)  # → [1, 4, 9]
```

---

### 3.6 `np.sqrt()`

**Description:** Computes the non-negative square root of each element in the array.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `x` | array_like | The input array whose square roots will be computed. Values must be non-negative. |

**Usage in notebook:**
```python
arr_sqrt = np.sqrt([1, 4, 9])  # → [1.0, 2.0, 3.0]
```

---

### 3.7 `np.exp()`

**Description:** Computes the exponential (e^x) of each element in the array. The base is Euler's number *e* ≈ 2.71828.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `x` | array_like | The input array. Each element is used as the exponent of *e*. |

**Usage in notebook:**
```python
arr_exp = np.exp([1, 2, 3])  # → [2.718..., 7.389..., 20.086...]
```

---

### 3.8 `np.log()`

**Description:** Computes the natural logarithm (base *e*) of each element. This is the inverse of `np.exp()`.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `x` | array_like | The input array. Values must be positive. |

**Usage in notebook:**
```python
arr_log = np.log([1, np.e, np.e**2])  # → [0.0, 1.0, 2.0]
```

---

### 3.9 `np.sin()`

**Description:** Computes the trigonometric sine of each element (assumes input is in radians).

| Parameter | Type | Purpose |
|-----------|------|---------|
| `x` | array_like | The input array of angles in **radians**. |

**Usage in notebook:**
```python
arr_sin = np.sin([0, np.pi/2, np.pi])  # → [0.0, 1.0, ~0.0]
```

---

### 3.10 `np.cos()`

**Description:** Computes the trigonometric cosine of each element (assumes input is in radians).

| Parameter | Type | Purpose |
|-----------|------|---------|
| `x` | array_like | The input array of angles in **radians**. |

**Usage in notebook:**
```python
arr_cos = np.cos([0, np.pi/2, np.pi])  # → [1.0, ~0.0, -1.0]
```

---

## 4. Statistical Operations

### 4.1 `np.mean()`

**Description:** Computes the arithmetic mean (average) of array elements along the specified axis.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `a` | array_like | The input array containing the values to average. |
| `axis` | int or None, optional | The axis along which to compute the mean. `None` (default) computes the mean of the flattened array. |

**Usage in notebook:**
```python
mean_value = np.mean([1, 2, 3, 4, 5])  # → 3.0
mean = np.mean(data_array)
```

---

### 4.2 `np.median()`

**Description:** Computes the median (middle value) of an array along the specified axis.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `a` | array_like | The input array. |
| `axis` | int or None, optional | The axis along which to compute the median. `None` (default) computes over the entire flattened array. |

**Usage in notebook:**
```python
median_value = np.median([1, 2, 3, 4, 5])  # → 3.0
median = np.median(data_array)
```

---

### 4.3 `np.std()`

**Description:** Computes the standard deviation — a measure of the spread/dispersion of values from the mean.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `a` | array_like | The input array. |
| `axis` | int or None, optional | The axis along which to compute. `None` (default) computes over the flattened array. |
| `ddof` | int, optional | "Delta Degrees of Freedom." Use `ddof=0` (default) for **population** standard deviation, `ddof=1` for **sample** standard deviation. |

**Usage in notebook:**
```python
std_deviation = np.std([1, 2, 3, 4, 5])
std_dev = np.std(data_array, ddof=0)
```

---

### 4.4 `np.var()`

**Description:** Computes the variance — the average of the squared deviations from the mean. Variance = std squared.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `a` | array_like | The input array. |
| `axis` | int or None, optional | The axis along which to compute. `None` (default) computes over the flattened array. |
| `ddof` | int, optional | "Delta Degrees of Freedom." Use `ddof=0` (default) for population variance, `ddof=1` for sample variance. |

**Usage in notebook:**
```python
variance = np.var([1, 2, 3, 4, 5])
variance = np.var(data_array, ddof=0)
```

---

### 4.5 `np.percentile()`

**Description:** Computes the nth percentile of the data along the specified axis. The nth percentile is the value below which n% of the data falls.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `a` | array_like | The input array. |
| `q` | float or array of floats | The percentile(s) to compute. Must be between `0` and `100` inclusive. |
| `axis` | int or None, optional | The axis along which to compute. `None` (default) computes over the flattened array. |

**Usage in notebook:**
```python
percentile_50 = np.percentile([1, 2, 3, 4, 5], 50)  # → 3.0 (same as median)
q75, q25 = np.percentile(data_array, [75, 25])       # For IQR calculation
```

---

### 4.6 `np.max()` / `np.min()`

**Description:** Returns the maximum / minimum value in an array along a given axis.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `a` | array_like | The input array. |
| `axis` | int or None, optional | Axis along which to operate. `None` (default) returns the global max/min. |

**Usage in notebook:**
```python
data_range = np.max(data_array) - np.min(data_array)
```

---

### 4.7 `np.unique()`

**Description:** Returns the sorted unique elements of an array. Optionally returns additional information like counts.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `ar` | array_like | The input array. |
| `return_counts` | bool, optional | If `True`, also returns the number of times each unique value appears. |
| `return_index` | bool, optional | If `True`, returns the indices of the first occurrences. |
| `return_inverse` | bool, optional | If `True`, returns indices to reconstruct the original array from the unique array. |

**Usage in notebook:**
```python
values, counts = np.unique(data_array, return_counts=True)
mode = values[np.argmax(counts)]
```

---

### 4.8 `np.argmax()`

**Description:** Returns the index of the maximum value in an array along a given axis. Used for finding the position of the largest element.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `a` | array_like | The input array. |
| `axis` | int or None, optional | The axis along which to find the argmax. `None` (default) operates on the flattened array. |

**Usage in notebook:**
```python
mode = values[np.argmax(counts)]  # Find index of highest count → mode value
```

---

## 5. Random Number Generation

### 5.1 `np.random.rand()`

**Description:** Generates an array of random floats uniformly distributed in the range [0.0, 1.0).

| Parameter | Type | Purpose |
|-----------|------|---------|
| `*d` | int(s) | The dimensions of the output array. If no arguments, returns a single float. |

**Usage in notebook:**
```python
random_number = np.random.rand()       # Single random float
random_array = np.random.rand(2, 3)    # 2x3 array of random floats
```

---

### 5.2 `np.random.randint()`

**Description:** Generates random integers from a discrete uniform distribution within a specified range.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `low` | int | The lowest integer to be drawn (inclusive). |
| `high` | int, optional | The upper bound (exclusive). If not provided, values are drawn from [0, low). |
| `size` | int or tuple of ints, optional | The shape of the output array. If `None`, returns a single int. |

**Usage in notebook:**
```python
random_int = np.random.randint(0, 10)               # Single int in [0, 10)
random_array = np.random.randint(0, 10, size=(2, 3)) # 2x3 array of ints in [0, 10)
```

---

### 5.3 `np.random.choice()`

**Description:** Generates random samples from a given 1-D array. Can sample with or without replacement.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `a` | 1-D array_like or int | If an array, a random sample is drawn from its elements. If an int, the sample is drawn from `np.arange(a)`. |
| `size` | int or tuple of ints, optional | The shape of the output array. If `None`, returns a single value. |
| `replace` | bool, optional | Whether sampling is with replacement (`True`, default) or without (`False`). |
| `p` | 1-D array_like, optional | The probability associated with each element in `a`. Must sum to 1. |

**Usage in notebook:**
```python
random_sample = np.random.choice(array)                          # Single sample
random_samples = np.random.choice(array, size=3, replace=True)   # With replacement
random_samples_no_replace = np.random.choice(array, size=3, replace=False)  # Without
```

---

### 5.4 `np.random.uniform()`

**Description:** Draws samples from a continuous uniform distribution over the interval [low, high).

| Parameter | Type | Purpose |
|-----------|------|---------|
| `low` | float, optional | Lower boundary of the output interval (inclusive). Default is `0.0`. |
| `high` | float, optional | Upper boundary of the output interval (exclusive). Default is `1.0`. |
| `size` | int or tuple of ints, optional | The shape of the output array. If `None`, returns a single float. |

**Usage in notebook:**
```python
random_number = np.random.uniform(1, 10)                  # Single float in [1, 10)
random_array = np.random.uniform(1, 10, size=(2, 3))       # 2x3 array
```

---

### 5.5 `np.random.normal()`

**Description:** Draws samples from a normal (Gaussian) distribution. Most real-world data follows or approximates this distribution.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `loc` | float, optional | The mean (center) of the distribution. Default is `0.0`. |
| `scale` | float, optional | The standard deviation (spread) of the distribution. Default is `1.0`. |
| `size` | int or tuple of ints, optional | The shape of the output array. If `None`, returns a single float. |

**Usage in notebook:**
```python
random_number = np.random.normal(loc=0, scale=1)              # Single sample from N(0,1)
random_array = np.random.normal(loc=0, scale=1, size=(2, 3))  # 2x3 array from N(0,1)
```

---

### 5.6 `np.random.exponential()`

**Description:** Draws samples from an exponential distribution. Commonly used to model wait times between events.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `scale` | float, optional | The scale parameter (the mean/expected value). Default is `1.0`. |
| `size` | int or tuple of ints, optional | The shape of the output array. If `None`, returns a single float. |

**Usage in notebook:**
```python
random_number = np.random.exponential(scale=1)
random_array = np.random.exponential(scale=1, size=(2, 3))
```

---

### 5.7 `np.random.binomial()`

**Description:** Draws samples from a binomial distribution. Models the number of successes in `n` independent yes/no experiments, each with probability `p`.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `n` | int | The number of trials (experiments). |
| `p` | float | The probability of success in each trial. Must be in [0, 1]. |
| `size` | int or tuple of ints, optional | The shape of the output array. If `None`, returns a single int. |

**Usage in notebook:**
```python
random_number = np.random.binomial(n=10, p=0.5)
random_array = np.random.binomial(n=10, p=0.5, size=(2, 3))
```

---

### 5.8 `np.random.poisson()`

**Description:** Draws samples from a Poisson distribution. Models the number of events occurring in a fixed interval of time/space.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `lam` | float, optional | The expected number of events (lambda, the rate). Default is `1.0`. |
| `size` | int or tuple of ints, optional | The shape of the output array. If `None`, returns a single int. |

**Usage in notebook:**
```python
random_number = np.random.poisson(lam=3)
random_array = np.random.poisson(lam=3, size=(2, 3))
```

---

### 5.9 `np.random.seed()`

**Description:** Seeds the random number generator for **reproducibility**. Using the same seed guarantees the same sequence of random numbers every time, essential for debugging and reproducible experiments.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `seed` | int or None | The seed value. Use an integer for reproducible results, or `None` for a random seed. |

**Usage in notebook:**
```python
np.random.seed(42)
```

---

## 6. Linear Algebra Functions

### 6.1 `np.dot()`

**Description:** Computes the dot product of two arrays. For 2-D arrays, this performs **matrix multiplication**. For 1-D arrays, it computes the inner product.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `a` | array_like | The first input array. |
| `b` | array_like | The second input array. For matrix multiplication, the number of columns of `a` must equal the number of rows of `b`. |

**Usage in notebook:**
```python
product_matrix = np.dot(matrix1, matrix2)
```

---

### 6.2 `np.linalg.inv()`

**Description:** Computes the multiplicative inverse of a square matrix. A matrix multiplied by its inverse produces the identity matrix.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `a` | array_like | The square matrix to invert. Must be non-singular (determinant != 0). |

**Usage in notebook:**
```python
inverse_matrix = np.linalg.inv(matrix)
```

> **Warning:** Raises `LinAlgError` if the matrix is singular (non-invertible).

---

### 6.3 `np.linalg.det()`

**Description:** Computes the determinant of a square matrix. The determinant indicates whether a matrix is invertible and measures the scaling factor of the linear transformation.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `a` | array_like | The input square matrix. |

**Usage in notebook:**
```python
determinant = np.linalg.det(matrix)  # → -2.0
```

---

### 6.4 `np.linalg.eig()`

**Description:** Computes the eigenvalues and right eigenvectors of a square matrix. Eigenvalues are scalars such that A*v = lambda*v, where v is the eigenvector.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `a` | array_like | The square matrix for which to compute eigenvalues and eigenvectors. |

**Returns:**
- `eigenvalues` — 1-D array of eigenvalues.
- `eigenvectors` — 2-D array where each column is an eigenvector corresponding to the respective eigenvalue.

**Usage in notebook:**
```python
eigenvalues, eigenvectors = np.linalg.eig(matrix)
```

---

### 6.5 `np.linalg.solve()`

**Description:** Solves a system of linear equations `A*x = b` for x. More numerically stable and efficient than computing the inverse and then multiplying.

| Parameter | Type | Purpose |
|-----------|------|---------|
| `a` | array_like | The coefficient matrix `A`. Must be square and non-singular. |
| `b` | array_like | The right-hand side vector or matrix `b`. |

**Usage in notebook:**
```python
A = np.array([[2, 3], [4, -1]])
b = np.array([13, 3])
solution = np.linalg.solve(A, b)   # Solves 2x + 3y = 13, 4x - y = 3
```

---

## 7. Array Properties & Attributes

### 7.1 `ndarray.shape`

**Description:** A tuple giving the dimensions of the array. For a 2-D array with m rows and n columns, shape is `(m, n)`.

**Usage in notebook:**
```python
matrix.shape       # → (3, 3)
a1.shape           # → (2, 3)
a2.shape           # → (2,)
```

---

### 7.2 `ndarray.tolist()`

**Description:** Converts an ndarray to a regular Python list (or nested list for multi-dimensional arrays).

| Parameter | Type | Purpose |
|-----------|------|---------|
| *(none)* | — | This method takes no parameters. |

**Usage in notebook:**
```python
X.ravel().tolist()
Y.ravel().tolist()
```

---

## 8. Utility / Helper Functions

### 8.1 `np.e`

**Description:** A mathematical constant — Euler's number *e* (approximately 2.71828). Not a function, but a constant used in mathematical operations.

**Usage in notebook:**
```python
np.log([1, np.e, np.e**2])
```

---

### 8.2 `np.pi`

**Description:** A mathematical constant — pi (approximately 3.14159). Used extensively in trigonometric calculations.

**Usage in notebook:**
```python
np.sin([0, np.pi/2, np.pi])
np.cos([0, np.pi/2, np.pi])
```

---

### 8.3 Python Built-in Functions Used with NumPy

The notebook also uses several Python built-in functions in conjunction with NumPy:

| Function | Description | Usage in Notebook |
|----------|-------------|-------------------|
| `print()` | Outputs text/values to the console | Used in every cell to display results |
| `list()` | Converts an iterable to a Python list | `list(zip(X.ravel().tolist(), Y.ravel().tolist()))` |
| `zip()` | Combines multiple iterables element-wise into tuples | `zip(X.ravel().tolist(), Y.ravel().tolist())` |

---

## Summary Table — All NumPy Functions at a Glance

| # | Function | Category | Purpose |
|---|----------|----------|---------|
| 1 | `np.array()` | Array Creation | Create array from list/tuple |
| 2 | `np.zeros()` | Array Creation | Array filled with zeros |
| 3 | `np.ones()` | Array Creation | Array filled with ones |
| 4 | `np.empty()` | Array Creation | Uninitialized array |
| 5 | `np.arange()` | Array Creation | Evenly spaced values (by step) |
| 6 | `np.linspace()` | Array Creation | Evenly spaced values (by count) |
| 7 | `np.logspace()` | Array Creation | Log-scale spaced values |
| 8 | `np.meshgrid()` | Array Creation | Coordinate grid matrices |
| 9 | `np.eye()` | Array Creation | Identity matrix |
| 10 | `np.diag()` | Array Creation | Diagonal matrix |
| 11 | `.reshape()` | Manipulation | Change array shape |
| 12 | `.ravel()` | Manipulation | Flatten to 1-D |
| 13 | `.T` | Manipulation | Transpose |
| 14 | `np.swapaxes()` | Manipulation | Swap two axes |
| 15 | `np.concatenate()` | Manipulation | Join arrays along axis |
| 16 | `np.split()` | Manipulation | Split into sub-arrays |
| 17 | `np.hstack()` | Manipulation | Stack horizontally |
| 18 | `np.vstack()` | Manipulation | Stack vertically |
| 19 | `np.repeat()` | Manipulation | Repeat elements |
| 20 | `np.tile()` | Manipulation | Tile/repeat array |
| 21 | `np.add()` | Math Operations | Element-wise addition |
| 22 | `np.subtract()` | Math Operations | Element-wise subtraction |
| 23 | `np.multiply()` | Math Operations | Element-wise multiplication |
| 24 | `np.divide()` | Math Operations | Element-wise division |
| 25 | `np.power()` | Math Operations | Element-wise exponentiation |
| 26 | `np.sqrt()` | Math Operations | Square root |
| 27 | `np.exp()` | Math Operations | Exponential (e^x) |
| 28 | `np.log()` | Math Operations | Natural logarithm |
| 29 | `np.sin()` | Math Operations | Trigonometric sine |
| 30 | `np.cos()` | Math Operations | Trigonometric cosine |
| 31 | `np.mean()` | Statistics | Arithmetic mean |
| 32 | `np.median()` | Statistics | Median value |
| 33 | `np.std()` | Statistics | Standard deviation |
| 34 | `np.var()` | Statistics | Variance |
| 35 | `np.percentile()` | Statistics | Nth percentile |
| 36 | `np.max()` | Statistics | Maximum value |
| 37 | `np.min()` | Statistics | Minimum value |
| 38 | `np.unique()` | Statistics | Unique values and counts |
| 39 | `np.argmax()` | Statistics | Index of maximum |
| 40 | `np.random.rand()` | Random | Uniform floats in [0, 1) |
| 41 | `np.random.randint()` | Random | Random integers |
| 42 | `np.random.choice()` | Random | Sample from array |
| 43 | `np.random.uniform()` | Random | Uniform distribution |
| 44 | `np.random.normal()` | Random | Normal distribution |
| 45 | `np.random.exponential()` | Random | Exponential distribution |
| 46 | `np.random.binomial()` | Random | Binomial distribution |
| 47 | `np.random.poisson()` | Random | Poisson distribution |
| 48 | `np.random.seed()` | Random | Set seed for reproducibility |
| 49 | `np.dot()` | Linear Algebra | Dot product / matrix multiply |
| 50 | `np.linalg.inv()` | Linear Algebra | Matrix inverse |
| 51 | `np.linalg.det()` | Linear Algebra | Matrix determinant |
| 52 | `np.linalg.eig()` | Linear Algebra | Eigenvalues and eigenvectors |
| 53 | `np.linalg.solve()` | Linear Algebra | Solve linear system Ax = b |

---

> **Note:** This document covers all NumPy functions, methods, attributes, and constants used in the `Numpy.ipynb` notebook. Each entry includes the actual usage context from the notebook for easy cross-reference.
