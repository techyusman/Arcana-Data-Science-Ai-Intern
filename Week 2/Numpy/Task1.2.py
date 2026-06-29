"""

Memory: NumPy arrays use less memory because all elements have the same 
data type and are stored continuously in memory, 
while Python lists store references to individual objects, 
increasing memory overhead.


Operations: NumPy arrays perform fast vectorized computations 
(e.g., arr + 5, arr * 2) without explicit loops, 
whereas Python lists generally require iteration using for loops 
or list comprehensions for similar operations.



Observation:
The NumPy array performs the multiplication directly on all 
elements (vectorization), making the code simpler and much faster 
than using a Python list with a loop, especially for large datasets


"""