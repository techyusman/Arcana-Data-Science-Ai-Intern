# Import the pyplot module from matplotlib, which provides a MATLAB-like plotting framework.
# We usually import it with the alias 'plt' for convenience.
import matplotlib.pyplot as plt

# Import numpy for generating sample data easily.
# We usually import it with the alias 'np'.
import numpy as np





# ==========================================
# 1. Basic Line Plot
# ==========================================

# Generate an array of 10 evenly spaced values between 0 and 10 for the x-axis.
x = np.linspace(0, 10, 10)

# Calculate the y-values as the square of the x-values.
y = x ** 2

# Create a new figure (the window or page where the plot is drawn).
# figsize=(8, 4) sets the width to 8 inches and height to 4 inches.
plt.figure(figsize=(8, 4))

# Plot the x and y data.
# The 'b-' argument means use blue color ('b') and a solid line ('-').
# label="y = x^2" assigns a label to this line, which will be used in the legend.
plt.plot(x, y, 'b-', label="y = x^2")

# Set the title of the plot.
plt.title("Basic Line Plot")

# Set the label for the x-axis.
plt.xlabel("X-axis (Values)")

# Set the label for the y-axis.
plt.ylabel("Y-axis (Squared Values)")

# Display the legend to identify the plotted lines based on their labels.
plt.legend()

# Display the plot on the screen.
# Execution will pause here until the plot window is closed.
plt.show()







# ==========================================
# 2. Scatter Plot
# ==========================================

# Generate 50 random values for the x-axis.
x_scatter = np.random.rand(50)

# Generate 50 random values for the y-axis.
y_scatter = np.random.rand(50)

# Create a new figure with specific dimensions.
plt.figure(figsize=(6, 4))

# Create a scatter plot using the random x and y values.
# color='red' sets the color of the points, and marker='o' uses circles as points.
plt.scatter(x_scatter, y_scatter, color='red', marker='o', label="Random Points")

# Add a title to the scatter plot.
plt.title("Scatter Plot Example")

# Add a label to the x-axis.
plt.xlabel("Random X")

# Add a label to the y-axis.
plt.ylabel("Random Y")

# Add a grid to make it easier to read values.
plt.grid(True)

# Show the legend.
plt.legend()

# Display the scatter plot.
plt.show()







# ==========================================
# 3. Bar Chart
# ==========================================

# Define a list of categories for the bar chart.
categories = ['Category A', 'Category B', 'Category C', 'Category D']

# Define the corresponding values for each category.
values = [15, 30, 45, 10]

# Create a new figure for the bar chart.
plt.figure(figsize=(7, 5))

# Create a bar chart with the categories on the x-axis and values on the y-axis.
# color='skyblue' fills the bars with a sky blue color.
plt.bar(categories, values, color='skyblue')

# Add a title to the bar chart.
plt.title("Bar Chart Example")

# Add labels to the axes.
plt.xlabel("Categories")
plt.ylabel("Values")

# Display the bar chart.
plt.show()








# ==========================================
# 4. Histogram
# ==========================================

# Generate 1000 random numbers from a normal (Gaussian) distribution.
data = np.random.randn(1000)

# Create a new figure for the histogram.
plt.figure(figsize=(7, 5))

# Create a histogram using the generated data.
# bins=30 divides the data into 30 intervals (bars).
# color='purple' fills the bars, and edgecolor='black' adds a border to each bar.
plt.hist(data, bins=30, color='purple', edgecolor='black')

# Add a title to the histogram.
plt.title("Histogram Example")

# Add an x-axis label.
plt.xlabel("Data Values")

# Add a y-axis label showing the count in each bin.
plt.ylabel("Frequency")

# Display the histogram.
plt.show()
