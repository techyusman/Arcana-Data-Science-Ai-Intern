import numpy as np

prices = np.array([120, 121, 119, 122, 118, 117, 115, 116, 118, 120])

print("Prices for days 2 to 6: ", prices[1:6])

print("Prices greater than 118: ", prices[prices > 118])

