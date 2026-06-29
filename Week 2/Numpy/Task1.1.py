import numpy as np

Temp_hourly = np.array([
    32.5, 33, 31, 30, 29.5, 27,
    29, 31, 33, 34, 36, 37,
    37, 36, 38, 40, 42, 41,
    40, 42, 43, 40, 36, 35
])

print("Average Hourly Temperature:", np.mean(Temp_hourly))
print("Maximum Hourly Temperature:", np.max(Temp_hourly))
print("Minimum Hourly Temperature:", np.min(Temp_hourly))

Temp_fahrenheit = (Temp_hourly * 9/5) + 32

print("Temperature in Fahrenheit: \n", Temp_fahrenheit)