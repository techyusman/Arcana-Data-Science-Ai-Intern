import numpy as np

monthly_sales = np.array([
  120, 135, 150, 160, 170, 180, 195, 200, 185, 175, 160, 210,  # Year 1
    130, 140, 155, 165, 175, 190, 200, 210, 195, 180, 165, 220,  # Year 2
    140, 150, 160, 175, 185, 195, 210, 220, 200, 190, 170, 230   # Year 3
])

monthly_sales = monthly_sales.reshape(3,12)

print("Total sales per year: ", np.sum(monthly_sales, axis=1))

print("Average monthly sales: \n", np.mean(monthly_sales, axis=0).round(1))


#argmax tells the index of the max value in the given array, so we add 1 to get the month number

best_month = np.argmax(np.mean(monthly_sales, axis=0))+1

print("Best month for sales:", best_month)
