import numpy as np

emp_salary_in_thousands = np.array([[120, 150, 250],
[150, 160, 260],  
[140, 145, 245]])

bonus = np.array([5, 10, 15])

print("Salaries after bonus (in thousands): ", emp_salary_in_thousands + bonus)
