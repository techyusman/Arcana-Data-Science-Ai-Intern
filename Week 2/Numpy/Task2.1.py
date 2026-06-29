import numpy as np

random_age = np.random.randint(18,60,1000)
#print(random_age)

agerange =random_age[(random_age > 24) & (random_age < 36)]

percentage = (len(agerange)/1000)*100

print("Percentage of people in age range 24-36: ", percentage)
