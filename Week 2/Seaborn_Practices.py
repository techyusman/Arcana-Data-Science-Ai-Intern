import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


#we can directly improt dataset using seaborn as well it doesnt requires explicit data to download
Data = sns.load_dataset('penguins')

#Check its first 5 head of dataset
print(Data.head())

#counting species 
print(Data['species'].value_counts())

#counting island
print(Data['island'].value_counts())

# the ways of styling themes are as follows :
# white
# dark
# whitegird
# darkgird
# ticks

sns.set_style("whitegrid")
#create the scatterplot based on flipper length and bill length based on island
sns.scatterplot(data = Data, x = "flipper_length_mm", y ="bill_length_mm", hue="island")
#plt.show()
