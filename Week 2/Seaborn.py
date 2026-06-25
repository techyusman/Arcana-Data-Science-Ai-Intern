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

#sns.set_style("whitegrid")
#create the scatterplot based on flipper length and bill length based on island
#sns.scatterplot(data = Data, x = "flipper_length_mm", y ="bill_length_mm", hue="island")

# Creating Stripplot
#sns.stripplot(data = Data, x = "species" , y = "bill_length_mm", hue = "island", dodge=True)


# Creating swarmplot
#sns.swarmplot(data = Data, x = "species" , y = "bill_length_mm", hue = "island")

# Visualizing one features using histogram and distigustion based on sex

#sns.histplot(Data, x = "body_mass_g", hue = "sex", multiple="stack")

# line plot 
#sns.lineplot(Data, x = "body_mass_g" , y = "flipper_length_mm", hue = "sex")


# Joint plot combining two different plot, historgram and scatter plot
#sns.jointplot(Data, x="body_mass_g", y= "flipper_length_mm", hue="sex")

# Bar Plot
#sns.barplot(Data, x="species", y = "bill_length_mm", hue = "sex",)

#box plot, it help us to identify the outlier in the data

#sns.boxplot(Data, x="species", y = "body_mass_g", hue = "sex")


# KDE - For estimating the propability density function of continous. 
# Or in simple terms : a smooth curve that shows the distribution of data. 
# it helps you see where the data is peaks and how it spreads

#sns.kdeplot(Data, x= "body_mass_g", hue = "sex", fill=True)

#Heatmap : a 2d color- coded matrix that shows the relationships or patterns b/w two variables
# shows the correlations b/w features mainly usend in numarical values

columns = ["bill_length_mm", "bill_depth_mm","flipper_length_mm", "body_mass_g"]

sns.heatmap(Data[columns].corr() , annot=True)
plt.show()
