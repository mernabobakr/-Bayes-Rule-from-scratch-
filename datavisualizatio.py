# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 07:59:53 2020

@author: Bassmala
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split



data = pd.read_csv("IRIS.csv") 
print(data.info())
print(data.describe())

print(data["species"].value_counts())
#plotting histogram
data.hist(bins=50, figsize=(20,15))
plt.show()

 
#dividing data
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)


# let`s see now how much correlation between the attributes 
corr_matrix = data.corr()
#print(corr_matrix["sepal_length"].sort_values(ascending=False))


# find nan values
sample_incomplete_rows = data[data.isnull().any(axis=1)].head()
#print(sample_incomplete_rows)






