# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:47:37 2020

@author: Bassmala
"""

from sklearn.model_selection import train_test_split
from math import sqrt, pi, exp
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)

def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)

    return probabilities




#get prediction for every single row         
def Rowprediction(summaries, row):  
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_probability = None, -1
	for class_value, probability in probabilities.items():
		if  probability > best_probability:
			best_probability = probability
			best_label = class_value
	return best_label 





#get the system accuracy

def getAccuracy(test,summary):
    correct_predictions = 0
    for i in range(len(test)):
        predicted_label=Rowprediction(summary,test[i])
        if test[i][-1] == predicted_label:
            print("yes")
            correct_predictions += 1
    return (correct_predictions / float(len(test))) * 100.0 




       

df = pd.read_csv("IRIS.csv") 
df=pd.DataFrame(df)

attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "Classs"]
df.columns = attributes
classs = {'Iris-setosa': 0,'Iris-versicolor': 1, 'Iris-virginica': 2} 
df.Classs = [classs[item] for item in df.Classs]
#train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df,df["Classs"]):
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]

df = [list(row) for row in train_set.values]
dataset=separate_by_class(df)
summary = summarize_by_class(df)
test_set=[list(row) for row in test_set.values]
print(test_set[0])


            

probability=Rowprediction(summary,test_set[1])
#print(summary)
acc=getAccuracy(test_set,summary)
print(acc)












