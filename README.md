# Bayesian-rule-Merna Abobakr

**Implementation of Bayesian Rule in python to classify IRIS Flowers**

* I dowloaded the dataset from this link [kaggle](https://www.kaggle.com/arshid/iris-flower-dataset)
.which contains 5 attributes [sepal_length, sepal_width, petal_length, petal_width, species] with three different classes
[Iris-setosa,Iris-versicolo,Iris-virginica].



### Fistly in file "data visualization..py"
* After reading the csv data file, I'll use `.info()` to know some information about them as shown in the first screenshot below.



* Also to Check some statistics for these date ,such as [count, mean, min, std,...] using `.describe()`



* Plotting a histogram for each numerical attribute:



* Correlation Calculation to see how much the features are correlated to each other.

* See if the data has NAN values using `.isnull()` , and there are no missing values in this dataset so there is no requirement for imputation, and also the 4 continuous variables are 
approximately on the same scale so there is no normalization needed.


### secondly in file "main..py"

* Firstly divide the data into train and test using "StratifiedShuffleSplit" ,then use the train data to apply the functions of baye's .
* Secondly test the data and compare predicted output with real output to get the accuracy.
* The accuracy is 96.7%