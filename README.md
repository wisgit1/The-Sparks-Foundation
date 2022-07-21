# The-Sparks-Foundation
Task 1 - Prediction using Supervised Machine Learning
Simple Linear Regression

In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

import pandas as pd
data=pd.read_csv('data.csv')
print('Data imported sucessfully')
data.head()
# to check if the dataset contains null /missing values
data.isnull().sum()  
# to find no. of rows and columns
data.shape
data.info()
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# plotting the data to check its relationship 

plt.figure()
# plt.rcParams['figure.figsize'] = [16,9]   
data.plot(x='Hours',y='Scores',style='*',color = 'k')
plt.title('Hours vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Percentage scored')
plt.grid()
plt.show()
# to check the correlation 
data.corr()
import seaborn as sns
sns.heatmap(data.corr(),annot = True)
# data preparation
# using iloc we divide the data 
X=data.iloc[:,:-1].values   # independent variable
y=data.iloc[:,1].values     # dependent variable 
from sklearn.model_selection import train_test_split
# spliting the data into training and testing data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
# Training the algorithm 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr.coef_
lr.intercept_
# Plotting the regression line
line = lr.coef_*X+lr.intercept_
# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()
line = lr.coef_*X+lr.intercept_

# Plotting for the training data
plt.figure()
# plt.rcParams['figure.figsize'] = [16,9]
plt.scatter(X_train, y_train, color='red')
plt.plot(X,line,color='green')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()
# Plotting for the testing data
plt.figure()
# plt.rcParams['figure.figsize'] = [16,9]
plt.scatter(X_test, y_test, color='red')
plt.plot(X,line,color='green')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()
print(X_test) # Testing data - In Hours
y_pred = lr.predict(X_test) # Predicting the scores
y_pred
y_test
# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 
# You can also test with your own data
hours = 9.25
own_pred = lr.predict([[hours]])
print('The predicted score if a person studies for',hours,'hours is ',own_pred[0])
# Evaluating the model 
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 
