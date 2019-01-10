#import packages 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read the dataset

dataset= pd.read_csv('marks.csv')

#divide the dataset

x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

#splitting the data based on training and test

from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/5,random_state=0)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print(x_train)
print(y_train)

#impliment the classifiers based on simple linear regression

from sklearn.linear_model import LinearRegression
simpleLinearRegression=LinearRegression()
simpleLinearRegression.fit(x_train,y_train)

y_predict=simpleLinearRegression.predict(x_test)
print(x_test)

#sample=np.asarray(5.5)
#sample.reshape(-1,1)
#y_predict_val=simpleLinearRegression.predict(sample)
#print(y_predict_val)

#implement the graph for simple linear regression

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,simpleLinearRegression.predict(x_train))
         






