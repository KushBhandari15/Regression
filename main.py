# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:51:28 2022

@author: Kush Bhandari
"""

# Script to Visualize the the Expected vs.  
# Predicted Prices using Multiple Linear  
# Regression Housing Price Estimator 
import pandas as pd 
from sklearn.datasets import fetch_california_housing 

cali = fetch_california_housing() 
cali_df = pd.DataFrame(cali.data, columns=cali.feature_names) 
cali_df['MedHouseValue'] = pd.Series(cali.target) 

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split( cali.data, cali.target, random_state=11) 

from sklearn.linear_model import LinearRegression 

mu_regress = LinearRegression() 
mu_regress.fit(X=X_train, y=y_train) 
predicted = mu_regress.predict(X_test) 
expected = y_test 

from sklearn import metrics

'''z = zip(predicted[::1000], expected[::1000]) 
count=0
print(X_test.size)
for p, e in z: 
    print("Feature ",count,"has R2 score: ",metrics.r2_score(expected, predicted)) 
    print("Feature ",count,"has MSE score: ",metrics.mean_squared_error(expected, predicted))
    count=count+1
    print("\n")

cali = pd.DataFrame() 
cali_df['Expected'] = pd.Series(expected) 
cali_df['Predicted'] = pd.Series(predicted) 

'''

print("Using All features")
X = cali_df.iloc[:,0:8]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression() 
mu_regress.fit(x_train,y_train)
y_predict = mu_regress.predict(x_test)
print("R2 score: ",metrics.r2_score(y_test,y_predict))
print("MSE score: ",metrics.mean_squared_error(y_test,y_predict))
print("\n\n")

X = cali_df[['MedInc']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 0 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("Feature 0 has MSE score: ",metrics.mean_squared_error(y_test,y_pred))
print("\n")


X = cali_df[['HouseAge']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 1 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("Feature 1 has MSE score: ",metrics.mean_squared_error(y_test,y_pred))
print("\n")

X = cali_df[['AveRooms']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 2 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("Feature 2 has MSE score: ",metrics.mean_squared_error(y_test,y_pred))
print("\n")

X = cali_df[['AveBedrms']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 3 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("Feature 3 has MSE score: ",metrics.mean_squared_error(y_test,y_pred))
print("\n")

X = cali_df[['Population']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 4 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("Feature 4 has MSE score: ",metrics.mean_squared_error(y_test,y_pred))
print("\n")

X = cali_df[['AveOccup']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 5 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("Feature 5 has MSE score: ",metrics.mean_squared_error(y_test,y_pred))
print("\n")

X = cali_df[['Latitude']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 6 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("Feature 6 has MSE score: ",metrics.mean_squared_error(y_test,y_pred))
print("\n")

X = cali_df[['Longitude']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 7 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("Feature 7 has MSE score: ",metrics.mean_squared_error(y_test,y_pred))
print("\n")