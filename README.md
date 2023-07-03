# Regression

we will practice with simple linear regression with the California Housing Dataset.

You should be familiar with how to perform multiple linear regression for predicting housing prices using the California Housing dataset for both training and testing. 

In this lab exercise you will  perform single linear regressions with each feature (one column) of the data set individually and calculate two metrics for the predictions obtained from the linear regression. 

First split the dataset into training and testing sets, then select one feature. This was illustrated previously using DataFrames and applying simple linear regression. Next, train the model using that one feature and make predictions as you we have done in the multiple linear regression case study. Repeat this process for each of the eight features. Print out R2 score and mean squared error score for each of the separate simple linear regression steps.

Produce a table that can be used to determine which feature produced the best results. Write a short explanation of how you would evaluate the results of the simple linear regression estimators, and comparing them with the results from the multiple linear regression.

Deliverables: Upload your code and table for this assignment.

Example output table :

Prints out for each feature in the data the R2 score and the MSE score.

Multiple Linear Regression using All features  
R2 score : 0.6008983115964333
MSE score: 0.5350149774449118

Feature 0 has R2 score : 0.4630810035698606
          has MSE score 0.7197656965919478
Feature 1 has R2 score : 0.013185632224592903
          has MSE score 1.3228720450408296
Feature 2 has R2 score : 0.024105074271276283
          has MSE score 1.3082340086454287
Feature 3 has R2 score : -0.0011266270315772875
          has MSE score 1.3420583158224824
Feature 4 has R2 score : 8.471986797708997e-05
          has MSE score 1.3404344471369465
Feature 5 has R2 score : -0.00018326453581640756
          has MSE score 1.340793693098357
Feature 6 has R2 score : 0.020368890210145207
          has MSE score 1.3132425427841639
Feature 7 has R2 score : 0.0014837207852690382
          has MSE score 1.3385590192298276All
