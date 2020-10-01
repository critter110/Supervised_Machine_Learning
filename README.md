# Supervised_Machine_Learning
## Summary
  For this challenge we were given a large data set containing many instances of loan applications. Each instance had many columns containg details about the loan application. Using Jupyter Notebook as our enviornment, The CSV file was uploaded, read into a pandas dataframe, and cleaned up a little bit by droping some columns that were unrelated to our analysis.
  Next we split the cleaned dataframe into two seperate dataframes. The first was a dataframe of all the attributes (independent variables), and the second was only one column, the loan status, which is the dependent variable and the one we are trying to predict. Then using sklearn's train_test_split method, we split the attribute target dataframes into training and testing sets. 75% of the original loan instances are used for training the logistical regression model, and 25% of the original loan instances are used for testing the accuracy of the model.
Since there are a lot more good loans than bad, we used different techniques to even out the training data so there would be roughly equal amounts of good loans and bad loans. The first technique used was to randomly oversample the training data using imblearn's RandomOverSampler method. This method increases the number of bad loans by randomly selecting bad loans allowing duplicates to increase the number of bad loans the model has to work with. The equal amount of good loans and bad loans should give the model an easier time fitting the data. After randomly oversampling, we fit the logistical regression model to the training data. Then we create predictions using the testing data. We then compare our predictions with the known outcomes of our testing data. Then we compared our model's predictions with the known outcomes to assess the accuracy of our model. 
  The next method we used was imblearn's SMOTE oversampling method. This method again oversamples the under represented bad loans catagory, but this time instead of using duplicates to increase the number of instances, it creates new instances by interpolating the bad loan data. Again with equal numbers of good and bad loans, the same process was followed of fitting, predicting, and analyzing the accuracy of the predictions.
  The next method we used was imblearns ClusterCentroids undersampling method. This time, the method cuts down on the number of good loans in order to get an equal size of good and bad loans. It does this by grouping the good loan data into clusters with similar good loan instances grouped together. Then it takes the centroid of this cluster and uses data point instead of the actual loan instances. Once equal number of good and bad loans was achieved, the same fit, predict analyze procedures were followed.
  The final method we used was imblearn's SMOTEENN method which oversamples the under represented instances, and under samples the over represented instances. Again the same fit, predict, analyze procedure was used. 

## Analysis
We originally started with 68,470 low risk (good) loans and 347 high risk (bad) loans. After randomly oversampling the high risk loans, we had 51,344 of both for the logistic regression model to work with. After fitting and predicting, the model had a balanced accuracy score of 0.629. The confusion matrix is below. 

|                    | Predicted High Risk | Predicted Low risk | 
|--------------------|---------------------|--------------------|
| Actually high risk | 52                  | 27                 | 
| Actually low risk  | 6825                | 10301              | 


The classification report is Below

|                |pre       |rec       |spe        |f1       |geo       |iba       |sup    |
|----------------|----------|----------|-----------|---------|----------|----------|-------|
| high_risk      |0.01      | 0.66     | 0.60      |0.01     |0.63      | 0.40     |   79  |
|   low_risk     |  1.00    |  0.60    |  0.66     | 0.75    |  0.63    |  0.39    | 17126 |
| avg / total    |   0.99   |   0.60   |  0.66     | 0.75    |  0.63    |  0.39    | 17205 |

We can see that the random oversampling method is pretty bad at correctly predicting high-risk loans more so in the sense that it is overly safe and predicts that too many low risk loans are actually high risk. This isn't the worst case scenario since it is probably better to be on the safe side when predicting high-risk loans. 

Next using the SMOTE oversampling method, we again ended up with 51,344 of each good and bad loans. After fitting and predicting, we got a balanced accuracy score of 0.626. Pretty close to the randomly oversampled results. The confusion matrix and classification report are below. 

|                    | Predicted High Risk | Predicted Low risk | 
|--------------------|---------------------|--------------------|
| Actually high risk | 47                  | 32                 | 
| Actually low risk  | 5882                | 11244              | 


                  pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.59      0.66      0.02      0.62      0.39        79
   low_risk       1.00      0.66      0.59      0.79      0.62      0.39     17126

avg / total       0.99      0.66      0.60      0.79      0.62      0.39     17205

This model has the same problems as the random over sampling. It predicts a lot of actually low risk loans to be high risk. 

Next using the cluster centroid under sampling method, we ended up with only 268 instances of good loans and bad loans, one of the negative consequences of undersampling. After fitting and predicting, we got a balanced accuracy score of 0.505. The confusion matrix and classification report are below. 

|                    | Predicted High Risk | Predicted Low risk | 
|--------------------|---------------------|--------------------|
| Actually high risk | 49                  | 30                 | 
| Actually low risk  | 10447               | 6679               | 


                 pre       rec       spe        f1       geo       iba       sup

  high_risk       0.00      0.62      0.39      0.01      0.49      0.25        79
   low_risk       1.00      0.39      0.62      0.56      0.49      0.24     17126

avg / total       0.99      0.39      0.62      0.56      0.49      0.24     17205

We can see that this model is even worse than the previous two at being too eager to predict a high risk loan. 

Finally, using the combination over and under sampling method, we trained the model with 68,458 high risk loans, and 62,022 low risk loans. After fitting and predicting, we got a balanced accuracy score of 0.671. The best of any model so far. The confusion matrix and classification report are below. 

|                    | Predicted High Risk | Predicted Low risk | 
|--------------------|---------------------|--------------------|
| Actually high risk | 59                  | 20                 | 
| Actually low risk  | 6932                | 10194              | 


                  pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.75      0.60      0.02      0.67      0.45        79
   low_risk       1.00      0.60      0.75      0.75      0.67      0.44     17126

avg / total       0.99      0.60      0.75      0.74      0.67      0.44     17205

It is important to focus on the model's ability to predict high risk loans as that is the whole point of using the model. Although none of the models are very good at doing this, the combo over and under sampling method did this the best. 

