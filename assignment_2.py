from __future__ import print_function, division
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sklearn
import seaborn as sns

#-----Question 1.1-----
auto_numeric_loc = os.path.join(os.getcwd(), 'datasets', 'train_auto_numeric.csv')
auto_numeric = pd.read_csv(auto_numeric_loc)
print("The number of data points = %d\nThe number of attributes = %d" % (len(auto_numeric.index), len(auto_numeric.columns) - 1))

#-----Question 1.2-----
auto_numeric.head(8)

#-----Question 1.3-----
auto_numeric.describe()
1
#-----Question 1.4-----
plt.scatter(auto_numeric['engine-power'], auto_numeric['price'], alpha = 0.4 )
plt.title('A Scatter Plot of Price against Engine-power',fontsize = 12)
plt.xlabel('Engine-power', color = 'red')
plt.ylabel('Price', color = 'green')
plt.show()

#-----Question 1.5-----
# Do you think that engine-power alone is sufficient for predicting the price? Can you make any other observations on the data from the above plot? Please explain your answer in 2-3 sentences.
# Your answer goes here
# Engine-power alone is not sufficient for predicting the price. Because the engine and the price is not completely linearly related. For example, when the engine energy consumption increases, the price does not rise, in some cases, low energy consumption, the price is almost as high as the situation of high energy consumption.
# Most of the price is concentrated in the 6000-10000
#-----Question 1.6-----
dis = plt.hist(auto_numeric['price'], bins = 60)
plt.xlabel('car prices', color = 'red')
plt.ylabel('frequency', color = 'blue')
plt.title(r'Distribution of the Car Prices', fontsize = 12)
plt.show()
#-----Question 1.7-----
How would you preprocess it to improve the performance of linear regression? Don’t do it at this stage, but instead in one sentence explain why you would do what you suggested.
Your answer goes here We should remove some outliers, for example, some of the price frequency is very small but the value is much higher than most of the price distribution of the interval


#-----Question 1.8-----
x = auto_numeric['engine-power']
y = auto_numeric['price']
#-----Question 1.9-----
x = x.reshape(auto_numeric.shape[0], 1)
print(x.shape)

#------Question 1.10-----
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size=0.2, random_state=0)
#------Question 1.11-----
lm = LinearRegression(normalize = True)
model = lm.fit(x_train, y_train)
#------Question 1.12-----
x_pre = lm.coef_ * x_test + lm.intercept_
print("x_pre = coefficient * x_test + intercept = {0}".format(x_pre))
#------Question 1.13-----
# What happens to the price as one more unit of engine-power is added? By examining the magnitude of the regression coefficient is it possible to tell whether or not engine-power is an important influential variable on price? Explain your answer in 1-2 sentences.
# Your answer goes here
# Will not change much
# Engine-power is not an important influential variable on price. Because the coefficient is very small only 0.08. So will not have a great impact.
#------Question 1.14-----
x_train_pre = lm.predict(x_train)
plt.scatter(x_train, y_train, alpha = 0.4, color = 'blue')
plt.scatter(x_train, x_train_pre, color='red', marker = 4, alpha = 0.5)
plt.title('A Scatter Plot of Training Data and Regression line of Predictions on Training Set',fontsize = 8)
plt.xlabel('Engine-power', color = 'red')
plt.ylabel('Price', color = 'green')
plt.show()
#------Question 1.15-----
So far we have used Hold-out validation. Can you think of a disadvantage of using this method, especially when dealing with small datasets?
Your answer goes here When a small data set is used to train the model in this way and then predict on a large data set, it will be significantly different from the true value.
#------Question 1.16-----
kf = KFold(len(x), n_folds=5, shuffle = True, random_state = 0)

#------Question 1.17-----
count = 0
for train_index, test_index in kf:
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    count += 1
    print("the mean value of the price variable for the training instances in fold %d = " %(count), y_train.mean())
#------Question 1.18-----
lm_new = LinearRegression(normalize = True)
predicted_new = cross_val_predict(lm_new, x, y, cv =kf)
print(predicted_new.shape, y.shape)
#------Question 1.19-----
r2 = r2_score(y, predicted_new)
rmse = mean_squared_error(y, predicted_new) ** 0.5
mae = mean_absolute_error(y, predicted_new)

cc = np.corrcoef(y, predicted_new)[0, 1]
print("R2 = {}\nRMSE = {}\nMAE = {}\nCC = {}".format(r2, rmse, mae, cc))
#------Question 1.20-----
What do the above metrics intend to measure? Relate the values of CC, MAE and RMSE to the observations you made in Question 1.5. Explain your answer in 1-2 short paragraphs.
Your answer goes here They intend to estimate the accuracy of the model. The MAE and RMSE measure the errors between the true values and predicted values. The CC measures the correlation of the dataset.
In the above results, a is large, cc is small, so ep is not enough to predict the price. In the above results, The MAE and RMSE are large, cc is small, so energe-power is not enough to predict the price. So their relevance is not enough.
#------Question 1.21-----
dis = plt.hist(y - predicted_new, bins = 60)
plt.xlabel('residuals', color = 'red')
plt.ylabel('frequency', color = 'blue')
plt.title(r'the residuals of the linear regression model', fontsize = 12)
plt.show()
#------Question 1.22-----
auto_base_loc = os.path.join(os.getcwd(), 'datasets', 'train_auto_base.csv')
auto_base = pd.read_csv(auto_base_loc)

x_new = auto_base['engine-power']
y_new = auto_base['price']
x_new = x_new.reshape(auto_base.shape[0], 1)
kf = KFold(len(auto_base), n_folds = 5, shuffle = True, random_state = 0)

lm_base = LinearRegression(normalize = True)
predicted_base = cross_val_predict(lm_base, x_new, y_new, cv = kf)

r2_base = r2_score(y_new, predicted_base)
rmse_base = mean_squared_error(y_new, predicted_base) ** 0.5
mae_base = mean_absolute_error(y_new, predicted_base)
cc_base = np.corrcoef(y_new, predicted_base)[0,1]
print("R2 = {}\nRMSE = {}\nMAE = {}\nCC = {}".format(r2_base, rmse_base, mae_base, cc_base))
#------Question 1.23-----
plt.figure()
plt.subplot(121)
plt.scatter(y_new, predicted_base, alpha = 0.4, color = 'black')
plt.title('The Scatter of True vs. Predicted Prices')
plt.xlabel('true price', color = 'red')
plt.ylabel('predicted price', color = 'green')
plt.subplots_adjust(wspace = 0.3, right = 1.5)
plt.subplot(122)
plt.scatter(x_new, predicted_base, alpha = 0.4, color = 'blue')
plt.title('The Scatter of Engine-power vs. Predicted Price')
plt.xlabel('true engine-power', color = 'red')
plt.ylabel('predicted price',  color = 'green')
plt.show()
#------Question 1.24-----
What is the simplest baseline model for the purposes of regression? Relate your answer to the regression model you have just built as part of this question. Can the predictions of this model be justified given the procedure you followed to train it?
Your answer goes here The simplest linear regression model is the sum of y divided by the sum of x equal to the coefficient, the constant is zero. No, in the simplest linear regression, all predicted values are the same.
#------Question 1.25-----
auto_base.describe()
#-----Question 2.1-----
auto_numeric_attri = auto_numeric.drop(['price'],axis=1).columns
pairplot1 = sns.pairplot(data = auto_numeric, x_vars = auto_numeric_attri[0:4], y_vars = 'price', size = 3)
pairplot2 = sns.pairplot(data = auto_numeric, x_vars = auto_numeric_attri[4:8], y_vars = 'price', size = 3)
pairplot3 = sns.pairplot(data = auto_numeric, x_vars = auto_numeric_attri[8:12], y_vars = 'price', size = 3)
pairplot4 = sns.pairplot(data = auto_numeric, x_vars = auto_numeric_attri[12:], y_vars = 'price', size = 3)

#-----Question 2.2-----
lm_mlr = LinearRegression(normalize = True)
x_mlr = auto_numeric.drop(labels = 'price', axis = 1)
y_mlr = auto_numeric['price']
mlr_pre = cross_val_predict(lm_mlr, x, y, cv = 5)
kf = KFold(len(auto_numeric), n_folds = 5, shuffle = True, random_state = 0)
predicted_mlr = cross_val_predict(lm_mlr, x_mlr, y_mlr, cv = kf)
#-----Question 2.3-----
frmse_mlr = mean_squared_error(y_mlr, predicted_mlr) ** 0.5
mae_mlr = mean_absolute_error(y_mlr, predicted_mlr)
cc_mlr = np.corrcoef(y_mlr, predicted_mlr)[0,1]
print("RMSE = {}\nMAE = {}\nCC = {}".format(rmse_mlr, mae_mlr, cc_mlr))

#-----Question 2.4-----
# Comment on each metric display above in comparison to what you have obtained for the Simple Linear Regression model in Question 1.19.
# Your answer goes here The CC in Question 2.3 is twice as that in question 1.19, and the RMSE and MAE in 2.3 are smaller than those in 1.19. That means 2.3 are better at predicting the price.

# -----Question 2.5-----
dis = plt.hist(auto_numeric['engine-size'], bins = 50)
plt.xlabel('engine-size', color = 'red')
plt.ylabel('frequency', color = 'green')
plt.title(r' The Histogram for the Engine-size ', fontsize = 12)
plt.show()
# -----Question 2.6-----
# Is the distribution expected to cause a problem for regression? Explain your answer in 2-3 sentences.
# Your answer goes here
# Yes. Beacuse the distribution is dense and most of them just in the range of 3-10, using it to predict the price became difficult and the outliers is obvious.

# -----Question 2.7-----
auto_numeric2 = auto_numeric.copy(deep=True)
auto_numeric2['engine-size'] = np.log(auto_numeric['engine-size'])
plt.hist(np.sqrt(auto_numeric2['engine-size']), bins = 40)
plt.title('The histogram of the transformed attribute')
plt.xlabel('engine-size')
plt.ylabel('frequency')
plt.show()

#-----Question 2.8-----
lr_tr = LinearRegression(normalize=True)
x_tr = auto_numeric2.drop(['price'],axis=1)
kf = KFold(len(auto_numeric), n_folds = 5, shuffle = True, random_state = 0)
predicted_tr = cross_val_predict(lr_tr, x_tr, y, cv = kf)
rmse_tr = mean_squared_error(predicted_tr, y) ** 0.5
mae_tr = mean_absolute_error(y, predicted_tr)
cc_tr = np.corrcoef(predicted_tr,y)[1,0]
r2_tr = r2_score(predicted_tr, y)
print("R2 = {}\nRMSE = {}\nMAE = {}\nCC = {}".format(r2_tr, rmse_tr, mae_tr, cc_tr))
#-----Question 2.9-----
How has the performance of your model changed? Explain your answer in 1-2 sentences.
Your answer goes here
The model became better, because the error of model became smaller and the CC became larger than before.
#-----Question 2.10-----
pd.set_option('max_columns', 30)
auto_full_loc = os.path.join(os.getcwd(), 'datasets', 'train_auto_full.csv')
auto_full = pd.read_csv(auto_full_loc, delimiter = ',')
print("the number of samples is: {}".format(auto_full.shape[0]))
print("the number of attributes is: {}".format(auto_full.shape[1] - 1))
auto_full.head(20)
#----Question 2.11-----
# This dataset contains a mixture of numeric and nominal attributes. Name the variables that you think are categorical. Why can we not use the nominal attributes in their current form for the purposes of regression?
# Your answer goes here
# The categorical variables: make, fuel-type, aspiration, body-style, drive-wheels, engine-location, engine-type, fuel-system and symboling. In the regression, we can only use the positive norminal attributes which are string type or negative norminal attributes.
#-----Question 2.12-----
auto_full_edit = auto_full.copy(deep=True)
attributes = ['make', 'fuel-type', 'aspiration', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'fuel-system', 'symboling']
labelE = []
for i in range(len(attributes)):
    labelE.append(LabelEncoder())
for index,item in enumerate(attributes):
    labelE[index].fit(auto_full_edit[item])
    auto_full_edit[item] = labelE[index].transform(auto_full_edit[item])
enc = OneHotEncoder(categorical_features=np.array([1,2,3,4,5,6,7,12,13,15,23]))
x = auto_full_edit.drop(['price'],axis=1).values
enc.fit(x)
x_enc = enc.transform(x).toarray()
print("the shape of X_enc is: ",x_enc.shape)
#-----Question 2.13-----
y = auto_full_edit['price']
lm_full = LinearRegression(normalize=True)
kf = KFold(auto_full_edit.shape[0], n_folds=5,shuffle=True,random_state=0)
y_pred = cross_val_predict(lm_full, x_enc, y ,cv=kf)
rmse_full = mean_squared_error(y_pred, y) ** 0.5
mae_full = mean_absolute_error(y, y_pred)
cc_full = np.corrcoef(y_pred,y)[1,0]
r2_full = r2_score(y_pred,y)
print("R2 = {}\nRMSE = {}\nMAE = {}\nCC = {}".format(r2_full, rmse_full, mae_full, cc_full))
#-----Question 2.14-----
How does this more complex model perform with respect to your best performing model from either question 2.3 or 2.8? List one advantage and one disadvantage of using the more complex model.
Your answer goes here
The model in question 2.8 is better than that in question 2.3.
The advantage is it performs less mistakes in training data.
The disadvantage is adding too many addtional attributes which may ask more calculating time and outliers in training data will cause more mistakes.
#-----Question 2.15-----
dtr = DecisionTreeRegressor(random_state=0)
kf = KFold(auto_full_edit.shape[0], n_folds=5,shuffle=True,random_state=0)
y_pred = cross_val_predict(dtr, x_enc, y ,cv=kf)
rmse_dtr = mean_squared_error(y_pred, y) ** 0.5
mae_dtr = mean_absolute_error(y, y_pred)
cc_dtr = np.corrcoef(y_pred,y)[1,0]
r2_dtr = r2_score(y_pred,y)
print("R2 = {}\nRMSE = {}\nMAE = {}\nCC = {}".format(r2_dtr, rmse_dtr, mae_dtr, cc_dtr))
