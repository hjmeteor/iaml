from __future__ import print_function, division
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sklearn
import seaborn as sns

#-----Question 1.1-----
#Load the datasets train_20news_partA.csv
train_20news_partA_loc = os.path.join(os.getcwd(), 'datasets',
                                      'train_20news_partA.csv')
train_20news_partA = pd.read_csv(train_20news_partA_loc)

#Load the datasets train_20news_partB.csv
train_20news_partB_loc = os.path.join(os.getcwd(), 'datasets',
                                      'train_20news_partB.csv')
train_20news_partB = pd.read_csv(train_20news_partB_loc)

#-----Question 1.2-----
#Display basic information for dataset A such as number of columns, type, and memory usage
train_20news_partA.info()

#-----Question 1.3-----
# #How many data points and how many attributes are there in the dataset that we can use to model the target variable class?

# Your answer goes here
# There are 2129 data points and 521 attributes.

#-----Question 1.4-----
train_20news_partA.describe()

#-----Question 1.5-----
train_20news_partA.head(7)

#-----Question 1.6-----
import re
attribute = pd.Series(train_20news_partA.columns)
#Precompiled the regular expressions
re_extract_name = re.compile('^(w\d{0,3})\_(\w+)$')
#Using regular expressions to match the actual name of the attributes
for n in range(len(attribute) - 1):
    attribute[n] = re_extract_name.match(attribute[n]).group(2)
attribute.head(100)
#-----Question 1.7-----
ax = sns.stripplot(
    x=train_20news_partA['w10_home'],
    data=train_20news_partA,
    jitter=True,
    alpha=0.55)


#-----Question 1.8-----
def scatter_jitter(arr1, arr2, jitter=0.2):
    """ Plots a joint scatter plot of two arrays by adding small noise to each example.
    Noise is proportional to variance in each dimension. """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    arr1 = arr1 + jitter * arr1.std(
        axis=0) * np.random.standard_normal(arr1.shape)
    arr2 = arr2 + jitter * arr2.std(
        axis=0) * np.random.standard_normal(arr2.shape)
    plt.scatter(arr1, arr2, marker=4)


scatter_jitter(
    train_20news_partA['w10_home'],
    train_20news_partA['w12_internet'],
    jitter=0.2)
plt.title('Joint Distribution of Two Variables', fontsize=12)
plt.xlabel('w10_home', color='red')
plt.ylabel('w12_internet', color='green')
#-----Question 1.9-----
news_A_clean = pd.DataFrame(train_20news_partA)
t20newspA_mean = train_20news_partA.mean()
t20newspA_std = train_20news_partA.std()
t20newspA_index = len(train_20news_partA.index)
t20newspA_column = len(train_20news_partA.columns)
#Used in the construction of approximate 95% confidence intervals.
n = 1.96
#The upper and lower of the confidence intervals
CI_upper = t20newspA_mean + n * t20newspA_std
CI_lower = t20newspA_mean - n * t20newspA_std
attribute = pd.Series(train_20news_partA.columns)
#The outliers are out of the confidence intervals(>mean+1.96*std or <mean-1.96*std)
for c in range(len(attribute) - 1):
    news_A_clean = news_A_clean.ix[(news_A_clean[attribute[c]] <= CI_upper[
        c]) & (news_A_clean[attribute[c]] >= CI_lower[c])]
print(news_A_clean)

#------Question 1.10-----
news_A_clean_lengh = len(news_A_clean.index)
excluded_outliers = t20newspA_index - news_A_clean_lengh
print("news_A_clean_lengh = %d\nexcluded_outliers = %d" %
      (news_A_clean_lengh, excluded_outliers))

#-----Question 2.1-----
scatter_jitter(news_A_clean['w281_ico'], news_A_clean['w273_tek'], jitter=0.1)
plt.title('Joint Distribution of Two Variables', fontsize=12)
plt.xlabel('w281_ico', color='red')
plt.ylabel('w273_tek', color='green')
plt.show()

#-----Question 2.2-----
# What do you observe?
# How does that relate to the Naive Bayes assumption?
# What would be the main issue we would have to face if we didn't make this assumption?
# Your answer goes here:
# The value of different attributions in the same datapoint is the same for most of the time. For example, most of the points in the figure appear in (1,1),(2,2)...Only a small part of which appears in (1,2) (2,1).
# The Naive Bayes classifier uses the conditional independence assumption. However, as can be seen from the figure, these two attributions are not independent of each other. So it may affect the performance of the classifier.
# If the attributes are not conditionally independent, then the sample space will be large, the reality that the training set is not large enough compared with the sample space, and the directly using the frequency to calculate P(x1,x2...xN|y) is very difficult, more importantly, usually the probability is zero and not being observed are not the same.

#-----Question 2.3-----
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
features = news_A_clean.drop('class', axis=1)
targets = pd.Series(news_A_clean['class'])
gnb = GaussianNB()
fit = gnb.fit(features, targets.values)
accuracy = fit.score(features, targets.values)
print("The classification accuracy = %.12f" % accuracy)

#-----Question 2.4-----
from sklearn.preprocessing import normalize
features_pre = fit.predict(features)
cm = confusion_matrix(targets, features_pre)
cm_norm = normalize(cm, norm='l1')


def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(
            cm,
            xticklabels=classes,
            yticklabels=classes,
            vmin=0.,
            vmax=1.,
            annot=True)
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


classes = [1, 2, 3, 4, 5]
plot_confusion_matrix(cm_norm, classes=classes)

# -----Question 2.5-----
# Comment on the confusion matrix from the previous question. Does it look like what you would have expected? Explain.
# Your answer goes here
# It look like what I would have expected. Because the classifaction accuracy is 0.889048302248, most of the classification is correct. But the classification of class 2 there are some deviations.

# -----Question 2.6-----
unclean_features = train_20news_partA.drop('class', axis=1)
unclean_targets = pd.Series(train_20news_partA['class'])
gnb = GaussianNB()
unclean_fit = gnb.fit(unclean_features, unclean_targets.values)
unclean_accuracy = unclean_fit.score(unclean_features, unclean_targets.values)
print("The classification accuracy on the unclean training dataset = %.12f" %
      unclean_accuracy)

# -----Question 2.7-----
# Comment on the above results (Questions 2.3 & 2.6). In particular explain why you think that cleaning the data helps in this case.
# Your answer goes here
# Based on the performance of the classifier's training on a dataset that is not cleaned, outliers have a large effect on the performance of the classifier, since the values of the outliers are 100 times the values of normal datapoints. And there is no other value (eg 20,30,50) between the value of outliers and the normal value in datapoints. So outliers make the performance of the classifier declining very quickly.

#-----Question 2.8-----
features_test = train_20news_partB.drop('class', axis=1)
targets_test = pd.Series(train_20news_partB['class'])
features_test_pre = fit.predict(features_test)
cm_testb = confusion_matrix(targets_test, features_test_pre)
cm_norm_testb = normalize(cm_testb, norm='l1')
classes = [1, 2, 3, 4, 5]
plot_confusion_matrix(cm_norm_testb, classes=classes)
accuracy_testb = accuracy_score(targets_test, features_test_pre)
print("The classification accuracy on the test dataset = %.12f" %
      accuracy_testb)

#-----Question 2.9-----
# Comment on the results from the previous question. Do you think this is an acceptable level of performance? Which are the easiest and most difficult classes to predict correctly?
# Your answer goes here
# I think this is an acceptable level of performance.
# The class 1 is the easiest class to predict correctly. The class 2 is the most difficult class to predict correctly.


#-----Question 2.10-----
# What is a reasonable baseline against which to compare the classiffication performance? Hint: What is the simplest classiffier you can think of and what would its performance be on this dataset?
# Your answer goes here
# I think most frequent classifier could be a reasonable baseline.
#----Question 2.11-----
def mostfrequent(targets, class_num):
    count = np.zeros((1, class_num))
    for t in targets:
        for n in range(class_num - 1):
            if t == n + 1:
                count[0][n] += 1
    max = np.max(count)
    for c in range(class_num - 1):
        if count[0][c] == max:
            return (c + 1)


mf = mostfrequent(targets, 5)
print("Class %d is the most frequent class in the training dataset" % mf)


def mfpredict(mf, targets):
    return [mf for t in targets]


pre_test = mfpredict(mf, targets_test)
accuracy_baseline = accuracy_score(targets_test, pre_test)
print("The baseline performance(most frequent classifier's accuracy) = %.12f" %
      accuracy_baseline)
#-----Question 2.12-----
from sklearn.ensemble import RandomForestClassifier
X_tr = news_A_clean.drop('class', axis=1)
y_tr = pd.Series(news_A_clean['class'])
rf = RandomForestClassifier(n_estimators=50).fit(X=X_tr, y=y_tr)
X_ts = train_20news_partB.drop('class', axis=1)
y_ts = train_20news_partB['class']
rf_prediction = rf.predict(X=X_ts)
# Your code goes here
print(
    'Classification accuracy on the test set by using a Random Forest:',
    accuracy_score(
        y_ts, rf.predict(X=X_ts)))
plt.figure()
cm = confusion_matrix(y_ts, rf_prediction)
cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
classes = [1, 2, 3, 4, 5]
plot_confusion_matrix(cm_norm, classes=classes)
