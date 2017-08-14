#!/usr/bin/python

# Supress warnings
import warnings
warnings.filterwarnings("ignore")

# Import necessary libraries
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import tester

import pprint
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import operator

from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif

from sklearn.neighbors import KNeighborsClassifier
from sklearn import grid_search

### Preparation

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Transform dictionary to the Pandas DataFrame  
df = pd.DataFrame.from_dict(data_dict, orient = 'index')
# Names of the features 
features = list(df.keys())

# Dataframe information
df = df.replace("NaN", np.nan)
#df.info()

count = sum(df.count())
count_nan = df.isnull().sum().sum()

print "\nTotal not-null data points:", count
print 'Total NaN: {} ({:.2%})'.format(count_nan, float(count_nan)/float(count+count_nan))

# Allocation between classes
poi_types = df.poi.value_counts()
poi_types.index=['non-POI', 'POI']
print "Allocation:\n", poi_types
 
# Upper fence value to detect outliers
upper = df.quantile(.25) + 1.5 * (df.quantile(.75)-df.quantile(.25))

# We should not use features such as email_address and poi for outliers analysis
features_filt = features
features_filt.remove('email_address')
features_filt.remove('poi')

# Finding outliers
upper_o = pd.DataFrame((df[features_filt] > upper[features_filt]).sum(axis = 1), columns = ['outlier_columns_max']).\
    sort_values('outlier_columns_max', ascending = 0)  

# Removing an outlier
df = df.drop(['TOTAL'])
df = df.drop(['THE TRAVEL AGENCY IN THE PARK'])

# Removing too sparcely populated features
df = df.drop(['loan_advances','director_fees','restricted_stock_deferred', 'email_address'], 1)

# Adding new features
df['f_bonus'] = df['bonus']/df['total_payments']
df['f_salary'] = df['salary']/df['total_payments']
df['f_long_term_incentive'] = df['long_term_incentive']/df['total_payments']
df['f_exercised_stock_options'] = df['exercised_stock_options']/df['total_stock_value']
df['f_restricted_stock'] = df['restricted_stock']/df['total_stock_value']
df['f_from_poi'] = df['from_poi_to_this_person']/df['to_messages']
df['f_to_poi'] = df['from_this_person_to_poi']/df['from_messages']
df['f_shared_receipt_with_poi'] = df['shared_receipt_with_poi']/df['to_messages']

## Scale features 

df = df.fillna(df.mean())

# Preserving the dataframe structure
i = df.index
c = df.columns

# Scaling
scaler = MinMaxScaler()
scaler.fit(df)

df = DataFrame(scaler.transform(df), index=i, columns=c)

# Convert to a dictionary and use preprocessing functions. 
my_list = df.to_dict(orient='records')
my_dataset = {}

for index, values in zip(i, my_list):
    my_dataset[index] = values

# We will need a full list of features to select the best ones  
features_list_full = my_dataset.itervalues().next().keys()

# POI at the first position for further use with preprocessing functions 
features_list_full.remove('poi')
features_used = list(features_list_full)
features_list_full.insert(0, 'poi')

# Use preprocessing functions 
data = featureFormat(my_dataset, features_list_full)
labels, features = targetFeatureSplit(data)

# Selecting the most significant features 
# Create a dictionary to populate 
best_f_dict = dict.fromkeys(features_used, 0)

# Create a Stratified ShuffleSplit cross-validato with 10 splits (folds)
# 10 is a default value, which should be enough for a dataset containing 144 values
cv = StratifiedShuffleSplit(labels, 10)

# Try classifiers on folds and select the best features
for train_idx, test_idx in cv:
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )
        
    # Create a Decision Tree classifier 
    clf = tree.DecisionTreeClassifier()    
    # Fit the classifier using training set, and test on test set
    clf.fit(features_train, labels_train)
    # Detect most important features from the tree
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]   
    # Get 10 most important features of the Decision tree 
    for i in range(10):
        best_f_val = features_used[indices[i]]
        best_f_dict[best_f_val] = best_f_dict[best_f_val] + 1

    # Detect most important features using SelectKBest
    selector = SelectKBest(chi2, k=10)
    selector.fit(features_train, labels_train)
    features_train_transformed = selector.transform(features_train) 
    support = np.asarray(selector.get_support())
        
    # Get an array with the most important features 
    features_arr = np.asarray(features_used)
    columns_support = features_arr[support]
    for val in columns_support: 
        best_f_dict[val] = best_f_dict[val] + 1
        
# Get the list of sorted features sorted by their importance 
best_f_list = sorted(best_f_dict.iteritems(), key=operator.itemgetter(1), reverse=True)

print "The most important features:"
pprint.pprint(best_f_list)

# Save the list of features for future reference, sorted by their importance
features_filtered = []
for val in best_f_list: 
    features_filtered.append(val[0])

# Final feature list after several rounds of testing 
features_filtered = ['exercised_stock_options', 'total_stock_value', 'f_shared_receipt_with_poi', 'bonus', 'total_payments', 'f_to_poi', 'f_long_term_incentive', 'other', 'salary', 'restricted_stock', 'f_from_poi', 'deferred_income', 'f_restricted_stock', 'expenses', 'deferral_payments']

# Prepare features list for tester and for data split 
features_tester = list(features_filtered)
features_tester.insert(0, 'poi')

data = featureFormat(my_dataset, features_tester)
labels, features = targetFeatureSplit(data)

## Decision Tree Classifier 

# Create a function which return shuffled testing and training sets 
def shuffle_split(labels, features): 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []   
    # 10 is a default value, which should be enough for a dataset containing 144 values
    cv = StratifiedShuffleSplit(labels, 10)
    # Create leatures and labels shuffled dataset
    for train_idx, test_idx in cv:    
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
    return features_train, features_test, labels_train, labels_test

# Get the shuffled sets
features_train, features_test, labels_train, labels_test = shuffle_split(labels, features)

# Create and use a classifier 
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(features_train, labels_train)

# Call tester 
tester.dump_classifier_and_data(clf_tree, my_dataset, features_tester)
tester.main() 

## GaussianNB with feature selection and PCA
clf_gnb_n = GaussianNB()

# feature selection
features_list2 = ['poi'] + range(3)
my_dataset1 = pd.DataFrame(SelectKBest(f_classif, k=8).fit_transform(df, df.poi), index = df.index)

#PCA
pca = PCA(n_components=3)
my_dataset2 = pd.DataFrame(pca.fit_transform(my_dataset1), index=df.index)
my_dataset2.insert(0, "poi", df.poi)
my_dataset2 = my_dataset2.to_dict(orient = 'index')  

# Change the structure 
features_list_full_pca = my_dataset2.itervalues().next().keys()
# POI at the first position for further use with preprocessing functions 
features_list_full_pca.remove('poi')
features_list_full_pca.insert(0, 'poi')

# Use preprocessing functions 
data_pca = featureFormat(my_dataset2, features_list_full_pca)
labels_pca, features_pca = targetFeatureSplit(data_pca)

# Get the shuffled sets
features_train, features_test, labels_train, labels_test = shuffle_split(labels_pca, features_pca)

# GaussianNB implementation on the reformatted data
clf_gnb = GaussianNB()
clf_gnb.fit(features_train, labels_train)

tester.dump_classifier_and_data(clf_gnb, my_dataset2, features_list_full_pca)
tester.main()

## GaussianNB without feature selection and PCA

# Get the shuffled sets
features_train, features_test, labels_train, labels_test = shuffle_split(labels, features)

# GaussianNB
clf_gnb = GaussianNB()
clf_gnb.fit(features_train, labels_train)
pred = clf_gnb.predict(features_test)

tester.dump_classifier_and_data(clf_gnb, my_dataset, features_tester)
tester.main()

## kNN 

# Get the shuffled sets
features_train, features_test, labels_train, labels_test = shuffle_split(labels, features)

clf_knn = KNeighborsClassifier()
clf_knn.fit(features_train, labels_train)

# Call tester 
tester.dump_classifier_and_data(clf_knn, my_dataset, features_tester)
tester.main() 

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Parameters for fine tuning  
parameters = {'splitter':('best','random'), 
    'min_samples_split':(2, 5, 10, 15), 
    'min_samples_leaf':(1, 3, 6, 8, 10)}

# Create and use a classifier for fitting
clf_tree = tree.DecisionTreeClassifier()
clf_tree = grid_search.GridSearchCV(clf_tree, parameters) 

clf_tree.fit(features_train, labels_train)

print "Best parameters for the selected algorithm:", clf_tree.best_params_

# Using the parameters. Finalise: best parameters and algorithm
clf_tree = tree.DecisionTreeClassifier(min_samples_split = 2,
                             splitter = 'best',
                             min_samples_leaf = 1)

clf_tree.fit(features_train, labels_train)

### Evaluation 

print "\nFinal evaluation\n"

# Call tester 
tester.dump_classifier_and_data(clf_tree, my_dataset, features_tester)
tester.main() 
