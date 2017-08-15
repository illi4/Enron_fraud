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

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif

from sklearn.neighbors import KNeighborsClassifier
from sklearn import grid_search

####### 0. Preparation

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

# Function to remove derived or original values depending on their strength
feature_singlets = {'f_bonus':'bonus', 'f_salary':'salary', 
                    'f_long_term_incentive':'long_term_incentive', 
                    'f_exercised_stock_options':'exercised_stock_options',
                    'f_restricted_stock':'restricted_stock'
                    }

feature_pairs = {'f_from_poi':['from_poi_to_this_person', 'to_messages'], 
                 'f_to_poi':['from_this_person_to_poi', 'from_messages'], 
                 'f_shared_receipt_with_poi':['shared_receipt_with_poi', 'to_messages']
                }

def rm_excessive_features(features_source, feature_pairs, feature_singlets):
    rm_values = set()
    
    for key, value in feature_pairs.iteritems(): 
        # Calculating original and derived weight 
        # Comparing by weighted average
        w_org = (features_source[value[0]] + features_source[value[1]])/2
        w_drv = features_source[key]
        if w_drv >= w_org: 
            #print 'Removing feature:', value[0], value[1]
            rm_values.add(value[0])
            rm_values.add(value[1])                                  
        else: 
            #print 'Removing feature:', key
            rm_values.add(key)  
    
    for key, value in feature_singlets.iteritems(): 
        w_org = features_source[value] 
        w_drv = features_source[key]
        if w_drv >= w_org: 
            #print 'Removing feature:', value
            rm_values.add(value)                            
        else: 
            #print 'Removing feature:', key
            rm_values.add(key)     
            
    # Removing less significant features
    for rm_val in rm_values: 
        features_source.pop(rm_val, None)    
    
    return features_source

# Create a function to return combinations of the most significant features 
def select_features(features_used, features, labels): 
    # Create a dictionary to populate from feature_used array, and a results list

    best_f_dict = dict.fromkeys(features_used, 0)
    results = []

    # Create a Stratified ShuffleSplit cross-validaton with 10 splits (folds)
    # 10 is a default value, which should be enough for a dataset containing 144 values
    cv = StratifiedShuffleSplit(labels, 10)
    #print "\nFeature engineering:"
    
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

    # Remove less significant features 
    best_f_dict = rm_excessive_features(best_f_dict, feature_pairs, feature_singlets)      
            
    # Get the list of sorted features sorted by their importance 
    best_f_list = sorted(best_f_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    
    #print "\nThe most important features by scores:"
    #pprint.pprint(best_f_list)

    # Save the list of features for future reference, sorted by their importance
    features_filtered = []
    for val in best_f_list: 
        features_filtered.append(val[0])
        
    results = features_filtered 
    return results 

#### 1. Generate a list of sets of the most significant features to evaluate performance on different combinations 
features_filtered_set = []
for i in range (0, 5):
    best_set_found = select_features(features_used, features, labels)
    print "\nFeature combination", i, best_set_found
    features_filtered_set.append(best_set_found)	

#### 2. Classifiers testing and selection 

# Function which return shuffled testing and training sets 
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
    return features_train, features_test, labels_train, labels_test

# Function to produce shuffled sets for each of identified features combination and to fit / evaluate a classifier on each
def prepare_evaluate(my_dataset, features_filtered_set, clf):
    i = 1
    for features_filtered in features_filtered_set:       
        # Prepare features list for tester and for data split 
        features_tester = list(features_filtered)
        features_tester.insert(0, 'poi')
        data = featureFormat(my_dataset, features_tester)
        labels, features = targetFeatureSplit(data)
        # Get the shuffled sets
        features_train, features_test, labels_train, labels_test = shuffle_split(labels, features)
        print "Evaluating classifier on the feature set", i
        # Fit the classifier
        clf.fit(features_train, labels_train)
        # Call tester 
        tester.test_classifier(clf, my_dataset, features_tester)
        i += 1

# A. Create and use a classifier - decision tree 
clf_tree = tree.DecisionTreeClassifier()
prepare_evaluate(my_dataset, features_filtered_set, clf_tree)

# B. Create and use a classifier - GaussianNB
clf_gnb = GaussianNB()
prepare_evaluate(my_dataset, features_filtered_set, clf_gnb)

# C. GaussianNB with feature selection and PCA
pca = PCA(n_components=3)
clf_gnb_n = GaussianNB()
pipe = Pipeline([('pca', pca), ('model', clf_gnb_n)])
prepare_evaluate(my_dataset, features_filtered_set, pipe)

# D. kNN
clf_knn = KNeighborsClassifier()
prepare_evaluate(my_dataset, features_filtered_set, clf_knn)

 
# Final feature list after several rounds of testing 
features_filtered = ['f_bonus', 'deferral_payments', 'f_from_poi', 'f_salary', 'deferred_income', 'other', 'f_shared_receipt_with_poi', 'from_messages', 'expenses', 'from_this_person_to_poi', 'long_term_incentive', 'total_payments', 'f_exercised_stock_options', 'f_restricted_stock', 'total_stock_value']


### 3. Final: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation.  


# Parameters for fine tuning  
parameters = {'splitter':('best','random'), 
    'min_samples_split':(2, 5, 10, 15), 
    'min_samples_leaf':(1, 3, 6, 8, 10)}

# Update the dataset
features_tester = list(features_filtered)
features_tester.insert(0, 'poi')
data = featureFormat(my_dataset, features_tester)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = shuffle_split(labels, features)

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
