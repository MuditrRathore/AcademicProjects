# -*- coding: utf-8 -*-
"""
A machine learning project for loan prediction

K. Steinkamp

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble


# Load datasets
# -------------
train = pd.read_csv('train_data.csv')
print(type(train))
test = pd.read_csv('test_data.csv')


# Basic data exploration and plots
# --------------------------------
print(train.head(5))
train.describe()
train['Education'].value_counts()

cols = (['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
         'Loan_Amount_Term'])  # numeric features
for c in cols:
    train.hist(column=c, bins=50)
    # train.boxplot(column=c, by = 'Gender')

pd.crosstab(train['Credit_History'], train['Gender'], margins=True)


# Impute missing values
# ---------------------
train_mod = train
test_mod = test

# Impute 'LoanAmount' with median values
train_mod['LoanAmount'] = train_mod['LoanAmount'].fillna(train_mod['LoanAmount'].median())
test_mod['LoanAmount'] = test_mod['LoanAmount'].fillna(test_mod['LoanAmount'].median())

# For now, impute 'Credit_History' simply with '1.0' (the majority)
train_mod['Credit_History'].fillna(1.0, inplace=True)
test_mod['Credit_History'].fillna(1.0, inplace=True)

# For now, impute 'Gender' simply with 'Male' (the majority)
train_mod['Gender'].fillna('Male', inplace=True)
test_mod['Gender'].fillna('Male', inplace=True)


# Label encoding
# --------------
number = LabelEncoder()

train_mod['Gender'] = number.fit_transform(train_mod['Gender'].astype(str))
test_mod['Gender'] = number.transform(test_mod['Gender'].astype(str))

train_mod['Education'] = number.fit_transform(train_mod['Education'].astype(str))
test_mod['Education'] = number.transform(test_mod['Education'].astype(str))

train_mod['Loan_Status'] = number.fit_transform(train_mod['Loan_Status'].astype(str))


# Feature Engineering
# -------------------
train_mod['TotalIncome'] = train_mod['ApplicantIncome'] + train_mod['CoapplicantIncome']
test_mod['TotalIncome'] = test_mod['ApplicantIncome'] + test_mod['CoapplicantIncome']

# Perform log transformation of TotalIncome to make it closer to normal
train_mod['TotalIncome_log'] = np.log(train_mod['TotalIncome'])
test_mod['TotalIncome_log'] = np.log(test_mod['TotalIncome'])

train_mod.hist(column='TotalIncome', bins=40)
train_mod.hist(column='TotalIncome_log', bins=40)


# Model Building: Logistic Regression
# -----------------------------------
# Create object of Logistic Regression
model = sklearn.linear_model.LogisticRegression()

# Select predictors
predictors = ['Credit_History', 'Education', 'Gender', 'TotalIncome_log', 'LoanAmount']

# Converting predictors and outcome to numpy array
x_train = train_mod[predictors].values
y_train = train_mod['Loan_Status'].values

# Fit model
model.fit(x_train, y_train)

# Predict Loan Status for test data
x_test = test_mod[predictors].values
predicted = model.predict(x_test)
predicted = number.inverse_transform(predicted)
test_mod['Loan_Status'] = predicted

# Write to file
test_mod.to_csv('LoanPrediction_LogReg.csv', columns=['Loan_ID', 'Loan_Status'])


# Model Building: Decision Tree
# -----------------------------
# Create object of Decision Tree
model = sklearn.tree.DecisionTreeClassifier()

# Select predictors
predictors = ['Credit_History', 'Education', 'Gender', 'TotalIncome_log', 'LoanAmount']

# Converting predictors and outcome to numpy array
x_train = train_mod[predictors].values
y_train = train_mod['Loan_Status'].values

# Fit model
model.fit(x_train, y_train)

# Predict Loan Status for test data
x_test = test_mod[predictors].values
predicted = model.predict(x_test)
predicted = number.inverse_transform(predicted)
test_mod['Loan_Status'] = predicted

# Write to file
test_mod.to_csv('LoanPrediction_DecisionTree.csv', columns=['Loan_ID', 'Loan_Status'])


# Model Building: Random Forest
# -----------------------------
# Create object of Random Forest
model = sklearn.ensemble.RandomForestClassifier()

# Select all predictors
predictors = ['Credit_History', 'Education', 'Gender', 'TotalIncome_log', 'LoanAmount']

# Converting predictors and outcome to numpy array
x_train = train_mod[predictors].values
y_train = train_mod['Loan_Status'].values

# Fit model
model.fit(x_train, y_train)

# Feature importance
featimp = pd.Series(model.feature_importances_, index=predictors).sort_values(ascending=False)
print(featimp)

# Predict Loan Status for test data
x_test = test_mod[predictors].values
predicted = model.predict(x_test)
predicted = number.inverse_transform(predicted)
test_mod['Loan_Status'] = predicted

# Write to file
test_mod.to_csv('LoanPrediction_RandomForest.csv', columns=['Loan_ID', 'Loan_Status'])

