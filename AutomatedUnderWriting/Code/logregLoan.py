# -*- coding: utf-8 -*-
"""
A machine learning project for loan prediction

K. Steinkamp

"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation, metrics
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
from sklearn.ensemble import ExtraTreesClassifier


# Define modelfit function
# ------------------------
def modelfit(model, dtrain, dtest, predictors, performCV=True, printFeatImp=True, n_cvfolds=10):
    # Fit the model to the data
    model.fit(dtrain[predictors], dtrain['Disbursed'])

    # Predict training set:
    dtrain_predictions = model.predict(dtrain[predictors])
    dtrain_predprob = model.predict_proba(dtrain[predictors])[:, 1]

    # Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(model, dtrain[predictors],
                                                    dtrain['Disbursed'], cv=n_cvfolds,
                                                    scoring='roc_auc')

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    if performCV:
        print("CV Score : Mean %.7g | Std %.7g | Min %.7g | Max %.7g" % (np.mean(cv_score),
                                                                         np.std(cv_score),
                                                                         np.min(cv_score),
                                                                         np.max(cv_score)))

    # Print Feature Importance:
    if printFeatImp:
        feat_imp = pd.Series(model.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()



# Load datasets
# -------------
train = pd.read_csv('train_data.csv')
#print(type(train))
test = pd.read_csv('test_data.csv')


# Basic data exploration and plots
# --------------------------------
#print(train.head(5))
train.describe()
train['Education'].value_counts()

cols = (['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
         'Loan_Amount_Term'])  # numeric features
#for c in cols:
#    train.hist(column=c, bins=50)
#    # train.boxplot(column=c, by = 'Gender')

pd.crosstab(train['Education'], train['Gender'], margins=True, normalize='columns')
pd.crosstab(train['Credit_History'], train['Property_Area'], margins=True, normalize='columns')
plt.scatter(train['LoanAmount'],train['Credit_History'])
plt.show()

# Impute missing values
# ---------------------
train_mod = train.copy()
test_mod = test.copy()

# Exclude observations with missing 'Credit_History'
train_mod = train_mod.dropna(subset=['Credit_History']).reset_index()

# Impute 'LoanAmount' with median values
train_mod['LoanAmount'] = train_mod['LoanAmount'].fillna(train_mod['LoanAmount'].median())
test_mod['LoanAmount'] = test_mod['LoanAmount'].fillna(test_mod['LoanAmount'].median())

# For now, impute 'Gender' simply with 'Male' (the majority)
train_mod['Gender'] = train_mod['Gender'].fillna('Male')
test_mod['Gender'] = test_mod['Gender'].fillna('Male')


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

# train_mod.hist(column='TotalIncome', bins=40)
# train_mod.hist(column='TotalIncome_log', bins=40)


# Model Building: Logistic Regression
# -----------------------------------
# Create object of Logistic Regression
#model = sklearn.linear_model.LogisticRegression()
model = sklearn.ExtraTreesClassifier(n_estimators=250, random_state=0)

# Select predictors
# predictors = ['Credit_History', 'Education', 'Gender', 'TotalIncome_log', 'LoanAmount']
predictors = ['Credit_History', 'Education', 'TotalIncome_log', 'LoanAmount']

# Converting predictors and outcome to numpy array
x_train = train_mod[predictors].values
y_train = train_mod['Loan_Status'].values

# Coss-validation
# Simple K-Fold cross validation. 10 folds.
cv = cross_validation.KFold(len(train_mod), n_folds=10)

cv_score = cross_validation.cross_val_score(model, train_mod[predictors],
                                            train_mod['Loan_Status'], cv=10,
                                            scoring='roc_auc')
results = []
for traincv, testcv in cv:
    model.fit(x_train[traincv, :], y_train[traincv])
    x_test = train_mod.ix[testcv, predictors]
    predicted = model.predict(x_test)
    results.append(sum(abs(predicted - train_mod.ix[testcv, 'Loan_Status'].values))/len(testcv))

print("\nCV Results: " + str(np.mean(100*np.array(results))) + "% wrong predictions")
print("\nCV Score: " + str(np.mean(cv_score)))

feat_imp = pd.Series(model.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()


##Fit model to whole training dataset
#model.fit(x_train, y_train)
#
## Predict Loan Status for test data
#x_test = test_mod[predictors].values
#predicted = model.predict(x_test)
#predicted = number.inverse_transform(predicted)
#test_mod['Loan_Status'] = predicted
#
## Write to file
#test_mod.to_csv('LoanPrediction_LogReg2.csv', columns=['Loan_ID', 'Loan_Status'])

