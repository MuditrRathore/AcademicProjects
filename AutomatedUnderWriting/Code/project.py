import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import matplotlib.pyplot as plt

train = pd.read_csv('credit_train.csv')
test = pd.read_csv('credit_test.csv')

#summarizes the data
#-------------------

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

# Feature Engineering
# -------------------
train_mod['TotalIncome'] = train_mod['ApplicantIncome'] + train_mod['CoapplicantIncome']
test_mod['TotalIncome'] = test_mod['ApplicantIncome'] + test_mod['CoapplicantIncome']

# Perform log transformation of TotalIncome to make it closer to normal
train_mod['TotalIncome_log'] = np.log(train_mod['TotalIncome'])
test_mod['TotalIncome_log'] = np.log(test_mod['TotalIncome'])


h1 = train_mod.hist(column='TotalIncome', bins=40)
plt.savefig('Income.png')
h2 = train_mod.hist(column='TotalIncome_log', bins=40)
plt.savefig('testplot.png')

#Logistic Regression
model = sklearn.linear_model.LogisticRegression()
predictors = ['Credit_History', 'Education', 'Gender', 'TotalIncome_log', 'LoanAmount']
x_train = train_mod[predictors].values
y_train = train_mod['Loan_Status'].values

model.fit(x_train, y_train)
#predict
x_test = test_mod[predictors].values
predicted = model.predict(x_test)
test_mod['Loan_Status'] = predicted

print(test['Loan_Status'])
print(accuracy_score(test['Loan_Status'], predicted, normalize=True))
