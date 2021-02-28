import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing
from sklearn.model_selection import cross_val_score
#from fancyimpute import MICE
# Load the data
data = pd.read_csv("L2/file.csv",low_memory=False)
df = data[0:1000]
#print(df.head())

"""
#to view dataset composition
pos = df[df["loan_status"] == "Charged Off"].shape[0]
neg = df[df["loan_status"] == "Fully Paid"].shape[0]
print(pos,"\n",neg)

plt.figure(figsize=(8, 6))
k = sns.countplot(df["loan_status"])
plt.xticks((0, 1), ["Paid", "Defaulter"])
plt.xlabel("")
plt.ylabel("Count")
plt.title("Class counts", y=1, fontdict={"fontsize": 20})
plt.show()
k.figure.savefig("distribution.png")
"""

#encoding the data
number = LabelEncoder()
for c in df.columns:
	if c != "Unnamed" and c != "installment":
		df[c] = number.fit_transform(df[c].astype(str))


# Create dummy variables from the feature purpose
df = pd.get_dummies(df, columns=["purpose"], drop_first=True)

# Create binary features to check if the example is has missing values for all features that have missing values
for feature in df.columns:
  if np.any(np.isnan(df[feature])):
    df["is_" + feature + "_missing"] = np.isnan(df[feature]) * 1

# Original Data
X = df.loc[:, df.columns != "loan_status"].values
y = df.loc[:, df.columns == "loan_status"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print("Original data shapes:", X_train.shape, X_test.shape)

# Drop NA and remove binary columns

train_indices_na = np.max(np.isnan(X_train), axis=1)
test_indices_na = np.max(np.isnan(X_test), axis=1)
X_train_dropna, y_train_dropna = X_train[~train_indices_na, :][:, :-6], y_train[~train_indices_na]
X_test_dropna, y_test_dropna = X_test[~test_indices_na, :][:, :-6], y_test[~test_indices_na]
print("After dropping NAs:", X_train_dropna.shape, X_test_dropna.shape)


"""
# MICE data
mice = fancyimpute.MICE(verbose=0)
X_mice = mice.complete(X)
X_train_mice, X_test_mice, y_train_mice, y_test_mice = train_test_split(
  X_mice, y, test_size=0.2, shuffle=True, random_state=123, stratify=y)
print("MICE data shapes:", X_train_mice.shape, X_test_mice.shape)
"""

# Build random forest classifier
rf_clf = RandomForestClassifier(n_estimators=500, max_features=0.25, criterion="entropy", class_weight="balanced")
# Build base line model -- Drop NA's
pip_baseline = make_pipeline(RobustScaler(), rf_clf)
scores = cross_val_score(pip_baseline,X_train_dropna, y_train_dropna,scoring="roc_auc", cv=10)
print("Baseline model's average AUC:", scores.mean())

# Build model with mean imputation
pip_impute_mean = make_pipeline(Imputer(strategy="mean"), RobustScaler(), rf_clf)
scores = cross_val_score(pip_impute_mean, X_train, y_train, scoring="roc_auc", cv=10)
print("Mean imputation model's average AUC:", scores.mean())

# Build model with median imputation
pip_impute_median = make_pipeline(Imputer(strategy="median"), RobustScaler(), rf_clf)
scores = cross_val_score(pip_impute_median,
                         X_train, y_train,
                         scoring="roc_auc", cv=10)
print("Median imputation model's average AUC:", scores.mean())

# Build model using MICE imputation
"""
pip_impute_mice = make_pipeline(RobustScaler(), rf_clf)
scores = cross_val_score(pip_impute_mice,
                         X_train_mice, y_train_mice,
                         scoring="roc_auc", cv=10)
print("MICE imputation model's average AUC:", scores.mean())
"""

# Impute the missing data using features means
imp = Imputer()
imp.fit(X_train)
X_train = imp.transform(X_train)
X_test = imp.transform(X_test)

# Standardize the data
std = RobustScaler()
std.fit(X_train)
X_train = std.transform(X_train)
X_test = std.transform(X_test)

# Implement RandomUnderSampler
random_undersampler = RandomUnderSampler()
X_res, y_res = random_undersampler.fit_sample(X_train, y_train)

# Shuffle the data
perms = np.random.permutation(X_res.shape[0])
X_res = X_res[perms]
y_res = y_res[perms]

# Define base learners
xgb_clf = xgb.XGBClassifier(objective="binary:logistic",
                            learning_rate=0.03,
                            n_estimators=500,
                            max_depth=1,
                            subsample=0.4,
                            random_state=123)
svm_clf = SVC(gamma=0.1,
              C=0.01,
              kernel="poly",
              degree=3,
              coef0=10.0,
              probability=True)
rf_clf = RandomForestClassifier(n_estimators=300,
                                max_features="sqrt",
                                criterion="gini",
                                min_samples_leaf=5,
                                class_weight="balanced")

# Define meta-learner
logreg_clf = LogisticRegression(penalty="l2", C=100, fit_intercept=True)
# Fitting voting clf --> average ensemble
voting_clf = VotingClassifier([("xgb", xgb_clf),
                               ("svm", svm_clf),
                               ("rf", rf_clf)],
                              voting="soft",
                              flatten_transform=True)
voting_clf.fit(X_res, y_res)
xgb_model, svm_model, rf_model = voting_clf.estimators_
models = {"xgb": xgb_model,
          "svm": svm_model,
          "rf": rf_model,
          "avg_ensemble": voting_clf}

# Build first stack of base learners
first_stack = make_pipeline(voting_clf,
                            FunctionTransformer(lambda X: X[:, 1::2]))

# Use CV to generate meta-features
meta_features = cross_val_predict(first_stack, X_res, y_res, cv=10, method="transform")

# Refit the first stack on the full training set
first_stack.fit(X_res, y_res)

# Fit the meta learner
second_stack = logreg_clf.fit(meta_features, y_res)

# Plot ROC and PR curves using all models and test data
"""
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for name, model in models.items():
  model_probs = model.predict_proba(X_test)[:, 1:]
  model_auc_score = roc_auc_score(y_test, model_probs)
  fpr, tpr, _ = roc_curve(y_test, model_probs)
  precision, recall, _ = precision_recall_curve(y_test, model_probs)
  axes[0].plot(fpr, tpr, label=f"{name}, auc = {model_auc_score:.3f}")
  axes[1].plot(recall, precision, label=f"{name}")

stacked_probs = second_stack.predict_proba(first_stack.transform(X_test))[:, 1:]
stacked_auc_score = roc_auc_score(y_test, stacked_probs)
fpr, tpr, _ = roc_curve(y_test, stacked_probs)
precision, recall, _ = precision_recall_curve(y_test, stacked_probs)
axes[0].plot(fpr, tpr, label=f"stacked_ensemble, auc = {stacked_auc_score:.3f}")
axes[1].plot(recall, precision, label="stacked_ensembe")
axes[0].legend(loc="lower right")
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].set_title("ROC curve")
axes[1].legend()
axes[1].set_xlabel("recall")
axes[1].set_ylabel("precision")
axes[1].set_title("PR curve")
plt.tight_layout()
"""