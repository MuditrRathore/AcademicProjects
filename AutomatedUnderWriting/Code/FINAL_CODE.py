import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Imputer
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix

from sklearn.ensemble.partial_dependence import plot_partial_dependence
# Load the data
df = pd.read_csv("L2/file.csv",low_memory=False)
# Check both the datatypes and if there is missing values print(f"Data types:\n{11 * '-'}")
print(f"{df.dtypes}\n")
print(f"Sum of null values in each feature:\n{35 * '-'}")
print(f"{df.isnull().sum()}")
df.head()
df['int_rate'].replace(regex=True,inplace=True,to_replace=r'\%',value=r'')
df['revol_util'].replace(regex=True,inplace=True,to_replace=r'\%',value=r'')

# Get number of positve and negative examples
pos = df[df["loan_status"] == "Charged Off"].shape[0]
neg = df[df["loan_status"] == "Fully Paid"].shape[0]
print(f"Positive examples = {pos}")
print(f"Negative examples = {neg}")
print(f"Proportion of positive to negative examples = {(pos / neg) * 100:.2f}%")


plt.figure(figsize=(8, 6))
sns.countplot(df["loan_status"])
plt.xticks((0, 1), ["Charged Off", "Fully Paid"])
plt.xlabel("")
plt.ylabel("Count")
plt.title("Class counts", y=1, fontdict={"fontsize": 20});
plt.show()


df = pd.get_dummies(df, columns=["purpose"], drop_first=True)

number = LabelEncoder()

df["loan_status"] = number.fit_transform(df["loan_status"])


for c in df.columns:
	if c != "Unnamed" and c != "installment" and c != "purpose":
		df[c] = number.fit_transform(df[c].astype(str))


#
for feature in df.columns:
	if np.any(np.isnan(df[feature])):
		df["is_" + feature + "_missing"] = np.isnan(df[feature])*1

#reducing size for testing
df = df.sample(n=4000, frac=None, replace=False, weights=None, random_state=None, axis=None)

# Original Data
X = df.loc[:, df.columns != "loan_status"].values
y = df.loc[:, df.columns == "loan_status"].values.flatten()
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, shuffle=True, random_state=123,stratify = y)
print(f"Original data shapes: {X_train.shape, X_test.shape}")

# Drop NA and remove binary columns
train_indices_na = np.max(np.isnan(X_train), axis=1)
test_indices_na = np.max(np.isnan(X_test), axis=1)
X_train_dropna, y_train_dropna = X_train[~train_indices_na, :][:, :-6], y_train[~train_indices_na]
X_test_dropna, y_test_dropna = X_test[~test_indices_na, :][:, :-6], y_test[~test_indices_na]
print(f"After dropping NAs: {X_train_dropna.shape, X_test_dropna.shape}")


"""
not working becaus of tensorflow dependencies
# MICE data
mice = fancyimpute.MICE(verbose=0)
X_mice = mice.complete(X)
X_train_mice, X_test_mice, y_train_mice, y_test_mice = train_test_split(
  X_mice, y, test_size=0.2, shuffle=True, random_state=123, stratify=y)
print(f"MICE data shapes: {X_train_mice.shape, X_test_mice.shape}")
"""
"""

# Build random forest classifier
rf_clf = RandomForestClassifier(n_estimators=500, max_features=0.25,
                                criterion="entropy", class_weight="balanced")
# Build base line model -- Drop NA's
pip_baseline = make_pipeline(RobustScaler(), rf_clf)
scores = cross_val_score(pip_baseline,
                         X_train_dropna, y_train_dropna,
                         scoring="roc_auc", cv=10)
print(f"Baseline model's average AUC: {scores.mean():.3f}")

# Build model with mean imputation
pip_impute_mean = make_pipeline(Imputer(strategy="mean"), RobustScaler(), rf_clf)
scores = cross_val_score(pip_impute_mean, X_train, y_train, scoring="roc_auc", cv=10)
print(f"Mean imputation model's average AUC: {scores.mean():.3f}")

# Build model with median imputation
pip_impute_median = make_pipeline(Imputer(strategy="median"), RobustScaler(), rf_clf)
scores = cross_val_score(pip_impute_median,
                         X_train, y_train,
                         scoring="roc_auc", cv=10)
print(f"Median imputation model's average AUC: {scores.mean():.3f}")

#works perfectly till here

#works fine not needed while testing

# fit RF to plot feature importances
rf_clf.fit(RobustScaler().fit_transform(
  Imputer(strategy="median").fit_transform(X_train)), y_train)

# Plot features importance
importances = rf_clf.feature_importances_
indices = np.argsort(rf_clf.feature_importances_)[::-1]
pos = np.arange(len(importances))
plt.figure(figsize=(12, 6))
plt.bar(pos, importances[indices], align="center")
plt.xticks(range(1, 334),
           df.columns[df.columns != "loan_status"][indices],
           rotation=90)
plt.title("Feature Importance", {"fontsize": 16});
plt.show()
"""
# Drop generated binary features
X_train = X_train[:, :14]
X_test = X_test[:, :14]

"""
#testing the best sampling technique
#works fine not needed for testing

# Build model with no sampling
pip_orig = make_pipeline(Imputer(strategy="mean"),
                         RobustScaler(),
                         rf_clf)
scores = cross_val_score(pip_orig,
                         X_train, y_train,
                         scoring="roc_auc", cv=10)
print(f"Original model's average AUC: {scores.mean():.3f}")

# Build model with undersampling
pip_undersample = imb_make_pipeline(Imputer(strategy="mean"),
                                    RobustScaler(),
                                    RandomUnderSampler(),
                                    rf_clf)
scores = cross_val_score(pip_undersample,
                         X_train, y_train,
                         scoring="roc_auc", cv=10)
print(f"Under-sampled model's average AUC: {scores.mean():.3f}")

# Build model with oversampling
pip_oversample = imb_make_pipeline(Imputer(strategy="mean"),
                                   RobustScaler(),
                                   RandomOverSampler(),
                                   rf_clf)
scores = cross_val_score(pip_oversample,
                         X_train, y_train,
                         scoring="roc_auc", cv=10)
print(f"Over-sampled model's average AUC: {scores.mean():.3f}")

# Build model with EasyEnsemble
resampled_rf = BalancedBaggingClassifier(base_estimator=rf_clf,
                                         n_estimators=10,
                                         random_state=123)
pip_resampled = make_pipeline(Imputer(strategy="mean"),
                              RobustScaler(),
                              resampled_rf)
scores = cross_val_score(pip_resampled,
                         X_train, y_train,
                         scoring="roc_auc", cv=10)
print(f"EasyEnsemble model's average AUC: {scores.mean():.3f}")

# Build model with SMOTE
pip_smote = imb_make_pipeline(Imputer(strategy="mean"),
                              RobustScaler(),
                              SMOTE(),
                              rf_clf)
scores = cross_val_score(pip_smote,
                         X_train, y_train,
                         scoring="roc_auc", cv=10)
print(f"SMOTE model's average AUC: {scores.mean():.3f}")

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
logreg_clf = LogisticRegression(penalty="l2", C=100, fit_intercept=True)

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

#metalearner-2
xgb_clf = xgb.XGBClassifier(objective="binary:logistic",
                            learning_rate=0.03,
                            n_estimators=500,
                            max_depth=1,
                            subsample=0.4,
                            random_state=123)
# Fitting voting clf --> average ensemble
voting_clf = VotingClassifier([("logreg", logreg_clf),
                               ("svm", svm_clf),
                               ("rf", rf_clf)],
                              voting="soft",
                              flatten_transform=True)
voting_clf.fit(X_res, y_res)
logreg_model, svm_model, rf_model = voting_clf.estimators_
models = {"logreg": logreg_model,
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
second_stack = xgb_clf.fit(meta_features, y_res)
test_meta = cross_val_predict(first_stack, X_test, y_test, cv=10, method="transform")
acc = accuracy_score(y_test,second_stack.predict(test_meta))
print("Accuracy score using stacking ensemble model", acc)
acc2 = accuracy_score(y_test,voting_clf.predict(X_test))
print("Accuracy score using averaging ensemble model", acc2)

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
plt.show()

from mlens.visualization import corrmat

probs_df = pd.DataFrame(meta_features, columns=["xgb", "svm", "rf"])
corrmat(probs_df.corr(), inflate=True)


second_stack_probs = second_stack.predict_proba(first_stack.transform(X_test))
second_stack_preds = second_stack.predict(first_stack.transform(X_test))
conf_mat = confusion_matrix(y_test, second_stack_preds)

plt.figure(figsize=(16, 8))
plt.matshow(conf_mat, cmap=plt.cm.Reds, alpha=0.2)
for i in range(2):
  for j in range(2):
    plt.text(x=j, y=i, s=conf_mat[i, j], ha="center", va="center")
plt.title("Confusion matrix", y=1.1, fontdict={"fontsize": 20})
plt.xlabel("Predicted", fontdict={"fontsize": 14})
plt.ylabel("Actual", fontdict={"fontsize": 14})
plt.show()


gbrt = GradientBoostingClassifier(loss="deviance",
                                  learning_rate=0.1,
                                  n_estimators=100,
                                  max_depth=3,
                                  random_state=123)
gbrt.fit(X_res, y_res)
fig, axes = plot_partial_dependence(gbrt,
                                    X_res,
                                    np.argsort(gbrt.feature_importances_)[::-1][:8],
                                    n_cols=4,
                                    feature_names=df.columns[:14],
                                    figsize=(14, 8))
plt.subplots_adjust(top=0.9)
plt.suptitle("Partial dependence plots of borrower not fully paid\n" + 
             "the loan based on top most influential features")
plt.show()
for ax in axes:
	ax.set_xticks(())