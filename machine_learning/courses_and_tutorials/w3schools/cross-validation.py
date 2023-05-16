from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    StratifiedKFold,
    LeaveOneOut,
    LeavePOut,
    ShuffleSplit,
)

# Split the data into k smaller sets, train on k-1 sets and keep the last for testing

X, y = datasets.load_iris(return_X_y=True)

clf = DecisionTreeClassifier(random_state=42)

k_folds = KFold(n_splits=5)

scores = cross_val_score(clf, X, y, cv=k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

# Stratified K-Fold - Allows for handling of unbalanced datasets

sk_folds = StratifiedKFold(n_splits=5)

scores = cross_val_score(clf, X, y, cv=sk_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

# Leave-One-Out

loo = LeaveOneOut()

scores = cross_val_score(clf, X, y, cv=loo)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

# Leave-P-Out - More nuanced version of Leave-One-Out

lpo = LeavePOut(p=2)

scores = cross_val_score(clf, X, y, cv=lpo)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

# Shuffle split - leaves out a percentage of the data for testing

ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits=5)

scores = cross_val_score(clf, X, y, cv=ss)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
