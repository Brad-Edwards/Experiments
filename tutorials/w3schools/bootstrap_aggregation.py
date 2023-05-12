from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt

data = datasets.load_wine(as_frame=True)

X = data.data
y = data.target

# Base classifier performance
(
    X_train,
    X_test,
    y_train,
    y_test,
) = train_test_split(X, y, test_size=0.25, random_state=22)

dtree = DecisionTreeClassifier(random_state=22)
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)

print(
    "Train data accuracy: ",
    accuracy_score(y_true=y_train, y_pred=dtree.predict(X_train)),
)
print("Test data accuracy: ", accuracy_score(y_true=y_test, y_pred=y_pred))


# Usually grid search is used to tune the number of estimators

estimator_range = [2, 4, 6, 8, 10, 12, 14, 16]

models = []
scores = []

for n_estimators in estimator_range:
    clf = BaggingClassifier(n_estimators=n_estimators, random_state=22)
    clf.fit(X_train, y_train)
    models.append(clf)
    scores.append(accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))

# Generate the plot of scores against number of estimators
plt.figure(figsize=(9, 6))
plt.plot(estimator_range, scores)

# Adjust labels and font
plt.xlabel("n_estimators", fontsize=18)
plt.ylabel("score", fontsize=18)
plt.tick_params(labelsize=16)

plt.show()

# The plot shows 12 estimators is the best choice, so let's test against out of the bag data

oob_model = BaggingClassifier(n_estimators=12, random_state=22, oob_score=True)
oob_model.fit(X_train, y_train)

print(oob_model.oob_score_)

# Decision trees from bagging classifier

clf = BaggingClassifier(n_estimators=12, oob_score=True, random_state=22)

clf.fit(X_train, y_train)

plt.figure(figsize=(30, 20))

plot_tree(clf.estimators_[0], feature_names=X.columns)

plt.show()
