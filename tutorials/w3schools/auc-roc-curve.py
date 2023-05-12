import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

n = 10000
ratio = 0.95
n_0 = int((1 - ratio) * n)
n_1 = int(ratio * n)

y = np.array([0] * n_0 + [1] * n_1)

# Probabilities obtained from a hypothetical model that always predicts the majority class. This would be accurate for one class, but no good for actual use.

y_proba = np.array([1] * n)
y_pred = y_proba > 0.5

print(f"accuracy score: {accuracy_score(y, y_pred)}")
cf_mat = confusion_matrix(y, y_pred)
print(f"confusion matrix: {cf_mat}")
print(f"class 0 accuracy: {cf_mat[0][0]/n_0}")
print(f"class 1 accuracy: {cf_mat[1][1]/n_1}")

# Probabilities obtained from a hypothetical model that doesn't always predict the mode. Not as accurate for the majority group, but being more balanced it's actually useful.

y_proba_2 = np.array(
    np.random.uniform(0, 0.7, n_0).tolist() + np.random.uniform(0.7, 1, n_1).tolist()
)
y_pred_2 = y_proba_2 > 0.5

print(f"accuracy score: {accuracy_score(y, y_pred_2)}")
cf_mat = confusion_matrix(y, y_pred_2)
print(f"confusion matrix: {cf_mat}")
print(f"class 0 accuracy: {cf_mat[0][0]/n_0}")
print(f"class 1 accuracy: {cf_mat[1][1]/n_1}")


def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


# Model 1
plot_roc_curve(y, y_proba)
print(f"roc auc score: {roc_auc_score(y, y_proba)}")
plt.show()

# Model 2
plot_roc_curve(y, y_proba_2)
print(f"roc auc score: {roc_auc_score(y, y_proba_2)}")
plt.show()

# Probabilities

n = 10000
y = np.array([0] * n + [1] * n)
#
y_prob_1 = np.array(
    np.random.uniform(0.25, 0.5, n // 2).tolist()
    + np.random.uniform(0.3, 0.7, n).tolist()
    + np.random.uniform(0.5, 0.75, n // 2).tolist()
)
y_prob_2 = np.array(
    np.random.uniform(0, 0.4, n // 2).tolist()
    + np.random.uniform(0.3, 0.7, n).tolist()
    + np.random.uniform(0.6, 1, n // 2).tolist()
)

print(f"model 1 accuracy score: {accuracy_score(y, y_prob_1>.5)}")
print(f"model 2 accuracy score: {accuracy_score(y, y_prob_2>.5)}")

print(f"model 1 AUC score: {roc_auc_score(y, y_prob_1)}")
print(f"model 2 AUC score: {roc_auc_score(y, y_prob_2)}")

plot_roc_curve(y, y_prob_1)
plt.show()

plot_roc_curve(y, y_prob_2)
plt.show()
