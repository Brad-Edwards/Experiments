import numpy as np
from sklearn import linear_model

# X represents the size of a tumour in cm. The capital X is used to represent a matrix of features, this is a convention in machine learning

X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)

# X must be reshaped into a column vector for the logistic regression to work
# y represents whether the tumour is malignant (1) or benign (0)

y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X, y)

print("Tumour is malignant if the probability is greater than 0.5, benign if less than 0.5")

# predict if a tumour with a size of 2.5cm is malignant or benign

print(logr.predict(np.array([2.5]).reshape(-1,1)))

# predict if a tumour with a size of 3.5cm is malignant or benign

print(logr.predict(np.array([3.5]).reshape(-1,1)))

# Coefficients of the logistic regression is the expected changed in log-odds for a one unit increase in X

log_odds = logr.coef_
odds = np.exp(log_odds)

print("The odds of a tumour being malignant increase by a factor of " + str(odds[0][0]) + " for every one unit increase in size")

# The coefficient and intercept values can be used to calculate the probability of a tumour being malignant for any given size

def logit2prob(logr, x):
    log_odds = logr.coef_ * x + logr.intercept_
    odds = np.exp(log_odds)
    prob = odds / (1 + odds)
    return prob

probabilities = logit2prob(logr, X)

print("The probability of a tumour being malignant for a size of X[0] is " + str(probabilities[0][0]))
print("The probability of a tumour being malignant for a size of X[1] is " + str(probabilities[1][0]))

print("All probabilities for X: " + str(probabilities))