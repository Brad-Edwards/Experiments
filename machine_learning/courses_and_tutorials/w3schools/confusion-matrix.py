import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

actual = np.random.binomial(1, 0.9, size = 1000)
predicted = np.random.binomial(1, 0.9, size = 1000)
confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()

# Accuracy - How often is it correct? True positives + True negatives / total

Accuracy = metrics.accuracy_score(actual, predicted)

# Precision - Of the positives predicted, what percent are correct? True positives / (True positives + False positives)

Precision = metrics.precision_score(actual, predicted)

# Sensitivity (Recall) - Of the positives, what percent are correct? True positives / (True positives + False negatives)

Sensitivity = metrics.recall_score(actual, predicted)

# Specificity - Of the negatives, what percent are correct? True negatives / (True negatives + False positives)

Specificity = metrics.recall_score(actual, predicted, pos_label = 0)

# F1 Score - Harmonic mean of precision and recall. 2 * (Precision * Recall) / (Precision + Recall)
# F1 Score is better than accuracy when there is an uneven class distribution, considers false positives and false negatives

F1_Score = metrics.f1_score(actual, predicted)

print({"Accuracy": Accuracy, "Precision": Precision, "Sensitivity": Sensitivity, "Specificity": Specificity, "F1 Score": F1_Score})

# ROC Curve - Receiver Operating Characteristic Curve
# Plots the true positive rate (sensitivity) against the false positive rate (1 - specificity)

fpr, tpr, thresholds = metrics.roc_curve(actual, predicted)

plt.plot(fpr, tpr)
plt.show()