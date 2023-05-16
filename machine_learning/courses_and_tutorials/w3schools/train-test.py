import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

plt.scatter(x, y)
plt.suptitle('Customer data')
plt.xlabel('Time before making purchase (minutes)')
plt.ylabel('Amount spent (dollars)')
plt.show()

train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]

print('\nThe training data should look like the original data or we have a biased training set\n')
plt.scatter(train_x, train_y)
plt.suptitle('Training data')
plt.xlabel('Time before making purchase (minutes)')
plt.ylabel('Amount spent (dollars)')
plt.show()

print('\nThe testing data should look like the original data or we have a biased testing set\n')
plt.scatter(test_x, test_y)
plt.suptitle('Testing data')
plt.xlabel('Time before making purchase (minutes)')
plt.ylabel('Amount spent (dollars)')
plt.show()

print('\nHere is where you need to know your models, so you have an idea what regression model would be the best fit.\n')
print('In this case, we will use polynomial regression\n')
model = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
line = numpy.linspace(0, 6, 100)
plt.scatter(train_x, train_y)
plt.plot(line, model(line))
plt.suptitle('Polynomial regression of training data')
plt.xlabel('Time before making purchase (minutes)')
plt.ylabel('Amount spent (dollars)')
plt.show()

r2_score_train = r2_score(train_y, model(train_x))
print('\nR-squared value for training data: ', r2_score_train)

print('\nHow well does the model fit the test data?\n')
r2 = r2_score(test_y, model(test_x))
print('R-squared value for test data: ', r2)

print('\nBoth are ok r squared values.\n')
print('How much will a customer spend if they are in the store for 5 minutes?\n')
print('Amount spent: ', model(5))