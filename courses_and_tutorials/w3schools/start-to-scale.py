# This code and many values are adapted straight from the w3schools tutorial 
# on Machine Learning in Python. Think of these as my notes on the tutorial.
# https://www.w3schools.com/python/python_ml_getting_started.asp

import numpy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# Some useful functions

def slope_intercept(x):
    return slope * x + intercept

# Distributions with mean value x, standard deviation y, and number of values z
uniformDistribution = numpy.random.uniform(0.0, 5.0, 100000)
# Gaussian distribution
normalDistribution = numpy.random.normal(5.0, 1.0, 100000) 

# Some data to use
car_data_file = "./car_data.csv"
maximumWarpSpeed = numpy.random.uniform(5.0, 1.0, 10)
maximumWarpSpeed.sort()
normalX = numpy.random.normal(5.0, 1.0, 100000)
normalY = numpy.random.normal(10.0, 2.0, 100000)
starshipAge = numpy.random.uniform(25.0, 4.0, 10)
starshipAge.sort()
turtleAge = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
flipOversPerYear = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
coffeeConsumption = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
bugsPerHour = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
totalLinesOfCode = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
qualityOfCode = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

print('\n\nDescriptive Statistics for Maximum Warp Speed Data\n')
print('Mean: ', numpy.mean(maximumWarpSpeed))
print('Median: ', numpy.median(maximumWarpSpeed))
print('Mode: ', stats.mode(maximumWarpSpeed))
print('Standard Deviation: ', numpy.std(maximumWarpSpeed))
print('Variance: ', numpy.var(maximumWarpSpeed))
print('50th Percentile: ', numpy.percentile(maximumWarpSpeed, 50))
print('95th Percentile: ', numpy.percentile(maximumWarpSpeed, 95))
print('\n')

print('Histogram of uniform distribution\n')
plt.hist(uniformDistribution, 100)
plt.suptitle('Uniform Distribution')
plt.show()

print('Histogram of normal distribution\n')
plt.hist(normalDistribution, 100)
plt.suptitle('Normal Distribution')
plt.show()

# Linear Regression
print('Linear Regression of maximum warp speed vs starship age\n')
slope, intercept, r, p, std_err = stats.linregress(maximumWarpSpeed, starshipAge)
print('Slope: ', slope)
print('Intercept: ', intercept)
print('R: ', r)
print('P: ', p)
print('Standard Error: ', std_err)
print('R should be high since this is a good fit for linear regression')
plt.scatter(maximumWarpSpeed, starshipAge)
model = list(map(slope_intercept, maximumWarpSpeed))
plt.plot(maximumWarpSpeed, model)
plt.suptitle('Maximum Warp Speed vs Starship Age')
plt.xlabel('Maximum Warp Speed')
plt.ylabel('Starship Age')
plt.show()

print('\nPredict maximum warp speed of a starship that is 15 years old')
print('Age: {}, Maximum Warp Speed: {}'.format(15, slope_intercept(15)))
print('\nPredict maximum warp speed of a starship that is 2 years old')
print('Age: {}, Maximum Warp Speed: {}'.format(2, slope_intercept(2)))

print('\npredict age of a starship that has a maximum warp speed of 8')
inverse_slope, inverse_intercept, inverse_r, inverse_p, inverse_std_err = stats.linregress(starshipAge, maximumWarpSpeed)
print('Maximum Warp Speed: {}, Age: {}'.format(8, slope_intercept(8)))

print('\nLinear Regression of turtle age vs flip overs per year\n')
print('This data is (deliberately) not a good fit for linear regression')
slope, intercept, r, p, std_err = stats.linregress(turtleAge, flipOversPerYear)
print('Slope: ', slope)
print('Intercept: ', intercept)
print('R: ', r)
print('P: ', p)
print('Standard Error: ', std_err)
print('R should be low since this is not a good fit for linear regression')
plt.scatter(turtleAge, flipOversPerYear)
model = list(map(slope_intercept, turtleAge))
plt.plot(turtleAge, model)
plt.suptitle('Turtle Age vs Flip Overs Per Year')
plt.xlabel('Turtle Age')
plt.ylabel('Flip Overs Per Year')
plt.show()

print('\nPolynomial regression of coffee consumption vs bugs per hour\n')
model = numpy.poly1d(numpy.polyfit(coffeeConsumption, bugsPerHour, 3))
line = numpy.linspace(1, 22, 100)
plt.scatter(coffeeConsumption, bugsPerHour)
plt.plot(line, model(line))
plt.suptitle('Coffee Consumption vs Bugs Per Hour')
plt.xlabel('Coffee Consumption')
plt.ylabel('Bugs Per Hour')
print('Are they correlated? What\'s the R-squared value?')
print('R-squared: ', r2_score(bugsPerHour, model(coffeeConsumption)))
print('\nPredict bugs per hour for a coffee consumption of 11')
print('Coffee Consumption: {}, Bugs Per Hour: {}'.format(11, model(11)))
print('Coffee Consumption: {}, Bugs Per Hour: {}'.format(6, model(6)))
plt.show()

print('\nPolynomial regression of total lines of code vs quality of code\n')
print('This data is (deliberately) not a good fit for polynomial regression')
model = numpy.poly1d(numpy.polyfit(totalLinesOfCode, qualityOfCode, 3))
line = numpy.linspace(1, 100, 100)
plt.scatter(totalLinesOfCode, qualityOfCode)
plt.plot(line, model(line))
plt.suptitle('Total Lines of Code vs Quality of Code')
plt.xlabel('Total Lines of Code')
plt.ylabel('Quality of Code')
print('Are they correlated? What\'s the R-squared value?')
print('R-squared: ', r2_score(qualityOfCode, model(totalLinesOfCode)))
plt.show()

print('\nMultiple linear regression on some fake car data\n')
df = pandas.read_csv(car_data_file)
X = df[['Weight', 'Volume']]
y = df['CO2']
regr = linear_model.LinearRegression()
regr.fit(X, y)
predictedC02 = regr.predict([[2300, 1300]])
print('Predicted C02 for a vehicle weighing 2300kg with a 1.3L engine: ', predictedC02)
print('Coefficients: ', regr.coef_)

print('\nMultiple linear regression but scaling the data first\n')
scale = StandardScaler()
scaledX = scale.fit_transform(X)
regr = linear_model.LinearRegression()
regr.fit(scaledX, y)
scaled = scale.transform([[2300, 1300]])
predictedC02 = regr.predict([scaled[0]])
print('Predicted C02 for a vehicle weighing 2300kg with a 1.3L engine: ', predictedC02)
