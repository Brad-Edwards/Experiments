import pandas as pd
from sklearn import linear_model

# Converting non numerical data to numerical data for use in machine learning models

# One Hot Encode data - create dummy variables for each category
# This is useful for data that is not ordinal (i.e. there is no order to the data)

cars = pd.read_csv("car_data.csv")

# Note, can use drop_first=True to drop the first column of the dummy variables, saving space.
# This is because the first column can be inferred from the other columns (i.e. if all other columns are 0, the first column must be 1, provided that at least one column MUST be 1)

ohe_cars = pd.get_dummies(cars[["Car"]])

X = pd.concat([cars[["Volume", "Weight"]], ohe_cars], axis=1)
y = cars["CO2"]

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedCO2 = regr.predict(
    [[2300, 1300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
)

print(predictedCO2)
