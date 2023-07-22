# Copyright (c) 2023 Brad Edwards
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# This example uses the Combined Cycle Power Plant dataset from the UCI Machine Learning Repository.
# The point of this example is to show how to use the LinearRegression model from scikit-learn to perform regression,
# and to show how much easier, faster, and less error-prone it is to do this way instead of with a custom implementation.

data = pd.read_csv("CCPP_data.csv")

features = data.drop(columns=["PE"])
target = data["PE"]

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

scaler = StandardScaler()

features_train_scaled = scaler.fit_transform(features_train)

features_test_scaled = scaler.transform(features_test)

model_lr = LinearRegression()

model_lr.fit(features_train_scaled, target_train)

target_pred_lr = model_lr.predict(features_test_scaled)

mse_lr = mean_squared_error(target_test, target_pred_lr)
mae_lr = mean_absolute_error(target_test, target_pred_lr)
r2_lr = r2_score(target_test, target_pred_lr)

poly = PolynomialFeatures(degree=2)

features_train_poly = poly.fit_transform(features_train_scaled)

features_test_poly = poly.transform(features_test_scaled)

model_pr = LinearRegression()

model_pr.fit(features_train_poly, target_train)

target_pred_pr = model_pr.predict(features_test_poly)

mse_pr = mean_squared_error(target_test, target_pred_pr)
mae_pr = mean_absolute_error(target_test, target_pred_pr)
r2_pr = r2_score(target_test, target_pred_pr)

# MSE - Lower the value, better the model
# MAE - Lower the value, better the model
# R2 - Higher the value, better the model
print("Linear Regression with Feature Scaling:")
print(f"    Mean Squared Error (MSE): {mse_lr}")
print(f"    Mean Absolute Error (MAE): {mae_lr}")
print(f"    R-squared (R2): {r2_lr}")
print()
print("Polynomial Regression:")
print(f"    Mean Squared Error (MSE): {mse_pr}")
print(f"    Mean Absolute Error (MAE): {mae_pr}")
print(f"    R-squared (R2): {r2_pr}")
