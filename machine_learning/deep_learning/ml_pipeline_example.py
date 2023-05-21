# This file contains a simple implementation of Logistic Regression using Gradient Descent.
#
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class MLModel:
    """
    The purpose of this class is to provide a structured way to handle common tasks in a machine learning workflow.
    These tasks include loading data, preprocessing data, training a model, and evaluating the model's performance.

    Preprocessing involves preparing the raw data to be input into a machine learning model. This often includes
    filling missing values (imputing) and converting categorical data into a format that can be understood
    by the model (one-hot encoding). Imputing is necessary because most machine learning models cannot handle
    missing values. One-hot encoding is used to convert categorical data into a binary format that can be used
    by the model.

    The model used in this class is a RandomForestRegressor. A random forest is a type of ensemble machine learning
    model that operates by constructing a multitude of decision trees at training time and outputting the mean
    prediction of the individual trees for regression tasks.

    The performance of the model is evaluated using cross-validation. Cross-validation is a resampling procedure
    used to evaluate machine learning models on a limited data sample. The procedure has a single parameter
    called k that refers to the number of groups that a given data sample is to be split into.
    Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning
    model on unseen data.

    The metric used to evaluate the model's performance is the Mean Absolute Error (MAE). The MAE is a measure
    of how close predictions are to the actual outcomes. It's the average over the test sample of the absolute
    differences between prediction and actual observation where all individual differences have equal weight.

    The class provides methods to perform each of these tasks, and the methods are designed to be
    called in sequence. For example, you would first call load_data, then preprocess_data, then train_model,
    and finally evaluate_model.
    """

    def __init__(self):
        """
        Initializes the MLModel class with the following private variables:

        self._df: This will hold the training data loaded from a CSV file.
        It's initialized as None and will be a pandas DataFrame once data is loaded.

        self._y: This will hold the target values from the training data.
        It's initialized as None and will be a pandas Series once data is loaded and the target column is set.

        self._numeric_features: This will hold the names of the numeric feature columns in the training data.
        It's initialized as None and will be a pandas Index object once data is loaded.

        self._categorical_features: This will hold the names of the categorical feature columns in the training data.
        It's initialized as None and will be a pandas Index object once data is loaded.

        self._preprocessor: This will hold a sklearn ColumnTransformer object that preprocesses the data by
        filling missing values and encoding categorical features. It's initialized as None and will
        be set in the preprocess_data method.

        self._pipeline: This will hold a sklearn Pipeline object that includes the preprocessor and the model.
        It's initialized as None and will be set in the train_model method.

        self._test_df: This will hold the test data loaded from a CSV file. It's initialized as None
        and will be a pandas DataFrame once test data is loaded.
        """
        self._df = None
        self._y = None
        self._numeric_features = None
        self._categorical_features = None
        self._preprocessor = None
        self._pipeline = None
        self._test_df = None

    def load_data(self, train_path, test_path, target_column):
        """
        Loads training and test data from CSV files and sets the target column.

        1. Loads the training data from a CSV file located at the path specified by 'train_path'.
        The data is stored in a pandas DataFrame.

        2. Drops any rows in the training data where the value in the target column is missing.
        This is done because machine learning models cannot handle target values that are missing.

        3. Sets the target values (self._y) by extracting the column specified by 'target_column'
        from the training data.

        4. Removes the target column from the training data (self._df), as it's now stored
        separately in self._y.

        5. Loads the test data from a CSV file located at the path specified by 'test_path'.
        The test data is stored in a separate pandas DataFrame (self._test_df).

        6. Identifies the numeric and categorical features in the training data. This is done by
        selecting columns of certain data types. Numeric features are those with data type
        'int64' or 'float64', and categorical features are those with data type 'object'. The names
        of these columns are stored in self._numeric_features and self._categorical_features, respectively.

        Parameters:

        train_path (str): The path to the CSV file containing the training data.

        test_path (str): The path to the CSV file containing the test data.

        target_column (str): The name of the column in the training data that contains the target values.
        """
        self._df = pd.read_csv(train_path)
        self._df = self._df.dropna(subset=[target_column])
        self._y = self._df[target_column]
        self._df = self._df.drop(columns=[target_column])

        # Load test data
        self._test_df = pd.read_csv(test_path)

        # Identify numeric and categorical features
        self._numeric_features = self._df.select_dtypes(
            include=["int64", "float64"]
        ).columns
        self._categorical_features = self._df.select_dtypes(include=["object"]).columns

    def preprocess_data(
        self,
        numeric_imputer_strategy="median",
        numeric_fill_value=None,
        categorical_imputer_strategy="most_frequent",
        categorical_fill_value=None,
        handle_unknown="ignore",
    ):
        """
        Preprocesses the training data by filling missing values and encoding categorical features.

        This method performs several important steps:

        1. For numeric features, missing values are filled using a strategy specified by
        'numeric_imputer_strategy'. By default, this is set to "median", which means that missing values
        are filled with the median value of the respective feature. If 'numeric_fill_value' is specified,
        missing values are filled with this value instead.

        2. For categorical features, missing values are filled using a strategy specified by
        'categorical_imputer_strategy'. By default, this is set to "most_frequent", which means that
        missing values are filled with the most frequent category of the respective feature. If
        'categorical_fill_value' is specified, missing values are filled with this value instead.

        3. After filling missing values, categorical features are one-hot encoded. This means that each
        unique category in a feature is turned into a separate binary feature (0 or 1). The 'handle_unknown'
        parameter determines how the encoder handles unknown categories that might appear in the test data.
        By default, it's set to "ignore", which means that the encoder will ignore unknown categories
        and not throw an error.

        4. The preprocessing steps for numeric and categorical features are combined into a single sklearn
        ColumnTransformer. This allows us to apply different preprocessing steps to different types of
        features in one go.

        Parameters:

        numeric_imputer_strategy (str, optional): The strategy to use for filling missing values in
        numeric features. Default is "median".

        numeric_fill_value (any, optional): The value to use for filling missing values in numeric
        features. Default is None.

        categorical_imputer_strategy (str, optional): The strategy to use for filling missing values
        in categorical features. Default is "most_frequent".

        categorical_fill_value (any, optional): The value to use for filling missing values in
        categorical features. Default is None.

        handle_unknown (str, optional): The method to use for handling unknown categories in the
        one-hot encoder. Default is "ignore".
        """
        numeric_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy=numeric_imputer_strategy, fill_value=numeric_fill_value
                    ),
                )
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy=categorical_imputer_strategy,
                        fill_value=categorical_fill_value,
                    ),
                ),
                ("onehot", OneHotEncoder(handle_unknown=handle_unknown)),
            ]
        )

        self._preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self._numeric_features),
                ("cat", categorical_transformer, self._categorical_features),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy=categorical_imputer_strategy,
                        fill_value=categorical_fill_value,
                    ),
                ),
                ("onehot", OneHotEncoder(handle_unknown=handle_unknown)),
            ]
        )

        self._preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self._numeric_features),
                ("cat", categorical_transformer, self._categorical_features),
            ]
        )

    def train_model(self, n_estimators=100, random_state=0):
        """
        Trains a RandomForestRegressor model on the preprocessed data.

        This method performs several important steps:

        1. A RandomForestRegressor model is created. The number of trees in the forest (n_estimators)
        and the seed used by the random number generator (random_state) can be specified. By default,
        the forest consists of 100 trees and the random seed is set to 0.

        2. The RandomForestRegressor model is added to a pipeline. The pipeline includes the
        preprocessor defined in the preprocess_data method and the model. The preprocessor will be
        applied to the data first, and then the processed data will be used to train the model.
        Using a pipeline ensures that the same preprocessing steps are applied to the training data
        and any future data the model will make predictions on.

        3. The pipeline is fitted to the training data. This involves applying the preprocessing
        steps to the training data and then training the model on the processed data.

        Parameters:

        n_estimators (int, optional): The number of trees in the forest. Default is 100.

        random_state (int, optional): The seed used by the random number generator. Default is 0.
        """
        model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )

        self._pipeline = Pipeline(
            steps=[("preprocessor", self._preprocessor), ("model", model)]
        )

        self._pipeline.fit(self._df, self._y)

    def evaluate_model(self, cv=5, scoring="neg_mean_absolute_error"):
        """
        Evaluates the model using cross-validation and returns the mean absolute error.

        Cross-validation is a statistical method used to estimate the skill of machine
        learning models. It is commonly used in applied machine learning to compare and
        select a model for a given predictive modeling problem because it is easy to
        understand, easy to implement, and results in skill estimates that generally
        have a lower bias than other methods.

        During cross-validation, the data is split into 'cv' groups or 'folds'. Then,
        for each unique group:

        1. The model is trained using 'cv-1' groups as training data;

        2. The resulting model is validated on the remaining part of the
        data (i.e., it is used as a test set to compute a performance measure
        such as accuracy).

        The performance measure reported by cross-validation is then the average of
        the performance measure computed in each experiment.

        In this method, the performance measure is the mean absolute error (MAE).
        The MAE is a measure of how close predictions are to the actual outcomes.
        It's the average over the test sample of the absolute differences between
        prediction and actual observation where all individual differences have
        equal weight.

        Parameters:

        cv (int, optional): The number of folds in the cross-validation. Default is 5.

        scoring (str, optional): A string indicating the scoring metric. The default is "neg_mean_absolute_error".

        Returns:

        float: The mean absolute error of the model.
        """
        scores = -1 * cross_val_score(
            self._pipeline, self._df, self._y, cv=cv, scoring=scoring
        )
        return scores.mean()

    def check_cv_values(self, cv_values, scoring="neg_mean_absolute_error"):
        """
        Checks different values of the cv parameter in cross-validation and returns the mean absolute error for each.

        This method is used to tune the 'cv' parameter in the cross-validation process. The 'cv' parameter
        determines the number of folds in the cross-validation.

        For each value of 'cv' in 'cv_values', the method evaluates the model using cross-validation and
        computes the mean absolute error (MAE). The MAE is a measure of how close predictions are to the
        actual outcomes. It's the average over the test sample of the absolute differences between prediction and
        actual observation where all individual differences have equal weight.

        The method returns a dictionary where the keys are the 'cv' values and the values are the corresponding MAEs.
        Additionally, the dictionary includes a 'best' key, which corresponds to the 'cv' value that resulted
        in the lowest MAE.

        Parameters:

        cv_values (list): A list of integers representing the 'cv' values to check.

        scoring (str, optional): A string indicating the scoring metric. The default is "neg_mean_absolute_error".

        Returns:
        dict: A dictionary where the keys are the 'cv' values and the values are the corresponding MAEs.
        The dictionary also includes a 'best' key, which corresponds to the 'cv' value that resulted in the lowest MAE.
        """
        results = {}
        for cv in cv_values:
            mae = self.evaluate_model(cv=cv, scoring=scoring)
            results[cv] = mae

        # Find the cv value with the lowest MAE
        best_cv = min(results, key=results.get)
        results["best"] = best_cv

        return results


def main():
    # Load data
    model = MLModel()
    model.load_data(
        train_path="./data/home_data/train.csv",
        test_path="./data/home_data/test.csv",
        target_column="SalePrice",
    )

    # Preprocess data
    model.preprocess_data()

    # Train and evaluate model with defaults
    model.train_model(n_estimators=50, random_state=0)
    mae = model.evaluate_model()
    print("Mean Absolute Error with defaults:", mae)
    print("Next step may take several minutes...")
    # Check different values of the cv parameter
    cv_values = [10, 20, 30, 40, 50]
    results = model.check_cv_values(cv_values)
    print("Results (MAE for each cv_value and best cv_value):", results)


if __name__ == "__main__":
    main()
