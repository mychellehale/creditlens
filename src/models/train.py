from typing import Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(dat: pd.DataFrame, target: str, split_ratio: float = 0.2) -> Tuple:
    '''
    Splits a DataFrame into train and test sets for features and target.

    :param dat: Input DataFrame containing features and target column
    :type dat: pd.DataFrame
    :param target: Name of the target column to predict
    :type target: str
    :param split_ratio: Proportion of data to use for the test set. Default is 0.2 (20%)
    :type split_ratio: float
    :return: X_train, X_test, y_train, y_test
    :rtype: Tuple of (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series)
    '''
    X = dat.drop(target, axis=1)
    y = dat[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
    return X_train, X_test, y_train, y_test


def data_model(X_train: pd.DataFrame, y_train: pd.Series, model: Any) -> Any:
    '''
    Trains a sklearn-compatible model on the provided training data.
    Generic and reusable — works with XGBoost, LightGBM, CatBoost, or any sklearn estimator.

    :param X_train: Training features
    :type X_train: pd.DataFrame
    :param y_train: Training target
    :type y_train: pd.Series
    :param model: An instantiated, unfitted sklearn-compatible model
    :type model: Any
    :return: Fitted model
    :rtype: Any
    '''
    fitted = model.fit(X_train, y_train)
    return fitted