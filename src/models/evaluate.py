from typing import Any
import pandas as pd
from sklearn.metrics import classification_report, _ranking 


def evaluate_model(fitted_model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> str:
    '''
    Evaluates a fitted classifier against the test set and returns a classification report.
    Reports precision, recall, F1-score, and support for each class.

    :param fitted_model: A fitted sklearn-compatible classifier
    :type fitted_model: Any
    :param X_test: Test features
    :type X_test: pd.DataFrame
    :param y_test: True labels for the test set
    :type y_test: pd.Series
    :return: Classification report as a formatted string
    :rtype: str
    '''
    return classification_report(y_test, fitted_model.predict(X_test))

def auc_eval(fitted_model: Any, X_test: pd.DataFrame, y_test:pd.Series) -> Any:
    '''
    Getting AUC-PR curve and score for fitted model.

    :param fitted_model: A fitted sklearn-compatible classifier
    :type fitted_model: Any
    :param X_test: Test features
    :type X_test: pd.DataFrame
    :param y_test: True labels for the test set
    :type y_test: pd.Series
    :return: AUC-PR curve and score for fitted model
    :rtype: tuple
    '''
    model_predict = fitted_model.predict_proba(X_test)
    auc_score = _ranking.average_precision_score(y_test, model_predict[:,1])
    auc_curve = _ranking.precision_recall_curve(y_test, model_predict[:,1])
    return(auc_curve, auc_score)