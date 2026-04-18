import shap
import pandas as pd

def shap_exp_xgb(X_test: pd.DataFrame, fitted_model):
    '''
    Generates SHAP values for a tree-based model (XGBoost, LightGBM, CatBoost)
    and renders a summary plot of feature importances.

    :param X_test: Test data (features only, no target)
    :type X_test: pd.DataFrame
    :param fitted_model: A fitted tree-based model
    :return: SHAP values array for each feature
    :rtype: np.ndarray
    '''
    explainer = shap.TreeExplainer(fitted_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    return(shap_values)

def shap_exp_other(X_test: pd.DataFrame, fitted_model):
    '''
    Generates SHAP values for non-tree-based models (e.g. logistic regression, neural nets)
    using the model-agnostic shap.Explainer, and renders a summary plot.

    :param X_test: Test data (features only, no target)
    :type X_test: pd.DataFrame
    :param fitted_model: A fitted non-tree-based model
    :return: SHAP values array for each feature
    :rtype: np.ndarray
    '''
    explainer = shap.Explainer(fitted_model)
    shap_values = explainer(X_test).values
    shap.summary_plot(shap_values, X_test)
    return(shap_values)