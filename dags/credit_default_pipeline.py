"""
Daily retraining pipeline for CreditLens credit default risk model.

Pulls the UCI Credit Card Default dataset from Kaggle, preprocesses it, and
trains a CatBoost classifier with RandomizedSearchCV tuned for recall on class 1
(default). Experiment parameters and metrics are logged to MLflow. The best
estimator is saved to /tmp for downstream use.

Tasks: load_data >> process_data >> run_model
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.data import load
from src.models import train, explain, evaluate
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier as xgb
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import shap
import seaborn #I like the visuals better than matplotlib. Is that ok? if not import matplotlib
import sys
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import pickle as pickle
import mlflow

file_path = "uciml/default-of-credit-card-clients-dataset"

def load_data():
    """Download raw dataset from Kaggle and write to /tmp as parquet."""
    df = load.get_kaggle_data(file_path)
    df.to_parquet("/tmp/credit_raw.parquet")

def process_data():
    """Rename target column, compute class imbalance weight, and split into train/test parquet files."""
    df = pd.read_parquet("/tmp/credit_raw.parquet")
    df = df.rename(columns={'default.payment.next.month': 'target'})
    df = df.drop(columns=['ID'], errors='ignore')
    X = df.drop('target', axis = 1)
    y = df['target']

    y_negs = y[y==0].count()
    y_pos = y[y==1].count()

    weight_adjust = y_negs/y_pos
    X_train, X_test, y_train, y_test = train.split_data(df, 'target', 0.2)
    X_train.to_parquet("/tmp/X_train.parquet")
    X_test.to_parquet("/tmp/X_test.parquet")
    y_train.to_frame().to_parquet("/tmp/y_train.parquet")
    y_test.to_frame().to_parquet("/tmp/y_test.parquet")
    with open("/tmp/weight_adjust.txt", "w") as f:
        f.write(str(weight_adjust))

def run_model():
    """Tune CatBoost with RandomizedSearchCV, log results to MLflow, and pickle the best estimator."""
    X_train = pd.read_parquet("/tmp/X_train.parquet")
    X_test = pd.read_parquet("/tmp/X_test.parquet")
    y_train = pd.read_parquet("/tmp/y_train.parquet").squeeze()
    y_test = pd.read_parquet("/tmp/y_test.parquet").squeeze()
    with open("/tmp/weight_adjust.txt", "r") as f:
        weight_adjust = float(f.read())
    param_grid = {
        "max_depth": [4, 5, 6],
        "n_estimators": range(100, 500),
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    }
    search = RandomizedSearchCV(
        estimator=CatBoostClassifier(scale_pos_weight=weight_adjust, random_seed=42, verbose=0),
        param_distributions=param_grid,
        n_iter=10,
        scoring="recall",
        random_state=42,
    ).fit(X_train, y_train)
    model = search.best_estimator_
    tuned_vals = evaluate.evaluate_model(model, X_test, y_test, output_dict=True)
    mlflow.set_experiment("creditlens-default-risk")
    with mlflow.start_run(run_name="CatBoost-scheduled"):
        mlflow.log_param("max_depth", search.best_params_['max_depth'])
        mlflow.log_param("n_estimators", search.best_params_['n_estimators'])
        mlflow.log_param("learning_rate", search.best_params_['learning_rate'])
        mlflow.log_metric("recall_class_1", tuned_vals['1']['recall'])
        mlflow.log_metric("precision_class_1", tuned_vals['1']['precision'])
        mlflow.log_metric("f1_class_1", tuned_vals['1']['f1-score'])
    with open("/tmp/best_model_tuned.pkl", "wb") as f:
        pickle.dump(model, f)

with DAG(
    dag_id = 'credit_default_pipeline',
    start_date = datetime(2026, 4, 7),
    schedule = "@daily",
    catchup = False,
) as dag:
    load_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
        )
    process_task = PythonOperator(
        task_id="process_data",
        python_callable=process_data,
        )
    run_task = PythonOperator(
        task_id="run_model",
        python_callable=run_model,
        )
    load_task >> process_task >> run_task