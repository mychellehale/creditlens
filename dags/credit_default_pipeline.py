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

file_path = "uciml/default-of-credit-card-clients-dataset"

def load_data():
    df = load.get_kaggle_data(file_path)
    df.to_parquet("/tmp/credit_raw.parquet")

def process_data():
    df = pd.read_parquet("/tmp/credit_raw.parquet")
    df = df.rename(columns={'default.payment.next.month': 'target'})
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
    X_train = pd.read_parquet("/tmp/X_train.parquet")
    y_train= pd.read_parquet("/tmp/y_train.parquet")
    with open("/tmp/weight_adjust.txt", "r") as f:
        weight_adjust = float(f.read())
    hyperparam_tuning = RandomizedSearchCV(estimator = xgb(scale_pos_weight=weight_adjust, base_score = 0.5),
                                        n_iter=10,
                                        param_distributions= {'max_depth': [4,5,6],
                                                                'n_estimators': range(100,500),
                                                                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3]},
                                        scoring='recall').fit(X_train, y_train)

    model_tuned = train.data_model(X_train, y_train, 
                            xgb(scale_pos_weight=weight_adjust, 
                                base_score = 0.5,
                                max_depth = hyperparam_tuning.best_params_['max_depth'],
                                n_estimators = hyperparam_tuning.best_params_['n_estimators'],
                                learning_rate = hyperparam_tuning.best_params_['learning_rate']))
    with open("/tmp/model_tuned.pkl", "wb") as f:
        pickle.dump(model_tuned, f)

def save_values():
    X_test = pd.read_parquet("/tmp/X_test.parquet")
    y_test = pd.read_parquet("/tmp/y_test.parquet")
    # Save Model values
    with open("/tmp/model_tuned.pkl", "rb") as f:
        model_tuned = pickle.load(f)
    tuned_vals = evaluate.evaluate_model(model_tuned, X_test, y_test)
    with open("/tmp/evaluation_results.txt", "w") as f:
        f.write(str(tuned_vals))

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
    save_task = PythonOperator(
        task_id="save_values",
        python_callable=save_values,
        )

    load_task >> process_task >> run_task >> save_task