import sys
from pathlib import Path
import pickle
from tqdm import tqdm

import mlflow
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from datetime import datetime

# sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.load import get_kaggle_data
from src.models.evaluate import evaluate_model, auc_eval

# Load data
df = get_kaggle_data("uciml/default-of-credit-card-clients-dataset")
df = df.rename(columns={"default.payment.next.month": "target"})
X = df.drop(["ID", "target"], axis=1)
y = df["target"]
weight_adjust = float((y == 0).sum() / (y == 1).sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("creditlens-default-risk")

models = [
    ("XGBoost", XGBClassifier(scale_pos_weight=weight_adjust, random_state=42)),
    ("LightGBM", lgb.LGBMClassifier(scale_pos_weight=weight_adjust, random_state=42)),
    ("CatBoost", CatBoostClassifier(scale_pos_weight=weight_adjust, verbose=0, random_seed=42)),
]

param_grid = {
    "max_depth": [4, 5, 6],
    "n_estimators": range(100, 500),
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
}

best_recall = 0.0
best_model = None
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
best_model_name = None

for name, estimator in tqdm(models, desc="Training models"):
    search = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid,
                                n_iter=10, scoring="recall", random_state=42).fit(X_train, y_train)
    tuned_vals = evaluate_model(search.best_estimator_, X_test, y_test, output_dict= True) 
    auc_pr = auc_eval(search.best_estimator_, X_test=X_test, y_test=y_test)
    with mlflow.start_run(run_name = name):
        mlflow.log_param("max_depth", search.best_params_['max_depth'])
        mlflow.log_param("n_estimators", search.best_params_['n_estimators'])
        mlflow.log_param("learning_rate", search.best_params_['learning_rate'])
        mlflow.log_metric("recall_class_1", tuned_vals['1']['recall'])
        mlflow.log_metric("precision_class_1", tuned_vals['1']['precision'])
        mlflow.log_metric("F1_class_1", tuned_vals['1']['f1-score'])
        mlflow.log_metric("auc_pr", auc_pr[1])
        if best_recall < tuned_vals['1']['recall']:
            best_recall = tuned_vals['1']['recall']
            best_model = search
            best_model_name = name
    with open(f"models/best_model_{timestamp}.pkl", "wb") as f:
        pickle.dump(best_model, f)

print(f'All Done. The best model was: {best_model_name} with recall {best_recall:.3f}')


