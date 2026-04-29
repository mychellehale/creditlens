from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load import get_kaggle_data
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Report

from datetime import datetime 

df = get_kaggle_data("uciml/default-of-credit-card-clients-dataset")
df = df.rename(columns={"default.payment.next.month": "target"})
df = df.drop(columns=["ID"])

X = df.drop("target", axis=1)
y = df["target"]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reference = X_train.copy()
current = X_test.copy()

definition = DataDefinition()
reference_dataset = Dataset.from_pandas(reference, data_definition=definition)
current_dataset = Dataset.from_pandas(current, data_definition=definition)

report = Report(metrics=[DataDriftPreset()])
result = report.run(current_data=current_dataset, reference_data=reference_dataset)

output_path = Path(f"reports/data_drift_report_{timestamp}.html")
result.save_html(str(output_path))

print(f"Drift report saved to {output_path}")
