# CreditLens

CreditLens is an end-to-end credit default risk pipeline built on the UCI Credit Card Default dataset. It predicts whether a cardholder will default on their next payment and returns a probability score along with a risk rating. The project covers the full ML lifecycle: exploratory data analysis, model training and evaluation, explainability, experiment tracking, and a production-ready scoring API.

---

## The Problem

Missing a credit default is more costly than a false alarm. A bank that flags too many good customers loses business, but a bank that misses defaults loses money. CreditLens treats these as unequal errors and optimizes accordingly. Recall on class 1 (default) is the north-star metric throughout.

---

## Data

- **Source:** UCI Credit Card Default dataset via Kaggle (`uciml/default-of-credit-card-clients-dataset`)
- **Size:** 30,000 records, 23 features
- **Class imbalance:** 78% non-default, 22% default (3.5:1 ratio)
- **Features:** Credit limit, demographics (age, sex, education, marriage), six months of payment status history, bill amounts, and payment amounts
- **Known limitation:** Data covers Taiwan credit card holders from 2005. It may not generalize to other geographies or current credit behavior.

---

## Methods

Three gradient boosting models were trained and compared: XGBoost, LightGBM, and CatBoost. Each was tuned with `RandomizedSearchCV` optimizing for recall, and class imbalance was handled via `scale_pos_weight`. All experiments are logged in MLflow.

SHAP values were computed for all three models to explain which features drive predictions. Payment status in the most recent month (PAY_0) is consistently the strongest predictor across all models.

EDA identified potential data quality issues in the EDUCATION feature (undocumented categories 0, 5, and 6) and confirmed that payment history separates defaulters from non-defaulters more cleanly than bill amounts or demographics.

---

## Key Results

| Model | Recall (Class 1) | Precision (Class 1) | F1 (Class 1) | AUC-PR |
|---|---|---|---|---|
| XGBoost | 0.628 | **0.459** | 0.530 | **0.548** |
| LightGBM | 0.622 | 0.459 | 0.528 | 0.545 |
| **CatBoost** | **0.634** | 0.459 | **0.533** | 0.547 |

CatBoost has the highest recall and was selected as the production model.

---

## Project Structure

```
creditlens/
├── dags/                  # Airflow DAG for scheduled retraining
├── docs/                  # EU AI Act compliance framing
├── models/                # Saved model artifacts (.pkl)
├── notebooks/             # EDA, model training, and evaluation notebook
├── scripts/               # MLflow training script
├── src/
│   ├── api/               # FastAPI scoring endpoint
│   ├── data/              # Data loading utilities
│   └── models/            # Training, evaluation, and SHAP modules
├── Dockerfile
└── pyproject.toml
```

---

## How to Run

### Setup

```bash
git clone https://github.com/mychellehale/creditlens.git
cd creditlens
uv sync
```

### Train models and log to MLflow

```bash
python -m scripts.train_all_mlflow
mlflow ui  # view results at http://127.0.0.1:5000
```

### Start the scoring API

```bash
uvicorn src.api:app --reload
# docs at http://127.0.0.1:8000/docs
```

### Run with Docker

```bash
docker build -t creditlens .
docker run -p 8000:8000 creditlens
```

---

## Future Work

- Evidently drift report to monitor distribution shift between training and production data
- EDUCATION feature cleanup: group undocumented categories (0, 5, 6) into "other"
- Threshold tuning: the default 0.5 threshold may not be optimal for recall-focused deployment
- Expanded compliance documentation for production use under the EU AI Act

---

## Compliance

CreditLens is classified as a high-risk AI system under EU AI Act Annex III, Section 5(b). See [docs/eu_ai_act_compliance.md](docs/eu_ai_act_compliance.md) for a full mapping against Articles 9, 10, 13, and 14.
