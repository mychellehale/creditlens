# EU AI Act Compliance Framing — CreditLens

**Status:** Compliance-aware development. This document maps CreditLens against key EU AI Act articles. It does not assert full regulatory compliance, which would require independent conformity assessment and legal review.

---

## Classification

CreditLens is an end-to-end ML pipeline that predicts a user's probability of defaulting on their next credit card payment. It uses demographic data, payment history, and monthly balances as inputs, and returns both a probability and a risk rating (low, medium, or high).

Credit scoring systems are explicitly listed in Annex III, Section 5(b) of the EU AI Act as high-risk AI systems. An incorrect default prediction could prevent someone from accessing new lines of credit or increasing their credit limit, directly affecting their financial rights. Because of this, CreditLens is subject to Articles 9 through 15.

---

## Article 9 — Risk Management System

**What it requires:** A continuous process to identify, estimate, and mitigate known and foreseeable risks throughout the system lifecycle.

**How CreditLens addresses it:**

- The north-star metric for model selection is recall on class 1 (default). Missing a default is more costly than a false alarm. At scale, missed defaults represent tens of thousands of dollars in unrecovered losses, so the model is optimized for sensitivity to the positive class.
- The training dataset has a 3.5:1 class imbalance. Without correction, any model trained on this data will be biased toward predicting no default, since that is the majority class. All three model architectures use `scale_pos_weight` to account for this.
- Three models were trained and evaluated independently: XGBoost, LightGBM, and CatBoost. LightGBM outperformed the others on recall. Comparing multiple architectures reduces the risk of locking in a poor modeling choice.
- Drift monitoring via Evidently is planned but not yet implemented.

**Gaps:** Risk management is currently undocumented outside of this framing document. Drift detection is still in progress. No formal failure mode analysis has been conducted.

---

## Article 10 — Data and Data Governance

**What it requires:** Training data must be relevant, representative, and examined for bias. Data governance practices must be documented.

**How CreditLens addresses it:**

- The dataset is the UCI Credit Card Default dataset, with over 30,000 records, sourced via Kaggle (uciml/default-of-credit-card-clients-dataset). It was originally published by Yeh and Lien (2009) and covers credit card holders in Taiwan in 2005.
- The 22% default rate and class imbalance are documented in the EDA section of the notebook and addressed through class weighting during training.
- EDA was conducted to examine potential bias across SEX, EDUCATION, and MARRIAGE. SEX showed a negligible difference in default rate between groups. EDUCATION categories 0, 5, and 6 are not defined in the data dictionary and have been flagged for grouping into the "other" category before any production use.
- The ID column was dropped from model features because it carries no predictive signal.

**Gaps:** The dataset covers only Taiwan and only reflects behavior from 2005. It may not generalize well to other geographies or current credit conditions. No formal bias audit has been conducted. A production deployment would need retraining on current and geographically relevant data.

---

## Article 13 — Transparency and Provision of Information to Users

**What it requires:** Operators must have enough information to understand the system's purpose, performance, limitations, and outputs.

**How CreditLens addresses it:**

- SHAP values are computed for XGBoost and CatBoost models. These explain which features and values most influence each prediction. PAY_0, the most recent payment status, is consistently the strongest predictor across models.
- All model runs are logged in MLflow with parameters and metrics including recall, precision, F1, and AUC-PR. This makes performance traceable and comparable across experiments.
- The FastAPI /predict endpoint returns a default probability, a binary prediction, and a risk level. These are interpretable outputs, not a single opaque score.
- Known limitations are documented here: the model was trained on 2005 Taiwanese data, and the inclusion of demographic features such as SEX, EDUCATION, and MARRIAGE would require legal review before use in any jurisdiction with anti-discrimination protections.

**Gaps:** No formal model card or operator instruction manual exists. A production system would need these before deployment.

---

## Article 14 — Human Oversight

**What it requires:** Humans must be able to understand outputs, detect anomalies, intervene, and override the system when necessary.

**How CreditLens addresses it:**

- The API returns a recommendation, not a binding decision. Final credit decisions are left to human operators.
- The /docs interface lets developers input their own feature values and observe how changes affect the predicted probability and risk level. This supports understanding of model behavior before any action is taken on outputs.
- MLflow provides a full audit trail of all experiment runs, including parameters, metrics, and saved model artifacts.
- Evidently drift detection is planned to alert when production data diverges from the training distribution.

**Gaps:** No formal human-in-the-loop workflow exists. A production deployment would also need override logging to track cases where operators disagree with model recommendations.

---

## Summary

CreditLens is built with its high-risk classification in mind. The current implementation addresses risk management through metric selection and class weighting, data governance through EDA and documented limitations, transparency through SHAP and MLflow, and human oversight through interpretable API outputs and an audit trail.

Full compliance would require a formal conformity assessment, an independent bias audit, an operator instruction manual, and override logging. These gaps are acknowledged and would need to be addressed before any regulated deployment.
