# 🤖 AI Usage & Verification Log: SkillLens

This document logs the AI-assisted workflows, code generation steps, architectural decisions, and human verification methodologies applied during the development of **SkillLens**.

---

## 1. Project Overview & Scope
- **Application:** SkillLens - Student Placement Readiness & Explainability Web Application.
- **Role Executed:** Principal Machine Learning Engineer & Full-Stack Streamlit Developer.
- **Core Technology Stack:** Python 3.12+, Streamlit, Scikit-Learn, XGBoost, SHAP, Plotly, Pytest.

---

## 2. AI Assistance Log

| Component | AI Contribution | Verification & Quality Assurance Step |
| :--- | :--- | :--- |
| **Schema Parser & Preprocessing (`src/preprocessing.py`)** | Generated automated column detection, target leakage prevention, and scikit-learn `ColumnTransformer` pipeline. | Inspected raw Kaggle dataset columns, verified handling of missing columns, verified out-of-vocabulary categories in unit tests. |
| **Model Training & Evaluation (`src/model.py`)** | Architected Baseline (Logistic Regression) vs Advanced (XGBoost) training routines, metric computations, and failure case extraction. | Verified that both models evaluate on the identical frozen test set without data leakage. Verified ROC-AUC, PR curve, and confusion matrix dimensions. |
| **Explainability Engine (`src/explainability.py`)** | Developed SHAP TreeExplainer wrapper, local driver waterfall extraction (+/- factor impacts), narrative generation, and cohort benchmark comparisons. | Validated mathematical bounds of SHAP contributions; implemented heuristic fallback if SHAP native C-extensions encounter runtime errors. |
| **Failure Analysis Logger (`src/model.py`)** | Created automated failure case selector extracting >=20 misclassified/uncertain samples with qualitative root-cause diagnostic hypotheses. | Verified extraction of False Positives, False Negatives, and Borderline cases with rich candidate context. |
| **Streamlit Web Application (`app.py`)** | Designed modern 5-tab responsive UI, Plotly gauges, What-If simulation sliders, cohort radar charts, and batch prediction uploader. | Tested UI rendering, interactive state management, real-time delta score calculations, and CSV batch processing. |
| **Automated Test Suite (`tests/test_pipeline.py`)** | Generated 10 unit tests covering schema parsing, missing data imputation, unseen categories, model metrics, and explainability integrity. | Executed `pytest` suite in virtual environment with 100% test pass rate. |

---

## 3. Key Design Decisions & Guardrails

1. **Strict Prevention of Data Leakage:**
   - Evaluated column patterns to prevent target leakage (e.g. `Salary`, `Package`, `Company_Name` automatically quarantined if present).
   - Dataset splitting is stratified and executed **before** fitting the preprocessing pipeline; the test set is frozen to `data/test_frozen.csv`.

2. **Fault-Tolerant Inference:**
   - The preprocessing pipeline ensures that if an inference payload lacks certain fields or contains unmapped strings, default schema statistics are applied without raising uncaught exceptions.

3. **Interpretable Metric Scaling:**
   - Raw numeric values are preserved for display in user diagnostics and recommendations, while standardized z-scores are passed to linear and gradient boosting classifiers.
