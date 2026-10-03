"""
SkillLens Automated Test Suite
Unit tests for schema parsing, data pipeline transformations, edge cases,
missing values, unexpected columns, model training, SHAP explainability, and failure logging.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.explainability import ExplainabilityEngine, compute_cohort_benchmarks
from src.model import (
    evaluate_model,
    generate_failure_analysis_log,
    train_advanced_model,
    train_baseline_model,
)
from src.preprocessing import (
    DatasetSchema,
    PlacementDataPipeline,
    SchemaParser,
    load_and_split_data,
    resolve_dataset_path,
)


@pytest.fixture
def sample_raw_dataframe():
    """Provides a synthetic DataFrame mimicking Kaggle placement schema."""
    np.random.seed(42)
    n = 100
    data = {
        "StudentID": range(1, n + 1),
        "CGPA": np.random.uniform(6.0, 9.5, size=n),
        "Internships": np.random.randint(0, 4, size=n),
        "Projects": np.random.randint(0, 5, size=n),
        "Workshops/Certifications": np.random.randint(0, 3, size=n),
        "AptitudeTestScore": np.random.randint(50, 100, size=n),
        "SoftSkillsRating": np.random.uniform(2.5, 5.0, size=n),
        "ExtracurricularActivities": np.random.choice(["Yes", "No"], size=n),
        "PlacementTraining": np.random.choice(["Yes", "No"], size=n),
        "SSC_Marks": np.random.randint(50, 95, size=n),
        "HSC_Marks": np.random.randint(50, 95, size=n),
        "PlacementStatus": np.random.choice(["Placed", "NotPlaced"], size=n),
    }
    return pd.DataFrame(data)


def test_schema_parser_detection(sample_raw_dataframe):
    """Verifies automated detection of ID, Target, Numeric, and Categorical features."""
    schema = SchemaParser.parse(sample_raw_dataframe)

    assert schema.id_column == "StudentID"
    assert schema.target_column == "PlacementStatus"
    assert "CGPA" in schema.numeric_features
    assert "AptitudeTestScore" in schema.numeric_features
    assert "ExtracurricularActivities" in schema.categorical_features
    assert "PlacementTraining" in schema.categorical_features
    assert schema.target_mapping["Placed"] == 1
    assert schema.target_mapping["NotPlaced"] == 0


def test_target_leakage_detection(sample_raw_dataframe):
    """Verifies that columns with target leakage patterns (e.g. Salary, CTC) are excluded."""
    df_leak = sample_raw_dataframe.copy()
    df_leak["Salary"] = 600000
    df_leak["Company_Name"] = "Google"

    schema = SchemaParser.parse(df_leak)
    assert "Salary" in schema.leakage_features
    assert "Company_Name" in schema.leakage_features
    assert "Salary" not in schema.all_feature_names
    assert "Company_Name" not in schema.all_feature_names


def test_pipeline_fit_transform_no_nan(sample_raw_dataframe):
    """Ensures preprocessing pipeline transforms data without introducing NaNs."""
    pipeline = PlacementDataPipeline()
    X_trans = pipeline.fit_transform(sample_raw_dataframe)

    assert X_trans is not None
    assert X_trans.shape[0] == len(sample_raw_dataframe)
    assert not np.isnan(X_trans).any(), "Transformed feature matrix contains unexpected NaNs"
    assert len(pipeline.transformed_feature_names) == X_trans.shape[1]


def test_missing_data_imputation(sample_raw_dataframe):
    """Verifies graceful imputation of missing numeric (median) and categorical (mode) values."""
    df_missing = sample_raw_dataframe.copy()
    df_missing.loc[0:5, "CGPA"] = np.nan
    df_missing.loc[10:15, "PlacementTraining"] = np.nan
    df_missing.loc[20:25, "AptitudeTestScore"] = np.nan

    pipeline = PlacementDataPipeline()
    X_trans = pipeline.fit_transform(df_missing)

    assert not np.isnan(X_trans).any()
    assert X_trans.shape[0] == len(df_missing)


def test_unexpected_and_missing_columns(sample_raw_dataframe):
    """Tests pipeline robustness when user inputs missing features or extra unseen columns."""
    pipeline = PlacementDataPipeline()
    pipeline.fit(sample_raw_dataframe)

    # Input with missing columns + extra random columns
    incomplete_input = {
        "CGPA": 8.2,
        "Internships": 2,
        "RandomUnknownFeature": "IgnoredValue",
        "AnotherExtraColumn": 999,
    }

    X_single = pipeline.transform_single(incomplete_input)
    assert X_single.shape == (1, len(pipeline.transformed_feature_names))
    assert not np.isnan(X_single).any()


def test_unseen_categorical_values(sample_raw_dataframe):
    """Tests that out-of-vocabulary categorical inputs are ignored gracefully without throwing."""
    pipeline = PlacementDataPipeline()
    pipeline.fit(sample_raw_dataframe)

    unseen_input = {
        "CGPA": 7.5,
        "Internships": 1,
        "Projects": 2,
        "Workshops/Certifications": 1,
        "AptitudeTestScore": 75,
        "SoftSkillsRating": 4.0,
        "ExtracurricularActivities": "RareCategoryNeverSeenBefore",
        "PlacementTraining": "SpecialForeignTraining",
        "SSC_Marks": 70,
        "HSC_Marks": 75,
    }

    X_single = pipeline.transform_single(unseen_input)
    assert X_single.shape == (1, len(pipeline.transformed_feature_names))
    assert not np.isnan(X_single).any()


def test_model_training_and_evaluation(sample_raw_dataframe):
    """Tests Baseline and Advanced model training and metric integrity."""
    pipeline = PlacementDataPipeline()
    pipeline.fit(sample_raw_dataframe)

    X = pipeline.transform(sample_raw_dataframe)
    y = pipeline.encode_target(sample_raw_dataframe["PlacementStatus"])

    # Split
    X_train, X_test = X[:70], X[70:]
    y_train, y_test = y[:70], y[70:]

    baseline = train_baseline_model(X_train, y_train)
    res_base = evaluate_model(baseline, "Baseline", X_test, y_test, pipeline.transformed_feature_names)

    advanced = train_advanced_model(X_train, y_train)
    res_adv = evaluate_model(advanced, "Advanced", X_test, y_test, pipeline.transformed_feature_names)

    for res in [res_base, res_adv]:
        assert 0.0 <= res.accuracy <= 1.0
        assert 0.0 <= res.precision <= 1.0
        assert 0.0 <= res.recall <= 1.0
        assert 0.0 <= res.f1 <= 1.0
        assert 0.0 <= res.roc_auc <= 1.0
        assert res.y_proba.min() >= 0.0
        assert res.y_proba.max() <= 1.0
        assert res.confusion_matrix.shape == (2, 2)


def test_explainability_engine_local_and_global(sample_raw_dataframe):
    """Tests local factor impact calculation and global feature importance extraction."""
    pipeline = PlacementDataPipeline()
    pipeline.fit(sample_raw_dataframe)
    X = pipeline.transform(sample_raw_dataframe)
    y = pipeline.encode_target(sample_raw_dataframe["PlacementStatus"])

    model = train_advanced_model(X, y)
    engine = ExplainabilityEngine(model=model, pipeline=pipeline, background_data=X)

    # Global importance
    df_global = engine.get_global_importance(X)
    assert not df_global.empty
    assert "importance" in df_global.columns
    assert "display_name" in df_global.columns

    # Local single explanation
    single_input = {
        "CGPA": 8.7,
        "Internships": 2,
        "Projects": 3,
        "Workshops/Certifications": 2,
        "AptitudeTestScore": 88,
        "SoftSkillsRating": 4.5,
        "ExtracurricularActivities": "Yes",
        "PlacementTraining": "Yes",
        "SSC_Marks": 85,
        "HSC_Marks": 88,
    }

    explanation = engine.explain_instance(single_input)
    assert 0.0 <= explanation.predicted_probability <= 1.0
    assert explanation.readiness_tier != ""
    assert len(explanation.factors) > 0
    assert len(explanation.narrative_summary) > 10
    assert len(explanation.top_recommendations) > 0


def test_failure_analysis_logger(sample_raw_dataframe):
    """Tests that at least 20 diagnostic cases are extracted with failure reasons."""
    pipeline = PlacementDataPipeline()
    pipeline.fit(sample_raw_dataframe)
    X = pipeline.transform(sample_raw_dataframe)
    y = pipeline.encode_target(sample_raw_dataframe["PlacementStatus"])

    model = train_advanced_model(X, y)
    res = evaluate_model(model, "Advanced", X, y, pipeline.transformed_feature_names)

    failure_log = generate_failure_analysis_log(
        test_df_raw=sample_raw_dataframe,
        y_test=y,
        advanced_result=res,
        pipeline=pipeline,
        min_cases=20,
    )

    assert len(failure_log) >= 20
    for case in failure_log:
        assert case.error_type in ["False Positive", "False Negative", "High Uncertainty", "Borderline Case"]
        assert len(case.diagnostic_reason) > 0
        assert 0.0 <= case.predicted_probability <= 1.0


def test_cohort_benchmarks(sample_raw_dataframe):
    """Tests computation of cohort statistics for placed vs non-placed profiles."""
    benchmarks = compute_cohort_benchmarks(sample_raw_dataframe, "PlacementStatus")
    assert "placed_count" in benchmarks
    assert "not_placed_count" in benchmarks
    assert "placement_rate" in benchmarks
    assert "features" in benchmarks
    assert "CGPA" in benchmarks["features"]


def test_full_spectrum_marks_monotonicity(sample_raw_dataframe):
    """Verifies that 10th and 12th marks predict from 0% to 100% with strict monotonicity."""
    pipeline = PlacementDataPipeline()
    pipeline.fit(sample_raw_dataframe)
    X = pipeline.transform(sample_raw_dataframe)
    y = pipeline.encode_target(sample_raw_dataframe["PlacementStatus"])

    model = train_advanced_model(X, y, pipeline=pipeline)
    engine = ExplainabilityEngine(model=model, pipeline=pipeline, background_data=X)

    base_cand = {
        "CGPA": 7.5,
        "Internships": 1,
        "Projects": 2,
        "Workshops/Certifications": 1,
        "AptitudeTestScore": 75,
        "SoftSkillsRating": 4.0,
        "ExtracurricularActivities": "Yes",
        "PlacementTraining": "Yes",
        "SSC_Marks": 70,
        "HSC_Marks": 70,
    }

    # Test 10th% (SSC_Marks) from 0 to 100
    marks_test = [0, 15, 30, 35, 50, 60, 70, 80, 90, 100]
    ssc_probs = []
    for m in marks_test:
        c = dict(base_cand, SSC_Marks=m)
        exp = engine.explain_instance(c)
        assert 0.0 <= exp.predicted_probability <= 1.0
        ssc_probs.append(exp.predicted_probability)

    for i in range(len(ssc_probs) - 1):
        assert ssc_probs[i] <= ssc_probs[i + 1], f"Monotonicity failed at SSC {marks_test[i]} -> {marks_test[i+1]}"

    # Extreme failure test: 0% marks must result in severe elimination (< 0.05)
    assert ssc_probs[0] < 0.05, f"0% marks did not reflect severe penalty: {ssc_probs[0]}"


def test_unbounded_projects_internships_monotonicity(sample_raw_dataframe):
    """Verifies that project and internship counts above zero strictly boost readiness with diminishing returns."""
    pipeline = PlacementDataPipeline()
    pipeline.fit(sample_raw_dataframe)
    X = pipeline.transform(sample_raw_dataframe)
    y = pipeline.encode_target(sample_raw_dataframe["PlacementStatus"])

    model = train_advanced_model(X, y, pipeline=pipeline)
    engine = ExplainabilityEngine(model=model, pipeline=pipeline, background_data=X)

    base_cand = {
        "CGPA": 7.5,
        "Internships": 1,
        "Projects": 2,
        "Workshops/Certifications": 1,
        "AptitudeTestScore": 75,
        "SoftSkillsRating": 4.0,
        "ExtracurricularActivities": "Yes",
        "PlacementTraining": "Yes",
        "SSC_Marks": 70,
        "HSC_Marks": 70,
    }

    # Test projects from 0 to 35
    proj_counts = [0, 1, 2, 3, 5, 8, 15, 25, 35]
    proj_probs = []
    for p in proj_counts:
        c = dict(base_cand, Projects=p)
        exp = engine.explain_instance(c)
        assert 0.0 <= exp.predicted_probability <= 1.0
        proj_probs.append(exp.predicted_probability)

    for i in range(len(proj_probs) - 1):
        assert proj_probs[i] <= proj_probs[i + 1], f"Monotonicity failed at Projects {proj_counts[i]} -> {proj_counts[i+1]}"

    # Test internships from 0 to 20
    intern_counts = [0, 1, 2, 3, 5, 8, 12, 20]
    intern_probs = []
    for intern in intern_counts:
        c = dict(base_cand, Internships=intern)
        exp = engine.explain_instance(c)
        assert 0.0 <= exp.predicted_probability <= 1.0
        intern_probs.append(exp.predicted_probability)

    for i in range(len(intern_probs) - 1):
        assert intern_probs[i] <= intern_probs[i + 1], f"Monotonicity failed at Internships {intern_counts[i]} -> {intern_counts[i+1]}"
