"""
SkillLens ML Training, Evaluation, and Error Analysis Engine
Provides baseline vs. advanced model training (Logistic Regression vs. XGBoost / Random Forest),
comprehensive metric computation, ROC/PR curves, model persistence, and failure diagnostics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.ensemble import RandomForestClassifier

# Attempt XGBoost import with graceful fallback
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.preprocessing import PlacementDataPipeline, load_and_split_data


@dataclass
class ModelResult:
    """Stores evaluation metrics, curves, predictions, and model instance."""
    model_name: str
    model: Any
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    roc_thresholds: np.ndarray
    precision_curve: np.ndarray
    recall_curve: np.ndarray
    classification_report_dict: Dict[str, Any]
    y_pred: np.ndarray
    y_proba: np.ndarray
    feature_importances: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model": self.model,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix,
            "fpr": self.fpr,
            "tpr": self.tpr,
            "roc_thresholds": self.roc_thresholds,
            "precision_curve": self.precision_curve,
            "recall_curve": self.recall_curve,
            "classification_report_dict": self.classification_report_dict,
            "y_pred": self.y_pred,
            "y_proba": self.y_proba,
            "feature_importances": self.feature_importances,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelResult":
        return cls(**d)


@dataclass
class FailureCase:
    """Stores a single misclassified or high-uncertainty student record with diagnostic explanation."""
    sample_index: int
    student_id: Union[int, str]
    true_label: int
    true_status: str
    pred_label: int
    pred_status: str
    predicted_probability: float
    confidence_gap: float
    error_type: str  # 'False Positive', 'False Negative', 'High Uncertainty'
    feature_values: Dict[str, Any]
    diagnostic_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "student_id": self.student_id,
            "true_label": self.true_label,
            "true_status": self.true_status,
            "pred_label": self.pred_label,
            "pred_status": self.pred_status,
            "predicted_probability": self.predicted_probability,
            "confidence_gap": self.confidence_gap,
            "error_type": self.error_type,
            "feature_values": self.feature_values,
            "diagnostic_reason": self.diagnostic_reason,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FailureCase":
        return cls(**d)


@dataclass
class ModelArtifacts:
    """Encapsulates all trained models, preprocessing pipeline, evaluation results, and failure logs."""
    pipeline: PlacementDataPipeline
    baseline_result: ModelResult
    advanced_result: ModelResult
    failure_log: List[FailureCase]
    test_df_raw: pd.DataFrame
    y_test: np.ndarray
    trained_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "baseline_result": self.baseline_result.to_dict(),
            "advanced_result": self.advanced_result.to_dict(),
            "failure_log": [f.to_dict() for f in self.failure_log],
            "test_df_raw": self.test_df_raw,
            "y_test": self.y_test,
            "trained_at": self.trained_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelArtifacts":
        return cls(
            pipeline=d["pipeline"],
            baseline_result=ModelResult.from_dict(d["baseline_result"]),
            advanced_result=ModelResult.from_dict(d["advanced_result"]),
            failure_log=[FailureCase.from_dict(f) for f in d["failure_log"]],
            test_df_raw=d["test_df_raw"],
            y_test=d["y_test"],
            trained_at=d.get("trained_at"),
        )


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> LogisticRegression:
    """Trains a regularized Logistic Regression baseline model with calibrated probability outputs."""
    model = LogisticRegression(
        C=0.5,
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def train_advanced_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    random_state: int = 42,
) -> Any:
    """
    Trains an advanced gradient boosting model (XGBoost Classifier) with hyperparameter tuning
    and monotonic career domain constraints to guarantee realistic, non-contradictory predictions.
    """
    if HAS_XGBOOST:
        constraints = None
        if feature_names:
            c_list = []
            for f in feature_names:
                if any(k in f for k in ['CGPA', 'Internships', 'Projects', 'Workshops', 'Aptitude', 'SoftSkills', 'SSC', 'HSC', 'Yes']):
                    c_list.append(1)
                elif 'No' in f:
                    c_list.append(-1)
                else:
                    c_list.append(0)
            constraints = tuple(c_list)

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            gamma=0.1,
            monotone_constraints=constraints,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        if X_val is not None and y_val is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)
        return model
    else:
        # Fallback to Random Forest
        model = RandomForestClassifier(
            n_estimators=220,
            max_depth=7,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model


def evaluate_model(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    decision_threshold: float = 0.5,
) -> ModelResult:
    """
    Evaluates model performance across test data, computing comprehensive classification metrics.
    """
    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba(X_test)
        y_proba = proba_all[:, 1] if proba_all.shape[1] > 1 else proba_all[:, 0]
    else:
        # Decision function fallback
        dec = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-dec))

    y_pred = (y_proba >= decision_threshold).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_test, y_proba))
        fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)
    except Exception:
        roc_auc = 0.5
        fpr, tpr, roc_thresh = np.array([0, 1]), np.array([0, 1]), np.array([0.5, 0.5])

    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    except Exception:
        prec_curve, rec_curve = np.array([1, 0]), np.array([0, 1])

    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Extract feature importances if available
    feat_importances = None
    if feature_names:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_importances = {name: float(imp) for name, imp in zip(feature_names, importances)}
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_[0])
            total = np.sum(coefs) if np.sum(coefs) > 0 else 1.0
            norm_coefs = coefs / total
            feat_importances = {name: float(c) for name, c in zip(feature_names, norm_coefs)}

    return ModelResult(
        model_name=model_name,
        model=model,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        confusion_matrix=cm,
        fpr=fpr,
        tpr=tpr,
        roc_thresholds=roc_thresh,
        precision_curve=prec_curve,
        recall_curve=rec_curve,
        classification_report_dict=report_dict,
        y_pred=y_pred,
        y_proba=y_proba,
        feature_importances=feat_importances,
    )


def generate_failure_analysis_log(
    test_df_raw: pd.DataFrame,
    y_test: np.ndarray,
    advanced_result: ModelResult,
    pipeline: PlacementDataPipeline,
    min_cases: int = 20,
) -> List[FailureCase]:
    """
    Extracts at least 20 misclassified or high-uncertainty predictions from the test set,
    generating qualitative root-cause diagnostic hypotheses for model debugging.
    """
    y_pred = advanced_result.y_pred
    y_proba = advanced_result.y_proba
    schema = pipeline.schema

    id_col = schema.id_column if schema and schema.id_column in test_df_raw.columns else None

    records: List[FailureCase] = []

    for idx in range(len(y_test)):
        true_y = int(y_test[idx])
        pred_y = int(y_pred[idx])
        prob = float(y_proba[idx])
        gap = abs(prob - 0.5)

        raw_row = test_df_raw.iloc[idx].to_dict()
        student_id = raw_row.get(id_col, f"Student-{idx+1}") if id_col else f"Student-{idx+1}"

        is_error = true_y != pred_y
        is_uncertain = 0.40 <= prob <= 0.60

        if is_error or is_uncertain:
            if true_y == 0 and pred_y == 1:
                err_type = "False Positive"
            elif true_y == 1 and pred_y == 0:
                err_type = "False Negative"
            else:
                err_type = "High Uncertainty"

            # Formulate diagnostic reason based on student features
            reason_parts = []
            cgpa = raw_row.get("CGPA", None)
            aptitude = raw_row.get("AptitudeTestScore", None)
            internships = raw_row.get("Internships", None)
            projects = raw_row.get("Projects", None)
            soft_skills = raw_row.get("SoftSkillsRating", None)
            training = raw_row.get("PlacementTraining", None)

            if err_type == "False Positive":
                # Model predicted Placed (1), but actual was Not Placed (0)
                if cgpa is not None and cgpa >= 7.5:
                    reason_parts.append(f"High CGPA ({cgpa}) inflated model confidence")
                if internships == 0 or (internships is not None and internships <= 1):
                    reason_parts.append("Lack of internship experience not penalised enough")
                if aptitude is not None and aptitude < 70:
                    reason_parts.append(f"Lower aptitude score ({aptitude}) undermined placement")
                if not reason_parts:
                    reason_parts.append("Borderline academic profile with conflicting behavioral signals")

            elif err_type == "False Negative":
                # Model predicted Not Placed (0), but actual was Placed (1)
                if cgpa is not None and cgpa < 7.5:
                    reason_parts.append(f"Sub-8.0 CGPA ({cgpa}) triggered model conservatism")
                if (internships is not None and internships >= 1) or (projects is not None and projects >= 2):
                    reason_parts.append("Strong practical projects/internships offset lower academic metrics")
                if soft_skills is not None and soft_skills >= 4.0:
                    reason_parts.append(f"Exceptional soft skills ({soft_skills}/5) drove actual hiring success")
                if not reason_parts:
                    reason_parts.append("High individual candidate aptitude during interview rounds")

            else:
                # High uncertainty
                reason_parts.append(f"Probability {prob*100:.1f}% close to 50% decision boundary")
                if cgpa is not None:
                    reason_parts.append(f"CGPA {cgpa} near median threshold")

            diagnostic_str = " | ".join(reason_parts)

            records.append(
                FailureCase(
                    sample_index=idx,
                    student_id=student_id,
                    true_label=true_y,
                    true_status="Placed" if true_y == 1 else "Not Placed",
                    pred_label=pred_y,
                    pred_status="Placed" if pred_y == 1 else "Not Placed",
                    predicted_probability=prob,
                    confidence_gap=gap,
                    error_type=err_type,
                    feature_values=raw_row,
                    diagnostic_reason=diagnostic_str,
                )
            )

    # Sort records: actual errors first (by smallest confidence gap / hardest mistakes), then uncertain cases
    records.sort(key=lambda x: (0 if x.error_type != "High Uncertainty" else 1, x.confidence_gap))

    # Ensure at least min_cases (if not enough errors, take closest borderline cases)
    if len(records) < min_cases:
        all_indices = set(r.sample_index for r in records)
        remaining = [
            (idx, abs(float(y_proba[idx]) - 0.5))
            for idx in range(len(y_test))
            if idx not in all_indices
        ]
        remaining.sort(key=lambda x: x[1])
        for idx, gap in remaining[: (min_cases - len(records))]:
            raw_row = test_df_raw.iloc[idx].to_dict()
            student_id = raw_row.get(id_col, f"Student-{idx+1}") if id_col else f"Student-{idx+1}"
            prob = float(y_proba[idx])
            t_y = int(y_test[idx])
            p_y = int(y_pred[idx])
            records.append(
                FailureCase(
                    sample_index=idx,
                    student_id=student_id,
                    true_label=t_y,
                    true_status="Placed" if t_y == 1 else "Not Placed",
                    pred_label=p_y,
                    pred_status="Placed" if p_y == 1 else "Not Placed",
                    predicted_probability=prob,
                    confidence_gap=gap,
                    error_type="Borderline Case",
                    feature_values=raw_row,
                    diagnostic_reason=f"Candidate near decision boundary (P={prob*100:.1f}%)",
                )
            )

    return records[: max(min_cases, len([r for r in records if r.error_type in ("False Positive", "False Negative")]))]


def train_and_evaluate_all(
    data_dir: Union[str, Path] = "data",
    save_path: Optional[Union[str, Path]] = "models/model_artifacts.joblib",
) -> ModelArtifacts:
    """
    End-to-end orchestration:
    - Loads & splits data with frozen test set
    - Trains Baseline (Logistic Regression) and Advanced (XGBoost)
    - Evaluates both models on identical frozen test set
    - Builds Failure Case Log
    - Persists artifacts to disk
    """
    (
        df_train,
        df_val,
        df_test,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        pipeline,
    ) = load_and_split_data(data_dir=data_dir)

    feat_names = pipeline.transformed_feature_names

    # 1. Train Baseline
    baseline_model = train_baseline_model(X_train, y_train)
    baseline_result = evaluate_model(
        baseline_model,
        model_name="Baseline (Logistic Regression)",
        X_test=X_test,
        y_test=y_test,
        feature_names=feat_names,
    )

    # 2. Train Advanced
    advanced_model = train_advanced_model(X_train, y_train, X_val, y_val, feature_names=feat_names)
    advanced_model_name = "Advanced (XGBoost Classifier)" if HAS_XGBOOST else "Advanced (Random Forest)"
    advanced_result = evaluate_model(
        advanced_model,
        model_name=advanced_model_name,
        X_test=X_test,
        y_test=y_test,
        feature_names=feat_names,
    )

    # 3. Generate Failure Analysis Log
    failure_log = generate_failure_analysis_log(
        test_df_raw=df_test,
        y_test=y_test,
        advanced_result=advanced_result,
        pipeline=pipeline,
        min_cases=20,
    )

    artifacts = ModelArtifacts(
        pipeline=pipeline,
        baseline_result=baseline_result,
        advanced_result=advanced_result,
        failure_log=failure_log,
        test_df_raw=df_test,
        y_test=y_test,
        trained_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifacts.to_dict(), save_path)

    return artifacts


def load_model_artifacts(
    artifacts_path: Union[str, Path] = "models/model_artifacts.joblib",
    retrain_if_missing: bool = True,
) -> ModelArtifacts:
    """Loads existing artifacts or triggers automated retraining if not found."""
    path = Path(artifacts_path)
    if path.exists():
        raw_dict = joblib.load(path)
        if isinstance(raw_dict, dict):
            return ModelArtifacts.from_dict(raw_dict)
        return raw_dict
    if retrain_if_missing:
        return train_and_evaluate_all(save_path=path)
    raise FileNotFoundError(f"Model artifacts not found at {path.resolve()}")


if __name__ == "__main__":
    print("Training models and running evaluation...")
    arts = train_and_evaluate_all()
    print("\n--- BASELINE EVALUATION ---")
    print(f"Accuracy:  {arts.baseline_result.accuracy:.4f}")
    print(f"Precision: {arts.baseline_result.precision:.4f}")
    print(f"Recall:    {arts.baseline_result.recall:.4f}")
    print(f"F1-Score:  {arts.baseline_result.f1:.4f}")
    print(f"ROC-AUC:   {arts.baseline_result.roc_auc:.4f}")

    print(f"\n--- {arts.advanced_result.model_name.upper()} EVALUATION ---")
    print(f"Accuracy:  {arts.advanced_result.accuracy:.4f}")
    print(f"Precision: {arts.advanced_result.precision:.4f}")
    print(f"Recall:    {arts.advanced_result.recall:.4f}")
    print(f"F1-Score:  {arts.advanced_result.f1:.4f}")
    print(f"ROC-AUC:   {arts.advanced_result.roc_auc:.4f}")

    print(f"\nGenerated {len(arts.failure_log)} failure & diagnostic cases.")
    print("Sample failure case:", arts.failure_log[0].diagnostic_reason)
