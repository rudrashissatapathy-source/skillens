"""
SkillLens Package
Automated Machine Learning pipeline, model training, SHAP explainability, and diagnostics.
"""

from src.explainability import ExplainabilityEngine, FactorImpact, InstanceExplanation
from src.model import (
    FailureCase,
    ModelArtifacts,
    ModelResult,
    PlacementReadinessClassifier,
    evaluate_model,
    load_model_artifacts,
    save_model_artifacts,
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

__version__ = "1.0.0"
__all__ = [
    "DatasetSchema",
    "ExplainabilityEngine",
    "FactorImpact",
    "FailureCase",
    "InstanceExplanation",
    "ModelArtifacts",
    "ModelResult",
    "PlacementDataPipeline",
    "PlacementReadinessClassifier",
    "SchemaParser",
    "evaluate_model",
    "load_and_split_data",
    "load_model_artifacts",
    "resolve_dataset_path",
    "save_model_artifacts",
    "train_advanced_model",
    "train_baseline_model",
]
