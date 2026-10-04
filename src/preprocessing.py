"""
SkillLens Data Preprocessing & Pipeline Module
Handles automated schema parsing, data validation, cleaning, imputing,
categorical encoding, feature scaling, target mapping, and train/val/test splitting.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os

# Ensure project root is in sys.path when executed directly as a script
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DatasetSchema:
    """Stores the parsed metadata and schema configuration of the dataset."""
    id_column: Optional[str] = None
    target_column: Optional[str] = None
    numeric_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    leakage_features: List[str] = field(default_factory=list)
    all_feature_names: List[str] = field(default_factory=list)
    target_mapping: Dict[Any, int] = field(default_factory=dict)
    transformed_feature_names: List[str] = field(default_factory=list)
    feature_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def resolve_dataset_path(data_dir: Union[str, Path] = "data") -> Path:
    """
    Locates the placement dataset CSV file within the data directory,
    handling common naming variants like .csv or .csv.csv.
    """
    base_dir = Path(data_dir)
    possible_paths = [
        base_dir / "kaggle_placement_data.csv",
        base_dir / "kaggle_placement_data.csv.csv",
        base_dir / "placement_data.csv",
        Path("kaggle_placement_data.csv"),
        Path("kaggle_placement_data.csv.csv"),
    ]

    for p in possible_paths:
        if p.exists() and p.is_file():
            return p.resolve()

    # Fallback search for any .csv file in data directory
    if base_dir.exists() and base_dir.is_dir():
        csv_files = list(base_dir.glob("*.csv*"))
        if csv_files:
            return csv_files[0].resolve()

    raise FileNotFoundError(
        f"Could not locate Kaggle placement dataset in {base_dir.resolve()}. "
        f"Checked: {[str(p) for p in possible_paths]}"
    )


class SchemaParser:
    """
    Automated schema detection engine that inspects raw tabular data,
    identifies IDs, target columns, numeric/categorical features, and potential target leakage.
    """

    KNOWN_ID_PATTERNS = {"id", "studentid", "student_id", "rollno", "roll_number", "index"}
    KNOWN_TARGET_PATTERNS = {"placementstatus", "status", "placement_status", "placed", "placement"}
    KNOWN_LEAKAGE_PATTERNS = {"salary", "package", "ctc", "company", "placed_company", "offer_letter"}

    @classmethod
    def parse(cls, df: pd.DataFrame, explicit_target: Optional[str] = None) -> DatasetSchema:
        """
        Inspects the dataframe and returns a DatasetSchema with categorized features.
        """
        schema = DatasetSchema()
        normalized_cols = {col: col.strip().lower().replace(" ", "").replace("_", "").replace("/", "") for col in df.columns}

        # 1. Identify ID Column
        for orig_col, norm_col in normalized_cols.items():
            if norm_col in cls.KNOWN_ID_PATTERNS or "studentid" in norm_col or orig_col.lower() == "id":
                schema.id_column = orig_col
                break

        # 2. Identify Target Column
        if explicit_target and explicit_target in df.columns:
            schema.target_column = explicit_target
        else:
            for orig_col, norm_col in normalized_cols.items():
                if norm_col in cls.KNOWN_TARGET_PATTERNS or "placementstatus" in norm_col:
                    schema.target_column = orig_col
                    break
            if schema.target_column is None:
                # Fallback: check columns with 'placed' or 'status'
                for orig_col, norm_col in normalized_cols.items():
                    if "status" in norm_col or "placed" in norm_col:
                        schema.target_column = orig_col
                        break

        # 3. Identify Target Leakage Columns
        for orig_col, norm_col in normalized_cols.items():
            if orig_col in (schema.id_column, schema.target_column):
                continue
            if any(leak in norm_col for leak in cls.KNOWN_LEAKAGE_PATTERNS):
                schema.leakage_features.append(orig_col)

        # 4. Identify Features (Numeric vs Categorical)
        excluded_cols = set([c for c in [schema.id_column, schema.target_column] if c is not None] + schema.leakage_features)
        feature_cols = [c for c in df.columns if c not in excluded_cols]
        schema.all_feature_names = feature_cols

        for col in feature_cols:
            series = df[col]
            # If dtype is numeric or convertible to numeric with high cardinality
            if pd.api.types.is_numeric_dtype(series):
                # If binary/low-cardinality boolean, check if categorical or numeric
                unique_vals = series.dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                    schema.categorical_features.append(col)
                else:
                    schema.numeric_features.append(col)
            else:
                # String / object / category
                schema.categorical_features.append(col)

        # 5. Build Target Mapping
        if schema.target_column:
            unique_targets = df[schema.target_column].dropna().unique()
            mapping = {}
            for val in unique_targets:
                str_val = str(val).strip().lower()
                if str_val in {"placed", "yes", "1", "true", "p", "pass", "selected"}:
                    mapping[val] = 1
                elif str_val in {"notplaced", "not placed", "no", "0", "false", "np", "fail", "rejected"}:
                    mapping[val] = 0
                else:
                    # Default assignment if unknown
                    mapping[val] = 1 if len(mapping) == 0 else 0
            schema.target_mapping = mapping

        # 6. Extract Feature Statistics (min, max, median, mean, categories) for UI validation
        for col in schema.numeric_features:
            series = pd.to_numeric(df[col], errors="coerce")
            raw_min = float(series.min()) if not series.empty else 0.0
            raw_max = float(series.max()) if not series.empty else 100.0

            # Domain-calibrated UI limits for full spectrum prediction
            if col in {"SSC_Marks", "HSC_Marks"}:
                feat_min = 0.0
                feat_max = 100.0
            elif col == "Internships":
                feat_min = 0.0
                feat_max = max(25.0, raw_max)
            elif col == "Projects":
                feat_min = 0.0
                feat_max = max(40.0, raw_max)
            elif col in {"Workshops/Certifications", "Certifications"}:
                feat_min = 0.0
                feat_max = max(30.0, raw_max)
            elif col == "CGPA":
                feat_min = min(0.0, raw_min)
                feat_max = 10.0
            elif col == "AptitudeTestScore":
                feat_min = 0.0
                feat_max = 100.0
            elif col == "SoftSkillsRating":
                feat_min = 1.0
                feat_max = 5.0
            else:
                feat_min = raw_min
                feat_max = raw_max

            schema.feature_statistics[col] = {
                "type": "numeric",
                "min": feat_min,
                "max": feat_max,
                "mean": float(series.mean()) if not series.empty else 50.0,
                "median": float(series.median()) if not series.empty else 50.0,
                "std": float(series.std()) if not series.empty else 1.0,
                "default": float(series.median()) if not series.empty else 50.0,
            }

        for col in schema.categorical_features:
            unique_cats = [str(x) for x in df[col].dropna().unique().tolist()]
            if not unique_cats:
                unique_cats = ["No", "Yes"]
            mode_val = str(df[col].mode().iloc[0]) if not df[col].empty else unique_cats[0]
            schema.feature_statistics[col] = {
                "type": "categorical",
                "categories": unique_cats,
                "default": mode_val,
            }

        return schema


class PlacementDataPipeline:
    """
    Full data preprocessing and transformation pipeline.
    Maintains scikit-learn transformers for numeric scaling and categorical one-hot encoding,
    ensuring zero data leakage and deterministic behavior across train, test, and inference.
    """

    def __init__(self, schema: Optional[DatasetSchema] = None):
        self.schema = schema
        self.preprocessor: Optional[ColumnTransformer] = None
        self.transformed_feature_names: List[str] = []
        self.is_fitted: bool = False

    def build_preprocessor(self) -> ColumnTransformer:
        """Constructs the ColumnTransformer pipeline based on schema features."""
        if not self.schema:
            raise ValueError("Schema must be initialized before building preprocessor.")

        transformers = []

        if self.schema.numeric_features:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipe, self.schema.numeric_features))

        if self.schema.categorical_features:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipe, self.schema.categorical_features))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return self.preprocessor

    def fit(self, X: pd.DataFrame) -> "PlacementDataPipeline":
        """Fits the imputation, scaling, and encoding transformers on training data."""
        if self.schema is None:
            self.schema = SchemaParser.parse(X)

        if self.preprocessor is None:
            self.build_preprocessor()

        # Clean X to ensure only expected features are used
        X_clean = self._extract_features(X)
        self.preprocessor.fit(X_clean)

        # Generate transformed feature names
        feature_names = []
        if self.schema.numeric_features:
            feature_names.extend(self.schema.numeric_features)
        if self.schema.categorical_features:
            try:
                cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["encoder"]
                encoded_cats = cat_encoder.get_feature_names_out(self.schema.categorical_features)
                feature_names.extend(list(encoded_cats))
            except Exception:
                for col in self.schema.categorical_features:
                    feature_names.append(f"{col}_encoded")

        self.transformed_feature_names = feature_names
        self.schema.transformed_feature_names = feature_names
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms feature data using fitted preprocessor."""
        if not self.is_fitted or self.preprocessor is None:
            raise RuntimeError("Pipeline must be fitted before calling transform.")

        X_clean = self._extract_features(X)
        return self.preprocessor.transform(X_clean)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fits and transforms feature data in one step."""
        return self.fit(X).transform(X)

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures that all schema features exist in the DataFrame with correct types and normalized column names."""
        if not self.schema:
            raise ValueError("Schema not found.")

        df_copy = df.copy()
        # Normalized lookup for incoming columns to handle case/spacing differences (e.g. 'cgpa', 'Aptitude Test Score')
        norm_incoming = {
            str(col).strip().lower().replace(" ", "").replace("_", "").replace("/", ""): col
            for col in df_copy.columns
        }

        for col in self.schema.all_feature_names:
            if col not in df_copy.columns:
                norm_key = str(col).strip().lower().replace(" ", "").replace("_", "").replace("/", "")
                if norm_key in norm_incoming:
                    df_copy[col] = df_copy[norm_incoming[norm_key]]
                else:
                    # Impute missing column with default from schema stats
                    stat = self.schema.feature_statistics.get(col, {})
                    df_copy[col] = stat.get("default", 0 if col in self.schema.numeric_features else "No")

        return df_copy[self.schema.all_feature_names]

    def encode_target(self, y_raw: pd.Series) -> np.ndarray:
        """Maps target column values to binary integer vector (0/1)."""
        if self.schema is None or not self.schema.target_mapping:
            # Fallback numeric coercion
            return pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int).values

        # Apply mapping
        mapped = y_raw.map(self.schema.target_mapping)
        # Fill any unmapped values with 0
        return mapped.fillna(0).astype(int).values

    def transform_single(self, input_dict: Dict[str, Any]) -> np.ndarray:
        """
        Preprocesses a single student's input dictionary for inference.
        Returns a 2D numpy array of shape (1, n_features).
        """
        df_single = pd.DataFrame([input_dict])
        return self.transform(df_single)

    def save(self, filepath: Union[str, Path]) -> None:
        """Persists the fitted pipeline object to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PlacementDataPipeline":
        """Loads a persisted pipeline object from disk."""
        return joblib.load(filepath)


def load_and_split_data(
    csv_path: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    frozen_test_path: Optional[Union[str, Path]] = "data/test_frozen.csv",
) -> Tuple[
    pd.DataFrame,  # train_raw
    pd.DataFrame,  # val_raw
    pd.DataFrame,  # test_raw
    np.ndarray,    # X_train
    np.ndarray,    # y_train
    np.ndarray,    # X_val
    np.ndarray,    # y_val
    np.ndarray,    # X_test
    np.ndarray,    # y_test
    PlacementDataPipeline,
]:
    """
    Comprehensive workflow function:
    1. Loads dataset from path
    2. Parses schema
    3. Splits data into train (70%), validation (15%), and frozen test (15%)
    4. Persists the frozen test set to disk
    5. Fits preprocessor ONLY on train split
    6. Returns raw split DataFrames, transformed numpy feature matrices, target vectors, and pipeline.
    """
    if csv_path is None:
        if data_dir is not None:
            csv_path = resolve_dataset_path(data_dir=data_dir)
        else:
            csv_path = resolve_dataset_path()
    else:
        csv_path = Path(csv_path)

    df_raw = pd.read_csv(csv_path)

    # 1. Parse schema
    schema = SchemaParser.parse(df_raw)
    if not schema.target_column:
        raise ValueError(f"Target column could not be detected in dataset at {csv_path}")

    # 2. Check if frozen test set already exists
    frozen_path = Path(frozen_test_path) if frozen_test_path else None
    
    # Split train+val vs test
    total_eval_size = test_size + val_size
    df_train_val, df_test = train_test_split(
        df_raw,
        test_size=test_size,
        random_state=random_state,
        stratify=df_raw[schema.target_column] if df_raw[schema.target_column].nunique() > 1 else None,
    )

    # If frozen test set requested, save test set
    if frozen_path:
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        if not frozen_path.exists():
            df_test.to_csv(frozen_path, index=False)

    # Split train vs val
    val_relative_size = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=df_train_val[schema.target_column] if df_train_val[schema.target_column].nunique() > 1 else None,
    )

    # 3. Fit Pipeline on Train only
    pipeline = PlacementDataPipeline(schema=schema)
    pipeline.fit(df_train)

    # 4. Transform feature matrices
    X_train = pipeline.transform(df_train)
    y_train = pipeline.encode_target(df_train[schema.target_column])

    X_val = pipeline.transform(df_val)
    y_val = pipeline.encode_target(df_val[schema.target_column])

    X_test = pipeline.transform(df_test)
    y_test = pipeline.encode_target(df_test[schema.target_column])

    return (
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
    )


if __name__ == "__main__":
    print("Testing preprocessing module...")
    path = resolve_dataset_path()
    print(f"Found dataset at: {path}")
    df = pd.read_csv(path)
    schema = SchemaParser.parse(df)
    print(f"Detected ID: {schema.id_column}")
    print(f"Detected Target: {schema.target_column}")
    print(f"Numeric Features ({len(schema.numeric_features)}): {schema.numeric_features}")
    print(f"Categorical Features ({len(schema.categorical_features)}): {schema.categorical_features}")
    print(f"Leakage Features: {schema.leakage_features}")
    print(f"Target Mapping: {schema.target_mapping}")

    (df_tr, df_v, df_te, X_tr, y_tr, X_v, y_v, X_te, y_te, pipe) = load_and_split_data()
    print(f"Train Shape: {X_tr.shape}, Val Shape: {X_v.shape}, Test Shape: {X_te.shape}")
    print(f"Transformed Features ({len(pipe.transformed_feature_names)}): {pipe.transformed_feature_names}")
    print("Preprocessing test completed successfully!")
