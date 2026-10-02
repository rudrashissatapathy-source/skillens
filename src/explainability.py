"""
SkillLens Explainability Engine
Computes Global Feature Importance and Local SHAP / Impact Values for individual candidate predictions.
Generates human-interpretable factor breakdowns (+/- drivers) and actionable skill recommendations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.preprocessing import PlacementDataPipeline


@dataclass
class FactorImpact:
    """Represents the positive or negative contribution of a single feature to a candidate's readiness."""
    feature: str
    feature_display_name: str
    value: Any
    impact_score: float  # SHAP value or contribution
    direction: str       # 'positive' (boosts placement) or 'negative' (reduces placement)
    actionable_tip: str  # Practical advice to improve or leverage this factor


@dataclass
class InstanceExplanation:
    """Local explanation container for a single student's placement prediction."""
    predicted_probability: float
    readiness_tier: str
    base_value: float
    factors: List[FactorImpact]
    positive_factors: List[FactorImpact]
    negative_factors: List[FactorImpact]
    narrative_summary: str
    top_recommendations: List[str]


class ExplainabilityEngine:
    """
    Computes global importance metrics and granular local explanations
    using SHAP TreeExplainer / LinearExplainer with robust fallbacks.
    """

    FRIENDLY_FEATURE_NAMES = {
        "CGPA": "Academic CGPA",
        "Internships": "Completed Internships",
        "Projects": "Technical Projects",
        "Workshops/Certifications": "Certifications & Workshops",
        "AptitudeTestScore": "Aptitude Assessment Score",
        "SoftSkillsRating": "Soft Skills & Communication (1-5)",
        "ExtracurricularActivities": "Extracurricular Involvement",
        "PlacementTraining": "Placement Preparation Training",
        "SSC_Marks": "10th %",
        "HSC_Marks": "12th %",
    }

    ACTIONABLE_TIPS = {
        "CGPA": "Maintain consistent semester exam scores; academic cutoff filters are strictly enforced by tier-1 recruiters.",
        "Internships": "Complete an industry internship or virtual work experience to validate practical domain execution.",
        "Projects": "Deploy full-stack or domain-specific capstone projects with public GitHub repositories & live demos.",
        "Workshops/Certifications": "Earn accredited professional certifications (e.g., AWS, Azure, GCP, or specialized developer certs).",
        "AptitudeTestScore": "Practice quantitative aptitude, logical reasoning, and data interpretation with timed mock assessments.",
        "SoftSkillsRating": "Participate in mock HR interviews, group discussions, and technical presentation workshops.",
        "ExtracurricularActivities": "Engage in student clubs, hackathons, or leadership roles to demonstrate team dynamics.",
        "PlacementTraining": "Enroll in campus placement training programs and technical interview bootcamps.",
        "SSC_Marks": "Solid foundational score; highlight problem-solving progression across your engineering tenure.",
        "HSC_Marks": "Strong analytical foundation; showcase mathematical rigor in project documentation.",
    }

    def __init__(
        self,
        model: Any,
        pipeline: PlacementDataPipeline,
        background_data: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.pipeline = pipeline
        self.background_data = background_data
        self.feature_names = pipeline.transformed_feature_names
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self) -> None:
        """Initializes the appropriate SHAP explainer based on model architecture."""
        if not HAS_SHAP:
            return

        try:
            # Tree-based models (XGBoost, RandomForest)
            if hasattr(self.model, "feature_importances_") or "XGB" in str(type(self.model)) or "Forest" in str(type(self.model)):
                self.explainer = shap.TreeExplainer(self.model)
            elif hasattr(self.model, "coef_"):
                # Linear models
                if self.background_data is not None and len(self.background_data) > 0:
                    bg_sample = self.background_data[: min(100, len(self.background_data))]
                    self.explainer = shap.LinearExplainer(self.model, bg_sample)
                else:
                    self.explainer = shap.Explainer(self.model)
        except Exception:
            # Fallback to general Explainer
            try:
                if self.background_data is not None:
                    bg_sample = self.background_data[: min(50, len(self.background_data))]
                    self.explainer = shap.Explainer(self.model.predict_proba, bg_sample)
            except Exception:
                self.explainer = None

    def get_global_importance(self, X_sample: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Computes global feature importances, returning a formatted DataFrame.
        """
        if self.explainer is not None and X_sample is not None and len(X_sample) > 0:
            try:
                sample_limit = min(300, len(X_sample))
                shap_vals = self.explainer.shap_values(X_sample[:sample_limit])
                if isinstance(shap_vals, list):
                    shap_matrix = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                elif hasattr(shap_vals, "values"):
                    shap_matrix = shap_vals.values
                else:
                    shap_matrix = shap_vals

                # Handle multi-class shape (n_samples, n_features, n_classes)
                if len(shap_matrix.shape) == 3:
                    shap_matrix = shap_matrix[:, :, 1]

                mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
                df_imp = pd.DataFrame({
                    "feature": self.feature_names,
                    "importance": mean_abs_shap,
                })
                df_imp["display_name"] = df_imp["feature"].map(
                    lambda f: self._get_display_name(f)
                )
                df_imp = df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)
                total = df_imp["importance"].sum()
                df_imp["relative_pct"] = (df_imp["importance"] / (total if total > 0 else 1.0)) * 100
                return df_imp
            except Exception:
                pass

        # Native feature importances fallback
        if hasattr(self.model, "feature_importances_"):
            raw_imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            coefs = np.abs(self.model.coef_[0])
            raw_imp = coefs
        else:
            raw_imp = np.ones(len(self.feature_names)) / len(self.feature_names)

        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": raw_imp,
        })
        df_imp["display_name"] = df_imp["feature"].map(lambda f: self._get_display_name(f))
        df_imp = df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)
        total = df_imp["importance"].sum()
        df_imp["relative_pct"] = (df_imp["importance"] / (total if total > 0 else 1.0)) * 100
        return df_imp

    def explain_instance(self, raw_input_dict: Dict[str, Any]) -> InstanceExplanation:
        """
        Calculates local SHAP impact for a single candidate, categorizing positive and negative drivers.
        """
        # 1. Transform input
        X_trans = self.pipeline.transform_single(raw_input_dict)
        
        # 2. Get prediction probability
        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(X_trans)[0, 1])
        else:
            dec = float(self.model.decision_function(X_trans)[0])
            proba = float(1 / (1 + np.exp(-dec)))

        # Assign readiness tier
        if proba >= 0.75:
            tier = "High Readiness (Placement Ready)"
        elif proba >= 0.45:
            tier = "Moderate Readiness (Targeted Skill Refinement Needed)"
        else:
            tier = "Low Readiness (High Academic & Practical Intervention Needed)"

        # 3. Compute local SHAP / feature contributions
        feature_impacts: List[FactorImpact] = []
        base_val = 0.5

        if self.explainer is not None:
            try:
                shap_obj = self.explainer(X_trans)
                if hasattr(shap_obj, "values"):
                    vals = shap_obj.values[0]
                    # If 2D (features, classes)
                    if len(vals.shape) == 2:
                        vals = vals[:, 1]
                    base_val = float(shap_obj.base_values[0]) if not isinstance(shap_obj.base_values[0], (list, np.ndarray)) else float(shap_obj.base_values[0][1])
                elif isinstance(shap_obj, list):
                    vals = shap_obj[1][0] if len(shap_obj) > 1 else shap_obj[0][0]
                else:
                    vals = np.array(shap_obj)[0]

                # Map transformed feature values back to raw feature concepts
                feature_impacts = self._aggregate_shap_to_raw_features(raw_input_dict, vals)
            except Exception:
                feature_impacts = self._heuristic_feature_contributions(raw_input_dict, X_trans)
        else:
            feature_impacts = self._heuristic_feature_contributions(raw_input_dict, X_trans)

        # Sort factors
        positive_factors = [f for f in feature_impacts if f.impact_score > 0]
        negative_factors = [f for f in feature_impacts if f.impact_score < 0]

        positive_factors.sort(key=lambda x: abs(x.impact_score), reverse=True)
        negative_factors.sort(key=lambda x: abs(x.impact_score), reverse=True)

        # Generate narrative summary
        narrative = self._generate_narrative(proba, tier, positive_factors, negative_factors)
        recommendations = self._generate_recommendations(negative_factors, raw_input_dict)

        return InstanceExplanation(
            predicted_probability=proba,
            readiness_tier=tier,
            base_value=base_val,
            factors=feature_impacts,
            positive_factors=positive_factors,
            negative_factors=negative_factors,
            narrative_summary=narrative,
            top_recommendations=recommendations,
        )

    def _aggregate_shap_to_raw_features(
        self,
        raw_input: Dict[str, Any],
        transformed_shap_vals: np.ndarray,
    ) -> List[FactorImpact]:
        """
        Aggregates one-hot encoded SHAP values back to their original high-level domain feature.
        """
        raw_feature_impacts: Dict[str, float] = {}

        for feat_name, val in zip(self.feature_names, transformed_shap_vals):
            # Check if this transformed feature originates from a categorical column
            matched_raw = None
            if self.pipeline.schema:
                for cat_col in self.pipeline.schema.categorical_features:
                    if feat_name.startswith(cat_col):
                        matched_raw = cat_col
                        break
                if matched_raw is None:
                    for num_col in self.pipeline.schema.numeric_features:
                        if feat_name == num_col:
                            matched_raw = num_col
                            break

            if matched_raw is None:
                matched_raw = feat_name

            raw_feature_impacts[matched_raw] = raw_feature_impacts.get(matched_raw, 0.0) + float(val)

        impacts: List[FactorImpact] = []
        for raw_col, impact_val in raw_feature_impacts.items():
            disp_name = self.FRIENDLY_FEATURE_NAMES.get(raw_col, raw_col)
            val = raw_input.get(raw_col, "N/A")
            direction = "positive" if impact_val >= 0 else "negative"
            tip = self.ACTIONABLE_TIPS.get(raw_col, "Continue optimizing this core competency.")

            impacts.append(
                FactorImpact(
                    feature=raw_col,
                    feature_display_name=disp_name,
                    value=val,
                    impact_score=float(impact_val),
                    direction=direction,
                    actionable_tip=tip,
                )
            )

        impacts.sort(key=lambda x: abs(x.impact_score), reverse=True)
        return impacts

    def _heuristic_feature_contributions(
        self,
        raw_input: Dict[str, Any],
        X_trans: np.ndarray,
    ) -> List[FactorImpact]:
        """Fallback explanation heuristic when SHAP C-extensions are unavailable."""
        impacts: List[FactorImpact] = []
        schema = self.pipeline.schema

        for col, stats in (schema.feature_statistics.items() if schema else {}):
            val = raw_input.get(col, stats.get("default", 0))
            disp_name = self.FRIENDLY_FEATURE_NAMES.get(col, col)
            tip = self.ACTIONABLE_TIPS.get(col, "Focus on skill improvement.")

            if stats.get("type") == "numeric":
                mean = stats.get("mean", 50.0)
                std = stats.get("std", 1.0)
                z_score = (float(val) - mean) / (std if std > 0 else 1.0)
                impact_score = z_score * 0.15
            else:
                # Categorical
                str_val = str(val).strip().lower()
                if str_val in {"yes", "true", "1"}:
                    impact_score = 0.12
                else:
                    impact_score = -0.10

            impacts.append(
                FactorImpact(
                    feature=col,
                    feature_display_name=disp_name,
                    value=val,
                    impact_score=float(impact_score),
                    direction="positive" if impact_score >= 0 else "negative",
                    actionable_tip=tip,
                )
            )

        impacts.sort(key=lambda x: abs(x.impact_score), reverse=True)
        return impacts

    def _get_display_name(self, feat: str) -> str:
        """Returns clean user-friendly label for a feature name."""
        for orig, friendly in self.FRIENDLY_FEATURE_NAMES.items():
            if feat.startswith(orig):
                return friendly
        return feat.replace("_", " ").title()

    def _generate_narrative(
        self,
        proba: float,
        tier: str,
        positives: List[FactorImpact],
        negatives: List[FactorImpact],
    ) -> str:
        """Constructs an intelligent natural-language summary for the student's profile."""
        pct = proba * 100.0
        pos_names = [f"**{p.feature_display_name}** ({p.value})" for p in positives[:2]]
        neg_names = [f"**{n.feature_display_name}** ({n.value})" for n in negatives[:2]]

        narrative = f"The candidate has a **{pct:.1f}% Placement Readiness Score**, classifying them in the **{tier}** category. "

        if pos_names:
            narrative += f"Primary strengths elevating the candidate's score include {', '.join(pos_names)}. "
        if neg_names:
            narrative += f"Conversely, significant headwinds reducing the placement likelihood include {', '.join(neg_names)}. "
        else:
            narrative += "No severe skill deficits were identified across the core evaluation metrics."

        return narrative

    def _generate_recommendations(
        self,
        negatives: List[FactorImpact],
        raw_input: Dict[str, Any],
    ) -> List[str]:
        """Generates prioritized, tailored action steps for student skill improvement."""
        recs = []
        for neg in negatives[:3]:
            feat = neg.feature
            val = neg.value
            if feat == "CGPA":
                recs.append(f"🎯 **Academic Priority:** Current CGPA is {val}. Target scoring > 8.0 in upcoming semesters to satisfy recruiter shortlisting thresholds.")
            elif feat == "Internships":
                recs.append(f"💼 **Practical Experience:** Candidate currently has {val} internship(s). Pursue at least 1-2 industry internships or open-source fellowships.")
            elif feat == "Projects":
                recs.append(f"🚀 **Project Portfolio:** Build and deploy 2+ full-stack production-grade projects showcasing problem-solving and software design.")
            elif feat == "AptitudeTestScore":
                recs.append(f"🧠 **Aptitude Mastery:** Aptitude score is {val}. Take weekly mock speed tests in quantitative math and logical reasoning.")
            elif feat == "SoftSkillsRating":
                recs.append(f"🗣️ **Communication:** Soft skills rating is {val}/5.0. Participate in Toastmasters, presentation workshops, or peer interview prep.")
            elif feat == "PlacementTraining":
                recs.append("📚 **Bootcamp Enrollment:** Candidate has not undergone structured placement training. Enrolling in mock interview cycles is strongly recommended.")
            elif feat == "Workshops/Certifications":
                recs.append("📜 **Certifications:** Earn 1-2 cloud or engineering industry certifications to substantiate practical skills.")
            else:
                recs.append(f"⚡ **{neg.feature_display_name}:** {neg.actionable_tip}")

        if not recs:
            recs.append("🌟 **Maintain Momentum:** Profile is highly competitive. Focus on mock executive interviews and salary negotiation strategies.")

        return recs


def compute_cohort_benchmarks(df_raw: pd.DataFrame, target_col: str = "PlacementStatus") -> Dict[str, Any]:
    """
    Computes statistical benchmarks comparing Placed vs. Not Placed students
    across all numeric and categorical dimensions for cohort comparison.
    """
    if target_col not in df_raw.columns:
        return {}

    placed_mask = df_raw[target_col].astype(str).str.lower().isin(["placed", "1", "yes", "true"])
    df_placed = df_raw[placed_mask]
    df_not_placed = df_raw[~placed_mask]

    benchmarks = {
        "placed_count": len(df_placed),
        "not_placed_count": len(df_not_placed),
        "placement_rate": len(df_placed) / len(df_raw) if len(df_raw) > 0 else 0.0,
        "features": {},
    }

    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if "StudentID" in numeric_cols:
        numeric_cols.remove("StudentID")

    for col in numeric_cols:
        p_mean = float(df_placed[col].mean()) if not df_placed.empty else 0.0
        np_mean = float(df_not_placed[col].mean()) if not df_not_placed.empty else 0.0
        overall_mean = float(df_raw[col].mean()) if not df_raw.empty else 0.0
        benchmarks["features"][col] = {
            "placed_mean": round(p_mean, 2),
            "not_placed_mean": round(np_mean, 2),
            "overall_mean": round(overall_mean, 2),
        }

    return benchmarks
