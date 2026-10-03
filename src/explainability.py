"""
SkillLens Explainability Engine
Computes Global Feature Importance and Local SHAP / Impact Values for individual candidate predictions.
Generates human-interpretable factor breakdowns (+/- drivers) and actionable skill recommendations.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path when executed directly as a script
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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
        # Auto-link pipeline to model if model is a PlacementReadinessClassifier
        if hasattr(self.model, "pipeline") and self.model.pipeline is None:
            self.model.pipeline = self.pipeline
        self.background_data = background_data
        self.feature_names = pipeline.transformed_feature_names
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self) -> None:
        """Initializes the appropriate SHAP explainer based on model architecture."""
        if not HAS_SHAP:
            return

        target_model = getattr(self.model, "base_model", self.model)

        try:
            # Tree-based models (XGBoost, RandomForest)
            if (
                hasattr(target_model, "feature_importances_")
                or "XGB" in str(type(target_model))
                or "Forest" in str(type(target_model))
            ):
                self.explainer = shap.TreeExplainer(target_model)
            elif hasattr(target_model, "coef_"):
                # Linear models
                if self.background_data is not None and len(self.background_data) > 0:
                    bg_sample = self.background_data[: min(100, len(self.background_data))]
                    self.explainer = shap.LinearExplainer(target_model, bg_sample)
                else:
                    self.explainer = shap.Explainer(target_model)
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

        # Assign realistic recruitment readiness tier (No sugarcoating)
        if proba >= 0.80:
            tier = "High Placement Probability (Top Tier Competitor)"
        elif proba >= 0.65:
            tier = "Viable Placement Prospect (Competitive with Gaps)"
        elif proba >= 0.50:
            tier = "Borderline / High Screening Elimination Risk"
        elif proba >= 0.35:
            tier = "Below Hiring Threshold (Probable Rejection)"
        else:
            tier = "Severe Placement Deficit (Immediate Overhaul Required)"

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

        # Enforce domain sanity: below-average or 0-values must NEVER be falsely labeled as strengths
        positive_factors = [
            f for f in feature_impacts
            if f.impact_score > 0 and self._is_genuine_strength(f.feature, f.value)
        ]

        # Liabilities: negative model impact OR metrics that fall below baseline recruiter cutoffs
        negative_factors = [
            f for f in feature_impacts
            if f.impact_score < 0 or self._is_genuine_liability(f.feature, f.value)
        ]

        positive_factors.sort(key=lambda x: abs(x.impact_score), reverse=True)
        negative_factors.sort(
            key=lambda x: (not self._is_genuine_liability(x.feature, x.value), -abs(x.impact_score))
        )

        # Generate narrative summary & actionable recommendations
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

    @staticmethod
    def _is_genuine_strength(feature: str, val: Any) -> bool:
        """
        Validates whether a feature value is genuinely an asset in hiring reality.
        Prevents counter-intuitive anomalies where SHAP background centering might assign
        a small positive contribution to a deficit (e.g. 0 internships or low CGPA).
        """
        try:
            num_val = float(val)
        except (ValueError, TypeError):
            s_val = str(val).strip().lower()
            return s_val in {"yes", "true", "1"}

        if feature == "Internships":
            return num_val >= 1
        elif feature == "Projects":
            return num_val >= 2
        elif feature == "CGPA":
            return num_val >= 7.5
        elif feature == "AptitudeTestScore":
            return num_val >= 78.0
        elif feature == "SoftSkillsRating":
            return num_val >= 4.0
        elif feature in {"SSC_Marks", "HSC_Marks"}:
            return num_val >= 75.0
        elif feature == "Workshops/Certifications":
            return num_val >= 1

        return True

    @staticmethod
    def _is_genuine_liability(feature: str, val: Any) -> bool:
        """
        Identifies metrics that fall below baseline industry screening cutoffs,
        ensuring they are highlighted as liabilities even if SHAP values are near zero.
        """
        try:
            num_val = float(val)
        except (ValueError, TypeError):
            s_val = str(val).strip().lower()
            return s_val in {"no", "false", "0"}

        if feature == "Internships":
            return num_val < 1
        elif feature == "Projects":
            return num_val < 2
        elif feature == "CGPA":
            return num_val < 7.2
        elif feature == "AptitudeTestScore":
            return num_val < 75.0
        elif feature == "SoftSkillsRating":
            return num_val < 4.0
        elif feature in {"SSC_Marks", "HSC_Marks"}:
            return num_val < 60.0
        elif feature == "Workshops/Certifications":
            return num_val < 1

        return False

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
        """Constructs an intelligent natural-language summary for the student's profile without sugarcoating."""
        pct = proba * 100.0
        pos_names = [f"**{p.feature_display_name}** ({p.value})" for p in positives[:2]]
        neg_names = [f"**{n.feature_display_name}** ({n.value})" for n in negatives[:3]]

        # Unvarnished recruiter assessment
        if proba >= 0.80:
            verdict = "🟢 **RECRUITER VERDICT: STRONG HIRE / SHORTLIST.** Profile exhibits strong technical readiness and satisfies premier campus recruitment cutoffs."
        elif proba >= 0.65:
            verdict = "🟡 **RECRUITER VERDICT: COMPETITIVE / SELECTIVE.** Viable candidate for general campus drives, but vulnerable in competitive technical shortlists."
        elif proba >= 0.50:
            verdict = "🟠 **RECRUITER VERDICT: BORDERLINE / SCREENING RISK.** High probability of elimination in initial technical or aptitude screening rounds."
        elif proba >= 0.35:
            verdict = "🔴 **RECRUITER VERDICT: BELOW HIRING THRESHOLD.** Significant deficiencies detected. Immediate remediation required before attending campus drives."
        else:
            verdict = "🚨 **RECRUITER VERDICT: IMMEDIATE REJECTION RISK.** Profile falls critically short of baseline industry standards across core dimensions."

        narrative = f"{verdict}<br><br>The candidate holds a **{pct:.1f}% Placement Readiness Score** ({tier}). "

        if pos_names:
            narrative += f"Demonstrated strengths include {', '.join(pos_names)}. "
        else:
            narrative += "No distinct competitive advantages were identified on the resume. "

        if neg_names:
            narrative += f"Primary disqualification risks and performance bottlenecks include {', '.join(neg_names)}."
        else:
            narrative += "Candidate meets or exceeds minimum screening thresholds across all tracked attributes."

        return narrative

    def _generate_recommendations(
        self,
        negatives: List[FactorImpact],
        raw_input: Dict[str, Any],
    ) -> List[str]:
        """Generates prioritized, tailored action steps for student skill improvement without sugarcoating."""
        recs = []

        # Check critical deal-breakers first
        cgpa = float(raw_input.get("CGPA", 7.0))
        interns = int(raw_input.get("Internships", 0))
        projects = int(raw_input.get("Projects", 0))
        aptitude = float(raw_input.get("AptitudeTestScore", 70))
        training = str(raw_input.get("PlacementTraining", "No")).strip().lower()

        if cgpa < 7.0:
            recs.append(f"🎯 **Academic Cutoff Alert (CGPA {cgpa:.1f}):** Most Tier-1 tech recruiters enforce a rigid 7.0 or 7.5 CGPA initial eligibility filter. Prioritize semester examinations immediately to cross 7.5.")
        if interns == 0:
            recs.append("💼 **Zero Industry Experience Deficit:** Having 0 internships is a severe resume disqualifier in campus placements. Secure at least one verified industry internship or open-source fellowship immediately.")
        if projects < 2:
            recs.append(f"🚀 **Insufficient Project Portfolio ({projects} project{'s' if projects != 1 else ''}):** Recruiters look for at least 2 deployed, full-stack or domain capstone projects with public GitHub repos to verify coding competency.")
        if aptitude < 75:
            recs.append(f"🧠 **Aptitude Screen Hazard ({aptitude:.0f}/90):** Candidate is likely to fail the first-round online assessment (OA). Daily timed practice on quantitative math, DI, and logical reasoning is mandatory.")
        if training in {"no", "0", "false"}:
            recs.append("📚 **Formal Placement Training Gap:** Candidate has not completed placement training. Enroll in mock interview bootcamps and algorithmic problem-solving sprints immediately.")

        # Fill remaining recs from other negatives if fewer than 4
        for neg in negatives:
            if len(recs) >= 4:
                break
            feat = neg.feature
            val = neg.value
            if feat == "SoftSkillsRating" and float(val) < 4.0:
                recs.append(f"🗣️ **Communication / HR Round Risk ({val}/5.0):** Low soft skills rating risks elimination in behavioral/managerial rounds. Engage in mock GDs and recorded technical presentation sessions.")
            elif feat == "ExtracurricularActivities" and str(val).lower() in {"no", "0"}:
                recs.append("🏆 **Extracurricular Deficit:** Lack of extracurricular involvement weakens resume impact; join tech clubs, hackathons, or leadership committees.")
            elif feat == "Workshops/Certifications" and int(val) == 0:
                recs.append("📜 **Missing Industry Credentials:** Complete 1-2 recognized cloud/engineering certifications (e.g. AWS, Azure, Google) to validate domain knowledge.")

        if not recs:
            recs.append("🌟 **Maintain Momentum:** Profile satisfies core placement cutoffs. Focus on high-level system design, behavioral storytelling, and targeted company research.")

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
