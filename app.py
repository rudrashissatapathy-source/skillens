"""
SkillLens: Machine Learning Web Application for Student Placement Readiness
Built with Streamlit, Scikit-Learn, XGBoost, SHAP, and Plotly.
"""

import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

# Custom Modules
from src.explainability import (
    ExplainabilityEngine,
    FactorImpact,
    compute_cohort_benchmarks,
)
from src.model import (
    ModelArtifacts,
    ModelResult,
    evaluate_model,
    load_model_artifacts,
    train_and_evaluate_all,
)
from src.preprocessing import (
    PlacementDataPipeline,
    SchemaParser,
    load_and_split_data,
    resolve_dataset_path,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SkillLens | Placement Readiness Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #312E81 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(90deg, #FFFFFF, #93C5FD, #C7D2FE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        margin-top: 0.25rem;
        color: #F8FAFC;
    }

    /* Tier Badges */
    .tier-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .tier-high {
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .tier-blue {
        background: rgba(59, 130, 246, 0.18);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .tier-med {
        background: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .tier-low {
        background: rgba(239, 68, 68, 0.18);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .tier-critical {
        background: rgba(220, 38, 38, 0.25);
        color: #FCA5A5;
        border: 1px solid rgba(220, 38, 38, 0.6);
    }

    /* Diagnostic Box */
    .diagnostic-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #6366F1;
        border-radius: 0 12px 12px 0;
        padding: 1.25rem 1.5rem;
        margin: 1.25rem 0;
        font-size: 0.98rem;
        line-height: 1.6;
        color: #E2E8F0;
    }

    /* Recommendation Card */
    .rec-card {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        font-size: 0.92rem;
    }

    /* Custom Streamlit Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CACHED DATA & MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing SkillLens Intelligence Pipeline...")
def get_cached_artifacts() -> ModelArtifacts:
    """Loads or trains ML artifacts with caching."""
    artifacts_path = Path("models/model_artifacts.joblib")
    return load_model_artifacts(artifacts_path=artifacts_path, retrain_if_missing=True)


@st.cache_data(show_spinner=False)
def get_dataset_benchmarks() -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads raw dataset and calculates statistical benchmarks."""
    path = resolve_dataset_path()
    df = pd.read_csv(path)
    benchmarks = compute_cohort_benchmarks(df, target_col="PlacementStatus")
    return df, benchmarks


@st.cache_resource(show_spinner=False)
def get_cached_explainer_engine(
    model_type: str,
    _artifacts: ModelArtifacts,
    _df_head: pd.DataFrame,
) -> ExplainabilityEngine:
    """Caches the ExplainabilityEngine instance so TreeExplainer is only created once."""
    active_model = (
        _artifacts.advanced_result.model
        if "XGBoost" in model_type
        else _artifacts.baseline_result.model
    )
    bg_data = _artifacts.pipeline.transform(_df_head)
    return ExplainabilityEngine(
        model=active_model,
        pipeline=_artifacts.pipeline,
        background_data=bg_data,
    )


@st.cache_data(show_spinner=False)
def get_cached_global_importance(
    model_type: str,
    _engine: ExplainabilityEngine,
    _X_sample: np.ndarray,
) -> pd.DataFrame:
    """Caches dataset-level global importance so SHAP tree traversals are not re-run on every slider change."""
    return _engine.get_global_importance(_X_sample)


def render_chart(fig, **kwargs):
    """Renders Plotly chart with width='stretch' to eliminate deprecation warnings in modern Streamlit."""
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kwargs)


def render_dataframe(df, **kwargs):
    """Renders DataFrame with width='stretch' to eliminate deprecation warnings in modern Streamlit."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


# Initialize models and data
try:
    artifacts: ModelArtifacts = get_cached_artifacts()
    df_raw, cohort_benchmarks = get_dataset_benchmarks()
    pipeline: PlacementDataPipeline = artifacts.pipeline
    schema = pipeline.schema
except Exception as e:
    st.error(f"Error initializing SkillLens application: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# SIDEBAR: CANDIDATE PROFILE INPUT & CONFIGURATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 **SkillLens Engine**")
    st.caption("AI-Powered Placement Readiness & Explainability")
    st.divider()

    # Model Selection & Threshold
    st.markdown("#### ⚙️ **Model Configuration**")
    selected_model_type = st.radio(
        "Active Classifier:",
        options=["Advanced (XGBoost)", "Baseline (Logistic Regression)"],
        index=0,
        help="Switch between Gradient Boosted Trees and Regularized Logistic Regression.",
    )

    threshold = st.slider(
        "Classification Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Adjust the probability cutoff for classifying a candidate as 'Placed'.",
    )

    active_model_result = (
        artifacts.advanced_result
        if "XGBoost" in selected_model_type
        else artifacts.baseline_result
    )
    active_model = active_model_result.model

    # Explainability Engine cached bound to active model
    explainer_engine = get_cached_explainer_engine(
        selected_model_type,
        artifacts,
        df_raw.head(200),
    )

    st.divider()

    # Preset Profiles Selector
    st.markdown("#### 👤 **Candidate Profile Input**")
    preset = st.selectbox(
        "Load Student Archetype / Preset:",
        options=[
            "Custom Input",
            "🌟 High Achiever (Top Tier)",
            "⚖️ Average Engineering Student",
            "⚠️ Borderline / At-Risk Candidate",
            "🚀 High Practical, Moderate Academic",
            "📚 High CGPA, Low Practical Experience",
        ],
        index=0,
    )

    # Preset values dictionary (calibrated to empirical cohort statistics)
    preset_values: Dict[str, Any] = {}
    if preset == "🌟 High Achiever (Top Tier)":
        preset_values = {
            "CGPA": 8.8,
            "Internships": 2,
            "Projects": 3,
            "Workshops/Certifications": 2,
            "AptitudeTestScore": 88,
            "SoftSkillsRating": 4.7,
            "ExtracurricularActivities": "Yes",
            "PlacementTraining": "Yes",
            "SSC_Marks": 85,
            "HSC_Marks": 88,
        }
    elif preset == "⚖️ Average Engineering Student":
        preset_values = {
            "CGPA": 7.7,
            "Internships": 1,
            "Projects": 2,
            "Workshops/Certifications": 1,
            "AptitudeTestScore": 80,
            "SoftSkillsRating": 4.4,
            "ExtracurricularActivities": "Yes",
            "PlacementTraining": "Yes",
            "SSC_Marks": 70,
            "HSC_Marks": 74,
        }
    elif preset == "⚠️ Borderline / At-Risk Candidate":
        preset_values = {
            "CGPA": 6.8,
            "Internships": 0,
            "Projects": 1,
            "Workshops/Certifications": 0,
            "AptitudeTestScore": 65,
            "SoftSkillsRating": 3.5,
            "ExtracurricularActivities": "No",
            "PlacementTraining": "No",
            "SSC_Marks": 58,
            "HSC_Marks": 62,
        }
    elif preset == "🚀 High Practical, Moderate Academic":
        preset_values = {
            "CGPA": 7.4,
            "Internships": 2,
            "Projects": 3,
            "Workshops/Certifications": 2,
            "AptitudeTestScore": 84,
            "SoftSkillsRating": 4.5,
            "ExtracurricularActivities": "Yes",
            "PlacementTraining": "Yes",
            "SSC_Marks": 68,
            "HSC_Marks": 72,
        }
    elif preset == "📚 High CGPA, Low Practical Experience":
        preset_values = {
            "CGPA": 8.8,
            "Internships": 0,
            "Projects": 1,
            "Workshops/Certifications": 0,
            "AptitudeTestScore": 72,
            "SoftSkillsRating": 3.8,
            "ExtracurricularActivities": "No",
            "PlacementTraining": "No",
            "SSC_Marks": 88,
            "HSC_Marks": 86,
        }

    # Dynamic Inputs from Schema
    student_input: Dict[str, Any] = {}

    # Numeric Inputs
    cgpa_stat = schema.feature_statistics.get("CGPA", {})
    student_input["CGPA"] = st.slider(
        "Academic CGPA (Out of 10):",
        min_value=float(cgpa_stat.get("min", 5.0)),
        max_value=float(cgpa_stat.get("max", 10.0)),
        value=float(preset_values.get("CGPA", cgpa_stat.get("default", 7.5))),
        step=0.1,
    )

    intern_stat = schema.feature_statistics.get("Internships", {})
    student_input["Internships"] = st.number_input(
        "Completed Internships:",
        min_value=int(intern_stat.get("min", 0)),
        max_value=int(intern_stat.get("max", 5)),
        value=int(preset_values.get("Internships", intern_stat.get("default", 1))),
        step=1,
    )

    proj_stat = schema.feature_statistics.get("Projects", {})
    student_input["Projects"] = st.number_input(
        "Technical Projects Count:",
        min_value=int(proj_stat.get("min", 0)),
        max_value=int(proj_stat.get("max", 6)),
        value=int(preset_values.get("Projects", proj_stat.get("default", 2))),
        step=1,
    )

    cert_stat = schema.feature_statistics.get("Workshops/Certifications", {})
    student_input["Workshops/Certifications"] = st.number_input(
        "Workshops / Certifications:",
        min_value=int(cert_stat.get("min", 0)),
        max_value=int(cert_stat.get("max", 5)),
        value=int(preset_values.get("Workshops/Certifications", cert_stat.get("default", 1))),
        step=1,
    )

    apt_stat = schema.feature_statistics.get("AptitudeTestScore", {})
    student_input["AptitudeTestScore"] = st.slider(
        "Aptitude Assessment Score (0-100):",
        min_value=int(apt_stat.get("min", 40)),
        max_value=int(apt_stat.get("max", 100)),
        value=int(preset_values.get("AptitudeTestScore", apt_stat.get("default", 75))),
        step=1,
    )

    soft_stat = schema.feature_statistics.get("SoftSkillsRating", {})
    student_input["SoftSkillsRating"] = st.slider(
        "Soft Skills Rating (1.0 - 5.0):",
        min_value=float(soft_stat.get("min", 1.0)),
        max_value=float(soft_stat.get("max", 5.0)),
        value=float(preset_values.get("SoftSkillsRating", soft_stat.get("default", 4.0))),
        step=0.1,
    )

    col_cat1, col_cat2 = st.columns(2)
    with col_cat1:
        extra_stat = schema.feature_statistics.get("ExtracurricularActivities", {})
        extra_cats = extra_stat.get("categories", ["No", "Yes"])
        student_input["ExtracurricularActivities"] = st.selectbox(
            "Extracurriculars:",
            options=extra_cats,
            index=extra_cats.index(preset_values.get("ExtracurricularActivities", extra_stat.get("default", "No")))
            if preset_values.get("ExtracurricularActivities", extra_stat.get("default", "No")) in extra_cats
            else 0,
        )

    with col_cat2:
        train_stat = schema.feature_statistics.get("PlacementTraining", {})
        train_cats = train_stat.get("categories", ["No", "Yes"])
        student_input["PlacementTraining"] = st.selectbox(
            "Placement Training:",
            options=train_cats,
            index=train_cats.index(preset_values.get("PlacementTraining", train_stat.get("default", "Yes")))
            if preset_values.get("PlacementTraining", train_stat.get("default", "Yes")) in train_cats
            else 0,
        )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        ssc_stat = schema.feature_statistics.get("SSC_Marks", {})
        student_input["SSC_Marks"] = st.number_input(
            "10th %:",
            min_value=int(ssc_stat.get("min", 40)),
            max_value=int(ssc_stat.get("max", 100)),
            value=int(preset_values.get("SSC_Marks", ssc_stat.get("default", 70))),
            step=1,
        )
    with col_m2:
        hsc_stat = schema.feature_statistics.get("HSC_Marks", {})
        student_input["HSC_Marks"] = st.number_input(
            "12th %:",
            min_value=int(hsc_stat.get("min", 40)),
            max_value=int(hsc_stat.get("max", 100)),
            value=int(preset_values.get("HSC_Marks", hsc_stat.get("default", 75))),
            step=1,
        )

    st.divider()
    # Batch CSV Upload
    st.markdown("#### 📁 **Batch Inference (CSV)**")
    uploaded_file = st.file_uploader("Upload Student Dataset (.csv):", type=["csv"])


# -----------------------------------------------------------------------------
# MAIN HEADER BANNER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1 class="hero-title">SkillLens : Career Intelligence & Placement Readiness</h1>
        <p class="hero-subtitle">
            Enterprise Machine Learning Pipeline with SHAP Explainability & Real-Time Simulation Sandbox
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# RUN EXPLANATION & INFERENCE FOR SINGLE STUDENT
# -----------------------------------------------------------------------------
explanation = explainer_engine.explain_instance(student_input)
pred_prob = explanation.predicted_probability
is_placed = pred_prob >= threshold

if pred_prob >= 0.80:
    tier_class = "tier-high"
    tier_label = "Top Tier Candidate"
    status_color = "#10B981"
    verdict_badge = "🟢 Strong Hire / Priority Shortlist"
    verdict_sub = "Satisfies premier recruitment benchmarks"
elif pred_prob >= 0.65:
    tier_class = "tier-blue"
    tier_label = "Competitive Prospect"
    status_color = "#3B82F6"
    verdict_badge = "🔵 Viable Prospect / Selective Shortlist"
    verdict_sub = "Competitive profile with minor gaps"
elif pred_prob >= 0.50:
    tier_class = "tier-med"
    tier_label = "Borderline / High Screening Risk"
    status_color = "#F59E0B"
    verdict_badge = "🟠 Borderline / First-Round Screening Hazard"
    verdict_sub = "High probability of elimination in resume/OA screen"
elif pred_prob >= 0.35:
    tier_class = "tier-low"
    tier_label = "Below Hiring Threshold"
    status_color = "#EF4444"
    verdict_badge = "🔴 Below Cutoff / Probable Rejection"
    verdict_sub = "Sub-threshold across multiple recruitment metrics"
else:
    tier_class = "tier-critical"
    tier_label = "Severe Placement Deficit"
    status_color = "#DC2626"
    verdict_badge = "🚨 Critical Disqualification Risk"
    verdict_sub = "Immediate resume elimination across campus drives"


# -----------------------------------------------------------------------------
# APPLICATION TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Readiness Dashboard",
    "🔬 What-If Simulation Sandbox",
    "📊 Model Performance & Failure Log",
    "📈 Dataset Insights",
    "📁 Batch Predictions",
])


# =============================================================================
# TAB 1: READINESS DASHBOARD
# =============================================================================
with tab1:
    # Top Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Placement Status</div>
                <div class="metric-value" style="color: {status_color};">
                    {'Placed' if is_placed else 'Not Placed'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Readiness Score</div>
                <div class="metric-value">{pred_prob * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Readiness Tier</div>
                <div style="margin-top: 0.5rem;">
                    <span class="tier-badge {tier_class}">{tier_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Recruiter Verdict</div>
                <div style="margin-top: 0.35rem; font-size: 0.95rem; font-weight: 700; color: {status_color};">
                    {verdict_badge}
                </div>
                <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 0.2rem;">{verdict_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge Chart & SHAP Waterfall Layout
    col_gauge, col_shap = st.columns([1.1, 1.9])

    with col_gauge:
        st.markdown("#### 🎯 **Readiness Score Gauge**")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=pred_prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Placement Likelihood (%)", "font": {"size": 18, "color": "#F8FAFC"}},
                delta={"reference": threshold * 100, "increasing": {"color": "#10B981"}, "decreasing": {"color": "#EF4444"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": status_color, "thickness": 0.28},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255,255,255,0.1)",
                    "steps": [
                        {"range": [0, 35], "color": "rgba(220, 38, 38, 0.25)"},
                        {"range": [35, 50], "color": "rgba(239, 68, 68, 0.18)"},
                        {"range": [50, 65], "color": "rgba(245, 158, 11, 0.18)"},
                        {"range": [65, 80], "color": "rgba(59, 130, 246, 0.18)"},
                        {"range": [80, 100], "color": "rgba(16, 185, 129, 0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "#6366F1", "width": 3},
                        "thickness": 0.8,
                        "value": threshold * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            font={"color": "#F8FAFC"},
        )
        render_chart(fig_gauge)

    with col_shap:
        st.markdown("#### 🔍 **Explainability Breakdown (SHAP Feature Drivers)**")
        # Prepare SHAP bar data
        factors_data = []
        for f in explanation.factors[:8]:
            factors_data.append({
                "Feature": f.feature_display_name,
                "Value": str(f.value),
                "Impact": f.impact_score,
                "Direction": "Boosts Placement (+)" if f.impact_score >= 0 else "Lowers Placement (-)",
                "Color": "#10B981" if f.impact_score >= 0 else "#EF4444",
            })
        df_factors = pd.DataFrame(factors_data)

        if not df_factors.empty:
            fig_shap = px.bar(
                df_factors,
                x="Impact",
                y="Feature",
                orientation="h",
                color="Direction",
                color_discrete_map={"Boosts Placement (+)": "#10B981", "Lowers Placement (-)": "#EF4444"},
                text="Value",
                title=f"Feature Contributions toward Prediction ({selected_model_type})",
            )
            fig_shap.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(l=10, r=10, t=40, b=10),
                font={"color": "#F8FAFC"},
                xaxis=dict(title="SHAP Impact Magnitude (Log-Odds / Probability)", gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(title="", autorange="reversed"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_shap.update_traces(textposition="outside", cliponaxis=False)
            render_chart(fig_shap)

    # Diagnostic Summary & AI Actionable Recommendations
    st.markdown("#### 🧠 **Automated Candidate Diagnostic & Action Plan**")
    st.markdown(
        f"""
        <div class="diagnostic-box">
            {explanation.narrative_summary}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_rec1, col_rec2 = st.columns(2)
    with col_rec1:
        st.markdown("##### 🚀 **Strengths & Competitive Advantages**")
        if explanation.positive_factors:
            for pf in explanation.positive_factors[:3]:
                st.markdown(
                    f"""
                    <div class="rec-card" style="border-left: 3px solid #10B981;">
                        <strong>{pf.feature_display_name}:</strong> Candidate has <code>{pf.value}</code> (+{abs(pf.impact_score):.3f} impact).<br>
                        <span style="color: #94A3B8; font-size: 0.88rem;">{pf.actionable_tip}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No distinct positive drivers detected. Focus on holistic skill acquisition.")

    with col_rec2:
        st.markdown("##### ⚠️ **Prioritized Improvement Roadmap**")
        for rec in explanation.top_recommendations:
            st.markdown(
                f"""
                <div class="rec-card" style="border-left: 3px solid #F59E0B;">
                    {rec}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Cohort Benchmark Radar & Comparison Table
    st.markdown("#### 📊 **Cohort Benchmarking (Candidate vs. Placed Peers)**")
    col_radar, col_bench_table = st.columns([1.2, 1.8])

    with col_radar:
        # Radar Chart for normalized numeric features
        radar_features = ["CGPA", "AptitudeTestScore", "SoftSkillsRating", "Projects", "Internships"]
        cand_vals = []
        placed_vals = []
        not_placed_vals = []

        for rf in radar_features:
            stat = schema.feature_statistics.get(rf, {})
            max_v = float(stat.get("max", 100))
            min_v = float(stat.get("min", 0))
            span = max_v - min_v if max_v > min_v else 1.0

            cand_norm = (float(student_input.get(rf, 0)) - min_v) / span
            cand_vals.append(cand_norm * 100)

            bench = cohort_benchmarks.get("features", {}).get(rf, {})
            p_mean = bench.get("placed_mean", 0)
            np_mean = bench.get("not_placed_mean", 0)
            placed_vals.append(((p_mean - min_v) / span) * 100)
            not_placed_vals.append(((np_mean - min_v) / span) * 100)

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=cand_vals + [cand_vals[0]], theta=radar_features + [radar_features[0]], fill="toself", name="Current Candidate", line=dict(color="#6366F1", width=2)))
        fig_radar.add_trace(go.Scatterpolar(r=placed_vals + [placed_vals[0]], theta=radar_features + [radar_features[0]], fill="toself", name="Avg Placed Peer", line=dict(color="#10B981", dash="dot")))
        fig_radar.add_trace(go.Scatterpolar(r=not_placed_vals + [not_placed_vals[0]], theta=radar_features + [radar_features[0]], fill="toself", name="Avg Not Placed Peer", line=dict(color="#EF4444", dash="dot")))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F8FAFC"},
            height=340,
            margin=dict(l=30, r=30, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        render_chart(fig_radar)

    with col_bench_table:
        table_rows = []
        for rf in ["CGPA", "AptitudeTestScore", "Projects", "Internships", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"]:
            bench = cohort_benchmarks.get("features", {}).get(rf, {})
            c_val = student_input.get(rf, "N/A")
            p_val = bench.get("placed_mean", "N/A")
            np_val = bench.get("not_placed_mean", "N/A")
            diff = round(float(c_val) - float(p_val), 2) if isinstance(c_val, (int, float)) and isinstance(p_val, (int, float)) else 0.0
            table_rows.append({
                "Evaluation Dimension": explainer_engine.FRIENDLY_FEATURE_NAMES.get(rf, rf),
                "Candidate Value": c_val,
                "Placed Cohort Avg": p_val,
                "Not Placed Cohort Avg": np_val,
                "Gap vs Placed Peers": f"{'+' if diff >= 0 else ''}{diff}",
            })
        df_bench_display = pd.DataFrame(table_rows)
        render_dataframe(df_bench_display, hide_index=True)


# =============================================================================
# TAB 2: WHAT-IF SIMULATION SANDBOX
# =============================================================================
with tab2:
    st.markdown("### 🔬 **Interactive What-If Simulation Sandbox**")
    st.markdown(
        "Experiment with hypothetical interventions in real-time. "
        "Observe how acquiring more projects, improving CGPA, or finishing placement bootcamps shifts candidate readiness."
    )

    # Synchronize simulation state when candidate profile changes
    cand_sig = (
        preset,
        float(student_input["CGPA"]),
        int(student_input["Internships"]),
        int(student_input["Projects"]),
        int(student_input["Workshops/Certifications"]),
        int(student_input["AptitudeTestScore"]),
        float(student_input["SoftSkillsRating"]),
        str(student_input["PlacementTraining"]),
        str(student_input["ExtracurricularActivities"]),
    )

    if st.session_state.get("last_synced_cand_sig") != cand_sig:
        st.session_state["last_synced_cand_sig"] = cand_sig
        st.session_state["sim_cgpa"] = float(student_input["CGPA"])
        st.session_state["sim_intern"] = int(student_input["Internships"])
        st.session_state["sim_proj"] = int(student_input["Projects"])
        st.session_state["sim_cert"] = int(student_input["Workshops/Certifications"])
        st.session_state["sim_apt"] = int(student_input["AptitudeTestScore"])
        st.session_state["sim_soft"] = float(student_input["SoftSkillsRating"])
        st.session_state["sim_train"] = student_input["PlacementTraining"]
        st.session_state["sim_extra"] = student_input["ExtracurricularActivities"]

    col_sim_controls, col_sim_results = st.columns([1.1, 1.9])

    with col_sim_controls:
        st.markdown("#### 🛠️ **Simulated Interventions**")
        if st.button("🔄 Reset Sandbox to Current Candidate Profile"):
            st.session_state["sim_cgpa"] = float(student_input["CGPA"])
            st.session_state["sim_intern"] = int(student_input["Internships"])
            st.session_state["sim_proj"] = int(student_input["Projects"])
            st.session_state["sim_cert"] = int(student_input["Workshops/Certifications"])
            st.session_state["sim_apt"] = int(student_input["AptitudeTestScore"])
            st.session_state["sim_soft"] = float(student_input["SoftSkillsRating"])
            st.session_state["sim_train"] = student_input["PlacementTraining"]
            st.session_state["sim_extra"] = student_input["ExtracurricularActivities"]
            st.rerun()

        sim_cgpa = st.slider("Simulated CGPA:", 5.0, 10.0, step=0.1, key="sim_cgpa")
        sim_intern = st.number_input("Simulated Internships:", 0, 5, step=1, key="sim_intern")
        sim_proj = st.number_input("Simulated Projects:", 0, 6, step=1, key="sim_proj")
        sim_cert = st.number_input("Simulated Certifications:", 0, 5, step=1, key="sim_cert")
        sim_apt = st.slider("Simulated Aptitude Score:", 40, 100, step=1, key="sim_apt")
        sim_soft = st.slider("Simulated Soft Skills:", 1.0, 5.0, step=0.1, key="sim_soft")
        sim_train = st.selectbox("Placement Training:", ["Yes", "No"], key="sim_train")
        sim_extra = st.selectbox("Extracurricular Activities:", ["Yes", "No"], key="sim_extra")

        simulated_student = {
            **student_input,
            "CGPA": sim_cgpa,
            "Internships": sim_intern,
            "Projects": sim_proj,
            "Workshops/Certifications": sim_cert,
            "AptitudeTestScore": sim_apt,
            "SoftSkillsRating": sim_soft,
            "PlacementTraining": sim_train,
            "ExtracurricularActivities": sim_extra,
        }

    with col_sim_results:
        # Run Simulated Inference
        sim_explanation = explainer_engine.explain_instance(simulated_student)
        sim_prob = sim_explanation.predicted_probability
        delta_prob = sim_prob - pred_prob

        st.markdown("#### ⚡ **Real-Time Outcome Delta**")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric(
                label="Baseline Score",
                value=f"{pred_prob * 100:.1f}%",
                delta=None,
            )
        with res_col2:
            st.metric(
                label="Simulated Score",
                value=f"{sim_prob * 100:.1f}%",
                delta=f"{delta_prob * 100:+.1f}%",
                delta_color="normal" if delta_prob >= 0 else "inverse",
            )
        with res_col3:
            sim_placed = sim_prob >= threshold
            st.markdown(
                f"""
                <div style="padding: 0.5rem; background: rgba(30,41,59,0.7); border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: #94A3B8;">SIMULATED OUTCOME</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: {'#10B981' if sim_placed else '#EF4444'};">
                        {'Placed' if sim_placed else 'Not Placed'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Before vs After Comparison Bar Chart
        comp_df = pd.DataFrame([
            {"Stage": "Current Profile", "Placement Probability": pred_prob * 100, "Color": "#6366F1"},
            {"Stage": "Simulated Profile", "Placement Probability": sim_prob * 100, "Color": "#10B981" if delta_prob >= 0 else "#EF4444"},
        ])
        fig_comp = px.bar(
            comp_df,
            x="Stage",
            y="Placement Probability",
            color="Stage",
            color_discrete_map={"Current Profile": "#6366F1", "Simulated Profile": "#10B981" if delta_prob >= 0 else "#EF4444"},
            text="Placement Probability",
            range_y=[0, 100],
            title="Readiness Score Evolution",
        )
        fig_comp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_comp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=260,
            margin=dict(l=20, r=20, t=40, b=20),
            font={"color": "#F8FAFC"},
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            showlegend=False,
        )
        render_chart(fig_comp)

        # Simulation Path Planner
        st.markdown("#### 🎯 **High-Leverage Trajectory to Placement Readiness (> 80%)**")
        trajectory_steps = []
        if sim_prob < 0.80:
            if sim_proj < 3:
                trajectory_steps.append("Build and publish **1 additional full-stack project** on GitHub.")
            if sim_intern < 2:
                trajectory_steps.append("Complete **1 industrial or virtual internship**.")
            if sim_apt < 85:
                trajectory_steps.append("Raise Aptitude Score to **85+** through targeted problem sets.")
            if sim_train == "No":
                trajectory_steps.append("Complete college **Campus Placement Training Bootcamp**.")
            if sim_cgpa < 8.0:
                trajectory_steps.append("Target a semester GPA lift to bring cumulative CGPA to **8.0+**.")
        else:
            trajectory_steps.append("Current profile already exceeds the target 80% placement threshold.")

        for idx, step in enumerate(trajectory_steps, 1):
            st.markdown(f"**Step {idx}:** {step}")


# =============================================================================
# TAB 3: MODEL PERFORMANCE & FAILURE LOG
# =============================================================================
with tab3:
    st.markdown("### 📊 **Model Benchmarking & Error Analysis Engine**")
    st.markdown(
        "Direct performance comparison between the **Baseline (Logistic Regression)** "
        "and **Advanced (XGBoost Classifier)** evaluated on the exact same frozen test set."
    )

    # 1. Performance Table
    b_res = artifacts.baseline_result
    a_res = artifacts.advanced_result

    perf_df = pd.DataFrame([
        {
            "Model Architecture": b_res.model_name,
            "Accuracy": f"{b_res.accuracy * 100:.2f}%",
            "Precision": f"{b_res.precision * 100:.2f}%",
            "Recall": f"{b_res.recall * 100:.2f}%",
            "F1-Score": f"{b_res.f1 * 100:.2f}%",
            "ROC-AUC": f"{b_res.roc_auc:.4f}",
        },
        {
            "Model Architecture": a_res.model_name,
            "Accuracy": f"{a_res.accuracy * 100:.2f}%",
            "Precision": f"{a_res.precision * 100:.2f}%",
            "Recall": f"{a_res.recall * 100:.2f}%",
            "F1-Score": f"{a_res.f1 * 100:.2f}%",
            "ROC-AUC": f"{a_res.roc_auc:.4f}",
        },
    ])
    render_dataframe(perf_df, hide_index=True)

    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.markdown("#### 📈 **ROC-AUC Curves**")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=b_res.fpr, y=b_res.tpr, name=f"Baseline (AUC = {b_res.roc_auc:.3f})", line=dict(color="#94A3B8", width=2, dash="dash")))
        fig_roc.add_trace(go.Scatter(x=a_res.fpr, y=a_res.tpr, name=f"Advanced XGBoost (AUC = {a_res.roc_auc:.3f})", line=dict(color="#6366F1", width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Chance", line=dict(color="rgba(255,255,255,0.2)", dash="dot")))

        fig_roc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            font={"color": "#F8FAFC"},
            xaxis=dict(title="False Positive Rate", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="True Positive Rate", gridcolor="rgba(255,255,255,0.08)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        render_chart(fig_roc)

    with col_cm:
        model_display_title = "XGBoost" if "XGBoost" in selected_model_type else "Logistic Regression"
        st.markdown(f"#### 🔲 **{model_display_title} Confusion Matrix (Threshold = {threshold:.2f})**")
        
        # Dynamically compute predictions at the selected threshold
        y_test_arr = artifacts.y_test
        y_proba_arr = active_model_result.y_proba
        y_pred_thresh = (y_proba_arr >= threshold).astype(int)
        dyn_cm = confusion_matrix(y_test_arr, y_pred_thresh)
        z_text = [[str(y) for y in x] for x in dyn_cm]

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=dyn_cm,
                x=["Predicted Not Placed", "Predicted Placed"],
                y=["Actual Not Placed", "Actual Placed"],
                text=z_text,
                texttemplate="%{text}",
                colorscale="Viridis",
                showscale=False,
            )
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            font={"color": "#F8FAFC"},
        )
        render_chart(fig_cm)

    st.divider()

    # Global Feature Importance Chart (Instant / Pre-computed from Model Artifacts)
    st.markdown("#### 🌐 **Dataset Global Feature Importance**")
    feat_imp = active_model_result.feature_importances
    if feat_imp:
        imp_rows = [
            {
                "feature": k,
                "importance": v,
                "display_name": explainer_engine.FRIENDLY_FEATURE_NAMES.get(k, k),
            }
            for k, v in feat_imp.items()
        ]
        df_global_imp = pd.DataFrame(imp_rows)
        df_global_imp = df_global_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)
        total = df_global_imp["importance"].sum()
        df_global_imp["relative_pct"] = (df_global_imp["importance"] / (total if total > 0 else 1.0)) * 100
    else:
        df_global_imp = get_cached_global_importance(
            selected_model_type,
            explainer_engine,
            artifacts.pipeline.transform(df_raw.head(50)),
        )

    fig_global = px.bar(
        df_global_imp,
        x="relative_pct",
        y="display_name",
        orientation="h",
        color_discrete_sequence=["#6366F1"],
        title=f"Global Feature Importance Weight Distribution ({model_display_title})",
    )
    fig_global.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        font={"color": "#F8FAFC"},
        xaxis=dict(title="Relative Importance (%)", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="", autorange="reversed"),
    )
    render_chart(fig_global)

    st.divider()

    # Failure Analysis Log Deep-Dive
    st.markdown("#### 🔍 **Deep-Dive Failure Analysis & Error Log (20 Sample Cases)**")
    st.caption("Inspection of test set predictions where the model exhibited high uncertainty or misclassification.")

    # Filter selector
    filter_type = st.selectbox(
        "Filter Error Category:",
        options=["All Cases", "False Positive", "False Negative", "High Uncertainty", "Borderline Case"],
    )

    filtered_cases = artifacts.failure_log
    if filter_type != "All Cases":
        filtered_cases = [c for c in filtered_cases if c.error_type == filter_type]

    # Convert cases to DataFrame
    case_rows = []
    for c in filtered_cases:
        row_dict = {
            "Student ID": c.student_id,
            "Error Type": c.error_type,
            "Actual Status": c.true_status,
            "Predicted Status": c.pred_status,
            "Placement Prob": f"{c.predicted_probability * 100:.1f}%",
            "CGPA": c.feature_values.get("CGPA", "N/A"),
            "Internships": c.feature_values.get("Internships", "N/A"),
            "Projects": c.feature_values.get("Projects", "N/A"),
            "Aptitude": c.feature_values.get("AptitudeTestScore", "N/A"),
            "Soft Skills": c.feature_values.get("SoftSkillsRating", "N/A"),
            "Diagnostic Root-Cause Hypothesis": c.diagnostic_reason,
        }
        case_rows.append(row_dict)

    df_cases_view = pd.DataFrame(case_rows)
    render_dataframe(df_cases_view, hide_index=True)


# =============================================================================
# TAB 4: DATASET INSIGHTS & EXPLORATORY ANALYTICS
# =============================================================================
with tab4:
    st.markdown("### 📈 **Exploratory Data Analysis (EDA)**")
    st.caption(f"Dataset Overview ({len(df_raw)} records loaded from Kaggle Placement Repository)")

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        st.markdown("#### 🎯 **Placement Distribution**")
        target_counts = df_raw["PlacementStatus"].value_counts().reset_index()
        target_counts.columns = ["Status", "Count"]
        fig_pie = px.pie(
            target_counts,
            values="Count",
            names="Status",
            color="Status",
            color_discrete_map={"Placed": "#10B981", "NotPlaced": "#EF4444"},
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F8FAFC"},
            height=320,
        )
        render_chart(fig_pie)

    with eda_col2:
        st.markdown("#### 🎓 **CGPA vs Aptitude Distribution by Status**")
        fig_scatter = px.scatter(
            df_raw.sample(min(800, len(df_raw)), random_state=42),
            x="CGPA",
            y="AptitudeTestScore",
            color="PlacementStatus",
            color_discrete_map={"Placed": "#10B981", "NotPlaced": "#EF4444"},
            opacity=0.7,
            title="CGPA vs Aptitude Score (Sample of 800 Candidates)",
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F8FAFC"},
            height=320,
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        )
        render_chart(fig_scatter)

    st.markdown("#### 📊 **Feature Boxplot Distributions Across Outcomes**")
    eda_feature = st.selectbox(
        "Select Feature for Distribution Comparison:",
        options=["CGPA", "AptitudeTestScore", "Projects", "Internships", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"],
    )
    fig_box = px.box(
        df_raw.sample(min(600, len(df_raw)), random_state=42),
        x="PlacementStatus",
        y=eda_feature,
        color="PlacementStatus",
        color_discrete_map={"Placed": "#10B981", "NotPlaced": "#EF4444"},
        points="outliers",
    )
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC"},
        height=320,
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        showlegend=False,
    )
    render_chart(fig_box)


# =============================================================================
# TAB 5: BATCH INFERENCE
# =============================================================================
with tab5:
    st.markdown("### 📁 **Batch Inference & Export**")
    st.markdown(
        "Score entire graduating cohorts or batches of student applicants simultaneously. "
        "Upload a `.csv` file matching the schema to generate predictions and readiness tiers."
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded CSV with {len(batch_df)} student records.")

            # Transform and predict
            X_batch = pipeline.transform(batch_df)
            if hasattr(active_model, "predict_proba"):
                batch_probas = active_model.predict_proba(X_batch)[:, 1]
            else:
                batch_dec = active_model.decision_function(X_batch)
                batch_probas = 1 / (1 + np.exp(-batch_dec))

            batch_preds = (batch_probas >= threshold).astype(int)

            result_df = batch_df.copy()
            result_df["Predicted_Placement_Status"] = np.where(batch_preds == 1, "Placed", "Not Placed")
            result_df["Placement_Probability_%"] = (batch_probas * 100).round(2)
            result_df["Readiness_Tier"] = np.where(
                batch_probas >= 0.75,
                "High Readiness",
                np.where(batch_probas >= 0.45, "Moderate Readiness", "Low Readiness"),
            )

            render_dataframe(result_df)

            # CSV Download
            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Scored Batch Predictions CSV",
                data=csv_buffer.getvalue(),
                file_name="skilllens_batch_placement_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error processing batch CSV: {e}")
    else:
        st.info("Upload a CSV file in the sidebar to process batch predictions.")
        # Provide sample template download
        sample_df = df_raw.head(5).copy()
        if "PlacementStatus" in sample_df.columns:
            sample_df = sample_df.drop(columns=["PlacementStatus"])
        sample_csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Sample Batch Input Template CSV",
            data=sample_csv,
            file_name="skilllens_sample_input_template.csv",
            mime="text/csv",
        )
