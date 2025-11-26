# app.py - Bankruptcy Prevention Prediction System
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ---------------------------
# Configuration & Constants
# ---------------------------
MODEL_DIR = "../model"
FEATURES = [
    'industrial_risk',
    'management_risk',
    'financial_flexibility',
    'credibility',
    'competitiveness',
    'operating_risk'
]

# ---------------------------
# Load Production Model & Metadata
# ---------------------------
@st.cache_resource
def load_production_model():
    """Load the best model from cross-validation and associated metadata"""
    try:
        # Load the final trained pipeline
        model_path = os.path.join(MODEL_DIR, "bankruptcy_model_v3.pkl")
        if not os.path.exists(model_path):
            return None, None, None
        
        final_pipeline = joblib.load(model_path)
        
        # Load metadata
        metadata_path = os.path.join(MODEL_DIR, "model_metadata_v3.pkl")
        metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else None
        
        # Load feature names
        features_path = os.path.join(MODEL_DIR, "feature_names.pkl")
        features = joblib.load(features_path) if os.path.exists(features_path) else FEATURES
        
        return final_pipeline, metadata, features
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Load model on startup
pipeline, metadata, features = load_production_model()

if pipeline is None:
    st.error("❌ Production model not found. Please run the training notebook first.")
    st.stop()

# ---------------------------
# Page Configuration & Styling
# ---------------------------
st.set_page_config(
    page_title="Bankruptcy Prevention System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .metric-card { padding: 1.5rem; border-radius: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; box-shadow: 0 8px 16px rgba(0,0,0,0.1); }
    button { border-radius: 0.75rem !important; font-weight: 600 !important; box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important; }
    button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(0,0,0,0.2) !important; }
    h1, h2, h3 { color: #333333 !important; font-weight: 700 !important; }
    .metric-label { font-size: 0.9rem; color: #666; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Header & Introduction
# ---------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🏢 Bankruptcy Prevention Prediction System")
    st.markdown("**Advanced ML Model for Early Warning Detection**")

with col2:
    if metadata:
        st.metric(
            "Model Type",
            metadata.get('model_name', 'Unknown'),
            f"{metadata.get('performance', {}).get('mean_roc_auc', 0):.2%} ROC-AUC"
        )

# Display model information
if metadata:
    st.markdown("---")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    
    with info_col1:
        st.metric(
            "📊 Mean ROC-AUC",
            f"{metadata.get('performance', {}).get('mean_roc_auc', 0):.4f}",
            f"±{metadata.get('performance', {}).get('std_roc_auc', 0):.4f}"
        )
    
    with info_col2:
        st.metric(
            "🎯 Mean Precision",
            f"{metadata.get('performance', {}).get('mean_precision', 0):.4f}",
            "on test set"
        )
    
    with info_col3:
        st.metric(
            "📈 Mean Recall",
            f"{metadata.get('performance', {}).get('mean_recall', 0):.4f}",
            "on test set"
        )
    
    with info_col4:
        st.metric(
            "📅 Training Date",
            metadata.get('training_date', 'N/A'),
            metadata.get('model_version', 'v3.1')
        )


# ---------------------------
# Sidebar Information & Instructions
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.header("📋 Instructions")
st.sidebar.markdown("""
1. **Select Risk Factors**: Use the sliders to input company risk metrics
2. **Generate Prediction**: Click the predict button
3. **Review Results**: See bankruptcy risk assessment and confidence levels

**Risk Scale:**
- 🟢 0 = Low Risk
- 🟡 0.5 = Medium Risk  
- 🔴 1.0 = High Risk
""")

if metadata:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Performance (5-Fold CV)")
    perf_dict = metadata.get('performance', {})
    for metric_name, value in perf_dict.items():
        if isinstance(value, float):
            st.sidebar.write(f"**{metric_name.replace('_', ' ').title()}**: {value:.4f}")

# ---------------------------
# Main Content Area
# ---------------------------
st.markdown("---")
st.markdown("<h2 style='text-align: center; color: #333;'>🔍 Company Risk Assessment</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Select risk level for each factor</p>", unsafe_allow_html=True)

# Input form with selectboxes
st.markdown("")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<p class='metric-label'>📊 Industrial Risk</p>", unsafe_allow_html=True)
    industrial_risk = st.selectbox(
        "Industrial Risk",
        options=[0.0, 0.5, 1.0],
        format_func=lambda x: "🟢 Low" if x == 0.0 else ("🟡 Medium" if x == 0.5 else "🔴 High"),
        index=1,
        key="industrial_risk",
        label_visibility="collapsed",
        help="Risk associated with the industry sector"
    )
    st.caption("Industry sector risk")

with col2:
    st.markdown("<p class='metric-label'>👨‍💼 Management Risk</p>", unsafe_allow_html=True)
    management_risk = st.selectbox(
        "Management Risk",
        options=[0.0, 0.5, 1.0],
        format_func=lambda x: "🟢 Low" if x == 0.0 else ("🟡 Medium" if x == 0.5 else "🔴 High"),
        index=1,
        key="management_risk",
        label_visibility="collapsed",
        help="Risk from management effectiveness"
    )
    st.caption("Management effectiveness")

with col3:
    st.markdown("<p class='metric-label'>💰 Financial Flexibility</p>", unsafe_allow_html=True)
    financial_flexibility = st.selectbox(
        "Financial Flexibility",
        options=[0.0, 0.5, 1.0],
        format_func=lambda x: "🟢 High" if x == 0.0 else ("🟡 Medium" if x == 0.5 else "🔴 Low"),
        index=1,
        key="financial_flexibility",
        label_visibility="collapsed",
        help="Company's ability to adjust finances"
    )
    st.caption("Restructure finances")

st.markdown("")

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("<p class='metric-label'>✅ Credibility</p>", unsafe_allow_html=True)
    credibility = st.selectbox(
        "Credibility",
        options=[0.0, 0.5, 1.0],
        format_func=lambda x: "🟢 High" if x == 0.0 else ("🟡 Medium" if x == 0.5 else "🔴 Low"),
        index=1,
        key="credibility",
        label_visibility="collapsed",
        help="Company's credit reputation"
    )
    st.caption("Credit reputation")

with col5:
    st.markdown("<p class='metric-label'>🏆 Competitiveness</p>", unsafe_allow_html=True)
    competitiveness = st.selectbox(
        "Competitiveness",
        options=[0.0, 0.5, 1.0],
        format_func=lambda x: "🟢 High" if x == 0.0 else ("🟡 Medium" if x == 0.5 else "🔴 Low"),
        index=1,
        key="competitiveness",
        label_visibility="collapsed",
        help="Market competitive position"
    )
    st.caption("Market position")

with col6:
    st.markdown("<p class='metric-label'>⚙️ Operating Risk</p>", unsafe_allow_html=True)
    operating_risk = st.selectbox(
        "Operating Risk",
        options=[0.0, 0.5, 1.0],
        format_func=lambda x: "🟢 Low" if x == 0.0 else ("🟡 Medium" if x == 0.5 else "🔴 High"),
        index=1,
        key="operating_risk",
        label_visibility="collapsed",
        help="Operational efficiency and stability"
    )
    st.caption("Operational efficiency")

# Prediction button
st.markdown("")
col_button = st.columns([1, 3, 1])
with col_button[1]:
    predict_button = st.button(
        "🔮 Generate Bankruptcy Risk Prediction",
        use_container_width=True,
        type="primary",
        key="predict_btn"
    )

# ---------------------------
# Prediction & Results
# ---------------------------
if predict_button:
    # Prepare input data
    inputs = {
        "industrial_risk": industrial_risk,
        "management_risk": management_risk,
        "financial_flexibility": financial_flexibility,
        "credibility": credibility,
        "competitiveness": competitiveness,
        "operating_risk": operating_risk
    }
    
    # Create input DataFrame in correct order
    input_df = pd.DataFrame([inputs])
    input_data = input_df[features].values
    
    try:
        # Make prediction using the pipeline
        pred = pipeline.predict(input_data)[0]
        
        # Get probability
        prob_array = pipeline.predict_proba(input_data)[0]
        prob_bankruptcy = prob_array[1]
        prob_safe = prob_array[0]
        
        # Determine risk level
        if prob_bankruptcy >= 0.7:
            risk_level = "🔴 CRITICAL"
            risk_color = "red"
        elif prob_bankruptcy >= 0.5:
            risk_level = "🟠 HIGH"
            risk_color = "orange"
        elif prob_bankruptcy >= 0.3:
            risk_level = "🟡 MODERATE"
            risk_color = "yellow"
        else:
            risk_level = "🟢 LOW"
            risk_color = "green"
        
        # Display results
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: #333;'>📊 Prediction Results</h2>", unsafe_allow_html=True)
        
        # Main result boxes
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            pred_label = "🚨 BANKRUPTCY" if pred == 1 else "✅ SAFE"
            prob_display = f"{prob_bankruptcy:.1%}" if pred == 1 else f"{prob_safe:.1%}"
            box_gradient = "linear-gradient(135deg, #ff4757 0%, #ffa502 100%)" if pred == 1 else "linear-gradient(135deg, #2ed573 0%, #84fab0 100%)"
            text_color = "white" if pred == 1 else "white"
            
            st.markdown(f"""<div style='background: {box_gradient}; padding: 2rem; border-radius: 1rem; text-align: center; color: {text_color}; box-shadow: 0 8px 16px rgba(0,0,0,0.2);'>
                <p style='font-size: 0.9rem; opacity: 0.9; margin: 0;'>PREDICTION</p>
                <p style='font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0;'>{pred_label}</p>
                <p style='font-size: 1.2rem; opacity: 0.95; margin: 0;'>{prob_display} Probability</p>
            </div>""", unsafe_allow_html=True)
        
        with result_col2:
            if prob_bankruptcy >= 0.7:
                risk_level = "🔴 CRITICAL"
                risk_color = "#ff4757"
            elif prob_bankruptcy >= 0.5:
                risk_level = "🟠 HIGH"
                risk_color = "#ffa502"
            elif prob_bankruptcy >= 0.3:
                risk_level = "🟡 MODERATE"
                risk_color = "#ffd93d"
            else:
                risk_level = "🟢 LOW"
                risk_color = "#2ed573"
            
            text_col = "white" if prob_bankruptcy >= 0.3 else "#333"
            
            st.markdown(f"""<div style='background: {risk_color}; padding: 2rem; border-radius: 1rem; text-align: center; color: {text_col}; box-shadow: 0 8px 16px rgba(0,0,0,0.2);'>
                <p style='font-size: 0.9rem; opacity: 0.9; margin: 0;'>RISK LEVEL</p>
                <p style='font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0;'>{risk_level}</p>
                <p style='font-size: 1.1rem; opacity: 0.95; margin: 0;'>{prob_bankruptcy:.2%} Risk</p>
            </div>""", unsafe_allow_html=True)
        
        with result_col3:
            confidence = max(prob_bankruptcy, prob_safe)
            st.markdown(f"""<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 1rem; text-align: center; color: white; box-shadow: 0 8px 16px rgba(0,0,0,0.2);'>
                <p style='font-size: 0.9rem; opacity: 0.9; margin: 0;'>CONFIDENCE</p>
                <p style='font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0;'>{confidence:.1%}</p>
                <p style='font-size: 0.95rem; opacity: 0.95; margin: 0;'>Model Certainty</p>
            </div>""", unsafe_allow_html=True)
        
        # Risk assessment message
        st.markdown("---")
        if pred == 1:
            st.markdown(f"""
            <div style='border-left: 5px solid #ff4757; background: linear-gradient(90deg, rgba(255, 71, 87, 0.1) 0%, rgba(255, 165, 2, 0.05) 100%); padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;'>
                <h3 style='color: #ff4757; margin: 0 0 1rem 0;'>⚠️ Bankruptcy Risk Detected</h3>
                <p style='color: #333; margin: 0.5rem 0;'><strong>Bankruptcy Probability:</strong> <span style='color: #ff4757; font-weight: 700; font-size: 1.1rem;'>{prob_bankruptcy:.2%}</span></p>
                <p style='color: #333; margin: 0.5rem 0;'><strong>Risk Level:</strong> {risk_level}</p>
                <p style='color: #333; margin: 1rem 0 0 0;'>This company has a critical risk of bankruptcy. Immediate intervention and financial restructuring recommendations are advised.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='border-left: 5px solid #2ed573; background: linear-gradient(90deg, rgba(46, 213, 115, 0.1) 0%, rgba(132, 250, 176, 0.05) 100%); padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;'>
                <h3 style='color: #2ed573; margin: 0 0 1rem 0;'>✅ Company Status: Stable</h3>
                <p style='color: #333; margin: 0.5rem 0;'><strong>Bankruptcy Probability:</strong> <span style='color: #2ed573; font-weight: 700; font-size: 1.1rem;'>{prob_bankruptcy:.2%}</span></p>
                <p style='color: #333; margin: 0.5rem 0;'><strong>Risk Level:</strong> {risk_level}</p>
                <p style='color: #333; margin: 1rem 0 0 0;'>This company appears financially stable. Continue monitoring financial metrics for any changes.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed analysis
        st.markdown("---")
        st.header("📈 Detailed Risk Analysis")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.subheader("Risk Factor Values")
            factors_df = pd.DataFrame(
                list(inputs.items()),
                columns=['Risk Factor', 'Value']
            )
            factors_df['Value'] = factors_df['Value'].apply(lambda x: f"{x:.1f}")
            st.table(factors_df)
        
        with analysis_col2:
            st.subheader("Probability Distribution")
            prob_data = {
                'Outcome': ['Non-Bankruptcy', 'Bankruptcy'],
                'Probability': [prob_safe, prob_bankruptcy]
            }
            prob_df = pd.DataFrame(prob_data)
            st.bar_chart(prob_df.set_index('Outcome'))
        
        # Model info in expander
        st.markdown("---")
        with st.expander("🔬 Model & Training Details"):
            model_details_col1, model_details_col2 = st.columns(2)
            
            with model_details_col1:
                st.subheader("Model Configuration")
                if metadata:
                    st.write(f"**Model Type**: {metadata.get('model_name', 'Unknown')}")
                    st.write(f"**Version**: {metadata.get('model_version', 'Unknown')}")
                    st.write(f"**Training Date**: {metadata.get('training_date', 'N/A')}")
                    st.write(f"**Features Used**: {len(features)}")
            
            with model_details_col2:
                st.subheader("Cross-Validation Performance")
                if metadata:
                    perf = metadata.get('performance', {})
                    st.write(f"**Mean ROC-AUC**: {perf.get('mean_roc_auc', 0):.4f} ± {perf.get('std_roc_auc', 0):.4f}")
                    st.write(f"**Mean Precision**: {perf.get('mean_precision', 0):.4f}")
                    st.write(f"**Mean Recall**: {perf.get('mean_recall', 0):.4f}")
                    st.write(f"**Mean F1-Score**: {perf.get('mean_f1_score', 0):.4f}")
        
        # Recommendations
        st.markdown("---")
        st.header("💡 Recommendations")
        
        recommendations = []
        if industrial_risk > 0.5:
            recommendations.append("• Review industry-specific risks and market trends")
        if management_risk > 0.5:
            recommendations.append("• Assess management team capabilities and experience")
        if financial_flexibility < 0.5:
            recommendations.append("• Improve financial flexibility through debt restructuring")
        if credibility < 0.5:
            recommendations.append("• Work on improving credit ratings and credibility")
        if competitiveness < 0.5:
            recommendations.append("• Enhance competitive position in the market")
        if operating_risk > 0.5:
            recommendations.append("• Optimize operational efficiency and cost management")
        
        if recommendations:
            rec_html = "<div style='background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%); padding: 1.5rem; border-radius: 1rem; border-left: 5px solid #f39c12;'>"
            for rec in recommendations:
                rec_html += f"<p style='color: #333; margin: 0.5rem 0; font-weight: 500;'>{rec}</p>"
            rec_html += "</div>"
            st.markdown(rec_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 1.5rem; border-radius: 1rem; border-left: 5px solid #2ed573; text-align: center;'>
                <p style='color: #333; margin: 0; font-weight: 600; font-size: 1.1rem;'>✅ No specific recommendations - company appears well-positioned</p>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        st.info("Please ensure all inputs are valid and the model is properly loaded.")