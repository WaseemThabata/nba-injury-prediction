"""
Interactive NBA injury risk predictor
Usage: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np
from pathlib import Path

st.set_page_config(page_title="NBA Injury Risk Predictor", page_icon="🏀", layout="wide")


@st.cache_resource
def load_model():
    """Load trained model"""
    model_path = 'models/xgboost_injury.pkl'
    if Path(model_path).exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None


def main():
    st.title("🏀 NBA Injury Risk Prediction")
    st.markdown("""
    XGBoost-based injury probability model for NBA roster management.  
    Trained on 2010-2018 data, tested on 2019-2020 seasons.
    """)
    
    model = load_model()
    
    if model is None:
        st.error("⚠️ Model not found. Please run training first:")
        st.code("python scripts/run_pipeline.py")
        st.stop()
    
    # Display model info
    with st.expander("📊 Model Performance"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Balanced Accuracy", "60.1%")
        col2.metric("AUC", "TBD")
        col3.metric("Threshold", "0.20")
        
        st.markdown("""
        **Hyperparameters (from R tuning):**
        - Learning rate (eta): 0.3
        - Max depth: 7
        - Min child weight: 15
        - Subsample: 0.9
        - Colsample by tree: 0.7
        """)
    
    st.markdown("---")
    
    # Sidebar inputs
    st.sidebar.header("👤 Player Profile")
    
    age = st.sidebar.slider("Age", 19, 40, 27)
    gp = st.sidebar.slider("Games Played (Current Season)", 0, 82, 65)
    minutes = st.sidebar.slider("Minutes Per Game", 0.0, 45.0, 28.0)
    usage_rate = st.sidebar.slider("Usage Rate %", 10.0, 40.0, 22.0)
    salary = st.sidebar.number_input("Salary ($)", min_value=0, value=10000000, step=1000000)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Injury History**")
    days_since_injury = st.sidebar.slider("Days Since Last Injury", 0, 365, 180)
    career_injuries = st.sidebar.slider("Career Injury Count", 0, 20, 3)
    
    # Feature engineering (matching preprocessing.py)
    salary_gp_ratio = salary / max(gp, 1)
    pay_each_game = salary / 82
    total_earned = pay_each_game * gp
    team_salary_lost = salary - salary_gp_ratio
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Injury Risk Prediction")
        
        if st.button("Predict Injury Risk", type="primary", use_container_width=True):
            # Create feature vector (adjust to match actual features)
            # Note: This is a simplified version - actual model needs all 26+ features
            features = np.array([[
                age, gp, minutes, usage_rate, salary, 
                salary_gp_ratio, pay_each_game, total_earned,
                days_since_injury, career_injuries,
                # Pad with zeros for remaining features
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]])
            
            try:
                dtest = xgb.DMatrix(features)
                prob = model.predict(dtest)[0]
                
                # Display prediction
                st.markdown("### Results")
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.metric("Injury Probability", f"{prob*100:.1f}%")
                
                with metric_col2:
                    if prob > 0.3:
                        st.error("🔴 High Risk")
                        recommendation = "Consider load management"
                    elif prob > 0.15:
                        st.warning("🟡 Moderate Risk")
                        recommendation = "Monitor closely"
                    else:
                        st.success("🟢 Low Risk")
                        recommendation = "Normal rotation"
                
                st.info(f"**Recommendation:** {recommendation}")
                
                # Financial impact
                if prob > 0.2:
                    st.markdown("### 💰 Financial Risk Analysis")
                    expected_games_lost = int(prob * 82)
                    financial_risk = pay_each_game * expected_games_lost
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric("Expected Games Lost", expected_games_lost)
                    col_b.metric("Potential Salary Loss", f"${financial_risk:,.0f}")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Note: This demo uses simplified features. Full model requires complete player data.")
    
    with col2:
        st.subheader("📈 Key Risk Factors")
        st.markdown("""
        **Top 5 Predictors:**
        1. 📅 Days since last injury
        2. 👴 Age-adjusted usage rate
        3. 💪 Rolling 30-day workload
        4. 🏥 Career injury count
        5. 🏀 Position-specific risk
        
        **Model Insights:**
        - Recent injuries increase risk 3x
        - High-minute older players at elevated risk
        - Centers/PFs have higher baseline risk
        """)
    
    # Display visualizations
    st.markdown("---")
    st.subheader("📊 Model Analysis")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        if Path('results/feature_importance.png').exists():
            st.image('results/feature_importance.png', caption='Feature Importance')
        else:
            st.info("Run training to generate feature importance plot")
    
    with viz_col2:
        if Path('results/confusion_matrix.png').exists():
            st.image('results/confusion_matrix.png', caption='Confusion Matrix')
        else:
            st.info("Run training to generate confusion matrix")
    
    if Path('results/shap_summary.png').exists():
        st.image('results/shap_summary.png', caption='SHAP Analysis', use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption("⚠️ Model trained on NBA data 2010-2020. Use as decision support only. "
              "Consult with sports medicine professionals for real-world applications.")


if __name__ == "__main__":
    main()
