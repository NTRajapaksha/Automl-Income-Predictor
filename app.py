"""
AutoML Inference Application - GitHub Codespaces
Load pre-trained model and make predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="Income Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND METADATA
# ============================================================================

@st.cache_resource
def load_model_and_metadata():
    """Load the trained model, preprocessor, and metadata"""
    try:
        # Load model
        with open('/workspaces/Automl-classification/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessor
        with open('/workspaces/Automl-classification/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Load metadata
        with open('/workspaces/Automl-classification/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load comparison data
        comparison_df = pd.read_csv('/workspaces/Automl-classification/model_comparison.csv')
        
        return model, preprocessor, metadata, comparison_df
    except FileNotFoundError as e:
        st.error(f"""
        ⚠️ **Missing Required Files!**
        
        Please upload the following files from your Colab training:
        - best_model.pkl
        - preprocessor.pkl
        - model_metadata.json
        - model_comparison.csv
        
        Missing file: {str(e)}
        """)
        st.stop()

# Load everything
model, preprocessor, metadata, comparison_df = load_model_and_metadata()

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<h1 class="main-header">💰 Income Prediction System</h1>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; padding: 1rem;'>
    <p style='font-size: 1.2rem;'>
        Using <b>{metadata['best_model_name']}</b> trained on UCI Adult Income Dataset
        <br>
        <span style='color: #667eea;'>Accuracy: {metadata['accuracy']:.2%}</span> | 
        <span style='color: #764ba2;'>AUC: {metadata['auc']:.2%}</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - MODEL INFO
# ============================================================================

with st.sidebar:
    st.header("📊 Model Information")
    
    st.metric("Model", metadata['best_model_name'])
    st.metric("Test Accuracy", f"{metadata['accuracy']:.2%}")
    st.metric("AUC Score", f"{metadata['auc']:.2%}")
    
    with st.expander("🔧 Best Hyperparameters"):
        st.json(metadata['best_params'])
    
    with st.expander("📈 All Model Results"):
        st.dataframe(comparison_df, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🎯 About")
    st.info("""
    This app uses a machine learning model trained on the UCI Adult Income dataset 
    to predict whether a person's income exceeds $50K/year.
    """)

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Prediction", 
    "📊 Batch Predictions", 
    "📈 Model Analytics",
    "🧪 What-If Analysis"
])

# ============================================================================
# TAB 1: SINGLE PREDICTION
# ============================================================================

with tab1:
    st.header("Make a Single Prediction")
    st.markdown("Enter the details below to predict income level:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        age = st.slider("Age", 17, 90, 30)
        sex = st.selectbox("Sex", ["Male", "Female"])
        race = st.selectbox("Race", [
            "White", "Black", "Asian-Pac-Islander", 
            "Amer-Indian-Eskimo", "Other"
        ])
        native_country = st.selectbox("Native Country", [
            "United-States", "Mexico", "Philippines", "Germany", 
            "Puerto-Rico", "Canada", "India", "Other"
        ])
    
    with col2:
        st.subheader("Education & Work")
        education = st.selectbox("Education", [
            "Bachelors", "HS-grad", "11th", "Masters", "9th",
            "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
            "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
            "Preschool", "12th"
        ])
        education_num = st.slider("Years of Education", 1, 16, 10)
        workclass = st.selectbox("Work Class", [
            "Private", "Self-emp-not-inc", "Local-gov", 
            "State-gov", "Self-emp-inc", "Federal-gov",
            "Without-pay", "Never-worked"
        ])
        occupation = st.selectbox("Occupation", [
            "Prof-specialty", "Craft-repair", "Exec-managerial",
            "Adm-clerical", "Sales", "Other-service", "Machine-op-inspct",
            "Transport-moving", "Handlers-cleaners", "Farming-fishing",
            "Tech-support", "Protective-serv", "Priv-house-serv", "Armed-Forces"
        ])
    
    with col3:
        st.subheader("Financial & Family")
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)
        marital_status = st.selectbox("Marital Status", [
            "Married-civ-spouse", "Never-married", "Divorced",
            "Separated", "Widowed", "Married-spouse-absent",
            "Married-AF-spouse"
        ])
        relationship = st.selectbox("Relationship", [
            "Husband", "Wife", "Own-child", "Not-in-family",
            "Other-relative", "Unmarried"
        ])
        capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
        capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
        fnlwgt = st.number_input("Final Weight (fnlwgt)", 10000, 1500000, 200000)
    
    if st.button("🔮 Predict Income", type="primary", use_container_width=True):
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'workclass': [workclass],
            'fnlwgt': [fnlwgt],
            'education': [education],
            'education-num': [education_num],
            'marital-status': [marital_status],
            'occupation': [occupation],
            'relationship': [relationship],
            'race': [race],
            'sex': [sex],
            'capital-gain': [capital_gain],
            'capital-loss': [capital_loss],
            'hours-per-week': [hours_per_week],
            'native-country': [native_country]
        })
        
        # Preprocess
        X_pred, _ = preprocessor.preprocess(input_data, target_col='income', is_training=False)
        
        # Make prediction
        prediction = model.predict(X_pred)[0]
        prediction_proba = model.predict_proba(X_pred)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            income_class = ">50K" if prediction == 1 else "≤50K"
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background-color: {"#d4edda" if prediction == 1 else "#f8d7da"}; border-radius: 10px;'>
                <h2 style='margin: 0;'>Predicted Income</h2>
                <h1 style='margin: 0; color: {"#155724" if prediction == 1 else "#721c24"};'>{income_class}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Probability ≤50K", f"{prediction_proba[0]:.1%}")
        
        with col3:
            st.metric("Probability >50K", f"{prediction_proba[1]:.1%}")
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prediction_proba[1] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence Score", 'font': {'size': 24}},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#ffcccc'},
                    {'range': [30, 70], 'color': '#ffffcc'},
                    {'range': [70, 100], 'color': '#ccffcc'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}}))
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: BATCH PREDICTIONS
# ============================================================================

with tab2:
    st.header("Batch Predictions")
    st.markdown("Upload a CSV file with multiple records for batch predictions")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file is not None:
        # Load data
        batch_df = pd.read_csv(uploaded_file)
        
        st.subheader("Uploaded Data Preview")
        st.dataframe(batch_df.head(), use_container_width=True)
        
        if st.button("🚀 Run Batch Predictions", type="primary"):
            with st.spinner("Processing predictions..."):
                # Preprocess
                X_batch, _ = preprocessor.preprocess(
                    batch_df, 
                    target_col='income' if 'income' in batch_df.columns else None,
                    is_training=False
                )
                
                # Predictions
                predictions = model.predict(X_batch)
                predictions_proba = model.predict_proba(X_batch)
                
                # Add to dataframe
                results_df = batch_df.copy()
                results_df['Predicted_Income'] = ['≤50K' if p == 0 else '>50K' for p in predictions]
                results_df['Probability_Low'] = predictions_proba[:, 0]
                results_df['Probability_High'] = predictions_proba[:, 1]
                
                st.success(f"✅ Processed {len(results_df)} predictions!")
                
                # Results summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(results_df))
                with col2:
                    st.metric("Predicted ≤50K", (predictions == 0).sum())
                with col3:
                    st.metric("Predicted >50K", (predictions == 1).sum())
                
                # Distribution chart
                fig = px.pie(
                    values=[(predictions == 0).sum(), (predictions == 1).sum()],
                    names=['≤50K', '>50K'],
                    title='Income Distribution in Batch',
                    color_discrete_sequence=['#ff6b6b', '#51cf66']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display results
                st.subheader("Prediction Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv'
                )
    else:
        st.info("👆 Upload a CSV file to get started with batch predictions")
        
        # Show sample format
        with st.expander("📋 View Required CSV Format"):
            sample_df = pd.DataFrame({
                'age': [39, 50],
                'workclass': ['State-gov', 'Self-emp-not-inc'],
                'fnlwgt': [77516, 83311],
                'education': ['Bachelors', 'Bachelors'],
                'education-num': [13, 13],
                'marital-status': ['Never-married', 'Married-civ-spouse'],
                'occupation': ['Adm-clerical', 'Exec-managerial'],
                'relationship': ['Not-in-family', 'Husband'],
                'race': ['White', 'White'],
                'sex': ['Male', 'Male'],
                'capital-gain': [2174, 0],
                'capital-loss': [0, 0],
                'hours-per-week': [40, 13],
                'native-country': ['United-States', 'United-States']
            })
            st.dataframe(sample_df)

# ============================================================================
# TAB 3: MODEL ANALYTICS
# ============================================================================

with tab3:
    st.header("Model Performance Analytics")
    
    # Model comparison
    st.subheader("📊 Model Comparison")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Accuracy Comparison', 'AUC Score Comparison'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    fig.add_trace(
        go.Bar(x=comparison_df['Model'], y=comparison_df['Accuracy'], 
               name='Accuracy', marker_color='lightblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=comparison_df['Model'], y=comparison_df['AUC'], 
               name='AUC', marker_color='lightgreen'),
        row=1, col=2
    )
    
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comparison table
    st.subheader("📋 Detailed Model Comparison")
    st.dataframe(
        comparison_df.style.background_gradient(cmap='RdYlGn', subset=['Accuracy', 'AUC']),
        use_container_width=True
    )
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.subheader("🔍 Feature Importance")
        
        feature_importance_df = pd.DataFrame({
            'Feature': metadata['feature_names'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(
            feature_importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Top 15 Most Important Features',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Model metadata
    with st.expander("🔧 Complete Model Configuration"):
        st.json(metadata)

# ============================================================================
# TAB 4: WHAT-IF ANALYSIS
# ============================================================================

with tab4:
    st.header("🧪 What-If Analysis")
    st.markdown("Explore how changing different features affects the prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Baseline Scenario")
        base_age = st.slider("Age", 17, 90, 40, key='base_age')
        base_education = st.slider("Education Years", 1, 16, 13, key='base_edu')
        base_hours = st.slider("Hours per Week", 1, 99, 40, key='base_hours')
    
    with col2:
        st.subheader("Modified Scenario")
        mod_age = st.slider("Age", 17, 90, base_age, key='mod_age')
        mod_education = st.slider("Education Years", 1, 16, base_education, key='mod_edu')
        mod_hours = st.slider("Hours per Week", 1, 99, base_hours, key='mod_hours')
    
    # Create comparison scenarios
    base_data = pd.DataFrame({
        'age': [base_age],
        'workclass': ['Private'],
        'fnlwgt': [200000],
        'education': ['Bachelors'],
        'education-num': [base_education],
        'marital-status': ['Married-civ-spouse'],
        'occupation': ['Exec-managerial'],
        'relationship': ['Husband'],
        'race': ['White'],
        'sex': ['Male'],
        'capital-gain': [0],
        'capital-loss': [0],
        'hours-per-week': [base_hours],
        'native-country': ['United-States']
    })
    
    mod_data = base_data.copy()
    mod_data['age'] = mod_age
    mod_data['education-num'] = mod_education
    mod_data['hours-per-week'] = mod_hours
    
    # Make predictions
    X_base, _ = preprocessor.preprocess(base_data, target_col='income', is_training=False)
    X_mod, _ = preprocessor.preprocess(mod_data, target_col='income', is_training=False)
    
    base_proba = model.predict_proba(X_base)[0][1]
    mod_proba = model.predict_proba(X_mod)[0][1]
    
    # Display comparison
    st.markdown("---")
    st.subheader("Comparison Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Baseline Probability", f"{base_proba:.1%}")
    with col2:
        st.metric("Modified Probability", f"{mod_proba:.1%}", 
                 delta=f"{(mod_proba - base_proba):.1%}")
    with col3:
        change = "Increased" if mod_proba > base_proba else "Decreased"
        st.metric("Change", change)
    
    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Baseline', 'Modified'],
        y=[base_proba * 100, mod_proba * 100],
        marker_color=['lightblue', 'lightcoral'],
        text=[f'{base_proba:.1%}', f'{mod_proba:.1%}'],
        textposition='auto',
    ))
    fig.update_layout(
        title='Probability of Income >50K Comparison',
        yaxis_title='Probability (%)',
        yaxis_range=[0, 100]
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit and Scikit-learn</p>
    <p>Model trained on UCI Adult Income Dataset</p>
</div>
""", unsafe_allow_html=True)