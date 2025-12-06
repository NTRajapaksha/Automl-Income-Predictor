"""
Streamlit Web Interface for AutoML Pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
from automl_pipeline import AdvancedAutoML
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="Advanced AutoML", page_icon="🤖", layout="wide")

# Title and description
st.title("🤖 Advanced AutoML Pipeline")
st.markdown("""
This interactive dashboard runs a complete AutoML pipeline that:
- Preprocesses your data automatically
- Trains multiple ML models with hyperparameter tuning
- Compares model performance
- Provides detailed insights and visualizations
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Data source selection
data_source = st.sidebar.radio(
    "Select Data Source",
    ["Default (Adult Income)", "Upload CSV"]
)

# Initialize session state
if 'automl' not in st.session_state:
    st.session_state.automl = None
if 'results' not in st.session_state:
    st.session_state.results = None

# File upload
uploaded_file = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Select target column
        target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    else:
        st.info("👆 Please upload a CSV file to continue")
        st.stop()
else:
    target_col = 'income'

# Task type
task_type = st.sidebar.selectbox("Task Type", ["classification", "regression"])

# Run button
if st.sidebar.button("🚀 Run AutoML Pipeline", type="primary"):
    with st.spinner("Running AutoML pipeline... This may take a few minutes."):
        try:
            # Initialize AutoML
            automl = AdvancedAutoML(target_column=target_col, task_type=task_type)
            
            # Load data
            if data_source == "Upload CSV":
                automl.df = pd.read_csv(uploaded_file)
            else:
                automl.load_data()
            
            # Run pipeline
            progress_bar = st.progress(0)
            st.text("Preprocessing data...")
            automl.preprocess()
            progress_bar.progress(20)
            
            st.text("Defining models...")
            automl.define_models()
            progress_bar.progress(40)
            
            st.text("Training and tuning models...")
            automl.train_and_tune()
            progress_bar.progress(80)
            
            st.text("Generating visualizations...")
            automl.visualize_results()
            progress_bar.progress(100)
            
            # Store in session state
            st.session_state.automl = automl
            st.session_state.results = automl.results
            
            st.success("✅ AutoML pipeline completed successfully!")
            st.balloons()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()

# Display results if available
if st.session_state.results:
    automl = st.session_state.automl
    results = st.session_state.results
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏆 Best Model", "📈 Comparisons", "🔍 Details"])
    
    with tab1:
        st.header("Pipeline Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Model", automl.best_model)
        with col2:
            st.metric("Best Accuracy", f"{automl.best_score:.4f}")
        with col3:
            st.metric("Models Trained", len(results))
        
        # Dataset info
        st.subheader("Dataset Information")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", automl.df.shape[0])
        col2.metric("Features", automl.df.shape[1] - 1)
        col3.metric("Training Samples", automl.X_train.shape[0])
        col4.metric("Test Samples", automl.X_test.shape[0])
        
        # Quick comparison
        st.subheader("Model Performance Summary")
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Test Accuracy': [r['test_accuracy'] for r in results.values()],
            'CV Score': [r['cv_score'] for r in results.values()],
            'AUC': [r['auc'] for r in results.values()]
        }).sort_values('Test Accuracy', ascending=False)
        
        st.dataframe(comparison_df, use_container_width=True)
    
    with tab2:
        st.header(f"Best Model: {automl.best_model}")
        
        best_result = results[automl.best_model]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Metrics")
            st.metric("Test Accuracy", f"{best_result['test_accuracy']:.4f}")
            st.metric("Cross-Validation Score", f"{best_result['cv_score']:.4f}")
            st.metric("AUC Score", f"{best_result['auc']:.4f}")
            
            st.subheader("Best Hyperparameters")
            st.json(best_result['best_params'])
        
        with col2:
            st.subheader("Confusion Matrix")
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(automl.y_test, best_result['predictions'])
            
            fig = px.imshow(cm, 
                          labels=dict(x="Predicted", y="Actual", color="Count"),
                          x=['Class 0', 'Class 1'],
                          y=['Class 0', 'Class 1'],
                          text_auto=True,
                          color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        if automl.best_model in ['Random Forest', 'Gradient Boosting']:
            st.subheader("Feature Importance")
            feature_imp = pd.DataFrame({
                'Feature': automl.X_train.columns,
                'Importance': best_result['model'].feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            
            fig = px.bar(feature_imp, x='Importance', y='Feature', 
                        orientation='h',
                        title='Top 15 Most Important Features')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Model Comparisons")
        
        # Accuracy comparison
        fig1 = px.bar(comparison_df, x='Model', y='Test Accuracy',
                     title='Test Accuracy Comparison',
                     color='Test Accuracy',
                     color_continuous_scale='Viridis')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Multi-metric comparison
        metrics_df = comparison_df.melt(id_vars=['Model'], 
                                       value_vars=['Test Accuracy', 'CV Score', 'AUC'],
                                       var_name='Metric', value_name='Score')
        
        fig2 = px.bar(metrics_df, x='Model', y='Score', color='Metric',
                     title='Multi-Metric Comparison',
                     barmode='group')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Radar chart
        fig3 = go.Figure()
        for model in comparison_df['Model']:
            fig3.add_trace(go.Scatterpolar(
                r=[comparison_df[comparison_df['Model']==model]['Test Accuracy'].values[0],
                   comparison_df[comparison_df['Model']==model]['CV Score'].values[0],
                   comparison_df[comparison_df['Model']==model]['AUC'].values[0]],
                theta=['Test Accuracy', 'CV Score', 'AUC'],
                fill='toself',
                name=model
            ))
        
        fig3.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Model Performance Radar Chart'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab4:
        st.header("Detailed Results")
        
        selected_model = st.selectbox("Select Model for Details", list(results.keys()))
        
        if selected_model:
            model_result = results[selected_model]
            
            st.subheader(f"{selected_model} Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Best Hyperparameters:**")
                st.json(model_result['best_params'])
                
                st.write("**Performance Metrics:**")
                st.write(f"- Test Accuracy: {model_result['test_accuracy']:.4f}")
                st.write(f"- CV Score: {model_result['cv_score']:.4f}")
                st.write(f"- AUC: {model_result['auc']:.4f}")
            
            with col2:
                from sklearn.metrics import classification_report
                st.write("**Classification Report:**")
                report = classification_report(automl.y_test, 
                                              model_result['predictions'],
                                              output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
        
        # Download results
        st.subheader("Download Results")
        
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Table",
            data=csv,
            file_name='automl_results.csv',
            mime='text/csv'
        )

else:
    st.info("👈 Configure your settings and click 'Run AutoML Pipeline' to get started!")
    
    # Show example
    st.subheader("Example Output Preview")
    st.image("https://via.placeholder.com/800x400.png?text=Model+Comparison+Charts+Will+Appear+Here", 
             caption="Visualizations will appear here after running the pipeline")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Scikit-learn")