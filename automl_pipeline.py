"""
Advanced AutoML Pipeline for UCI Adult Income Dataset
Predicts whether income exceeds $50K/year based on census data

Features:
- Intelligent preprocessing for mixed data types
- Multiple ML algorithms with hyperparameter tuning
- SHAP explainability analysis
- Comprehensive visualizations
- Automated feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not available. Install with: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Note: SHAP not available. Install with: pip install shap")


class AdvancedAutoML:
    def __init__(self, target_column='income', task_type='classification'):
        """
        Initialize AutoML pipeline for UCI Adult Income dataset
        
        Parameters:
        -----------
        target_column : str
            Name of target variable (default: 'income')
        task_type : str
            'classification' or 'regression'
        """
        self.target_column = target_column
        self.task_type = task_type
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.feature_importance = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self):
        """Load UCI Adult Income dataset"""
        print("="*70)
        print("LOADING UCI ADULT INCOME DATASET")
        print("="*70)
        
        # Column names for Adult dataset
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        
        # Load training data
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        self.df = pd.read_csv(url, names=columns, skipinitialspace=True, na_values='?')
        
        print(f"\n✓ Dataset loaded successfully")
        print(f"  • Total samples: {self.df.shape[0]:,}")
        print(f"  • Features: {self.df.shape[1] - 1}")
        print(f"  • Target: {self.target_column}")
        
        # Display target distribution
        print(f"\n📊 Target Distribution:")
        target_counts = self.df[self.target_column].value_counts()
        for label, count in target_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  • {label}: {count:,} ({percentage:.1f}%)")
        
        # Display missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️  Missing Values Detected:")
            for col, count in missing[missing > 0].items():
                percentage = (count / len(self.df)) * 100
                print(f"  • {col}: {count:,} ({percentage:.1f}%)")
        
        return self.df
    
    def engineer_features(self):
        """Create additional features from existing ones"""
        print("\n🔧 Engineering new features...")
        
        # Age groups
        self.df['age_group'] = pd.cut(self.df['age'], 
                                       bins=[0, 25, 35, 45, 55, 100],
                                       labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])
        
        # Capital features
        self.df['has_capital_gain'] = (self.df['capital-gain'] > 0).astype(int)
        self.df['has_capital_loss'] = (self.df['capital-loss'] > 0).astype(int)
        self.df['capital_total'] = self.df['capital-gain'] - self.df['capital-loss']
        
        # Work hours categories
        self.df['hours_category'] = pd.cut(self.df['hours-per-week'],
                                           bins=[0, 20, 40, 60, 100],
                                           labels=['Part-time', 'Full-time', 'Overtime', 'Excessive'])
        
        # Education level simplification
        education_mapping = {
            'Preschool': 'Dropout', '1st-4th': 'Dropout', '5th-6th': 'Dropout',
            '7th-8th': 'Dropout', '9th': 'Dropout', '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout',
            'HS-grad': 'High School', 'Some-college': 'Higher Ed', 'Assoc-voc': 'Higher Ed',
            'Assoc-acdm': 'Higher Ed', 'Bachelors': 'Bachelors', 'Masters': 'Masters',
            'Prof-school': 'Professional', 'Doctorate': 'Doctorate'
        }
        self.df['education_level'] = self.df['education'].map(education_mapping)
        
        print("  ✓ Created 7 new features")
        
    def preprocess(self):
        """Automated data preprocessing with domain-specific handling"""
        print("\n" + "="*70)
        print("PREPROCESSING DATA")
        print("="*70)
        
        # Engineer features first
        self.engineer_features()
        
        # Handle missing values intelligently
        print("\n🔄 Handling missing values...")
        
        # For categorical columns, fill with mode
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col != self.target_column and self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                self.df[col].fillna(mode_val, inplace=True)
        
        # For numerical columns, fill with median
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if col != self.target_column and self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        print(f"  ✓ Missing values handled")
        
        # Separate features and target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Clean target variable (remove any whitespace and convert to binary)
        y = y.str.strip()
        y = (y == '>50K').astype(int)  # 1 for >50K, 0 for <=50K
        
        print(f"\n📋 Feature Types:")
        print(f"  • Numerical features: {len(X.select_dtypes(include=['int64', 'float64']).columns)}")
        print(f"  • Categorical features: {len(X.select_dtypes(include=['object', 'category']).columns)}")
        
        # Encode categorical features
        print("\n🔤 Encoding categorical variables...")
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        print(f"  ✓ Encoded {len(categorical_cols)} categorical columns")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data with stratification
        print("\n✂️  Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  • Training set: {self.X_train.shape[0]:,} samples")
        print(f"  • Test set: {self.X_test.shape[0]:,} samples")
        print(f"  • Train class distribution: {np.bincount(self.y_train)}")
        print(f"  • Test class distribution: {np.bincount(self.y_test)}")
        
        # Scale features
        print("\n⚖️  Scaling features...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("  ✓ Features scaled using StandardScaler")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def define_models(self):
        """Define candidate models with optimized hyperparameter grids"""
        print("\n" + "="*70)
        print("DEFINING MODEL CANDIDATES")
        print("="*70)
        
        self.models = {
            'Logistic Regression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs']
                },
                'needs_scaling': True
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'needs_scaling': False
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                },
                'needs_scaling': False
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'needs_scaling': True
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsClassifier(n_jobs=-1),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                'needs_scaling': True
            },
            'Naive Bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7]
                },
                'needs_scaling': True
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'criterion': ['gini', 'entropy']
                },
                'needs_scaling': False
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                },
                'needs_scaling': False
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = {
                'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0]
                },
                'needs_scaling': False
            }
        
        print(f"\n✓ Defined {len(self.models)} models:")
        for i, name in enumerate(self.models.keys(), 1):
            print(f"  {i}. {name}")
    
    def train_and_tune(self):
        """Train all models with hyperparameter tuning and cross-validation"""
        print("\n" + "="*70)
        print("TRAINING AND TUNING MODELS")
        print("="*70)
        
        # Use stratified k-fold
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for idx, (name, model_info) in enumerate(self.models.items(), 1):
            print(f"\n[{idx}/{len(self.models)}] Training {name}...")
            print("-" * 50)
            
            try:
                # GridSearchCV for hyperparameter tuning
                grid_search = GridSearchCV(
                    model_info['model'],
                    model_info['params'],
                    cv=cv_strategy,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0
                )
                
                # Choose scaled or unscaled data
                if model_info['needs_scaling']:
                    X_train_use = self.X_train_scaled
                    X_test_use = self.X_test_scaled
                else:
                    X_train_use = self.X_train
                    X_test_use = self.X_test
                
                # Fit model
                grid_search.fit(X_train_use, self.y_train)
                
                # Make predictions
                y_pred = grid_search.predict(X_test_use)
                y_pred_proba = grid_search.predict_proba(X_test_use)
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                f1 = f1_score(self.y_test, y_pred)
                
                # Store results
                self.results[name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'cv_score': grid_search.best_score_,
                    'test_accuracy': accuracy,
                    'auc': auc,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'predictions_proba': y_pred_proba,
                    'needs_scaling': model_info['needs_scaling']
                }
                
                print(f"  ✓ Best Params: {grid_search.best_params_}")
                print(f"  ✓ CV AUC Score: {grid_search.best_score_:.4f}")
                print(f"  ✓ Test Accuracy: {accuracy:.4f}")
                print(f"  ✓ Test AUC: {auc:.4f}")
                print(f"  ✓ Test F1: {f1:.4f}")
                
                # Track best model (by AUC)
                if auc > self.best_score:
                    self.best_score = auc
                    self.best_model = name
                    
            except Exception as e:
                print(f"  ✗ Error training {name}: {str(e)}")
                continue
    
    def generate_shap_analysis(self):
        """Generate SHAP explanations for the best model"""
        if not SHAP_AVAILABLE:
            print("\n⚠️  SHAP not available. Skipping explainability analysis.")
            return
        
        print("\n" + "="*70)
        print("GENERATING SHAP EXPLAINABILITY ANALYSIS")
        print("="*70)
        
        try:
            best_model_obj = self.results[self.best_model]['model']
            
            # Use appropriate data (scaled or not)
            if self.results[self.best_model]['needs_scaling']:
                X_sample = self.X_test_scaled[:1000]
            else:
                X_sample = self.X_test.values[:1000]
            
            print(f"\n🔍 Analyzing {self.best_model}...")
            
            # Create explainer based on model type
            if self.best_model in ['Random Forest', 'XGBoost', 'Gradient Boosting', 'Decision Tree']:
                explainer = shap.TreeExplainer(best_model_obj)
            else:
                explainer = shap.KernelExplainer(best_model_obj.predict_proba, X_sample[:100])
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # For binary classification, use positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Create SHAP summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, 
                            feature_names=self.feature_names,
                            show=False, max_display=15)
            plt.title(f'SHAP Feature Importance - {self.best_model}', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("  ✓ SHAP analysis complete")
            print("  ✓ Saved: shap_summary.png")
            
        except Exception as e:
            print(f"  ✗ Error generating SHAP analysis: {str(e)}")
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "="*70)
        print("FINAL AUTOML REPORT")
        print("="*70)
        
        # Model comparison table
        print("\n📊 MODEL PERFORMANCE COMPARISON")
        print("-" * 70)
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'CV AUC': [r['cv_score'] for r in self.results.values()],
            'Test Accuracy': [r['test_accuracy'] for r in self.results.values()],
            'Test AUC': [r['auc'] for r in self.results.values()],
            'F1 Score': [r['f1_score'] for r in self.results.values()]
        }).sort_values('Test AUC', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
        # Best model details
        print(f"\n🏆 BEST MODEL: {self.best_model}")
        print("-" * 70)
        print(f"  • Test AUC: {self.results[self.best_model]['auc']:.4f}")
        print(f"  • Test Accuracy: {self.results[self.best_model]['test_accuracy']:.4f}")
        print(f"  • F1 Score: {self.results[self.best_model]['f1_score']:.4f}")
        print(f"  • CV AUC: {self.results[self.best_model]['cv_score']:.4f}")
        print(f"\n  Hyperparameters:")
        for param, value in self.results[self.best_model]['best_params'].items():
            print(f"    - {param}: {value}")
        
        # Detailed classification report
        print(f"\n📈 DETAILED CLASSIFICATION REPORT ({self.best_model})")
        print("-" * 70)
        y_pred_best = self.results[self.best_model]['predictions']
        print(classification_report(self.y_test, y_pred_best, 
                                   target_names=['<=50K', '>50K']))
        
        # Feature importance for tree-based models
        if self.best_model in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Decision Tree', 'AdaBoost']:
            print(f"\n🔍 TOP 15 MOST IMPORTANT FEATURES")
            print("-" * 70)
            best_model_obj = self.results[self.best_model]['model']
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': best_model_obj.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            
            for idx, row in feature_importance.iterrows():
                print(f"  {row['Feature']:.<45} {row['Importance']:.4f}")
        
        print("\n" + "="*70)
        return comparison_df
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Model comparison - AUC
        ax1 = fig.add_subplot(gs[0, :2])
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'AUC': [r['auc'] for r in self.results.values()]
        }).sort_values('AUC')
        
        bars = ax1.barh(comparison['Model'], comparison['AUC'])
        ax1.set_xlabel('AUC Score', fontsize=12)
        ax1.set_title('Model Performance Comparison (AUC)', fontsize=14, fontweight='bold')
        ax1.set_xlim([0.5, 1.0])
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.3, label='Random Guess')
        
        # Color best model
        for i, bar in enumerate(bars):
            if comparison.iloc[i]['Model'] == self.best_model:
                bar.set_color('gold')
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
        
        # Add value labels
        for i, (idx, row) in enumerate(comparison.iterrows()):
            ax1.text(row['AUC'] + 0.01, i, f"{row['AUC']:.4f}", 
                    va='center', fontsize=9)
        
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Confusion Matrix
        ax2 = fig.add_subplot(gs[0, 2])
        cm = confusion_matrix(self.y_test, self.results[self.best_model]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
        ax2.set_title(f'Confusion Matrix\n{self.best_model}', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        ax2.set_yticklabels(['<=50K', '>50K'], rotation=0)
        ax2.set_xticklabels(['<=50K', '>50K'])
        
        # 3. ROC Curve
        ax3 = fig.add_subplot(gs[1, 0])
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['predictions_proba'][:, 1])
            if name == self.best_model:
                ax3.plot(fpr, tpr, label=f'{name} (AUC={result["auc"]:.3f})', 
                        linewidth=3, color='gold')
            else:
                ax3.plot(fpr, tpr, label=f'{name} (AUC={result["auc"]:.3f})', 
                        alpha=0.5, linewidth=1)
        
        ax3.plot([0, 1], [0, 1], 'k--', label='Random Guess', alpha=0.3)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curves', fontsize=12, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=8)
        ax3.grid(alpha=0.3)
        
        # 4. Precision-Recall Curve
        ax4 = fig.add_subplot(gs[1, 1])
        for name, result in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, 
                                                          result['predictions_proba'][:, 1])
            if name == self.best_model:
                ax4.plot(recall, precision, label=name, linewidth=3, color='gold')
            else:
                ax4.plot(recall, precision, label=name, alpha=0.5, linewidth=1)
        
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(alpha=0.3)
        
        # 5. Feature Importance
        ax5 = fig.add_subplot(gs[1, 2])
        if self.best_model in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Decision Tree', 'AdaBoost']:
            best_model_obj = self.results[self.best_model]['model']
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model_obj.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)
            
            ax5.barh(feature_imp['feature'], feature_imp['importance'], color='steelblue')
            ax5.set_xlabel('Importance')
            ax5.set_title(f'Top 10 Features\n{self.best_model}', fontsize=12, fontweight='bold')
            ax5.grid(axis='x', alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model',
                    ha='center', va='center', fontsize=12)
            ax5.axis('off')
        
        # 6. Multi-metric comparison
        ax6 = fig.add_subplot(gs[2, :])
        metrics_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [r['test_accuracy'] for r in self.results.values()],
            'AUC': [r['auc'] for r in self.results.values()],
            'F1 Score': [r['f1_score'] for r in self.results.values()]
        })
        
        x = np.arange(len(metrics_df))
        width = 0.25
        
        bars1 = ax6.bar(x - width, metrics_df['Accuracy'], width, label='Accuracy', alpha=0.8)
        bars2 = ax6.bar(x, metrics_df['AUC'], width, label='AUC', alpha=0.8)
        bars3 = ax6.bar(x + width, metrics_df['F1 Score'], width, label='F1 Score', alpha=0.8)
        
        # Highlight best model
        best_idx = metrics_df[metrics_df['Model'] == self.best_model].index[0]
        bars1[best_idx].set_edgecolor('gold')
        bars1[best_idx].set_linewidth(3)
        bars2[best_idx].set_edgecolor('gold')
        bars2[best_idx].set_linewidth(3)
        bars3[best_idx].set_edgecolor('gold')
        bars3[best_idx].set_linewidth(3)
        
        ax6.set_ylabel('Score')
        ax6.set_title('Multi-Metric Performance Comparison', fontsize=14, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
        ax6.set_ylim([0, 1])
        
        plt.savefig('automl_results.png', dpi=300, bbox_inches='tight')
        print("\n  ✓ Saved: automl_results.png")
        plt.close()
        
    def run_pipeline(self):
        """Execute complete AutoML pipeline"""
        print("\n" + "="*70)
        print("🚀 ADVANCED AUTOML PIPELINE - UCI ADULT INCOME")
        print("="*70)
        
        # Load and preprocess
        self.load_data()
        self.preprocess()
        
        # Define and train models
        self.define_models()
        self.train_and_tune()
        
        # Generate SHAP analysis
        self.generate_shap_analysis()
        
        # Generate report and visualizations
        self.generate_report()
        self.visualize_results()
        
        print("\n" + "="*70)
        print("✅ AUTOML PIPELINE COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  • automl_results.png - Comprehensive visualizations")
        print("  • shap_summary.png - SHAP explainability analysis (if available)")
        print(f"\n🎯 Best Model: {self.best_model}")
        print(f"🎯 Test AUC: {self.best_score:.4f}")
        
        return self.results


# Main execution
if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         ADVANCED AUTOML FOR UCI ADULT INCOME DATASET          ║
    ║                                                               ║
    ║  Task: Predict whether income exceeds $50K/year               ║
    ║  Dataset: UCI Adult Income (Census Data)                      ║
    ║  Models: 8+ algorithms with hyperparameter tuning             ║
    ║  Features: Explainability, visualization, comprehensive       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize AutoML
    automl = AdvancedAutoML(target_column='income', task_type='classification')
    
    # Run complete pipeline
    results = automl.run_pipeline()
    
    # Example: Making predictions with best model
    print("\n" + "="*70)
    print("USAGE EXAMPLE")
    print("="*70)
    print("\nTo use the best model for predictions:")
    print(f"""
# Get the best model
best_model = automl.results['{automl.best_model}']['model']

# Prepare new data (same preprocessing)
# new_data = ... your preprocessed data ...

# Make predictions
if automl.results['{automl.best_model}']['needs_scaling']:
    predictions = best_model.predict(automl.scaler.transform(new_data))
else:
    predictions = best_model.predict(new_data)

# Get probability predictions
if automl.results['{automl.best_model}']['needs_scaling']:
    probabilities = best_model.predict_proba(automl.scaler.transform(new_data))
else:
    probabilities = best_model.predict_proba(new_data)
    """)