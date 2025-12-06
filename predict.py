"""
Example script for making predictions with trained AutoML model
Demonstrates how to use the model for new predictions
"""

import pandas as pd
import numpy as np
import joblib
from automl_pipeline import AdvancedAutoML

def train_and_save_model():
    """Train model and save for later use"""
    print("="*70)
    print("TRAINING AND SAVING MODEL")
    print("="*70)
    
    # Initialize and train
    automl = AdvancedAutoML(target_column='income')
    automl.run_pipeline()
    
    # Save best model and preprocessing objects
    best_model = automl.results[automl.best_model]['model']
    needs_scaling = automl.results[automl.best_model]['needs_scaling']
    
    print("\n💾 Saving model and preprocessing objects...")
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(automl.scaler, 'scaler.pkl')
    joblib.dump(automl.label_encoders, 'label_encoders.pkl')
    joblib.dump(automl.feature_names, 'feature_names.pkl')
    
    # Save metadata
    metadata = {
        'model_name': automl.best_model,
        'needs_scaling': needs_scaling,
        'test_auc': automl.best_score,
        'test_accuracy': automl.results[automl.best_model]['test_accuracy'],
        'feature_names': automl.feature_names
    }
    joblib.dump(metadata, 'model_metadata.pkl')
    
    print("✅ Saved:")
    print("  • best_model.pkl")
    print("  • scaler.pkl")
    print("  • label_encoders.pkl")
    print("  • feature_names.pkl")
    print("  • model_metadata.pkl")
    
    return automl

def load_model():
    """Load trained model and preprocessing objects"""
    print("\n📂 Loading model...")
    
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
    metadata = joblib.load('model_metadata.pkl')
    
    print(f"✅ Loaded model: {metadata['model_name']}")
    print(f"   AUC: {metadata['test_auc']:.4f}")
    print(f"   Accuracy: {metadata['test_accuracy']:.4f}")
    
    return model, scaler, label_encoders, feature_names, metadata

def create_sample_input():
    """Create sample input data for prediction"""
    # Example: Single person's information
    sample_data = {
        'age': [35],
        'workclass': ['Private'],
        'fnlwgt': [215646],
        'education': ['Bachelors'],
        'education-num': [13],
        'marital-status': ['Married-civ-spouse'],
        'occupation': ['Exec-managerial'],
        'relationship': ['Husband'],
        'race': ['White'],
        'sex': ['Male'],
        'capital-gain': [0],
        'capital-loss': [0],
        'hours-per-week': [40],
        'native-country': ['United-States']
    }
    
    return pd.DataFrame(sample_data)

def preprocess_new_data(df, label_encoders, feature_names):
    """Preprocess new data using saved encoders"""
    print("\n🔧 Preprocessing new data...")
    
    # Create engineered features (same as training)
    df['age_group'] = pd.cut(df['age'], 
                              bins=[0, 25, 35, 45, 55, 100],
                              labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])
    
    df['has_capital_gain'] = (df['capital-gain'] > 0).astype(int)
    df['has_capital_loss'] = (df['capital-loss'] > 0).astype(int)
    df['capital_total'] = df['capital-gain'] - df['capital-loss']
    
    df['hours_category'] = pd.cut(df['hours-per-week'],
                                   bins=[0, 20, 40, 60, 100],
                                   labels=['Part-time', 'Full-time', 'Overtime', 'Excessive'])
    
    education_mapping = {
        'Preschool': 'Dropout', '1st-4th': 'Dropout', '5th-6th': 'Dropout',
        '7th-8th': 'Dropout', '9th': 'Dropout', '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout',
        'HS-grad': 'High School', 'Some-college': 'Higher Ed', 'Assoc-voc': 'Higher Ed',
        'Assoc-acdm': 'Higher Ed', 'Bachelors': 'Bachelors', 'Masters': 'Masters',
        'Prof-school': 'Professional', 'Doctorate': 'Doctorate'
    }
    df['education_level'] = df['education'].map(education_mapping)
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col in label_encoders:
            # Handle unseen categories
            df[col] = df[col].astype(str)
            known_labels = set(label_encoders[col].classes_)
            df[col] = df[col].apply(lambda x: x if x in known_labels else label_encoders[col].classes_[0])
            df[col] = label_encoders[col].transform(df[col])
        else:
            # New column - use simple encoding
            df[col] = df[col].astype('category').cat.codes
    
    # Ensure same features as training
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Select only training features in correct order
    df = df[feature_names]
    
    print(f"✅ Preprocessed {len(df)} samples with {len(feature_names)} features")
    
    return df

def make_prediction(model, scaler, data, needs_scaling):
    """Make predictions on new data"""
    print("\n🎯 Making predictions...")
    
    # Scale if needed
    if needs_scaling:
        data_processed = scaler.transform(data)
    else:
        data_processed = data.values if isinstance(data, pd.DataFrame) else data
    
    # Predict
    predictions = model.predict(data_processed)
    probabilities = model.predict_proba(data_processed)
    
    return predictions, probabilities

def interpret_prediction(predictions, probabilities):
    """Interpret and display predictions"""
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"\nSample {i+1}:")
        print(f"  Predicted Income Class: {'> $50K' if pred == 1 else '<= $50K'}")
        print(f"  Confidence:")
        print(f"    • <= $50K: {prob[0]:.2%}")
        print(f"    • > $50K:  {prob[1]:.2%}")
        
        if pred == 1 and prob[1] > 0.7:
            print(f"  ✅ HIGH confidence: Likely earns over $50K")
        elif pred == 1 and prob[1] > 0.5:
            print(f"  ⚠️  MODERATE confidence: Might earn over $50K")
        elif pred == 0 and prob[0] > 0.7:
            print(f"  ✅ HIGH confidence: Likely earns $50K or less")
        else:
            print(f"  ⚠️  MODERATE confidence: Might earn $50K or less")

def batch_predict(csv_file):
    """Make predictions on a CSV file"""
    print(f"\n📄 Loading data from {csv_file}...")
    
    # Load model
    model, scaler, label_encoders, feature_names, metadata = load_model()
    
    # Load new data
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df)} samples")
    
    # Preprocess
    df_processed = preprocess_new_data(df.copy(), label_encoders, feature_names)
    
    # Predict
    predictions, probabilities = make_prediction(
        model, scaler, df_processed, metadata['needs_scaling']
    )
    
    # Add results to dataframe
    df['predicted_income'] = ['> $50K' if p == 1 else '<= $50K' for p in predictions]
    df['probability_high_income'] = probabilities[:, 1]
    df['confidence'] = probabilities.max(axis=1)
    
    # Save results
    output_file = csv_file.replace('.csv', '_predictions.csv')
    df.to_csv(output_file, index=False)
    print(f"✅ Saved predictions to {output_file}")
    
    # Summary
    print("\n" + "="*70)
    print("BATCH PREDICTION SUMMARY")
    print("="*70)
    print(f"Total Samples: {len(df)}")
    print(f"Predicted > $50K: {(predictions == 1).sum()} ({(predictions == 1).sum()/len(df)*100:.1f}%)")
    print(f"Predicted <= $50K: {(predictions == 0).sum()} ({(predictions == 0).sum()/len(df)*100:.1f}%)")
    print(f"Average Confidence: {probabilities.max(axis=1).mean():.2%}")
    
    return df

# Main execution
if __name__ == "__main__":
    import sys
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║              INCOME PREDICTION EXAMPLE SCRIPT                  ║
    ║                                                               ║
    ║  Usage:                                                       ║
    ║    1. Train and save: python predict.py train                 ║
    ║    2. Single predict: python predict.py predict               ║
    ║    3. Batch predict:  python predict.py batch <file.csv>      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) < 2:
        mode = 'predict'  # Default mode
    else:
        mode = sys.argv[1]
    
    if mode == 'train':
        # Train and save model
        automl = train_and_save_model()
        
    elif mode == 'predict':
        try:
            # Load model
            model, scaler, label_encoders, feature_names, metadata = load_model()
            
            # Create sample input
            print("\n📝 Creating sample input...")
            sample_df = create_sample_input()
            print("\nInput data:")
            print(sample_df.to_string(index=False))
            
            # Preprocess
            sample_processed = preprocess_new_data(
                sample_df.copy(), label_encoders, feature_names
            )
            
            # Predict
            predictions, probabilities = make_prediction(
                model, scaler, sample_processed, metadata['needs_scaling']
            )
            
            # Display results
            interpret_prediction(predictions, probabilities)
            
        except FileNotFoundError:
            print("❌ Model files not found!")
            print("   Please run: python predict.py train")
            
    elif mode == 'batch' and len(sys.argv) > 2:
        try:
            csv_file = sys.argv[2]
            results = batch_predict(csv_file)
            print("\n✅ Batch prediction complete!")
            
        except FileNotFoundError:
            print(f"❌ File not found: {sys.argv[2]}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
    else:
        print("❌ Invalid arguments!")
        print("   Usage: python predict.py [train|predict|batch <file.csv>]")
    
    print("\n" + "="*70)
    print("🎉 Done!")
    print("="*70)