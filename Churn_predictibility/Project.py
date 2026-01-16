"""
Customer Churn Prediction with XGBoost
Complete ML Pipeline for 120K+ Customer Records
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, accuracy_score, precision_score, recall_score, 
                             f1_score, precision_recall_curve)
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
print("\nGenerating synthetic customer dataset (120,000 records)...\n")

def generate_customer_data(n_samples=120000):
    """Generating realistic synthetic customer churn data"""
    
    data = {
        # Customer Demographics
        'customerID': [f'CUST{i:06d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.52, 0.48]),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70]),
        
        # Service Information
        'tenure': np.random.randint(1, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.42, 0.48, 0.10]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.44, 0.21]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.28, 0.51, 0.21]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.45, 0.21]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.45, 0.21]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.29, 0.50, 0.21]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.38, 0.41, 0.21]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.39, 0.40, 0.21]),
        
        # Contract Information
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.21, 0.24]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41]),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
                                         n_samples, p=[0.34, 0.16, 0.22, 0.28]),
    }
    
    df = pd.DataFrame(data)
    
    # Generate monthly and total charges based on tenure
    base_charge = np.random.uniform(18, 40, n_samples)
    service_multiplier = np.random.uniform(1.0, 3.5, n_samples)
    df['MonthlyCharges'] = np.round(base_charge * service_multiplier, 2)
    df['TotalCharges'] = np.round(df['MonthlyCharges'] * df['tenure'] + np.random.normal(0, 50, n_samples), 2)
    df['TotalCharges'] = df['TotalCharges'].clip(lower=0)
    
    # Generate churn based on realistic patterns
    churn_prob = 0.25  # Base churn probability
    
    # Adjust probability based on features
    prob_adjustments = np.zeros(n_samples)
    prob_adjustments += (df['tenure'] < 12) * 0.25  # New customers more likely to churn
    prob_adjustments += (df['Contract'] == 'Month-to-month') * 0.20
    prob_adjustments += (df['PaymentMethod'] == 'Electronic check') * 0.10
    prob_adjustments += (df['TechSupport'] == 'No') * 0.08
    prob_adjustments += (df['OnlineSecurity'] == 'No') * 0.07
    prob_adjustments += (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)) * 0.12
    prob_adjustments -= (df['tenure'] > 48) * 0.20  # Long-term customers less likely to churn
    prob_adjustments -= (df['Contract'] == 'Two year') * 0.18
    prob_adjustments -= (df['Partner'] == 'Yes') * 0.05
    
    final_prob = np.clip(churn_prob + prob_adjustments, 0.05, 0.80)
    df['Churn'] = np.random.binomial(1, final_prob)
    df['Churn'] = df['Churn'].map({0: 'No', 1: 'Yes'})
    
    return df

# Generate dataset
df = generate_customer_data(120000)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset Info:")
print(df.info())
print(f"\nChurn Distribution:")
print(df['Churn'].value_counts())
print(f"Churn Rate: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.2f}%")

# ==================== DATA PREPROCESSING ====================
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Create a copy for processing
df_processed = df.copy()

# Drop customerID as it's not a feature
df_processed = df_processed.drop('customerID', axis=1)

# Handle missing values (if any)
print(f"\nMissing values:\n{df_processed.isnull().sum()}")

# Convert TotalCharges to numeric (in case of any issues)
df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)

# Create binary target variable
df_processed['Churn_Binary'] = (df_processed['Churn'] == 'Yes').astype(int)

# Feature Engineering
print("\nEngineering additional features...")

# Average monthly charge
df_processed['AvgMonthlyCharge'] = df_processed['TotalCharges'] / (df_processed['tenure'] + 1)

# Service count
service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df_processed['ServiceCount'] = 0
for col in service_cols:
    df_processed['ServiceCount'] += (df_processed[col].isin(['Yes', 'DSL', 'Fiber optic'])).astype(int)

# Tenure categories
df_processed['TenureCategory'] = pd.cut(df_processed['tenure'], 
                                        bins=[0, 12, 24, 48, 72], 
                                        labels=['0-12', '13-24', '25-48', '49-72'])

# Encode categorical variables
label_encoders = {}
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Churn')  # Don't encode target

print(f"\nEncoding {len(categorical_cols)} categorical features...")

for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col + '_Encoded'] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

# Select features for modeling
feature_cols = [col for col in df_processed.columns if col.endswith('_Encoded') or 
                col in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 
                       'AvgMonthlyCharge', 'ServiceCount']]

X = df_processed[feature_cols]
y = df_processed['Churn_Binary']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# ==================== TRAIN-TEST SPLIT ====================
print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print(f"\nTraining set churn rate: {y_train.sum() / len(y_train) * 100:.2f}%")
print(f"Testing set churn rate: {y_test.sum() / len(y_test) * 100:.2f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== MODEL TRAINING ====================
print("\n" + "=" * 60)
print("XGBOOST MODEL TRAINING")
print("=" * 60)

# Calculate scale_pos_weight for imbalanced classes
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nClass imbalance ratio: {scale_pos_weight:.2f}")

# Initialize XGBoost model
print("\nTraining XGBoost classifier...")
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

# Train the model
xgb_model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=False)

print("✓ Model training complete!")

# ==================== MODEL EVALUATION ====================
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Make predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n{'Metric':<20} {'Score':<10}")
print("-" * 30)
print(f"{'Accuracy':<20} {accuracy:.4f}")
print(f"{'Precision':<20} {precision:.4f}")
print(f"{'Recall':<20} {recall:.4f}")
print(f"{'F1-Score':<20} {f1:.4f}")
print(f"{'ROC-AUC':<20} {roc_auc:.4f}")

print(f"\n\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# Cross-validation
print(f"\n\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Cross-validation ROC-AUC scores: {cv_scores}")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ==================== FEATURE IMPORTANCE ====================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(20, 12))

# 1. Feature Importance Plot
ax1 = plt.subplot(2, 3, 1)
top_features = feature_importance.head(15)
plt.barh(top_features['feature'], top_features['importance'], color='steelblue')
plt.xlabel('Importance Score')
plt.title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# 2. Confusion Matrix Heatmap
ax2 = plt.subplot(2, 3, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

# 3. ROC Curve
ax3 = plt.subplot(2, 3, 3)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")

# 4. Precision-Recall Curve
ax4 = plt.subplot(2, 3, 4)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall_curve, precision_curve, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.grid(True)

# 5. Prediction Distribution
ax5 = plt.subplot(2, 3, 5)
plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.6, label='No Churn', color='blue')
plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.6, label='Churn', color='red')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
plt.legend()

# 6. Model Performance Metrics Bar Chart
ax6 = plt.subplot(2, 3, 6)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
scores = [accuracy, precision, recall, f1, roc_auc]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
bars = plt.bar(metrics, scores, color=colors)
plt.ylim([0, 1])
plt.ylabel('Score')
plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('churn_prediction_results.png', dpi=300, bbox_inches='tight')

perform_tuning = False  # Set to True to perform grid search

if perform_tuning:
    print("\nPerforming grid search for optimal hyperparameters...")
    print("This may take several minutes...\n")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    grid_search = GridSearchCV(
        XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight),
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Train final model with best parameters
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]
    
    print(f"\nTuned Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_best):.4f}")

# Save the model
import pickle

model_artifacts = {
    'model': xgb_model,
    'scaler': scaler,
    'feature_columns': feature_cols,
    'label_encoders': label_encoders
}

with open('churn_prediction_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

feature_importance.to_csv('feature_importance.csv', index=False)
print("\n" + "=" * 60)
print("EXAMPLE PREDICTION FUNCTION")
print("=" * 60)

def predict_churn(customer_data_dict, model_artifacts):
    """
    Predict churn for a new customer
    
    Parameters:
    customer_data_dict: dict with customer features
    model_artifacts: dict containing model, scaler, etc.
    
    Returns:
    dict with prediction and probability
    """
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    feature_cols = model_artifacts['feature_columns']
    
    # Create DataFrame from input
    customer_df = pd.DataFrame([customer_data_dict])
    
    # Apply same preprocessing
    # (In production, you'd want to handle this more robustly)
    
    # Extract features
    X_new = customer_df[feature_cols]
    
    # Scale
    X_new_scaled = scaler.transform(X_new)
    
    # Predict
    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0][1]
    
    return {
        'will_churn': bool(prediction),
        'churn_probability': float(probability),
        'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'
    }

# Test prediction
print("\nExample prediction:")
sample_customer = X_test.iloc[0].to_dict()
result = predict_churn(sample_customer, model_artifacts)
print(f"Customer churn prediction: {result}")
print(f"Actual churn: {y_test.iloc[0]}")

# ==================== SUMMARY ====================
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

summary = f"""
Dataset: 120,000 customer records
Training samples: {len(X_train):,}
Testing samples: {len(X_test):,}
Features: {X.shape[1]}

Model: XGBoost Classifier
  - Estimators: 200
  - Max depth: 6
  - Learning rate: 0.1

Performance Metrics:
  - Accuracy: {accuracy:.4f}
  - Precision: {precision:.4f}
  - Recall: {recall:.4f}
  - F1-Score: {f1:.4f}
  - ROC-AUC: {roc_auc:.4f}

Top 3 Most Important Features:
  1. {feature_importance.iloc[0]['feature']} ({feature_importance.iloc[0]['importance']:.4f})
  2. {feature_importance.iloc[1]['feature']} ({feature_importance.iloc[1]['importance']:.4f})
  3. {feature_importance.iloc[2]['feature']} ({feature_importance.iloc[2]['importance']:.4f})
"""