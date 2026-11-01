import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import (
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 10 Rows:")
print(df.head(10))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nChurn Distribution:")
print(df['Churn'].value_counts())


df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'].str.strip(),
    errors='coerce'
)

print("\nMissing TotalCharges After Conversion:")
print(df['TotalCharges'].isnull().sum())

# Replace blank values with 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)

print("\nMissing TotalCharges After Handling:")
print(df['TotalCharges'].isnull().sum())

print("\nTotalCharges Data Type:")
print(df['TotalCharges'].dtype)

df = df.drop('customerID', axis=1)

# Encode Churn: Yes = 1, No = 0
df['Churn'] = df['Churn'].map({
    'Yes': 1,
    'No': 0
})

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# One-Hot Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

print("\nEncoded Dataset Shape:")
print(X.shape)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

print("\nTraining Churn Distribution:")
print(y_train.value_counts())
gb_model = GradientBoostingClassifier(
    random_state=42
)

gb_model.fit(X_train, y_train)


# AdaBoost Classifier
ada_model = AdaBoostClassifier(
    random_state=42
)

ada_model.fit(X_train, y_train)


# XGBoost Classifier
xgb_model = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)


print("\nDefault Boosting Models Trained Successfully!")

# Parameter Grid
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}

# Grid Search with 5-Fold Cross Validation
grid_search = GridSearchCV(
    estimator=XGBClassifier(
        random_state=42,
        eval_metric='logloss'
    ),
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Display Best Parameters
print("\nBest Parameters:")
print(grid_search.best_params_)

# Display Best F1 Score
print("\nBest Cross-Validation F1 Score:")
print(grid_search.best_score_)

# Get Best Tuned Model
best_xgb = grid_search.best_estimator_

print("\nXGBoost Tuning Completed Successfully!")
models = {
    'Gradient Boosting': gb_model,
    'AdaBoost': ada_model,
    'Default XGBoost': xgb_model,
    'Tuned XGBoost': best_xgb
}

results = []

for name, model in models.items():

    # Predictions
    y_pred = model.predict(X_test)

    # Predicted probabilities for ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })


# Display Comparison Table
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

importance = best_xgb.feature_importances_

# Create DataFrame
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance
})

# Select Top 5 Features
top_5 = feature_importance.sort_values(
    by='Importance',
    ascending=False
).head(5)

print("\nTop 5 Important Features:")
print(top_5.to_string(index=False))

# Plot Feature Importance
plt.figure(figsize=(9, 5))

plt.barh(
    top_5['Feature'],
    top_5['Importance']
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 5 Important Features - Tuned XGBoost")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

print("\n========== FINAL CONCLUSION ==========")

best_model = results_df.loc[
    results_df['F1-Score'].idxmax()
]

print("Best Model:", best_model['Model'])
print("Best F1-Score:", round(best_model['F1-Score'], 4))
print("Best Accuracy:", round(best_model['Accuracy'], 4))

print("\nAccuracy alone is misleading because the dataset")
print("contains more customers who stayed than customers who left.")
print("A model may achieve high accuracy while failing to identify")
print("customers who are actually leaving.")
