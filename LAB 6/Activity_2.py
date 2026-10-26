import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# 1. Load dataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/jiangtianying/Kaggle-Project-Heart-Disease-UCI/master/heart.csv"
)
df.replace(["?", "NA", "N/A", ""], np.nan, inplace=True)

print("Dataset Shape:", df.shape)
print("\nFirst 10 Rows:")
print(df.head(10))

print("\nMissing Values:")
print(df.isnull().sum())

# Find target column
target_col = next(
    (c for c in ["target", "HeartDisease", "heart_disease",
                 "num", "condition"] if c in df.columns),
    None
)

if target_col is None:
    raise ValueError("Please set target_col to your dataset's target column.")

df.dropna(subset=[target_col], inplace=True)

print("\nTarget Class Distribution:")
print(df[target_col].value_counts().sort_index())

# 2-3. Separate features and target
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# Convert numeric-looking columns
for col in X.columns:
    if X[col].dtype == "object":
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().sum() >= X[col].notna().sum() * 0.8:
            X[col] = converted

# Fill missing values and encode categorical features
numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(exclude=np.number).columns

if len(numeric_cols) > 0:
    X[numeric_cols] = SimpleImputer(
        strategy="median"
    ).fit_transform(X[numeric_cols])

if len(categorical_cols) > 0:
    X[categorical_cols] = SimpleImputer(
        strategy="most_frequent"
    ).fit_transform(X[categorical_cols])

    X = pd.get_dummies(
        X, columns=list(categorical_cols), drop_first=False
    )

if not pd.api.types.is_numeric_dtype(y):
    y = y.astype("category").cat.codes

# 4. Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# 5. Basic Random Forest
basic_model = RandomForestClassifier(random_state=42)
basic_model.fit(X_train, y_train)

basic_pred = basic_model.predict(X_test)
print("\nBasic Random Forest Accuracy:",
      round(accuracy_score(y_test, basic_pred), 4))

# 6. Parameter grid from the given table
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Hyperparameters:")
print(grid_search.best_params_)

print("Best Cross-validation Accuracy:",
      round(grid_search.best_score_, 4))

# 7. Evaluate best model
y_pred = best_model.predict(X_test)

print("\nTest Set Results:")
print("Accuracy:",
      round(accuracy_score(y_test, y_pred), 4))
print("Precision:",
      round(precision_score(y_test, y_pred, average="weighted",
                            zero_division=0), 4))
print("Recall:",
      round(recall_score(y_test, y_pred, average="weighted",
                         zero_division=0), 4))
print("F1-score:",
      round(f1_score(y_test, y_pred, average="weighted",
                     zero_division=0), 4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=best_model.classes_
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 8. Feature importance
importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 3 Important Features:")
print(importances.head(3))

importances.head(15).sort_values().plot(
    kind="barh", figsize=(10, 6)
)
plt.xlabel("Feature Importance")
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()
