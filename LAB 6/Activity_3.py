import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load dataset
df = pd.read_csv("train.csv")

# 2. Display dataset information
print("Dataset Shape:", df.shape)

print("\nFirst 10 Rows:")
print(df.head(10))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSalePrice Summary Statistics:")
print(df["SalePrice"].describe())

# 3. Separate features and target
X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
y = df["SalePrice"]

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# 4. Handle missing values and encode categorical data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Function to calculate evaluation metrics
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2 Score": r2_score(y_true, y_pred)
    }

# 6. Train basic Random Forest Regressor
basic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

basic_model.fit(X_train, y_train)

basic_pred = basic_model.predict(X_test)

basic_metrics = evaluate_model(y_test, basic_pred)

print("\nBasic Random Forest Results:")
for metric, value in basic_metrics.items():
    print(f"{metric}: {value:,.4f}")

# 7. Define parameter grid
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": [0.7, 1.0]
}

# Apply GridSearchCV
grid_search = GridSearchCV(
    estimator=basic_model,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Hyperparameters:")
print(grid_search.best_params_)

# Convert negative MSE into positive MSE and calculate RMSE
best_cv_mse = -grid_search.best_score_
best_cv_rmse = np.sqrt(best_cv_mse)

print("Best Cross-Validation MSE:", round(best_cv_mse, 2))
print("Best Cross-Validation RMSE:", round(best_cv_rmse, 2))

# 8. Evaluate optimized model on test data
best_pred = best_model.predict(X_test)

best_metrics = evaluate_model(y_test, best_pred)

print("\nOptimized Random Forest Results:")
for metric, value in best_metrics.items():
    print(f"{metric}: {value:,.4f}")

# Compare basic and optimized models
comparison = pd.DataFrame(
    [basic_metrics, best_metrics],
    index=["Default Model", "Optimized Model"]
)

print("\nModel Comparison:")
print(comparison.round(4))

# 9. Plot actual versus predicted house prices
plt.figure(figsize=(8, 6))

plt.scatter(y_test, best_pred, alpha=0.6)

min_price = min(y_test.min(), best_pred.min())
max_price = max(y_test.max(), best_pred.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    "r--"
)

plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()

# 10. Display feature importances
fitted_preprocessor = best_model.named_steps["preprocessor"]
fitted_rf = best_model.named_steps["model"]

feature_names = fitted_preprocessor.get_feature_names_out()

importances = pd.Series(
    fitted_rf.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\nTop 5 Most Important Features:")
print(importances.head(5))

# Plot top five features
plt.figure(figsize=(9, 5))

importances.head(5).sort_values().plot(kind="barh")

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 5 Feature Importances")
plt.tight_layout()
plt.show()
