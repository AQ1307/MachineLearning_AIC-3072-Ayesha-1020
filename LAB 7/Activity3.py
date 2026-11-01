import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv("insurance.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 10 Rows:")
print(df.head(10))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics of Charges:")
print(df['charges'].describe())

df['sex'] = df['sex'].map({
    'female': 0,
    'male': 1
})

# Encode smoker: no = 0, yes = 1
df['smoker'] = df['smoker'].map({
    'no': 0,
    'yes': 1
})

# One-Hot Encode region
df = pd.get_dummies(
    df,
    columns=['region'],
    drop_first=True,
    dtype=int
)

# Separate Features and Target
X = df.drop('charges', axis=1)
y = df['charges']

# Split Dataset (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

# Random Forest Regressor
rf = RandomForestRegressor(
    random_state=42
)

# Gradient Boosting Regressor
gb = GradientBoostingRegressor(
    random_state=42
)

# Train Models
ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

print("\nSingle Regression Models Trained Successfully!")

# Base Models
estimators = [
    ('ridge', ridge),
    ('rf', rf),
    ('gb', gb)
]

# Stacking Regressor
stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    cv=5
)

# Train Stacking Model
stack_model.fit(X_train, y_train)

print("\nStacking Regressor Trained Successfully!")

models = {
    'Ridge Regression': ridge,
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'Stacking Regressor': stack_model
}

results = []

for name, model in models.items():

    # Predict Test Data
    y_pred = model.predict(X_test)

    # Calculate Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R2 Score': r2
    })


# Display Comparison Table
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

final_model = stack_model.final_estimator_

print("\nFinal Estimator Coefficients:")

for name, coef in zip(
    ['Ridge', 'Random Forest', 'Gradient Boosting'],
    final_model.coef_
):
    print(name, ":", coef)

print("\nIntercept:", final_model.intercept_)

y_pred_stack = stack_model.predict(X_test)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred_stack,
    alpha=0.6
)

# Reference Line
min_value = min(y_test.min(), y_pred_stack.min())
max_value = max(y_test.max(), y_pred_stack.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    'r--'
)

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")

plt.title("Actual vs Predicted Charges - Stacking Regressor")

plt.tight_layout()
plt.show()

print("\n========== FINAL CONCLUSION ==========")

best_model = results_df.loc[
    results_df['RMSE'].idxmin()
]

print("Best Model:", best_model['Model'])
print("Lowest RMSE:", round(best_model['RMSE'], 4))
print("R2 Score:", round(best_model['R2 Score'], 4))

print("\nFinal Estimator Coefficients show the contribution")
print("of each base model to the stacking prediction.")

most_trusted = final_model.coef_.argmax()

base_models = [
    'Ridge Regression',
    'Random Forest',
    'Gradient Boosting'
]

print("Model with Highest Coefficient:", base_models[most_trusted])
