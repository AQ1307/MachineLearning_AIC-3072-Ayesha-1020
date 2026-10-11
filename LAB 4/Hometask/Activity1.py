import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# 1. Load the dataset
df = pd.read_csv("creditcard.csv")

print("Dataset Shape:", df.shape)
print("\nClass Distribution:")
print(df["Class"].value_counts())

# Split into features (X) and target (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# Train initial KNN model with k=5
pipeline.fit(X_train, y_train)

# Predict test data
y_pred = pipeline.predict(X_test)

# 5. Evaluate initial model
accuracy_initial = accuracy_score(y_test, y_pred)
f1_initial = f1_score(y_test, y_pred, zero_division=0)

print("\nInitial KNN Results (k=5)")
print("Accuracy:", round(accuracy_initial, 4))
print("F1-Score:", round(f1_initial, 4))

# Tune k using GridSearchCV
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best model and best k
best_model = grid_search.best_estimator_
print("\nBest k:", grid_search.best_params_["knn__n_neighbors"])
print("Best Cross-Validation F1:", round(grid_search.best_score_, 4))

# Evaluate tuned model on the same test set
y_pred_tuned = best_model.predict(X_test)

accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned, zero_division=0)

print("\nTuned KNN Results")
print("Accuracy:", round(accuracy_tuned, 4))
print("F1-Score:", round(f1_tuned, 4))

# Compare both models
comparison = pd.DataFrame({
    "Metric": ["Accuracy", "F1-Score"],
    "Initial KNN (k=5)": [accuracy_initial, f1_initial],
    "Tuned KNN": [accuracy_tuned, f1_tuned]
})

print("\nModel Comparison:")
print(comparison.to_string(index=False))
