# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Step 2: Load dataset
df = pd.read_csv("creditcard.csv")

print("Dataset shape:", df.shape)
print(df["Class"].value_counts())

# Step 3: Split dataset into features X and target y
X = df.drop("Class", axis=1)
y = df["Class"]

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 5: Create a pipeline with scaling, SMOTE, and KNN
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# Step 6: Train KNN with k=5
pipeline.fit(X_train, y_train)

# Predict test data
y_pred = pipeline.predict(X_test)

# Step 7: Evaluate the initial model
print("\nInitial KNN Model (k=5)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# Step 8: Tune k using GridSearchCV
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Step 9: Display the best parameters
print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation F1-Score:", grid_search.best_score_)

# Step 10: Evaluate the tuned model
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

print("\nTuned KNN Model")
print("Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("F1-Score:", f1_score(y_test, y_pred_tuned))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned))