import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

df = pd.read_csv("water_potability.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 10 Rows:")
print(df.head(10))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution of Potability:")
print(df['Potability'].value_counts())

X = df.drop('Potability', axis=1)
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

imputer = SimpleImputer(strategy='median')

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X.columns
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X.columns
)

print("\nMissing Values After Imputation:")
print("Training:", X_train.isnull().sum().sum())
print("Testing:", X_test.isnull().sum().sum())

# Logistic Regression with Scaling
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# SVC with Scaling
svc = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(probability=True))
])

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Individual Models
lr.fit(X_train, y_train)
svc.fit(X_train, y_train)
rf.fit(X_train, y_train)

print("\nIndividual Models Trained Successfully!")

# Hard Voting Classifier
hard_voting = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('svc', svc),
        ('rf', rf)
    ],
    voting='hard'
)

hard_voting.fit(X_train, y_train)

# Soft Voting Classifier
soft_voting = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('svc', svc),
        ('rf', rf)
    ],
    voting='soft'
)

soft_voting.fit(X_train, y_train)

print("Hard Voting and Soft Voting Trained Successfully!")

# Single Decision Tree
dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)


# Bagging with 10 Estimators
bagging_10 = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    max_samples=0.5,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

bagging_10.fit(X_train, y_train)


# Bagging with 100 Estimators
bagging_100 = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.5,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

bagging_100.fit(X_train, y_train)


# Pasting with 100 Estimators
pasting_100 = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.5,
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)

pasting_100.fit(X_train, y_train)

print("\nDecision Tree, Bagging and Pasting Trained Successfully!")

models = {
    'Logistic Regression': lr,
    'SVC': svc,
    'Random Forest': rf,
    'Hard Voting': hard_voting,
    'Soft Voting': soft_voting,
    'Decision Tree': dt,
    'Bagging (10)': bagging_10,
    'Bagging (100)': bagging_100,
    'Pasting (100)': pasting_100
}

results = []

for name, model in models.items():

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })


# Display Comparison Table
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

best_model_name = results_df.loc[
    results_df['F1-Score'].idxmax(), 'Model'
]

best_model = models[best_model_name]

y_pred_best = best_model.predict(X_test)

print("\nBest Model:", best_model_name)

cm = confusion_matrix(y_test, y_pred_best)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Not Potable (0)', 'Potable (1)']
)

disp.plot(cmap='Blues')

plt.title("Confusion Matrix of Best Model")
plt.show()

best_single = results_df[
    results_df['Model'].isin([
        'Logistic Regression',
        'SVC',
        'Random Forest',
        'Decision Tree'
    ])
].sort_values('F1-Score', ascending=False).iloc[0]

best_ensemble = results_df[
    results_df['Model'].isin([
        'Hard Voting',
        'Soft Voting',
        'Bagging (10)',
        'Bagging (100)',
        'Pasting (100)'
    ])
].sort_values('F1-Score', ascending=False).iloc[0]

print("\n========== FINAL CONCLUSION ==========")

print("Best Single Model:", best_single['Model'])
print("Best Single Model F1-Score:", round(best_single['F1-Score'], 4))

print("Best Ensemble Model:", best_ensemble['Model'])
print("Best Ensemble F1-Score:", round(best_ensemble['F1-Score'], 4))

if best_ensemble['F1-Score'] > best_single['F1-Score']:
    print("\nVoting or Bagging performed better than the best single model.")
else:
    print("\nVoting or Bagging did not beat the best single model.")
