import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# 1. Load dataset
data = pd.read_csv("Telco Customer Churn.csv")

# Convert TotalCharges to numeric
# Some values may be empty in the dataset
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing values
data = data.dropna()

# 2. Encode target variable
# No = 0, Yes = 1
data["Churn"] = data["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Select required features
X = data[
    [
        "tenure",
        "MonthlyCharges",
        "Contract",
        "InternetService"
    ]
]

y = data["Churn"]

# Encode categorical features
X = pd.get_dummies(
    X,
    columns=["Contract", "InternetService"],
    drop_first=True
)

# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Scale numerical features
scaler = StandardScaler()

X_train[["tenure", "MonthlyCharges"]] = scaler.fit_transform(
    X_train[["tenure", "MonthlyCharges"]]
)

X_test[["tenure", "MonthlyCharges"]] = scaler.transform(
    X_test[["tenure", "MonthlyCharges"]]
)

# 5. Train Logistic Regression model
model = LogisticRegression()

model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# Probability of churn
y_prob = model.predict_proba(X_test)[:, 1]

# 7. Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["No Churn", "Churn"])
plt.yticks([0, 1], ["No Churn", "Churn"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.show()

# 9. ROC Curve
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

auc = roc_auc_score(
    y_test,
    y_prob
)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC Curve (AUC = {auc:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.show()