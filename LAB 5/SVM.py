# Step 1: Import libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Step 2: Load the dataset
data_dir = "MRI_Dataset"  # Change to your dataset path

images = []
labels = []

for label in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, label)

    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)

            # Step 3: Convert image to grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                # Step 4: Resize image to 64x64
                img = cv2.resize(img, (64, 64))

                # Step 5: Flatten image into a 1D array
                img = img.flatten()

                images.append(img)
                labels.append(label)

# Convert lists to NumPy arrays
X = np.array(images)
y = np.array(labels)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# Step 6: Split data into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 7: Normalize features using StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Train SVM with RBF kernel
model = SVC(kernel='rbf', C=10, gamma='scale')

model.fit(X_train, y_train)

# Step 9: Predict test labels
y_pred = model.predict(X_test)

# Step 10: Evaluate performance
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 11: Display confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# Step 12: Display a few test images with actual and predicted labels
# Use original flattened test images before scaling
X_test_original = X_test.copy()

# Inverse transform to recover the original pixel values
X_test_original = scaler.inverse_transform(X_test_original)

plt.figure(figsize=(12, 6))

for i in range(min(6, len(X_test))):
    plt.subplot(2, 3, i + 1)

    # Reshape flattened image back to 64x64
    img = X_test_original[i].reshape(64, 64)

    plt.imshow(img, cmap='gray')
    plt.title(f"Actual: {y_test[i]}\nPredicted: {y_pred[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()