import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X = X_train.reshape(X_train.shape[0], 784)
y = y_train

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

columns = [f"pixel_{i}" for i in range(784)]
df = pd.DataFrame(X, columns=columns)
df["label"] = y

print("Dataset Shape:", df.shape)
print("\nFirst 10 Rows:")
print(df.head(10))

print("\nClass Distribution:")
print(df["label"].value_counts().sort_index())

# 2. Separate features and labels
X = df.drop("label", axis=1).values.astype(np.float32)
y = df["label"].values

# Normalize pixel values from 0-255 to 0-1
X = X / 255.0

# Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nOriginal Features:", X_scaled.shape[1])

# Use a reproducible sample for faster execution
SAMPLE_SIZE = 10000
rng = np.random.default_rng(42)
indices = rng.choice(len(X_scaled), SAMPLE_SIZE, replace=False)

X_sample = X_scaled[indices]
y_sample = y[indices]

# 3. Apply three PCA methods

# Method 1: Fixed 50 components
pca1 = PCA(n_components=50)
X_pca1 = pca1.fit_transform(X_sample)

# Method 2: Retain 95% variance
pca2 = PCA(n_components=0.95)
X_pca2 = pca2.fit_transform(X_sample)

# Method 3: All components
pca3 = PCA(n_components=None)
X_pca3 = pca3.fit_transform(X_sample)

# 4. Display PCA results
results = pd.DataFrame({
    "Method": [
        "Fixed 50 Components",
        "95% Variance",
        "All Components"
    ],
    "Original Features": [784, 784, 784],
    "Retained Components": [
        pca1.n_components_,
        pca2.n_components_,
        pca3.n_components_
    ],
    "Cumulative Explained Variance (%)": [
        pca1.explained_variance_ratio_.sum() * 100,
        pca2.explained_variance_ratio_.sum() * 100,
        pca3.explained_variance_ratio_.sum() * 100
    ]
})

print("\nPCA Comparison:")
print(results.round(2).to_string(index=False))

# 5. Plot cumulative explained variance
cumulative_variance = np.cumsum(
    pca3.explained_variance_ratio_
)

plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance * 100
)

plt.axhline(95, color="red", linestyle="--",
            label="95% Variance")
plt.axvline(50, color="green", linestyle="--",
            label="50 Components")

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("Cumulative Explained Variance - Fashion MNIST")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Visualize first two principal components
plot_df = pd.DataFrame({
    "PC1": X_pca3[:, 0],
    "PC2": X_pca3[:, 1],
    "Class": [class_names[i] for i in y_sample]
})

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=plot_df,
    x="PC1",
    y="PC2",
    hue="Class",
    palette="tab10",
    s=18,
    alpha=0.65
)

plt.title("Fashion MNIST: First Two Principal Components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(
    title="Clothing Class",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
plt.tight_layout()
plt.show()
