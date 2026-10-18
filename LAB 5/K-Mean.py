# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Step 2: Load and display dataset
df = pd.read_csv("Mall_Customers.csv")

print("Dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# Select required features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Step 3: Apply K-Means without scaling
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["KMeans_Original"] = kmeans.fit_predict(X)

# Plot original K-Means clusters
plt.figure(figsize=(8, 5))
plt.scatter(
    X.iloc[:, 0], X.iloc[:, 1],
    c=df["KMeans_Original"], cmap="viridis"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Clustering Without Scaling")
plt.show()

# Step 4: Apply DBSCAN
dbscan = DBSCAN(eps=10, min_samples=5)
df["DBSCAN_Original"] = dbscan.fit_predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(
    X.iloc[:, 0], X.iloc[:, 1],
    c=df["DBSCAN_Original"], cmap="rainbow"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("DBSCAN Clustering")
plt.show()

# Step 5: Scale features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Re-run K-Means after scaling
kmeans_scaled = KMeans(n_clusters=5, random_state=42, n_init=10)
df["KMeans_Scaled"] = kmeans_scaled.fit_predict(X_scaled)

# Plot scaled K-Means clusters using original axes
plt.figure(figsize=(8, 5))
plt.scatter(
    X.iloc[:, 0], X.iloc[:, 1],
    c=df["KMeans_Scaled"], cmap="viridis"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Clustering After Scaling")
plt.show()

# Step 6: Compare original and scaled results
print("\nOriginal K-Means Cluster Counts:")
print(df["KMeans_Original"].value_counts().sort_index())

print("\nScaled K-Means Cluster Counts:")
print(df["KMeans_Scaled"].value_counts().sort_index())

print("\nDid scaling change cluster assignments?")
print(
    (df["KMeans_Original"] != df["KMeans_Scaled"]).sum(),
    "customers have different numeric cluster labels."
)

# Step 7: Elbow Method
inertia = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(range(1, 11))
plt.show()

# Step 8: Interpret scaled clusters
print("\nCluster Interpretation (Scaled K-Means):")

cluster_summary = df.groupby("KMeans_Scaled")[
    ["Annual Income (k$)", "Spending Score (1-100)"]
].mean()

print(cluster_summary)

# Step 9: Add fake feature LuckyNumber (0-100)
np.random.seed(42)
df["LuckyNumber"] = np.random.randint(0, 101, size=len(df))

print("\nDataset with LuckyNumber:")
print(df.head())

# Cluster using the two original features plus LuckyNumber
X_lucky = df[
    ["Annual Income (k$)", "Spending Score (1-100)", "LuckyNumber"]
]

# Scale all three features
X_lucky_scaled = StandardScaler().fit_transform(X_lucky)

kmeans_lucky = KMeans(n_clusters=5, random_state=42, n_init=10)
df["KMeans_LuckyNumber"] = kmeans_lucky.fit_predict(X_lucky_scaled)

# Plot clusters using Annual Income vs Spending Score
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["KMeans_LuckyNumber"],
    cmap="viridis"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Clusters with LuckyNumber")
plt.show()

print("\nCluster counts with LuckyNumber:")
print(df["KMeans_LuckyNumber"].value_counts().sort_index())

print("\nTask completed successfully.")