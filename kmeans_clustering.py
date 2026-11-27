import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Crop_recommendation.csv")
CLUSTERED_DATA_PATH = os.path.join(BASE_DIR, "dataset", "Crop_clustered.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset Loaded Successfully")

# Features only
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]

# Elbow Method
wcss = []
print("Calculating WCSS...")

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Save Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "elbow_graph.png"))
plt.close()
print("Elbow Graph Saved")

# Final KMeans Model
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

# Save model and clustered dataset
joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
df.to_csv(CLUSTERED_DATA_PATH, index=False)
print("KMeans Model and Clustered Dataset Saved Successfully")

# Display crops per cluster
print("\n========== Crops in Each Cluster ==========\n")
for cluster in sorted(df["Cluster"].unique()):
    crops = df[df["Cluster"] == cluster]["label"].unique()
    print(f"Cluster {cluster}:")
    for crop in crops:
        print("   -", crop)
    print()

print("\nCluster Summary:")
print(df.groupby("Cluster")["label"].value_counts())