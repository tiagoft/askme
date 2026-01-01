import time

import faiss
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.semi_supervised import LabelSpreading


# Get the data
# data = load_iris()
# X = data.data
# y_true = data.target
X, y_true = make_blobs(
    n_samples=100000,
    n_features=1000,
    centers=3,
    cluster_std=0.60,
    random_state=0,
)
print(f"Data shape: {X.shape}")

# Make the k-means clusterer
scaler = Normalizer()
X_scaled = scaler.fit_transform(X)
print(f"Scaled data shape: {X_scaled.shape}")

# kmeans = KMeans(n_clusters=300, random_state=42)
# t0 = time.time()
# kmeans.fit(X_scaled)
# t1 = time.time()
# labels = kmeans.labels_
# print(f"Total number of labels: {len(labels)} (unique: {len(set(labels))})")

# print("Cluster counts:", {i: int((labels == i).sum()) for i in range(3)})
# print("Inertia:", kmeans.inertia_)
# print("Adjusted Rand Index:", adjusted_rand_score(y_true, labels))
# print(f"Clustering time: {t1 - t0} seconds")
# #print("Silhouette Score:", silhouette_score(X_scaled, labels))

# Part 2: a faiss implementation
# Assume X is your dataset as a NumPy array of shape (n_samples, n_features)
# Example: X = np.array([[1,2], [1,4], [1,0], [4,2], [4,4], [4,0]], dtype='float32')

# Convert to float32 (FAISS requires this type)
# X_scaled = X_scaled.astype('float32')

# Number of clusters
n_clusters = 300
d = X_scaled.shape[1]
res = faiss.StandardGpuResources()

# Create a GPU index directly
t0 = time.time()

# Step 2: Create IVF index
gpu_index = faiss.IndexFlatIP(d)
gpu_index = faiss.index_cpu_to_gpu(res, 0, gpu_index)

# Pass the GPU index into KMeans
kmeans = faiss.Kmeans(
    d=X_scaled.shape[1],
    k=n_clusters,
    niter=20,
    verbose=False,
    gpu=True,
)
kmeans.index = gpu_index  # already GPU

# Train the mode
t1 = time.time()
kmeans.train(X_scaled)

# Get cluster assignments
t2 = time.time()
distances, labels = kmeans.index.search(X_scaled, 1)  # search nearest centroid
labels = labels.flatten()

# Get cluster centers
centers = kmeans.centroids

# print("Cluster labels:", labels)
# print("Cluster centers:", centers)
print("Cluster counts:", {i: int((labels == i).sum()) for i in range(3)})
print("Adjusted Rand Index:", adjusted_rand_score(y_true, labels))
#print("Silhouette Score:", silhouette_score(X_scaled, labels))
print("Index building time:", t1 - t0, "seconds")
print("Training time:", t2 - t1, "seconds")



t0 = time.time()
D, I = gpu_index.search(centers, 1)  # search top-1 nearest neighbor
t1 = time.time()
medoid_indices = I.flatten()
#print("Closest dataset indices for each centroid:", medoid_indices)
print(f"Search time for closest centroids: {t1 - t0} seconds")

print("Done. Proceeding to label propagation...")
# Simple label propagation based on closest data point to each centroid

medoid_indices = I.flatten()

# --- Step 2: Create labels for medoids ---
# Example: use cluster IDs as labels
# labels = -np.ones(len(X), dtype=int)  # -1 means unlabeled
# for cluster_id, medoid_idx in enumerate(medoid_indices):
#     labels[medoid_idx] = cluster_id

# t0 = time.time()
# label_spread = LabelSpreading(kernel='knn', n_neighbors=2, alpha=0.8)
# label_spread.fit(X, labels)
# t1 = time.time()
# # Predicted labels for all points
# predicted_labels = label_spread.transduction_

t0 = time.time()
predicted_labels = fast_label_spreading(embeddings=X_scaled,
                     initial_labels=np.array(
                         [-1 if i not in medoid_indices else labels[i]
                          for i in range(len(X_scaled))]
                     ),
                     k=15,
                     alpha=0.2,
                     max_iter=30)
t1 = time.time()
# Predicted labels for all points

print("Adjusted Rand Index after label spreading:",
      adjusted_rand_score(y_true, predicted_labels))
print(f"Total time for label spreading: {t1 - t0} seconds")
print(f"Number of labeled points: {(predicted_labels != -1).sum()} out of {len(predicted_labels)}")
print(f"Accuracy on labeled points: {np.mean(predicted_labels[predicted_labels != -1] == labels[predicted_labels != -1])}")






# print("Adjusted Rand Index after label spreading:",
#       adjusted_rand_score(y_true, predicted_labels))
# print(f"Total time for label spreading: {t1 - t0} seconds")
# print(f"Number of labeled points: {(labels != -1).sum()} out of {len(labels)}")
# print(f"Accuracy on labeled points: {np.mean(predicted_labels[labels != -1] == labels[labels != -1])}")