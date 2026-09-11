# ============================================================
# EXPERIMENT 5.1
# K-MEANS CLUSTERING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


print("=" * 75)
print("EXPERIMENT 5.1")
print("AIM: To implement K-Means clustering.")
print("=" * 75)


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("gallstone.csv")

print("\nDATASET LOADED SUCCESSFULLY")
print("-" * 55)
print("Dataset Shape:", df.shape)

print("\nNumber of Missing Values:", df.isnull().sum().sum())


# ------------------------------------------------------------
# 2. SELECT CLUSTERING FEATURES
# ------------------------------------------------------------
# Gallstone Status is NOT used for clustering because it is
# the outcome/reference variable.

features = [
    "Age",
    "Height",
    "Weight",
    "Body Mass Index (BMI)",
    "Total Body Water (TBW)",
    "Extracellular Water (ECW)",
    "Intracellular Water (ICW)",
    "Extracellular Fluid/Total Body Water (ECF/TBW)",
    "Total Body Fat Ratio (TBFR) (%)",
    "Lean Mass (LM) (%)",
    "Body Protein Content (Protein) (%)",
    "Visceral Fat Rating (VFR)",
    "Bone Mass (BM)",
    "Muscle Mass (MM)",
    "Obesity (%)",
    "Total Fat Content (TFC)",
    "Visceral Fat Area (VFA)",
    "Visceral Muscle Area (VMA) (Kg)",
    "Hepatic Fat Accumulation (HFA)",
    "Glucose",
    "Total Cholesterol (TC)",
    "Low Density Lipoprotein (LDL)",
    "High Density Lipoprotein (HDL)",
    "Triglyceride",
    "Aspartat Aminotransferaz (AST)",
    "Alanin Aminotransferaz (ALT)",
    "Alkaline Phosphatase (ALP)",
    "Creatinine",
    "Glomerular Filtration Rate (GFR)",
    "C-Reactive Protein (CRP)",
    "Hemoglobin (HGB)",
    "Vitamin D"
]

X = df[features].copy()

print("\nNumber of Clustering Features:", X.shape[1])

print("\nClustering Features:")
for feature in features:
    print("-", feature)


# ------------------------------------------------------------
# 3. STANDARDIZE FEATURES
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFEATURE STANDARDIZATION COMPLETED")


# ------------------------------------------------------------
# 4. ELBOW METHOD
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ELBOW METHOD")
print("=" * 60)

inertia_values = []

k_values = range(2, 11)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia_values.append(kmeans.inertia_)

    print(
        "K =", k,
        "| Inertia =", kmeans.inertia_
    )


plt.figure(figsize=(9, 6))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
plt.title("Elbow Method for Optimal K")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "output/elbow_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 5. SILHOUETTE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SILHOUETTE ANALYSIS")
print("=" * 60)

silhouette_values = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_values.append(score)

    print(
        "K =", k,
        "| Silhouette Score =", score
    )


# Select K with highest silhouette score
best_k = list(k_values)[
    np.argmax(silhouette_values)
]

best_silhouette = max(
    silhouette_values
)

print("\nSelected Optimal K:", best_k)
print("Best Silhouette Score:", best_silhouette)


# ------------------------------------------------------------
# 6. SILHOUETTE SCORE GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "output/silhouette_scores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 7. FINAL K-MEANS MODEL
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(
    X_scaled
)

df["Cluster"] = cluster_labels


print("\n" + "=" * 60)
print("K-MEANS CLUSTERING RESULTS")
print("=" * 60)

print("\nOptimal Number of Clusters:", best_k)

print(
    "\nFinal Silhouette Score:",
    silhouette_score(
        X_scaled,
        cluster_labels
    )
)


# ------------------------------------------------------------
# 8. CLUSTER SIZE
# ------------------------------------------------------------

print("\nCluster Sizes:")
print(
    df["Cluster"].value_counts().sort_index()
)


# ------------------------------------------------------------
# 9. CLUSTER SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLUSTER SUMMARY")
print("=" * 60)

cluster_summary = df.groupby(
    "Cluster"
)[features].mean()

print(
    cluster_summary.round(2)
)


# Save cluster summary
cluster_summary.round(2).to_csv(
    "output/cluster_summary.csv"
)


# ------------------------------------------------------------
# 10. PCA FOR 2-D VISUALIZATION
# ------------------------------------------------------------

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

print("\nPCA Visualization Variance:")
print(
    "PC1:",
    pca.explained_variance_ratio_[0]
)

print(
    "PC2:",
    pca.explained_variance_ratio_[1]
)

print(
    "Total:",
    sum(pca.explained_variance_ratio_)
)


# ------------------------------------------------------------
# 11. CLUSTER VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

for cluster in sorted(
    df["Cluster"].unique()
):

    mask = (
        df["Cluster"] == cluster
    )

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cluster {cluster}",
        alpha=0.7
    )


plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "K-Means Clusters of Gallstone Patients"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "output/kmeans_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 12. CLUSTER VS GALLSTONE STATUS
# ------------------------------------------------------------
# This is analysis AFTER clustering.
# Gallstone Status was NOT used to create the clusters.

print("\n" + "=" * 60)
print("CLUSTER VS GALLSTONE STATUS")
print("=" * 60)

status_table = pd.crosstab(
    df["Cluster"],
    df["Gallstone Status"]
)

print(status_table)

status_table.to_csv(
    "output/cluster_vs_gallstone_status.csv"
)


# ------------------------------------------------------------
# 13. RESULT
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("RESULT")
print("=" * 75)

print(
    "Thus, K-Means clustering was successfully implemented "
    "on the Gallstone medical dataset."
)

print(
    "The optimal number of clusters was determined using "
    "Elbow Method and Silhouette Analysis."
)