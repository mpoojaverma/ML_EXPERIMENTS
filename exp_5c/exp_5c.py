# ============================================================
# EXPERIMENT 5.3
# HIERARCHICAL CLUSTERING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage


print("=" * 75)
print("EXPERIMENT 5.3")
print("AIM: To implement Hierarchical clustering.")
print("=" * 75)


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("gallstone.csv")

print("\nDATASET LOADED SUCCESSFULLY")
print("-" * 55)
print("Dataset Shape:", df.shape)

print(
    "\nNumber of Missing Values:",
    df.isnull().sum().sum()
)


# ------------------------------------------------------------
# 2. SELECT CLUSTERING FEATURES
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 3. STANDARDIZE FEATURES
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFEATURE STANDARDIZATION COMPLETED")


# ------------------------------------------------------------
# 4. HIERARCHICAL DENDROGRAM
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("HIERARCHICAL CLUSTERING")
print("=" * 60)

print("\nCreating dendrogram using Ward linkage...")

linkage_matrix = linkage(
    X_scaled,
    method="ward"
)


plt.figure(figsize=(12, 7))

dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=9,
    show_contracted=True
)

plt.title(
    "Hierarchical Clustering Dendrogram"
)

plt.xlabel("Cluster / Sample Groups")
plt.ylabel("Ward Linkage Distance")

plt.tight_layout()

plt.savefig(
    "output/hierarchical_dendrogram.png",
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

cluster_range = range(2, 11)

silhouette_values = []

for k in cluster_range:

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_values.append(score)

    print(
        "Clusters =", k,
        "| Silhouette Score =", score
    )


# Select the number of clusters
best_k = list(cluster_range)[
    np.argmax(silhouette_values)
]

best_score = max(
    silhouette_values
)

print(
    "\nSelected Optimal Number of Clusters:",
    best_k
)

print(
    "Best Silhouette Score:",
    best_score
)


# ------------------------------------------------------------
# 6. SILHOUETTE GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    list(cluster_range),
    silhouette_values,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")

plt.title(
    "Silhouette Analysis for Hierarchical Clustering"
)

plt.xticks(list(cluster_range))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "output/hierarchical_silhouette_scores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 7. FINAL HIERARCHICAL MODEL
# ------------------------------------------------------------

model = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward"
)

cluster_labels = model.fit_predict(
    X_scaled
)

df["Cluster"] = cluster_labels


# ------------------------------------------------------------
# 8. FINAL RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("HIERARCHICAL CLUSTERING RESULTS")
print("=" * 60)

print(
    "\nOptimal Number of Clusters:",
    best_k
)

print(
    "Final Silhouette Score:",
    silhouette_score(
        X_scaled,
        cluster_labels
    )
)


# ------------------------------------------------------------
# 9. CLUSTER SIZES
# ------------------------------------------------------------

print("\nCluster Sizes:")

print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 10. CLUSTER SUMMARY
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

cluster_summary.round(2).to_csv(
    "output/hierarchical_cluster_summary.csv"
)


# ------------------------------------------------------------
# 11. PCA FOR VISUALIZATION
# ------------------------------------------------------------

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

print("\n" + "=" * 60)
print("PCA VISUALIZATION")
print("=" * 60)

print(
    "PC1 Variance:",
    pca.explained_variance_ratio_[0]
)

print(
    "PC2 Variance:",
    pca.explained_variance_ratio_[1]
)

print(
    "Total Variance:",
    sum(
        pca.explained_variance_ratio_
    )
)


# ------------------------------------------------------------
# 12. CLUSTER VISUALIZATION
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
    "Hierarchical Clusters of Gallstone Patients"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "output/hierarchical_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 13. CLUSTER VS GALLSTONE STATUS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLUSTER VS GALLSTONE STATUS")
print("=" * 60)

status_table = pd.crosstab(
    df["Cluster"],
    df["Gallstone Status"]
)

print(status_table)

status_table.to_csv(
    "output/hierarchical_cluster_vs_gallstone_status.csv"
)


# ------------------------------------------------------------
# 14. RESULT
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("RESULT")
print("=" * 75)

print(
    "Thus, Hierarchical clustering was successfully "
    "implemented on the Gallstone medical dataset."
)

print(
    "The optimal number of clusters was determined "
    "using Silhouette Analysis."
)