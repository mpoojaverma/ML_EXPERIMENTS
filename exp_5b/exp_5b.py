# ============================================================
# EXPERIMENT 5.2
# GAUSSIAN MIXTURE MODEL
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


print("=" * 75)
print("EXPERIMENT 5.2")
print("AIM: To implement Gaussian Mixture Model for clustering.")
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
# 4. MODEL SELECTION USING BIC AND AIC
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("GAUSSIAN MIXTURE MODEL SELECTION")
print("=" * 60)

components = range(2, 11)

bic_values = []
aic_values = []

for k in components:

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=42,
        n_init=5
    )

    gmm.fit(X_scaled)

    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)

    bic_values.append(bic)
    aic_values.append(aic)

    print(
        "Components =", k,
        "| BIC =", bic,
        "| AIC =", aic
    )


# ------------------------------------------------------------
# 5. SELECT BEST NUMBER OF COMPONENTS
# ------------------------------------------------------------

best_k_bic = list(components)[
    np.argmin(bic_values)
]

best_k_aic = list(components)[
    np.argmin(aic_values)
]

print("\nBest Components according to BIC:", best_k_bic)
print("Best Components according to AIC:", best_k_aic)


# Use BIC as the primary model-selection criterion
best_k = best_k_bic

print(
    "\nSelected Number of Components:",
    best_k
)


# ------------------------------------------------------------
# 6. BIC AND AIC GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    list(components),
    bic_values,
    marker="o",
    label="BIC"
)

plt.plot(
    list(components),
    aic_values,
    marker="o",
    label="AIC"
)

plt.xlabel("Number of Components")
plt.ylabel("Information Criterion")
plt.title("GMM Model Selection using BIC and AIC")
plt.xticks(list(components))
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "output/gmm_model_selection.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 7. FINAL GMM MODEL
# ------------------------------------------------------------

gmm = GaussianMixture(
    n_components=best_k,
    covariance_type="full",
    random_state=42,
    n_init=10
)

gmm.fit(X_scaled)

cluster_labels = gmm.predict(
    X_scaled
)

df["Cluster"] = cluster_labels


# ------------------------------------------------------------
# 8. CLUSTER PROBABILITIES
# ------------------------------------------------------------

cluster_probabilities = gmm.predict_proba(
    X_scaled
)

print("\n" + "=" * 60)
print("GMM CLUSTERING RESULTS")
print("=" * 60)

print(
    "\nSelected Components:",
    best_k
)

print(
    "\nBIC:",
    gmm.bic(X_scaled)
)

print(
    "AIC:",
    gmm.aic(X_scaled)
)


# ------------------------------------------------------------
# 9. SILHOUETTE SCORE
# ------------------------------------------------------------

if len(np.unique(cluster_labels)) > 1:

    silhouette = silhouette_score(
        X_scaled,
        cluster_labels
    )

    print(
        "\nSilhouette Score:",
        silhouette
    )

else:

    silhouette = None

    print(
        "\nSilhouette Score: Not applicable"
    )


# ------------------------------------------------------------
# 10. CLUSTER SIZES
# ------------------------------------------------------------

print("\nCluster Sizes:")

print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 11. CLUSTER SUMMARY
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
    "output/gmm_cluster_summary.csv"
)


# ------------------------------------------------------------
# 12. PCA FOR VISUALIZATION
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
# 13. GMM CLUSTER VISUALIZATION
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
    "Gaussian Mixture Model Clusters of Gallstone Patients"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "output/gmm_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 14. CLUSTER VS GALLSTONE STATUS
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
    "output/gmm_cluster_vs_gallstone_status.csv"
)


# ------------------------------------------------------------
# 15. SAMPLE MEMBERSHIP PROBABILITIES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE CLUSTER MEMBERSHIP PROBABILITIES")
print("=" * 60)

probability_df = pd.DataFrame(
    cluster_probabilities,
    columns=[
        f"Cluster {i} Probability"
        for i in range(best_k)
    ]
)

print(
    probability_df.head(10).round(4)
)

probability_df.head(10).to_csv(
    "output/sample_membership_probabilities.csv",
    index=False
)


# ------------------------------------------------------------
# 16. RESULT
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("RESULT")
print("=" * 75)

print(
    "Thus, Gaussian Mixture Model clustering was successfully "
    "implemented on the Gallstone medical dataset."
)

print(
    "The number of mixture components was selected using "
    "the Bayesian Information Criterion (BIC)."
)