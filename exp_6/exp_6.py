import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


print("=" * 80)
print("EXPERIMENT 6")
print("AIM: To implement Principal Component Analysis (PCA) for dimensionality reduction.")
print("=" * 80)


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("air_quality_historical.csv")

print("\nDATASET LOADED SUCCESSFULLY")
print("-" * 50)
print("Dataset Shape:", df.shape)
print("Number of Columns:", df.shape[1])


# ------------------------------------------------------------
# 2. SELECT FEATURES
# ------------------------------------------------------------

features = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "aerosol_optical_depth",
    "dust",
    "uv_index"
]

X = df[features].copy()

print("\nFEATURES USED FOR PCA")
print("-" * 50)

for feature in features:
    print(feature)


# ------------------------------------------------------------
# 3. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMISSING VALUES")
print("-" * 50)
print(X.isnull().sum())


# ------------------------------------------------------------
# 4. HANDLE MISSING VALUES
# ------------------------------------------------------------

X = X.fillna(X.median())

print("\nMissing values after median imputation:")
print(X.isnull().sum().sum())


# ------------------------------------------------------------
# 5. STANDARDIZE FEATURES
# ------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFEATURE STANDARDIZATION")
print("-" * 50)
print("StandardScaler applied successfully.")


# ------------------------------------------------------------
# 6. APPLY PCA
# ------------------------------------------------------------

pca_full = PCA()

X_pca_full = pca_full.fit_transform(X_scaled)

explained_variance = pca_full.explained_variance_ratio_

cumulative_variance = np.cumsum(explained_variance)


# ------------------------------------------------------------
# 7. EXPLAINED VARIANCE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPLAINED VARIANCE")
print("=" * 70)

for i in range(len(explained_variance)):

    print(
        f"PC{i + 1}: "
        f"Explained Variance = {explained_variance[i]:.6f}    "
        f"Cumulative = {cumulative_variance[i]:.6f}"
    )


# ------------------------------------------------------------
# 8. FIND NUMBER OF COMPONENTS
# ------------------------------------------------------------

components_90 = np.argmax(
    cumulative_variance >= 0.90
) + 1

components_95 = np.argmax(
    cumulative_variance >= 0.95
) + 1


print("\n" + "=" * 70)
print("COMPONENT SELECTION")
print("=" * 70)

print("Components required for 90% variance:", components_90)
print("Components required for 95% variance:", components_95)


# ------------------------------------------------------------
# 9. REDUCE DATASET
# ------------------------------------------------------------

n_components = components_95

pca = PCA(n_components=n_components)

X_reduced = pca.fit_transform(X_scaled)

reduced_columns = [
    f"PC{i + 1}"
    for i in range(n_components)
]

reduced_df = pd.DataFrame(
    X_reduced,
    columns=reduced_columns
)


# Keep date for reference
if "date" in df.columns:
    reduced_df.insert(
        0,
        "date",
        df["date"].values
    )


# Keep AQI values for interpretation
if "us_aqi" in df.columns:
    reduced_df["us_aqi"] = df["us_aqi"].values

if "european_aqi" in df.columns:
    reduced_df["european_aqi"] = df["european_aqi"].values


# ------------------------------------------------------------
# 10. SAVE REDUCED DATASET
# ------------------------------------------------------------

reduced_df.to_csv(
    "output/pca_reduced_dataset.csv",
    index=False
)


# ------------------------------------------------------------
# 11. SAVE EXPLAINED VARIANCE
# ------------------------------------------------------------

variance_df = pd.DataFrame({

    "Component": [
        f"PC{i + 1}"
        for i in range(len(explained_variance))
    ],

    "Explained_Variance_Ratio":
        explained_variance,

    "Cumulative_Explained_Variance":
        cumulative_variance
})

variance_df.to_csv(
    "output/explained_variance.csv",
    index=False
)


# ------------------------------------------------------------
# 12. PCA LOADINGS
# ------------------------------------------------------------

loadings = pd.DataFrame(
    pca.components_.T,
    index=features,
    columns=reduced_columns
)

loadings.to_csv(
    "output/pca_loadings.csv"
)


print("\nSELECTED COMPONENTS:", n_components)

print(
    "Total Retained Variance:",
    pca.explained_variance_ratio_.sum()
)


# ------------------------------------------------------------
# 13. SCREE PLOT
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance,
    marker="o"
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")

plt.title(
    "PCA Scree Plot"
)

plt.xticks(
    range(1, len(explained_variance) + 1)
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "output/pca_scree_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ------------------------------------------------------------
# 14. CUMULATIVE EXPLAINED VARIANCE
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.axhline(
    0.90,
    linestyle="--",
    label="90% Variance"
)

plt.axhline(
    0.95,
    linestyle="--",
    label="95% Variance"
)

plt.xlabel(
    "Number of Principal Components"
)

plt.ylabel(
    "Cumulative Explained Variance"
)

plt.title(
    "Cumulative Explained Variance"
)

plt.xticks(
    range(1, len(cumulative_variance) + 1)
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "output/cumulative_explained_variance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ------------------------------------------------------------
# 15. 2D PCA VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    X_pca_full[:, 0],
    X_pca_full[:, 1],
    c=df["us_aqi"],
    alpha=0.7
)

plt.colorbar(
    scatter,
    label="US AQI"
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "Air Quality Data in PCA Space"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "output/pca_2d_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ------------------------------------------------------------
# 16. PCA FEATURE LOADINGS
# ------------------------------------------------------------

loading_2d = loadings.iloc[:, :2]

plt.figure(figsize=(10, 7))

for feature in loading_2d.index:

    plt.arrow(
        0,
        0,
        loading_2d.loc[feature, "PC1"],
        loading_2d.loc[feature, "PC2"],
        head_width=0.025,
        length_includes_head=True
    )

    plt.text(
        loading_2d.loc[feature, "PC1"] * 1.08,
        loading_2d.loc[feature, "PC2"] * 1.08,
        feature,
        fontsize=9
    )


plt.axhline(
    0,
    linewidth=0.8
)

plt.axvline(
    0,
    linewidth=0.8
)

plt.xlabel(
    "PC1 Loading"
)

plt.ylabel(
    "PC2 Loading"
)

plt.title(
    "PCA Feature Loadings"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "output/pca_feature_loadings.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ------------------------------------------------------------
# 17. FINAL RESULT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"PCA successfully reduced the {len(features)}-dimensional "
    f"air-quality feature space to {n_components} principal "
    "components while retaining at least 95% of the variance."
)

print("=" * 70)