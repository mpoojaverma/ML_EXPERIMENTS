# Experiment 6 — Principal Component Analysis (PCA)

## Aim

To implement Principal Component Analysis (PCA) for dimensionality reduction on an air-quality dataset.

## Dataset

**Dataset:** `air_quality_historical.csv`

**Domain:** Air Quality / Environmental Monitoring

The dataset contains historical air-quality observations with pollutant and environmental measurements.

## PCA Features

The following nine numerical air-quality/environmental variables are used as PCA inputs:

1. `pm10`
2. `pm2_5`
3. `carbon_monoxide`
4. `nitrogen_dioxide`
5. `sulphur_dioxide`
6. `ozone`
7. `aerosol_optical_depth`
8. `dust`
9. `uv_index`

The `date` column is treated as an identifier/time field and is not used as a PCA feature.

`us_aqi` and `european_aqi` are retained for interpretation/visualization and are not used as PCA input variables because AQI values are derived indices.

## Algorithm

PCA is a dimensionality-reduction technique that transforms correlated input variables into a smaller set of uncorrelated principal components.

### Steps

1. Load the air-quality dataset.
2. Select numerical air-quality/environmental features.
3. Check missing values.
4. Replace missing numerical values using the median of each feature.
5. Standardize the features using `StandardScaler`.
6. Fit PCA using all available components.
7. Calculate explained variance ratio.
8. Calculate cumulative explained variance.
9. Determine the number of components required to retain at least 90% and 95% variance.
10. Select the minimum number of components required for 95% variance.
11. Transform the standardized dataset into the reduced PCA space.
12. Save the reduced dataset.
13. Visualize explained variance and the first two principal components.
14. Analyze PCA feature loadings.

## Why Standardization?

The air-quality variables are measured on different numerical scales. Standardization places them on a comparable scale before PCA so that variables with larger numerical magnitudes do not dominate the principal components.

## Output Files

The `output/` folder contains:

- `output.txt` — complete execution output
- `pca_scree_plot.png` — explained variance of individual components
- `cumulative_explained_variance.png` — cumulative explained variance
- `pca_2d_visualization.png` — observations projected onto the first two PCs
- `pca_feature_loadings.png` — feature loadings for PC1 and PC2
- `explained_variance.csv` — explained and cumulative variance values
- `pca_loadings.csv` — PCA component loadings
- `pca_reduced_dataset.csv` — transformed/reduced dataset

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## How to Run

From the `exp_6` directory:

```bash
python exp_6.py
```

All generated results are saved automatically inside the `output/` directory.

## Result

The PCA implementation successfully reduces the dimensionality of the selected air-quality feature space while retaining at least 95% of the original standardized variance.
