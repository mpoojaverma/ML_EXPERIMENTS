# Machine Learning Lab Experiments

This repository contains the implementations, datasets, outputs, visualizations, and documentation for the Machine Learning Laboratory experiments.

Each experiment is organized in a separate folder containing its Python implementation, dataset, generated results, and experiment-specific README.

---

## Repository Structure

```text
ML_EXPERIMENTS/
│
├── README.md
├── requirements.txt
│
├── exp_1/
│   ├── exp_1.py
│   ├── music library songs.csv
│   ├── output/
│   └── README.md
│
├── exp_2/
│   ├── exp_2.py
│   ├── music library songs.csv
│   ├── output/
│   └── README.md
│
├── exp_3/
│   ├── exp_3.py
│   ├── flood_risk_dataset_india.csv
│   ├── output/
│   └── README.md
│
├── exp_4a/
│   ├── exp_4a.py
│   ├── flood_risk_dataset_india.csv
│   ├── output/
│   └── README.md
│
├── exp_4b/
│   ├── exp_4b.py
│   ├── flood_risk_dataset_india.csv
│   ├── output/
│   └── README.md
│
├── exp_5a/
│   ├── exp_5a.py
│   ├── gallstone.csv
│   ├── output/
│   └── README.md
│
├── exp_5b/
│   ├── exp_5b.py
│   ├── gallstone.csv
│   ├── output/
│   └── README.md
│
├── exp_5c/
│   ├── exp_5c.py
│   ├── gallstone.csv
│   ├── output/
│   └── README.md
│
└── exp_6/
    ├── exp_6.py
    ├── air_quality_historical.csv
    ├── output/
    └── README.md
```

Each experiment folder contains:

- Python implementation
- Dataset used for the experiment
- Generated output/results
- Graphs and visualizations where applicable
- Experiment-specific README

---

# Experiments

## Experiment 1 — Load and View a Dataset

**Folder:** `exp_1`

### Aim

To load and view a dataset using Python and Pandas.

### Dataset

**Music Library Songs Dataset**

### Operations Performed

- Load CSV dataset
- Create a copy of the dataset
- Display dataset shape
- Select relevant columns
- Display the first records
- Inspect dataset information
- Generate summary statistics
- Check missing values

### Technologies

- Python
- Pandas

---

## Experiment 2 — Summary and Statistical Analysis of a Dataset

**Folder:** `exp_2`

### Aim

To perform summary and exploratory statistical analysis of a dataset.

### Dataset

**Music Library Songs Dataset**

### Analysis Performed

- Dataset shape
- Column information
- Data types
- First and last records
- Unique values
- Missing values
- Summary statistics
- Top artists
- Album distribution
- Songs per artist
- Top albums
- Artist popularity
- Distribution analysis

### Visualizations

The experiment generates multiple graphs for exploring the music library dataset.

### Technologies

- Python
- Pandas
- NumPy
- Matplotlib

---

# Experiment 3 — Linear Regression

**Folder:** `exp_3`

### Aim

To implement Linear Regression to perform prediction.

### Dataset

**Flood Risk Dataset — India**

### Variables

**Independent Variable:**

`River Discharge (m³/s)`

**Dependent Variable:**

`Water Level (m)`

### Method

Ordinary Least Squares (OLS) Linear Regression was implemented using NumPy.

The dataset is divided into:

- 80% training data
- 20% testing data

### Evaluation Metrics

- Mean Squared Error (MSE)
- R² Score

### Visualization

A scatter plot of the actual observations and the fitted linear regression line is generated.

### Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

# Experiment 4a — Bayesian Logistic Regression

**Folder:** `exp_4a`

### Aim

To implement Bayesian Logistic Regression for classification.

### Dataset

**Flood Risk Dataset — India**

### Target Variable

`Flood Occurred`

### Features

The model uses selected environmental and geographical variables including:

- Rainfall
- Temperature
- Humidity
- River Discharge
- Water Level
- Elevation
- Population Density

### Method

The implementation uses:

- Feature standardization
- Logistic regression likelihood
- Gaussian prior
- Posterior distribution
- Metropolis-Hastings sampling
- Posterior mean estimation

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

### Visualization

Posterior coefficient distributions are visualized using histograms.

### Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

# Experiment 4b — Support Vector Machine

**Folder:** `exp_4b`

### Aim

To implement Support Vector Machine (SVM) for classification.

### Dataset

**Flood Risk Dataset — India**

### Target Variable

`Flood Occurred`

### Features

Selected environmental and geographical variables are used as input features.

### Method

An SVM classifier with an RBF kernel is implemented.

The features are standardized before training.

### Model Configuration

- Kernel: RBF
- C: 1.0
- Gamma: `scale`

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

### Visualization

A confusion matrix is generated to evaluate classification performance.

### Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

# Experiment 5a — K-Means Clustering

**Folder:** `exp_5a`

### Aim

To implement K-Means Clustering for grouping similar observations.

### Dataset

**Gallstone Medical Dataset**

### Dataset Characteristics

- 319 records
- 39 attributes
- 32 numerical features used for clustering

### Method

The clustering features are standardized before applying K-Means.

The number of clusters is evaluated from:

`K = 2 to 10`

Two methods are used for cluster selection:

- Elbow Method
- Silhouette Score

The value of K producing the highest silhouette score is selected.

### Additional Analysis

The resulting clusters are compared with the original `Gallstone Status` variable after clustering.

### Visualizations

- Elbow curve
- Silhouette score plot
- PCA-based cluster visualization

PCA is used only for visualization and not for performing the clustering itself.

### Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

# Experiment 5b — Gaussian Mixture Model

**Folder:** `exp_5b`

### Aim

To implement Gaussian Mixture Model (GMM) clustering.

### Dataset

**Gallstone Medical Dataset**

### Method

Gaussian Mixture Models are fitted with different numbers of components from:

`K = 2 to 10`

Model selection is performed using:

- Bayesian Information Criterion (BIC)
- Akaike Information Criterion (AIC)

The model with the lowest BIC is selected.

### Additional Analysis

The experiment calculates:

- Cluster assignments
- Cluster membership probabilities
- Silhouette score
- Cluster sizes
- Comparison with Gallstone Status

### Visualization

PCA is used to project the clustered observations into two dimensions for visualization.

### Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

# Experiment 5c — Hierarchical Clustering

**Folder:** `exp_5c`

### Aim

To implement Hierarchical Clustering for grouping similar observations.

### Dataset

**Gallstone Medical Dataset**

### Method

The numerical clustering features are standardized.

Ward's linkage method is used to construct the hierarchical clustering structure.

A dendrogram is generated to visualize the hierarchical relationships.

The number of clusters is evaluated from:

`K = 2 to 10`

Silhouette scores are calculated and the best-performing K is selected.

### Additional Analysis

The experiment produces:

- Cluster assignments
- Cluster sizes
- Cluster summaries
- Comparison with Gallstone Status

### Visualizations

- Hierarchical dendrogram
- Silhouette score plot
- PCA-based cluster visualization

PCA is used only for visualization.

### Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- SciPy

---

# Experiment 6 — Principal Component Analysis

**Folder:** `exp_6`

### Aim

To implement Principal Component Analysis (PCA) for dimensionality reduction.

### Dataset

**Air Quality Historical Dataset**

### Domain

Air Quality / Environmental Monitoring

### Dataset Characteristics

- 1,298 observations
- 12 columns
- 9 numerical air-quality/environmental variables used for PCA

### Features Used

The following variables are used as PCA inputs:

- `pm10`
- `pm2_5`
- `carbon_monoxide`
- `nitrogen_dioxide`
- `sulphur_dioxide`
- `ozone`
- `aerosol_optical_depth`
- `dust`
- `uv_index`

The `date` column is treated as a reference/time field.

`us_aqi` and `european_aqi` are retained for interpretation and visualization and are not used as PCA input variables.

### Preprocessing

The experiment performs:

1. Missing-value inspection
2. Median imputation
3. Feature standardization using `StandardScaler`

### PCA

PCA is first fitted using all available components.

The explained variance ratio and cumulative explained variance are calculated.

The number of components required to retain:

- 90% variance
- 95% variance

is determined.

The reduced dataset is then generated using the minimum number of components required to retain at least 95% of the variance.

### Results

For the current dataset:

- Original PCA dimensions: **9**
- Components required for 90% variance: **5**
- Components required for 95% variance: **6**
- Selected components: **6**
- Variance retained: **95.71%**

Thus, the dimensionality is reduced from 9 variables to 6 principal components while retaining approximately 95.71% of the standardized variance.

### Visualizations

- PCA Scree Plot
- Cumulative Explained Variance Plot
- 2D PCA Visualization
- PCA Feature Loading Plot

### Output Files

The experiment generates:

- `pca_scree_plot.png`
- `cumulative_explained_variance.png`
- `pca_2d_visualization.png`
- `pca_feature_loadings.png`
- `explained_variance.csv`
- `pca_loadings.csv`
- `pca_reduced_dataset.csv`
- `output.txt`

### Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

# Technologies Used

The repository primarily uses:

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **SciPy**

---

# Installation

Clone the repository:

```bash
git clone https://github.com/mpoojaverma/ML_EXPERIMENTS.git
```

Navigate to the repository:

```bash
cd ML_EXPERIMENTS
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# Running an Experiment

Navigate to the required experiment folder.

For example:

```bash
cd exp_6
```

Run the Python program:

```bash
python exp_6.py
```

The generated graphs, CSV files, and execution output are stored in the corresponding `output/` directory.

---

# Repository Organization

Each experiment is maintained independently.

This makes it possible to:

- Run experiments separately
- Keep datasets with their corresponding programs
- Store experiment-specific outputs
- Reproduce results
- Review implementations easily
- Maintain a clean laboratory record

---

# Author

**M. Pooja Verma**

B.Tech — Computer Science and Engineering  
SRM Institute of Science and Technology
