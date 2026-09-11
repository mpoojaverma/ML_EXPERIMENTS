# Experiment 5c – Hierarchical Clustering

## Aim

To implement Hierarchical Clustering for grouping similar patients into clusters.

## Dataset

**Gallstone Medical Dataset**

The dataset contains clinical, demographic, body-composition, and laboratory measurements of patients.

### Dataset Details

- Number of records: **319**
- Number of attributes: **39**
- Missing values: **0**

The dataset contains meaningful medical measurements related to body composition, metabolic characteristics, liver-related measurements, kidney function, blood parameters, and other clinical characteristics.

## Objective

The objective of this experiment is to group patients with similar clinical and body-composition characteristics using Hierarchical Clustering.

The experiment is performed as an **unsupervised learning task**.

The `Gallstone Status` variable is not used as an input feature during clustering. It is used only after clustering for post-clustering analysis.

## Clustering Features

The following 32 numerical features are used:

- `Age`
- `Height`
- `Weight`
- `Body Mass Index (BMI)`
- `Total Body Water (TBW)`
- `Extracellular Water (ECW)`
- `Intracellular Water (ICW)`
- `Extracellular Fluid/Total Body Water (ECF/TBW)`
- `Total Body Fat Ratio (TBFR) (%)`
- `Lean Mass (LM) (%)`
- `Body Protein Content (Protein) (%)`
- `Visceral Fat Rating (VFR)`
- `Bone Mass (BM)`
- `Muscle Mass (MM)`
- `Obesity (%)`
- `Total Fat Content (TFC)`
- `Visceral Fat Area (VFA)`
- `Visceral Muscle Area (VMA) (Kg)`
- `Hepatic Fat Accumulation (HFA)`
- `Glucose`
- `Total Cholesterol (TC)`
- `Low Density Lipoprotein (LDL)`
- `High Density Lipoprotein (HDL)`
- `Triglyceride`
- `Aspartat Aminotransferaz (AST)`
- `Alanin Aminotransferaz (ALT)`
- `Alkaline Phosphatase (ALP)`
- `Creatinine`
- `Glomerular Filtration Rate (GFR)`
- `C-Reactive Protein (CRP)`
- `Hemoglobin (HGB)`
- `Vitamin D`

## Target / Reference Variable

The dataset contains:

```text
Gallstone Status

This variable is excluded from the clustering features.

It is used only after clustering to examine the relationship between the discovered clusters and the recorded gallstone status.

Algorithm
Load the Gallstone medical dataset.
Check the dataset shape and missing values.
Select the numerical clinical and body-composition features.
Exclude Gallstone Status from the clustering input.
Standardize the selected features using StandardScaler.
Calculate the hierarchical linkage matrix using Ward's linkage method.
Generate a hierarchical clustering dendrogram.
Evaluate different numbers of clusters using Silhouette Score.
Select the number of clusters with the highest Silhouette Score.
Apply Agglomerative Hierarchical Clustering using the selected number of clusters.
Assign each patient to a cluster.
Calculate the size of each cluster.
Generate a cluster-wise summary of the clinical features.
Apply PCA to reduce the feature space to two dimensions for visualization.
Visualize the generated clusters.
Compare the generated clusters with Gallstone Status after clustering.
Hierarchical Clustering

Hierarchical Clustering creates a hierarchy of groups based on the similarity between observations.

This experiment uses Agglomerative Hierarchical Clustering.

The process starts by treating every observation as an individual cluster. The closest clusters are then repeatedly merged until the required number of clusters is obtained.

Ward Linkage

Ward's linkage method is used to determine which clusters should be merged.

The method attempts to minimize the increase in within-cluster variance when two clusters are combined.

The standardized feature values are used when calculating the distances between observations.

Dendrogram

A dendrogram is used to visualize the hierarchical structure of the observations.

The dendrogram shows how individual observations or groups are progressively merged.

Output:

output/hierarchical_dendrogram.png
Silhouette Analysis

Silhouette Analysis is used to determine an appropriate number of clusters.

The experiment evaluates:

Clusters = 2, 3, 4, ..., 10

For each value, the Silhouette Score is calculated.

A higher Silhouette Score generally indicates better-defined and better-separated clusters.

The number of clusters with the highest Silhouette Score is selected for the final hierarchical clustering model.

Output:

output/hierarchical_silhouette_scores.png
PCA Visualization

The clustering is performed using the complete standardized feature space.

PCA is used only to visualize the resulting clusters in two dimensions.

The first two principal components are used as the X and Y axes.

PCA is not used to perform the hierarchical clustering itself.

Output:

output/hierarchical_clusters.png
Cluster Analysis

After clustering, the mean value of each selected clinical feature is calculated for every cluster.

This allows the characteristics of the identified patient groups to be examined.

The cluster summary is saved as:

output/hierarchical_cluster_summary.csv
Cluster vs Gallstone Status

After the clustering process is completed, the generated clusters are compared with the recorded Gallstone Status.

This provides a post-clustering analysis of how the discovered patient groups relate to gallstone status.

The comparison is saved as:

output/hierarchical_cluster_vs_gallstone_status.csv

Gallstone Status is not used during model training.

Output Files
output/
├── hierarchical_dendrogram.png
├── hierarchical_silhouette_scores.png
├── hierarchical_clusters.png
├── hierarchical_cluster_summary.csv
├── hierarchical_cluster_vs_gallstone_status.csv
└── output.txt
hierarchical_dendrogram.png

Displays the hierarchical structure produced using Ward linkage.

hierarchical_silhouette_scores.png

Shows the Silhouette Score for different numbers of clusters.

hierarchical_clusters.png

Shows the generated clusters using the first two principal components for visualization.

hierarchical_cluster_summary.csv

Contains the mean clinical and body-composition measurements for each cluster.

hierarchical_cluster_vs_gallstone_status.csv

Contains the relationship between the generated clusters and the recorded gallstone status.

output.txt

Contains the complete terminal output produced during execution.

Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
SciPy
Result

Thus, Hierarchical Clustering was successfully implemented on the Gallstone medical dataset. The hierarchical structure was visualized using a dendrogram, and the optimal number of clusters was determined using Silhouette Analysis. The resulting patient clusters were subsequently visualized and analyzed.