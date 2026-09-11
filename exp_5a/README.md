# Experiment 5a – K-Means Clustering

## Aim

To implement K-Means clustering for grouping similar data points into clusters.

## Dataset

**Gallstone Medical Dataset**

The Gallstone dataset contains clinical, demographic, body-composition, and laboratory measurements of patients.

### Dataset Details

- Number of records: **319**
- Number of attributes: **39**
- Missing values: **0**

The dataset contains meaningful medical features related to body composition, metabolic measurements, liver-related measurements, kidney function, blood parameters, and other clinical characteristics.

## Objective

The objective of this experiment is to group patients into clusters based on their clinical and body-composition characteristics using the K-Means clustering algorithm.

The clustering is performed as an **unsupervised learning task**.

The `Gallstone Status` variable is **not used as an input feature for clustering**. It is used only after clustering to examine the relationship between the discovered clusters and the actual gallstone status.

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

This variable is not used during K-Means training.

It is used only for post-clustering analysis to compare the generated clusters with the recorded gallstone status.

Algorithm
Load the Gallstone medical dataset.
Check the dataset shape and missing values.
Select the numerical clinical and body-composition features.
Exclude Gallstone Status from the clustering input.
Standardize the selected features using StandardScaler.
Apply the Elbow Method for values of K from 2 to 10.
Calculate the inertia for each value of K.
Apply Silhouette Analysis for values of K from 2 to 10.
Select the number of clusters based on the highest Silhouette Score.
Train the final K-Means clustering model using the selected value of K.
Assign every patient to a cluster.
Calculate the size of each cluster.
Generate a cluster-wise summary of the clinical features.
Apply PCA to reduce the standardized feature space to two dimensions for visualization.
Visualize the resulting clusters.
Compare the generated clusters with Gallstone Status after clustering.
Feature Standardization

The clinical features have different units and numerical ranges.

For example:

Age is measured in years.
Height is measured in height units.
Weight is measured in weight units.
Glucose and cholesterol have different numerical scales.
Body-composition measurements have different ranges.

Therefore, StandardScaler is used before applying K-Means.

Standardization transforms the features so that they have approximately:

Mean = 0
Standard Deviation = 1

This prevents features with larger numerical ranges from dominating the distance calculations.

K-Means Clustering

K-Means is an unsupervised clustering algorithm that divides observations into K clusters.

The algorithm works by:

Selecting K initial cluster centroids.
Assigning each observation to the nearest centroid.
Recalculating the centroid of each cluster.
Reassigning observations based on the updated centroids.
Repeating the process until the clusters stabilize.

The objective is to minimize the within-cluster sum of squared distances.

Elbow Method

The Elbow Method is used to study how the clustering inertia changes as the number of clusters increases.

Inertia represents the within-cluster sum of squared distances.

The experiment evaluates:

K = 2, 3, 4, ..., 10

The resulting values are visualized using an Elbow Curve.

Output:

output/elbow_curve.png
Silhouette Analysis

Silhouette Score is used to evaluate how well-separated the clusters are.

The score considers:

How close an observation is to other observations within its own cluster.
How far the observation is from observations in other clusters.

A higher Silhouette Score generally indicates better-defined clustering.

The experiment calculates the Silhouette Score for:

K = 2, 3, 4, ..., 10

The value of K with the highest Silhouette Score is selected as the optimal number of clusters.

Output:

output/silhouette_scores.png
PCA Visualization

The clustering is performed using the complete standardized feature space.

Since the dataset contains 32 clustering features, PCA is used only to create a two-dimensional visualization of the resulting clusters.

PCA is not used to create the K-Means clusters.

The first two principal components are used as the X and Y axes for visualization.

Output:

output/kmeans_clusters.png
Cluster Analysis

After clustering, the mean value of each selected clinical feature is calculated for every cluster.

This helps identify differences in the characteristics of the patient groups.

The cluster summary is saved as:

output/cluster_summary.csv
Cluster vs Gallstone Status

After the clusters are generated, the clusters are compared with the Gallstone Status variable.

This analysis helps examine how the unsupervised clusters correspond to the recorded gallstone status.

The comparison is saved as:

output/cluster_vs_gallstone_status.csv

Gallstone Status is therefore used only for interpretation and is not used to train the clustering model.

Results

For the executed experiment:

Dataset size: 319 records
Clustering features: 32
Missing values: 0
Optimal number of clusters: 2
Best Silhouette Score: 0.2061301917
Cluster 0 size: 155
Cluster 1 size: 164

The Elbow Method and Silhouette Analysis were used to determine the clustering configuration.

Output Files
output/
├── elbow_curve.png
├── silhouette_scores.png
├── kmeans_clusters.png
├── cluster_summary.csv
├── cluster_vs_gallstone_status.csv
└── output.txt
elbow_curve.png

Shows the variation of K-Means inertia for different numbers of clusters.

silhouette_scores.png

Shows the Silhouette Score for different numbers of clusters.

kmeans_clusters.png

Shows the K-Means clusters using the first two principal components for visualization.

cluster_summary.csv

Contains the mean values of the selected clinical features for each cluster.

cluster_vs_gallstone_status.csv

Contains the relationship between the generated clusters and the recorded gallstone status.

output.txt

Contains the complete terminal output produced during execution.

Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Result

Thus, K-Means clustering was successfully implemented on the Gallstone medical dataset. The optimal number of clusters was determined using the Elbow Method and Silhouette Analysis, and the resulting patient clusters were visualized and analyzed.

