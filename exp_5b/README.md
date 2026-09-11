# Experiment 5b – Gaussian Mixture Model

## Aim

To implement Gaussian Mixture Model (GMM) for clustering.

## Dataset

**Gallstone Medical Dataset**

The dataset contains clinical, demographic, body-composition, and laboratory measurements of patients.

### Dataset Details

- Number of records: **319**
- Number of attributes: **39**
- Missing values: **0**

The dataset contains meaningful medical measurements related to body composition, metabolic characteristics, liver-related measurements, kidney function, blood parameters, and other clinical characteristics.

## Objective

The objective of this experiment is to group patients into clusters based on their clinical and body-composition characteristics using a Gaussian Mixture Model.

The experiment treats clustering as an **unsupervised learning problem**.

The `Gallstone Status` variable is not used as an input feature while training the clustering model. It is used only after clustering to examine the relationship between the generated clusters and the recorded gallstone status.

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

It is used only for post-clustering analysis to examine how the generated clusters correspond to the recorded gallstone status.

Algorithm
Load the Gallstone medical dataset.
Check the dataset shape and missing values.
Select the numerical clinical and body-composition features.
Exclude Gallstone Status from the clustering input.
Standardize the selected features using StandardScaler.
Fit Gaussian Mixture Models with different numbers of components.
Evaluate models using the Bayesian Information Criterion (BIC).
Evaluate models using the Akaike Information Criterion (AIC).
Select the number of mixture components using the minimum BIC value.
Train the final Gaussian Mixture Model using the selected number of components.
Assign each patient to the most probable cluster.
Calculate cluster membership probabilities.
Calculate the Silhouette Score for the generated clusters.
Calculate the size of each cluster.
Generate a cluster-wise summary of the clinical features.
Apply PCA to reduce the feature space to two dimensions for visualization.
Visualize the resulting clusters.
Compare the generated clusters with Gallstone Status after clustering.
Gaussian Mixture Model

A Gaussian Mixture Model represents the data as a mixture of multiple Gaussian probability distributions.

Each Gaussian component represents a potential cluster.

Unlike K-Means, which assigns each observation to one cluster based on the nearest centroid, GMM provides a probability of membership for each observation across the different clusters.

The model estimates:

Mean of each Gaussian component
Covariance of each component
Mixing probability of each component

The Gaussian Mixture Model is fitted using the Expectation-Maximization (EM) approach.

Model Selection

Different numbers of Gaussian components are evaluated:

K = 2, 3, 4, ..., 10

Two information criteria are calculated:

Bayesian Information Criterion (BIC)

BIC evaluates model fit while applying a penalty for model complexity.

A lower BIC value indicates a better model according to the BIC criterion.

Akaike Information Criterion (AIC)

AIC evaluates the model fit while penalizing model complexity.

A lower AIC value indicates a better model according to the AIC criterion.

In this experiment, BIC is used as the primary criterion for selecting the number of Gaussian components.

Model Configuration

The final Gaussian Mixture Model uses:

Covariance type: full
Random state: 42
Multiple initializations: 10

The number of components is selected automatically based on the minimum BIC value.

Silhouette Score

The Silhouette Score is calculated after clustering to evaluate the separation of the generated clusters.

A higher score generally indicates better-separated clusters.

The Silhouette Score is used as an additional clustering evaluation measure and is not the primary criterion for selecting the number of GMM components.

Cluster Membership Probabilities

One important advantage of GMM over K-Means is that GMM provides a probability distribution over cluster membership.

For every patient, the model calculates probabilities such as:

Cluster 0 Probability
Cluster 1 Probability
Cluster 2 Probability
...

The probabilities indicate how strongly the observation belongs to each Gaussian component.

The sample membership probabilities are saved as:

output/sample_membership_probabilities.csv
PCA Visualization

The clustering is performed using the complete standardized feature space.

PCA is used only to visualize the resulting clusters in two dimensions.

The first two principal components are used for visualization.

PCA is not used as the input to the GMM clustering model.

The visualization is saved as:

output/gmm_clusters.png
Cluster Analysis

The mean value of each selected clinical feature is calculated for every cluster.

This allows the characteristics of the patient groups to be examined.

The cluster summary is saved as:

output/gmm_cluster_summary.csv
Cluster vs Gallstone Status

After the GMM clustering is completed, the generated clusters are compared with the recorded Gallstone Status.

This provides a post-clustering interpretation of how the discovered patient groups relate to gallstone status.

The comparison is saved as:

output/gmm_cluster_vs_gallstone_status.csv

Gallstone Status is not used during model training.

Results

For the executed experiment:

Dataset size: 319 records
Number of clustering features: 32
Missing values: 0
Best number of components according to BIC: 4
Best number of components according to AIC: 10
Selected number of components: 4
BIC of selected model: 15111.73115219098
AIC of selected model: 6666.407508644574
Silhouette Score: 0.1070952101
Cluster Sizes
Cluster 0 → 41
Cluster 1 → 116
Cluster 2 → 40
Cluster 3 → 122

The model therefore produced four patient clusters based on the selected clinical and body-composition features.

PCA Visualization Results

The first two principal components explain:

PC1 Variance = 0.2753530473
PC2 Variance = 0.1988044347

Total Variance Explained = 0.4741574819

Thus, the first two principal components explain approximately 47.42% of the total standardized feature variance.

Output Files
output/
├── gmm_model_selection.png
├── gmm_clusters.png
├── gmm_cluster_summary.csv
├── gmm_cluster_vs_gallstone_status.csv
├── sample_membership_probabilities.csv
└── output.txt
gmm_model_selection.png

Shows the BIC and AIC values for different numbers of Gaussian components.

gmm_clusters.png

Shows the generated GMM clusters using the first two principal components for visualization.

gmm_cluster_summary.csv

Contains the mean values of the selected clinical features for each cluster.

gmm_cluster_vs_gallstone_status.csv

Contains the relationship between the generated clusters and the recorded gallstone status.

sample_membership_probabilities.csv

Contains the GMM cluster membership probabilities for sample observations.

output.txt

Contains the complete terminal output produced during execution.

Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Result

Thus, Gaussian Mixture Model clustering was successfully implemented on the Gallstone medical dataset. The number of mixture components was selected using the Bayesian Information Criterion (BIC), and the resulting patient clusters were analyzed and visualized.

