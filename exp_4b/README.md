# Experiment 4b – Support Vector Machine

## Aim

To implement Support Vector Machine (SVM) for classification.

## Dataset

**Flood Risk Dataset – India**

The dataset contains 10,000 records and 14 attributes related to environmental, geographical, demographic, and flood-related conditions in India.

## Objective

The objective of this experiment is to classify whether a flood occurred based on selected environmental and geographical features.

### Input Features

- `Rainfall (mm)`
- `Temperature (°C)`
- `Humidity (%)`
- `River Discharge (m³/s)`
- `Water Level (m)`
- `Elevation (m)`
- `Population Density`

### Target Variable

- `Flood Occurred`

## Algorithm

1. Load the Flood Risk Dataset.
2. Select the required input features and target variable.
3. Split the dataset into training and testing sets.
4. Standardize the input features using `StandardScaler`.
5. Create an SVM classifier.
6. Use the Radial Basis Function (RBF) kernel.
7. Train the SVM model using the training data.
8. Predict the target values for the test data.
9. Evaluate the model using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
10. Generate the confusion matrix.
11. Visualize the confusion matrix.

## SVM Model Parameters

The SVM classifier is implemented using:

- Kernel: `RBF`
- `C`: `1.0`
- Gamma: `scale`

The RBF kernel is used to handle non-linear relationships between the input features and the target classes.

## Feature Scaling

Feature scaling is performed using `StandardScaler` before training the SVM model.

This places the numerical features on a comparable scale and is important for distance- and margin-based algorithms such as SVM.

## Evaluation Metrics

### Accuracy

Measures the proportion of correctly classified observations out of all test observations.

### Precision

Measures the proportion of observations predicted as flood occurrences that are actually flood occurrences.

### Recall

Measures the proportion of actual flood occurrences that are correctly identified.

### F1 Score

The F1 score is the harmonic mean of precision and recall.

## Confusion Matrix

A confusion matrix is generated to show the classification performance for:

- No Flood
- Flood

The visualization is saved as:

```text
output/svm_confusion_matrix.png

