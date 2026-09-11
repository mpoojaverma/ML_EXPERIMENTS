# Experiment 4a – Bayesian Logistic Regression

## Aim

To implement Bayesian Logistic Regression for classification.

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

The target represents whether a flood occurred.

## Algorithm

1. Load the Flood Risk Dataset.
2. Select the required input features and target variable.
3. Split the dataset into training and testing sets.
4. Standardize the input features using `StandardScaler`.
5. Add an intercept term to the feature matrix.
6. Define the sigmoid function.
7. Define the log-posterior function using:
   - Logistic regression likelihood
   - Gaussian prior
8. Initialize the model parameters.
9. Apply the Metropolis-Hastings algorithm to generate posterior samples.
10. Discard the burn-in samples.
11. Calculate the posterior mean of the model coefficients.
12. Use the posterior mean coefficients to calculate prediction probabilities.
13. Classify the observations using a probability threshold of 0.5.
14. Evaluate the classification model using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
15. Visualize the posterior distributions of the coefficients.

## Bayesian Logistic Regression

Bayesian Logistic Regression estimates the probability of a binary outcome while treating the model parameters as probability distributions.

The sigmoid function is used to convert the linear combination of features into a probability:

```text
P(y = 1 | X) = 1 / (1 + e^(-Xθ))

A Gaussian prior is applied to the model coefficients.

The posterior distribution is obtained using the likelihood and prior:

Posterior ∝ Likelihood × Prior

Since the posterior distribution is not calculated directly, the Metropolis-Hastings algorithm is used to obtain samples from the posterior distribution.

Metropolis-Hastings Sampling

The experiment uses the Metropolis-Hastings Markov Chain Monte Carlo method.

Parameters used:

Total samples: 5000
Burn-in samples: 1000
Proposal standard deviation: 0.05
Random seed: 42

After sampling, the first 1000 samples are discarded as burn-in and the remaining samples are used to estimate the posterior mean of the coefficients.

Evaluation Metrics
Accuracy

Measures the proportion of correctly classified observations.

Precision

Measures how many observations predicted as flood occurrences are actually flood occurrences.

Recall

Measures how many actual flood occurrences are correctly identified.

F1 Score

The F1 score is the harmonic mean of precision and recall.

Visualization

The experiment generates a histogram showing the posterior distributions of the logistic regression coefficients.


Result

Thus, Bayesian Logistic Regression was successfully implemented for flood occurrence classification.