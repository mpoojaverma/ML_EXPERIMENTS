# ============================================================
# EXPERIMENT 4.1
# BAYESIAN LOGISTIC REGRESSION FOR CLASSIFICATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


print("=" * 75)
print("EXPERIMENT 4.1")
print("AIM: To implement Bayesian Logistic Regression for classification.")
print("=" * 75)


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("flood_risk_dataset_india.csv")

print("\nDATASET LOADED SUCCESSFULLY")
print("-" * 55)
print("Dataset Shape:", df.shape)


# ------------------------------------------------------------
# 2. SELECT FEATURES AND TARGET
# ------------------------------------------------------------

X = df[
    [
        "Rainfall (mm)",
        "Temperature (°C)",
        "Humidity (%)",
        "River Discharge (m³/s)",
        "Water Level (m)",
        "Elevation (m)",
        "Population Density"
    ]
]

y = df["Flood Occurred"]


print("\nSelected Features:")
print(list(X.columns))

print("\nTarget Variable:")
print("Flood Occurred")


# ------------------------------------------------------------
# 3. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# ------------------------------------------------------------
# 4. STANDARDIZE FEATURES
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Add intercept column
X_train_b = np.c_[np.ones((len(X_train_scaled), 1)), X_train_scaled]
X_test_b = np.c_[np.ones((len(X_test_scaled), 1)), X_test_scaled]


# ------------------------------------------------------------
# 5. SIGMOID FUNCTION
# ------------------------------------------------------------

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------
# 6. LOG POSTERIOR
# ------------------------------------------------------------

def log_posterior(theta, X, y, prior_variance=10.0):

    z = X @ theta
    p = sigmoid(z)

    # Log likelihood
    log_likelihood = np.sum(
        y * np.log(p + 1e-10)
        + (1 - y) * np.log(1 - p + 1e-10)
    )

    # Gaussian prior
    log_prior = -np.sum(theta ** 2) / (2 * prior_variance)

    return log_likelihood + log_prior


# ------------------------------------------------------------
# 7. METROPOLIS-HASTINGS SAMPLING
# ------------------------------------------------------------

np.random.seed(42)

n_samples = 5000
burn_in = 1000

theta_current = np.zeros(X_train_b.shape[1])

current_posterior = log_posterior(
    theta_current,
    X_train_b,
    y_train.values
)

samples = []

proposal_std = 0.05

accepted = 0

for i in range(n_samples):

    theta_proposed = (
        theta_current
        + np.random.normal(
            0,
            proposal_std,
            size=theta_current.shape
        )
    )

    proposed_posterior = log_posterior(
        theta_proposed,
        X_train_b,
        y_train.values
    )

    log_acceptance_ratio = (
        proposed_posterior
        - current_posterior
    )

    if (
        np.log(np.random.rand())
        < log_acceptance_ratio
    ):
        theta_current = theta_proposed
        current_posterior = proposed_posterior
        accepted += 1

    samples.append(theta_current.copy())


samples = np.array(samples)

# Remove burn-in samples
posterior_samples = samples[burn_in:]


print("\n" + "=" * 60)
print("BAYESIAN SAMPLING RESULTS")
print("=" * 60)

print("\nTotal Samples     :", n_samples)
print("Burn-in Samples   :", burn_in)
print("Posterior Samples :", len(posterior_samples))

acceptance_rate = accepted / n_samples

print("Acceptance Rate   :", acceptance_rate)


# ------------------------------------------------------------
# 8. POSTERIOR MEAN PARAMETERS
# ------------------------------------------------------------

theta_mean = np.mean(
    posterior_samples,
    axis=0
)

print("\nPosterior Mean Coefficients:")
print(theta_mean)


# ------------------------------------------------------------
# 9. PREDICTION
# ------------------------------------------------------------

probabilities = sigmoid(
    X_test_b @ theta_mean
)

y_pred = (
    probabilities >= 0.5
).astype(int)


# ------------------------------------------------------------
# 10. EVALUATION
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 60)
print("CLASSIFICATION RESULTS")
print("=" * 60)

print("\nAccuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1 Score  :", f1)


# ------------------------------------------------------------
# 11. POSTERIOR COEFFICIENT DISTRIBUTIONS
# ------------------------------------------------------------

feature_names = [
    "Intercept",
    "Rainfall",
    "Temperature",
    "Humidity",
    "River Discharge",
    "Water Level",
    "Elevation",
    "Population Density"
]

plt.figure(figsize=(10, 6))

for i in range(
    min(4, posterior_samples.shape[1])
):
    plt.hist(
        posterior_samples[:, i],
        bins=40,
        alpha=0.6,
        label=feature_names[i]
    )

plt.xlabel("Coefficient Value")
plt.ylabel("Frequency")
plt.title(
    "Posterior Distributions of Logistic Regression Coefficients"
)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "output/posterior_coefficients.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 12. RESULT
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("RESULT")
print("=" * 75)

print(
    "Thus, Bayesian Logistic Regression was successfully "
    "implemented for flood occurrence classification."
)