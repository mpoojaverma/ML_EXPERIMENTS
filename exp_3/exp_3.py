import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# EXPERIMENT 3
# LINEAR REGRESSION TO PERFORM PREDICTION
# ============================================================

print("=" * 70)
print("EXPERIMENT 3")
print("AIM: To implement Linear Regression to perform prediction.")
print("=" * 70)


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

df = pd.read_csv("flood_risk_dataset_india.csv")

print("\nDATASET LOADED SUCCESSFULLY")
print("-" * 50)

print("Dataset Shape:", df.shape)

print("\nSelected Variables:")
print("Independent Variable : River Discharge (m³/s)")
print("Dependent Variable   : Water Level (m)")


# ============================================================
# STEP 2: SELECT INDEPENDENT AND DEPENDENT VARIABLES
# ============================================================

X = df["River Discharge (m³/s)"].values
Y = df["Water Level (m)"].values

# Reshape X into a 2D array
X = X.reshape(-1, 1)


# ============================================================
# STEP 3: SPLIT DATASET
# ============================================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# ============================================================
# STEP 4: LINEAR REGRESSION USING ORDINARY LEAST SQUARES
# ============================================================

def linear_regression_ols(X, Y):

    # Add column of ones for intercept
    X_b = np.c_[
        np.ones((len(X), 1)),
        X
    ]

    # Normal Equation
    theta = np.linalg.inv(
        X_b.T.dot(X_b)
    ).dot(
        X_b.T
    ).dot(Y)

    return theta


# ============================================================
# STEP 5: PREDICTION FUNCTION
# ============================================================

def predict(X, theta):

    # Add column of ones for intercept
    X_b = np.c_[
        np.ones((len(X), 1)),
        X
    ]

    # Calculate predictions
    Y_pred = X_b.dot(theta)

    return Y_pred


# ============================================================
# STEP 6: TRAIN MODEL
# ============================================================

theta = linear_regression_ols(
    X_train,
    Y_train
)


# ============================================================
# STEP 7: PREDICT TEST DATA
# ============================================================

Y_pred = predict(
    X_test,
    theta
)


# ============================================================
# STEP 8: EVALUATION
# ============================================================

mse = mean_squared_error(
    Y_test,
    Y_pred
)

r2 = r2_score(
    Y_test,
    Y_pred
)


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("LINEAR REGRESSION RESULTS")
print("=" * 60)

print("\nIntercept  :", theta[0])
print("Coefficient:", theta[1])

print("\nMean Squared Error:", mse)
print("R2 Score          :", r2)


# ============================================================
# STEP 9: VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    X_test,
    Y_test,
    alpha=0.5,
    label="Actual Data"
)

# Sort X values for a proper regression line
sorted_index = np.argsort(
    X_test[:, 0]
)

plt.plot(
    X_test[sorted_index],
    Y_pred[sorted_index],
    linewidth=2,
    label="Linear Regression Line"
)

plt.xlabel(
    "River Discharge (m³/s)"
)

plt.ylabel(
    "Water Level (m)"
)

plt.title(
    "Linear Regression: River Discharge vs Water Level"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

# Save output
plt.savefig(
    "output/linear_regression.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    "Thus, the implementation of Linear Regression "
    "to perform prediction has been successfully executed."
)