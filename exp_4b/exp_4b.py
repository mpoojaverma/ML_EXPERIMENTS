# ============================================================
# EXPERIMENT 4.2
# SUPPORT VECTOR MACHINE FOR CLASSIFICATION
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("=" * 75)
print("EXPERIMENT 4.2")
print("AIM: To implement Support Vector Machine for classification.")
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
# 4. FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# 5. CREATE SVM MODEL
# ------------------------------------------------------------

model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)


# ------------------------------------------------------------
# 6. TRAIN MODEL
# ------------------------------------------------------------

model.fit(
    X_train_scaled,
    y_train
)

print("\nSVM MODEL TRAINED SUCCESSFULLY")


# ------------------------------------------------------------
# 7. PREDICTION
# ------------------------------------------------------------

y_pred = model.predict(
    X_test_scaled
)


# ------------------------------------------------------------
# 8. EVALUATION
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
print("SVM CLASSIFICATION RESULTS")
print("=" * 60)

print("\nKernel    :", "RBF")
print("C         :", 1.0)
print("Gamma     :", "scale")

print("\nAccuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1 Score  :", f1)


# ------------------------------------------------------------
# 9. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ------------------------------------------------------------
# 10. VISUALIZE CONFUSION MATRIX
# ------------------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Flood", "Flood"]
)

disp.plot()

plt.title("SVM Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "output/svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 11. RESULT
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("RESULT")
print("=" * 75)

print(
    "Thus, Support Vector Machine was successfully "
    "implemented for flood occurrence classification."
)