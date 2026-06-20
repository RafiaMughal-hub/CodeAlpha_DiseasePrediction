# ============================================================================
# HEART DISEASE PEDICTION PROJECT
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# ===========================================================================
# LOAD DATASET
# ===========================================================================

df = pd.read_csv("heart.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst Five Rows:")
print(df.head())

# ============================================================================
# FEATURES & TARGET
# ============================================================================

x = df.drop("target",  axis=1)
y = df["target"]

# ============================================================================
# TRAIN TEST SPLIT
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================================
# FEATURE SCALING 
# ============================================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================================
# LOGISTIC REGRESSION
# ============================================================================

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(
    y_test,
    lr_pred
)

print("\nLogistic Regression Accuracy:")
print(round(lr_accuracy, 4))

# ===========================================================================
# RANDOM FOREST
# ===========================================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print("\nRandom Forest Accuracy:")
print(round(rf_accuracy, 4))

# ============================================================================
# ROC-AUC SCORE
# ============================================================================

rf_probs = rf_model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(
    y_test,
    rf_probs
)

print("\nROC-AUC Score:")
print(round(roc_auc, 4))

# ============================================================================
# BEST MODEL 
# ============================================================================

if rf_accuracy > lr_accuracy:
    best_model = "Random Forest"
else:
    best_model = "Logistic Regression"

print("\nBest Model:")
print(best_model)

# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        rf_pred
    )
)

# ============================================================================
# CONFUSION MATRIX
# ============================================================================

cm = confusion_matrix(
    y_test,
    rf_pred
)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ============================================================================
# ACCURACY COMPARISON GRAPH
# ============================================================================

models = [
    "Logistic Regression"
    "Random Forest"
]

accuracies = [
    lr_accuracy,
    rf_accuracy
]

plt.figure(figsize=(6,4))

plt.bar(
    models,
    accuracies
)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.show()

# ============================================================================
# ROC CURVE
# ============================================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    rf_probs
)

plt.figure(figsize=(6,4))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

print("\nProject Completed Successfully!")