# 1. IMPORT LIBRARIES

%matplotlib inline

import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# 2. LOAD DATASET

# Place data.csv in the current working directory.
# Alternatively, replace this with the full path to your CSV file.

data = pd.read_csv("data.csv")

# Remove unwanted unnamed index columns
data = data.loc[:, ~data.columns.str.contains(r"^Unnamed")]

print("First five rows:")
display(data.head())

print("\nDataset shape:", data.shape)

print("\nColumn data types:")
print(data.dtypes)

if "diagnosis" not in data.columns:
    raise ValueError("The dataset must contain a 'diagnosis' column.")

print("\nDiagnosis counts:")
print(data["diagnosis"].value_counts())


# 3. DATA CLEANING

df = data.copy()

# Remove the ID column because it is not a tumor measurement
if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

# Standardize diagnosis labels
df["diagnosis"] = (
    df["diagnosis"].astype(str).str.strip().str.upper()
)

# Keep recognized benign and malignant labels
df = df[df["diagnosis"].isin(["B", "M"])].copy()

# Convert predictor columns to numeric values
feature_columns = [c for c in df.columns if c != "diagnosis"]

df[feature_columns] = df[feature_columns].apply(
    pd.to_numeric, errors="coerce"
)

print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Remove rows with missing values
df.dropna(inplace=True)

print("\nDataset shape after cleaning:", df.shape)

print("\nDiagnosis counts after cleaning:")
print(df["diagnosis"].value_counts())

print("\nDuplicate rows:", df.duplicated().sum())


# 4. SEPARATE FEATURES AND TARGET

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

print("\nFeature matrix shape:", X.shape)
print("Target variable shape:", y.shape)
print("\nFeature columns:")
print(X.columns.tolist())


# 5. CORRELATION HEATMAP

correlation = X.corr()

plt.figure(figsize=(18, 14))
sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    annot=False
)
plt.title("Correlation Heatmap of Tumor Features")
plt.tight_layout()
plt.show()


# 6. BOX PLOTS: BENIGN VS MALIGNANT

features_to_plot = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean"
]

features_to_plot = [
    c for c in features_to_plot if c in df.columns
]

fig, axes = plt.subplots(2, 5, figsize=(20, 9))
axes = axes.flatten()

for ax, feature in zip(axes, features_to_plot):
    sns.boxplot(
        data=df,
        x="diagnosis",
        y=feature,
        ax=ax
    )
    ax.set_title(feature)
    ax.set_xlabel("Diagnosis")

for ax in axes[len(features_to_plot):]:
    ax.set_visible(False)

plt.suptitle("Boxplots: Benign vs Malignant")
plt.tight_layout()
plt.show()


# 7. FEATURE DISTRIBUTION PLOTS

fig, axes = plt.subplots(2, 5, figsize=(20, 9))
axes = axes.flatten()

for ax, feature in zip(axes, features_to_plot):
    sns.histplot(
        data=df,
        x=feature,
        hue="diagnosis",
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
        ax=ax
    )
    ax.set_title(feature)

for ax in axes[len(features_to_plot):]:
    ax.set_visible(False)

plt.suptitle("Feature Distributions by Diagnosis")
plt.tight_layout()
plt.show()


# 8. ENCODE TARGET LABELS

# B = 0 (Benign), M = 1 (Malignant)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nLabel mapping:")
print({
    label: int(code)
    for label, code in zip(
        label_encoder.classes_,
        label_encoder.transform(label_encoder.classes_)
    )
})


# 9. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of features:", X_train.shape[1])


# 10. BUILD MODELS
# Pipelines ensure scaling is learned only from training data.
# Scaling is particularly important for SVM.

models = {
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced"
        ))
    ])
}


# 11. TRAIN AND EVALUATE MODELS

results = []
trained_models = {}

for name, model in models.items():

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    trained_models[name] = model

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label=1, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label=1, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, pos_label=1, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Malignant Precision": precision,
        "Malignant Recall": recall,
        "Malignant F1-Score": f1
    })

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Malignant Precision: {precision:.4f}")
    print(f"Malignant Recall: {recall:.4f}")
    print(f"Malignant F1-Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        zero_division=0
    ))

    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"]
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Diagnosis")
    plt.ylabel("Actual Diagnosis")
    plt.tight_layout()
    plt.show()


# 12. COMPARE MODEL PERFORMANCE

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
display(
    results_df.sort_values(
        by="Accuracy", ascending=False
    ).reset_index(drop=True)
)


# 13. CROSS-VALIDATION
# Evaluate generalization across training-data folds.

print("\nCross-validation scores:")

for name, model in models.items():

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print(
        f"{name}: Mean accuracy = {cv_scores.mean():.4f}, "
        f"Standard deviation = {cv_scores.std():.4f}"
    )


# 14. HYPERPARAMETER TUNING FOR SVM

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(class_weight="balanced"))
])

param_grid = {
    "classifier__C": [0.1, 1, 10],
    "classifier__kernel": ["linear", "rbf"],
    "classifier__gamma": ["scale", "auto"]
}

grid_search = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest SVM parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation accuracy:")
print(round(grid_search.best_score_, 4))


# 15. EVALUATE THE TUNED SVM ON THE TEST SET

best_svm = grid_search.best_estimator_
tuned_predictions = best_svm.predict(X_test)

print("\nTuned SVM Test Accuracy:")
print(round(accuracy_score(y_test, tuned_predictions), 4))

print("\nTuned SVM Classification Report:")
print(classification_report(
    y_test,
    tuned_predictions,
    labels=[0, 1],
    target_names=["Benign", "Malignant"],
    zero_division=0
))

tuned_cm = confusion_matrix(
    y_test, tuned_predictions, labels=[0, 1]
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    tuned_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"]
)
plt.title("Tuned SVM - Confusion Matrix")
plt.xlabel("Predicted Diagnosis")
plt.ylabel("Actual Diagnosis")
plt.tight_layout()
plt.show()


# 16. SAVE THE TUNED MODEL

import pickle

with open("breast_cancer_svm_model.pkl", "wb") as file:
    pickle.dump(best_svm, file)

print("\nModel saved as breast_cancer_svm_model.pkl")
