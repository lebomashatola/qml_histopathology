# ============================================================
# CLASSICAL ML BENCHMARK
# HACT EMBEDDINGS
# ============================================================

import os
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    f1_score,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = (
    "/home/lebo/Downloads/"
    "quantum-histopathology-qml/data/hact_embeddings"
)

RESULTS_DIR = (
    "/home/lebo/Downloads/"
    "quantum-histopathology-qml/results"
)

CLASSIFICATION_DIR = (
    "/home/lebo/Downloads/"
    "quantum-histopathology-qml/results/classification"
)

LOSS_DIR = (
    "/home/lebo/Downloads/"
    "quantum-histopathology-qml/results/loss"
)


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(
    CLASSIFICATION_DIR,
    exist_ok=True
)

os.makedirs(
    LOSS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

train = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "train.csv"
    )
)

val = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "val.csv"
    )
)

test = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "test.csv"
    )
)


# ============================================================
# FIND EMBEDDING COLUMNS
# ============================================================

embedding_columns = [
    column
    for column in train.columns
    if column.startswith("embedding_")
]


# ============================================================
# FEATURES
# ============================================================

X_train = train[
    embedding_columns
].values

X_val = val[
    embedding_columns
].values

X_test = test[
    embedding_columns
].values


# ============================================================
# LABELS
# ============================================================

y_train = train[
    "label"
].values

y_val = val[
    "label"
].values

y_test = test[
    "label"
].values


# ============================================================
# BASIC INFORMATION
# ============================================================

print()
print("=" * 50)
print("CLASSICAL ML BENCHMARK")
print("=" * 50)

print(
    f"Embedding dimensions: {len(embedding_columns)}"
)

print(
    f"Train samples: {len(X_train)}"
)

print(
    f"Validation samples: {len(X_val)}"
)

print(
    f"Test samples: {len(X_test)}"
)


# ============================================================
# STANDARDIZE
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_val = scaler.transform(
    X_val
)

X_test = scaler.transform(
    X_test
)


# ============================================================
# MODELS
# ============================================================

models = {

    "SVM":
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "MLP":
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=300,
            random_state=42
        )
}


# ============================================================
# TRAIN MODELS
# ============================================================

for name, model in models.items():

    print()
    print(f"Training {name}...")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    train_pred = model.predict(
        X_train
    )

    val_pred = model.predict(
        X_val
    )

    test_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # F1 SCORES
    # --------------------------------------------------------

    train_f1 = f1_score(
        y_train,
        train_pred
    )

    val_f1 = f1_score(
        y_val,
        val_pred
    )

    test_f1 = f1_score(
        y_test,
        test_pred
    )


    print(
        f"Train F1: {train_f1:.4f}"
    )

    print(
        f"Validation F1: {val_f1:.4f}"
    )

    print(
        f"Test F1: {test_f1:.4f}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_test,
        test_pred,
        target_names=[
            "Class 0",
            "Class 1"
        ]
    )


    report_path = os.path.join(
        CLASSIFICATION_DIR,
        f"{name.replace(' ', '_')}_classification_report.txt"
    )


    with open(
        report_path,
        "w"
    ) as file:

        file.write(
            f"Model: {name}\n\n"
        )

        file.write(
            f"Train F1: {train_f1:.4f}\n"
        )

        file.write(
            f"Validation F1: {val_f1:.4f}\n"
        )

        file.write(
            f"Test F1: {test_f1:.4f}\n\n"
        )

        file.write(
            "Test Classification Report\n"
        )

        file.write(
            "===========================\n\n"
        )

        file.write(
            report
        )


    # ========================================================
    # MLP LOSS
    # ========================================================

    if name == "MLP":

        loss_path = os.path.join(
            LOSS_DIR,
            "MLP_training_loss.png"
        )


        plt.figure(
            figsize=(8, 5)
        )


        plt.plot(
            model.loss_curve_
        )


        plt.xlabel(
            "Iteration"
        )

        plt.ylabel(
            "Loss"
        )

        plt.title(
            "MLP Training Loss"
        )

        plt.grid(
            True
        )

        plt.tight_layout()


        plt.savefig(
            loss_path,
            dpi=300
        )


        plt.close()


# ============================================================
# READ EVERY TXT FILE IN CLASSIFICATION DIRECTORY
# ============================================================

report_files = sorted(
    glob.glob(
        os.path.join(
            CLASSIFICATION_DIR,
            "*.txt"
        )
    )
)


print()
print(
    f"Classification reports found: {len(report_files)}"
)


# ============================================================
# EXTRACT F1 SCORES
# ============================================================

models_from_reports = []

train_f1_scores = []

val_f1_scores = []

test_f1_scores = []


for report_file in report_files:

    with open(
        report_file,
        "r"
    ) as file:

        text = file.read()


    # --------------------------------------------------------
    # MODEL NAME
    # --------------------------------------------------------

    model_name = os.path.basename(
        report_file
    )


    model_name = model_name.replace(
        "_classification_report.txt",
        ""
    )


    model_name = model_name.replace(
        "_",
        " "
    )


    # --------------------------------------------------------
    # FIND F1 SCORES
    # --------------------------------------------------------

    train_match = re.search(
        r"Train F1:\s*([0-9.]+)",
        text
    )

    val_match = re.search(
        r"Validation F1:\s*([0-9.]+)",
        text
    )

    test_match = re.search(
        r"Test F1:\s*([0-9.]+)",
        text
    )


    # --------------------------------------------------------
    # ADD REPORT TO PLOT
    # --------------------------------------------------------

    if (
        train_match
        and val_match
        and test_match
    ):

        models_from_reports.append(
            model_name
        )

        train_f1_scores.append(
            float(
                train_match.group(1)
            )
        )

        val_f1_scores.append(
            float(
                val_match.group(1)
            )
        )

        test_f1_scores.append(
            float(
                test_match.group(1)
            )
        )


# ============================================================
# CREATE F1 BAR PLOT
# ============================================================

x = np.arange(
    len(models_from_reports)
)

width = 0.25


plt.figure(
    figsize=(12, 7)
)


plt.bar(
    x - width,
    train_f1_scores,
    width,
    label="Train"
)


plt.bar(
    x,
    val_f1_scores,
    width,
    label="Validation"
)


plt.bar(
    x + width,
    test_f1_scores,
    width,
    label="Test"
)


plt.xlabel(
    "Model"
)

plt.ylabel(
    "F1 Score"
)

plt.title(
    "Classification F1 Scores"
)


plt.xticks(
    x,
    models_from_reports,
    rotation=30,
    ha="right"
)


plt.ylim(
    0,
    1
)


plt.legend()


plt.grid(
    axis="y"
)


plt.tight_layout()


# ============================================================
# SAVE F1 PLOT
# ============================================================

f1_plot_path = os.path.join(
    RESULTS_DIR,
    "classical_f1_scores.png"
)


plt.savefig(
    f1_plot_path,
    dpi=300
)


plt.close()


# ============================================================
# DONE
# ============================================================

print()
print("=" * 50)
print("CLASSICAL BENCHMARK COMPLETE")
print("=" * 50)

print()

print(
    f"F1 plot: {f1_plot_path}"
)

print(
    f"Classification reports: {CLASSIFICATION_DIR}"
)

print(
    f"MLP loss: {LOSS_DIR}"
)

print()
print("Done.")