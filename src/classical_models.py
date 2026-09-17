import re
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report


def run_classical_benchmark(train, val, test, embedding_columns, config, results_dir):

    results_dir = Path(results_dir)
    classification_dir = results_dir / "classification"
    loss_dir = results_dir / "loss"

    classification_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)


    # FEATURES AND LABELS

    X_train = train[embedding_columns].values
    X_val   = val[embedding_columns].values
    X_test  = test[embedding_columns].values

    y_train = train["label"].values
    y_val   = val["label"].values
    y_test  = test["label"].values

    print()
    print("=" * 50)
    print("CLASSICAL ML BENCHMARK")
    print("=" * 50)
    print(f"Embedding dimensions: {len(embedding_columns)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")


    # STANDARDIZE

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)


    # LOAD MODELS

    ccfg = config.get("classical", {})

    mlp_cfg = ccfg.get("mlp", {})
    rf_cfg  = ccfg.get("random_forest", {})

    models = {

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),

        "Random_Forest": RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            random_state=rf_cfg.get("random_state", 42)
        ),

        "MLP": MLPClassifier(
            hidden_layer_sizes=tuple(mlp_cfg.get("hidden_layer_sizes", [32, 16])),
            max_iter=mlp_cfg.get("max_iter", 300),
            random_state=mlp_cfg.get("random_state", 42)
        ),
    }


    # TRAIN MODELS

    for name, model in models.items():

        print()
        print(f"Training {name}...")

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred   = model.predict(X_val)
        test_pred  = model.predict(X_test)

        train_f1 = f1_score(y_train, train_pred)
        val_f1   = f1_score(y_val,   val_pred)
        test_f1  = f1_score(y_test,  test_pred)

        print(f"Train F1: {train_f1:.4f}")
        print(f"Validation F1: {val_f1:.4f}")
        print(f"Test F1: {test_f1:.4f}")

        report = classification_report(
            y_test,
            test_pred,
            target_names=["Class 0", "Class 1"]
        )

        report_path = classification_dir / f"{name}_classification_report.txt"

        with open(report_path, "w") as f:
            f.write(f"Model: {name}\n\n")
            f.write(f"Train F1: {train_f1:.4f}\n")
            f.write(f"Validation F1: {val_f1:.4f}\n")
            f.write(f"Test F1: {test_f1:.4f}\n\n")
            f.write("Test Classification Report\n")
            f.write("===========================\n\n")
            f.write(report)

        if name == "MLP":

            plt.figure(figsize=(8, 5))
            plt.plot(model.loss_curve_)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("MLP Training Loss")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(loss_dir / "MLP_training_loss.png", dpi=300)
            plt.close()


    # AGGREGATE F1 BAR CHART

    report_files = sorted(
        glob.glob(str(classification_dir / "*.txt"))
    )

    print()
    print(f"Classification reports found: {len(report_files)}")

    model_names    = []
    train_f1_list  = []
    val_f1_list    = []
    test_f1_list   = []

    for report_file in report_files:

        with open(report_file, "r") as f:
            text = f.read()

        model_name = (
            Path(report_file)
            .stem
            .replace("_classification_report", "")
            .replace("_", " ")
        )

        train_match = re.search(r"Train F1:\s*([0-9.]+)", text)
        val_match   = re.search(r"Validation F1:\s*([0-9.]+)", text)
        test_match  = re.search(r"Test F1:\s*([0-9.]+)", text)

        if train_match and val_match and test_match:
            model_names.append(model_name)
            train_f1_list.append(float(train_match.group(1)))
            val_f1_list.append(float(val_match.group(1)))
            test_f1_list.append(float(test_match.group(1)))

    x = np.arange(len(model_names))
    width = 0.25

    plt.figure(figsize=(12, 7))
    plt.bar(x - width, train_f1_list, width, label="Train")
    plt.bar(x,         val_f1_list,   width, label="Validation")
    plt.bar(x + width, test_f1_list,  width, label="Test")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.title("Classification F1 Scores")
    plt.xticks(x, model_names, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(results_dir / "classical_f1_scores.png", dpi=300)
    plt.close()


    print()
    print("=" * 50)
    print("CLASSICAL BENCHMARK COMPLETE")
    print("=" * 50)
    print(f"F1 plot: {results_dir / 'classical_f1_scores.png'}")
    print(f"Classification reports: {classification_dir}")
    print(f"MLP loss: {loss_dir}")
