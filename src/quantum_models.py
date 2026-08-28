import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, classification_report

from qiskit.circuit.library import zz_feature_map
from qiskit.circuit.library import efficient_su2
from qiskit.circuit.library import TwoLocal

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA, SPSA


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = (
    "/home/lebo/Downloads/"
    "quantum-histopathology-qml/data/hact_embeddings"
)

RESULTS_DIR = (
    "/home/lebo/Downloads/"
    "quantum-histopathology-qml/results"
)

MODE = "ideal"                  # ideal, noisy, hardware
ANSATZ = "efficient_su2"        # efficient_su2, two_local
ANSATZ_REPS = 2

OPTIMIZER = "COBYLA"            # COBYLA, SPSA
MAXITER = 30

SHOTS = 1024

IBM_CONFIG = "/home/lebo/.ibm_quantum.json"


# ============================================================
# DIRECTORIES
# ============================================================

for directory in [
    "pca",
    "loss",
    "circuits",
    "classification"
]:
    os.makedirs(
        os.path.join(RESULTS_DIR, directory),
        exist_ok=True
    )


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv")
)

val = pd.read_csv(
    os.path.join(DATA_DIR, "val.csv")
)

test = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv")
)


# ============================================================
# FEATURES AND LABELS
# ============================================================

embedding_columns = [
    column
    for column in train.columns
    if column.startswith("embedding_")
]

N_QUBITS = len(embedding_columns)

X_train = train[embedding_columns].values
y_train = train["label"].values

X_val = val[embedding_columns].values
y_val = val["label"].values

X_test = test[embedding_columns].values
y_test = test["label"].values


print(
    f"Mode: {MODE} | "
    f"Qubits: {N_QUBITS} | "
    f"Ansatz: {ANSATZ} | "
    f"Optimizer: {OPTIMIZER}"
)


# ============================================================
# PCA
# ============================================================

X_all = np.concatenate(
    [X_train, X_val, X_test]
)

y_all = np.concatenate(
    [y_train, y_val, y_test]
)

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_all
)

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    X_pca[y_all == 0, 0],
    X_pca[y_all == 0, 1],
    label="N"
)

plt.scatter(
    X_pca[y_all == 1, 0],
    X_pca[y_all == 1, 1],
    label="UDH / PB"
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "PCA of HACT Embeddings"
)

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "pca",
        "hact_embeddings_pca.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# SCALE EMBEDDINGS
# ============================================================

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

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
# ZZ FEATURE MAP
# ============================================================

feature_map = zz_feature_map(
    feature_dimension=N_QUBITS,
    reps=1
)


# ============================================================
# ANSATZ
# ============================================================

if ANSATZ == "efficient_su2":

    ansatz = efficient_su2(
        num_qubits=N_QUBITS,
        reps=ANSATZ_REPS
    )

elif ANSATZ == "two_local":

    ansatz = TwoLocal(
        num_qubits=N_QUBITS,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz",
        reps=ANSATZ_REPS
    )

else:

    raise ValueError(
        "ANSATZ must be 'efficient_su2' or 'two_local'"
    )


# ============================================================
# OPTIMIZER
# ============================================================

if OPTIMIZER == "COBYLA":

    optimizer = COBYLA(
        maxiter=MAXITER
    )

elif OPTIMIZER == "SPSA":

    optimizer = SPSA(
        maxiter=MAXITER
    )

else:

    raise ValueError(
        "OPTIMIZER must be 'COBYLA' or 'SPSA'"
    )


# ============================================================
# CIRCUIT
# ============================================================

circuit = feature_map.compose(
    ansatz
)


# ============================================================
# SAVE CIRCUIT
# ============================================================

circuit_file = os.path.join(
    RESULTS_DIR,
    "circuits",
    f"{MODE}_{ANSATZ}_{OPTIMIZER}.png"
)

try:

    circuit.draw(
        output="mpl",
        filename=circuit_file
    )

except Exception:
    pass


# ============================================================
# EXECUTION BACKEND
# ============================================================

pass_manager = None


# ------------------------------------------------------------
# IDEAL
# ------------------------------------------------------------

if MODE == "ideal":

    from qiskit.primitives import StatevectorSampler

    sampler = StatevectorSampler(
        default_shots=SHOTS
    )


# ------------------------------------------------------------
# NOISY
# ------------------------------------------------------------

elif MODE == "noisy":

    from qiskit_aer import AerSimulator

    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error
    )

    from qiskit.primitives import BackendSamplerV2

    from qiskit.transpiler.preset_passmanagers import (
        generate_preset_pass_manager
    )

    noise_model = NoiseModel()

    error_1q = depolarizing_error(
        0.01,
        1
    )

    error_2q = depolarizing_error(
        0.02,
        2
    )

    noise_model.add_all_qubit_quantum_error(
        error_1q,
        ["x", "sx"]
    )

    noise_model.add_all_qubit_quantum_error(
        error_2q,
        ["cx"]
    )

    backend = AerSimulator(
        noise_model=noise_model
    )

    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=1
    )

    sampler = BackendSamplerV2(
        backend=backend
    )


# ------------------------------------------------------------
# HARDWARE
# ------------------------------------------------------------

elif MODE == "hardware":

    from qiskit_ibm_runtime import (
        QiskitRuntimeService,
        SamplerV2
    )

    from qiskit.transpiler.preset_passmanagers import (
        generate_preset_pass_manager
    )

    with open(
        IBM_CONFIG,
        "r"
    ) as file:

        config = json.load(file)

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=config["token"],
        instance=config["instance"]
    )

    backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=N_QUBITS
    )

    print(
        f"QPU: {backend.name}"
    )

    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=1
    )

    sampler = SamplerV2(
        mode=backend
    )

    sampler.options.default_shots = SHOTS


else:

    raise ValueError(
        "MODE must be 'ideal', 'noisy', or 'hardware'"
    )


# ============================================================
# TRAINING LOSS
# ============================================================

loss_history = []


def callback(weights, loss):

    loss_history.append(
        loss
    )


# ============================================================
# VQC
# ============================================================

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    loss="cross_entropy",
    callback=callback,
    pass_manager=pass_manager
)


# ============================================================
# TRAIN
# ============================================================

print("Training...")

vqc.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

y_train_pred = vqc.predict(
    X_train
)

y_val_pred = vqc.predict(
    X_val
)

y_test_pred = vqc.predict(
    X_test
)


# ============================================================
# F1 SCORES
# ============================================================

train_f1 = f1_score(
    y_train,
    y_train_pred
)

val_f1 = f1_score(
    y_val,
    y_val_pred
)

test_f1 = f1_score(
    y_test,
    y_test_pred
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_test_pred,
    target_names=[
        "N",
        "UDH / PB"
    ]
)

report_file = os.path.join(
    RESULTS_DIR,
    "classification",
    f"{MODE}_{ANSATZ}_{OPTIMIZER}_classification_report.txt"
)

with open(
    report_file,
    "w"
) as file:

    file.write(
        f"Execution: {MODE}\n"
    )

    file.write(
        f"Qubits: {N_QUBITS}\n"
    )

    file.write(
        "Feature map: ZZFeatureMap\n"
    )

    file.write(
        f"Ansatz: {ANSATZ}\n"
    )

    file.write(
        f"Optimizer: {OPTIMIZER}\n\n"
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


# ============================================================
# LOSS PLOT
# ============================================================

loss_file = os.path.join(
    RESULTS_DIR,
    "loss",
    f"{MODE}_{ANSATZ}_{OPTIMIZER}_training_loss.png"
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    loss_history
)

plt.xlabel(
    "Iteration"
)

plt.ylabel(
    "Loss"
)

plt.title(
    f"VQC Training Loss - {MODE}"
)

plt.grid()
plt.tight_layout()

plt.savefig(
    loss_file,
    dpi=300
)

plt.close()


# ============================================================
# FINAL RESULT
# ============================================================

print(
    f"Train F1: {train_f1:.4f} | "
    f"Val F1: {val_f1:.4f} | "
    f"Test F1: {test_f1:.4f}"
)