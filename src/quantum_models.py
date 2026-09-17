import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, classification_report

from qiskit.circuit.library import zz_feature_map
from qiskit.circuit.library import efficient_su2
from qiskit.circuit.library import TwoLocal

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA, SPSA


_BASE_DIR = Path(__file__).resolve().parent.parent


def run_quantum_model(train, val, test, embedding_columns, config, results_dir):

    # ============================================================
    # CONFIG
    # ============================================================

    qcfg = config["quantum"]

    MODE       = qcfg["mode"]
    ANSATZ     = qcfg["ansatz"]
    ANSATZ_REPS = qcfg.get("ansatz_reps", 2)
    OPTIMIZER  = qcfg["optimizer"]
    MAXITER    = qcfg["maxiter"]
    SHOTS      = qcfg["shots"]
    IBM_CONFIG = str(_BASE_DIR / qcfg.get("ibm_config", "configs/credentials.json").lstrip("/"))

    results_dir = Path(results_dir)

    # ============================================================
    # DIRECTORIES
    # ============================================================

    for directory in ["pca", "loss", "circuits", "classification"]:
        (results_dir / directory).mkdir(parents=True, exist_ok=True)


    # ============================================================
    # FEATURES AND LABELS
    # ============================================================

    X_train = train[embedding_columns].values
    y_train = train["label"].values

    X_val = val[embedding_columns].values
    y_val = val["label"].values

    X_test = test[embedding_columns].values
    y_test = test["label"].values

    N_QUBITS = len(embedding_columns)

    print(
        f"Mode: {MODE} | "
        f"Qubits: {N_QUBITS} | "
        f"Ansatz: {ANSATZ} | "
        f"Optimizer: {OPTIMIZER}"
    )


    # ============================================================
    # PCA
    # ============================================================

    X_all = np.concatenate([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[y_all == 0, 0], X_pca[y_all == 0, 1], label="N")
    plt.scatter(X_pca[y_all == 1, 0], X_pca[y_all == 1, 1], label="UDH / PB")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of HACT Embeddings")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(results_dir / "pca" / "hact_embeddings_pca.png", dpi=300)
    plt.close()


    # ============================================================
    # SCALE EMBEDDINGS
    # ============================================================

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)


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
            "ansatz must be 'efficient_su2' or 'two_local'"
        )


    # ============================================================
    # OPTIMIZER
    # ============================================================

    if OPTIMIZER == "COBYLA":

        optimizer = COBYLA(maxiter=MAXITER)

    elif OPTIMIZER == "SPSA":

        optimizer = SPSA(maxiter=MAXITER)

    else:

        raise ValueError(
            "optimizer must be 'COBYLA' or 'SPSA'"
        )


    # ============================================================
    # CIRCUIT — save diagram
    # ============================================================

    circuit = feature_map.compose(ansatz)

    try:
        circuit.draw(
            output="mpl",
            filename=str(results_dir / "circuits" / f"{MODE}_{ANSATZ}_{OPTIMIZER}.png")
        )
    except Exception:
        pass


    # ============================================================
    # EXECUTION BACKEND
    # ============================================================

    pass_manager = None

    if MODE == "ideal":

        from qiskit.primitives import StatevectorSampler

        sampler = StatevectorSampler(default_shots=SHOTS)

    elif MODE == "noisy":

        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        from qiskit.primitives import BackendSamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["x", "sx"])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])

        backend = AerSimulator(noise_model=noise_model)
        pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
        sampler = BackendSamplerV2(backend=backend)

    elif MODE == "hardware":

        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        with open(IBM_CONFIG, "r") as f:
            ibm_creds = json.load(f)

        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=ibm_creds["token"],
            instance=ibm_creds["instance"]
        )

        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=N_QUBITS
        )

        print(f"QPU: {backend.name}")

        pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = SHOTS

    else:

        raise ValueError(
            "mode must be 'ideal', 'noisy', or 'hardware'"
        )


    # ============================================================
    # VQC
    # ============================================================

    loss_history = []

    def callback(weights, loss):
        loss_history.append(loss)

    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        loss="cross_entropy",
        callback=callback,
        pass_manager=pass_manager
    )

    print("Training...")

    vqc.fit(X_train, y_train)


    # ============================================================
    # PREDICTIONS & SCORES
    # ============================================================

    y_train_pred = vqc.predict(X_train)
    y_val_pred   = vqc.predict(X_val)
    y_test_pred  = vqc.predict(X_test)

    train_f1 = f1_score(y_train, y_train_pred)
    val_f1   = f1_score(y_val,   y_val_pred)
    test_f1  = f1_score(y_test,  y_test_pred)


    # ============================================================
    # CLASSIFICATION REPORT
    # ============================================================

    report = classification_report(
        y_test,
        y_test_pred,
        target_names=["N", "UDH / PB"]
    )

    report_file = results_dir / "classification" / f"{MODE}_{ANSATZ}_{OPTIMIZER}_classification_report.txt"

    with open(report_file, "w") as f:
        f.write(f"Execution: {MODE}\n")
        f.write(f"Qubits: {N_QUBITS}\n")
        f.write("Feature map: ZZFeatureMap\n")
        f.write(f"Ansatz: {ANSATZ}\n")
        f.write(f"Optimizer: {OPTIMIZER}\n\n")
        f.write(f"Train F1: {train_f1:.4f}\n")
        f.write(f"Validation F1: {val_f1:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n\n")
        f.write("Test Classification Report\n")
        f.write("===========================\n\n")
        f.write(report)


    # ============================================================
    # LOSS PLOT
    # ============================================================

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"VQC Training Loss - {MODE}")
    plt.grid()
    plt.tight_layout()
    plt.savefig(results_dir / "loss" / f"{MODE}_{ANSATZ}_{OPTIMIZER}_training_loss.png", dpi=300)
    plt.close()


    # ============================================================
    # RESULT
    # ============================================================

    print(
        f"Train F1: {train_f1:.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"Test F1: {test_f1:.4f}"
    )
