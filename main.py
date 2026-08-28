from pathlib import Path
import sys

import pandas as pd
import yaml


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


from quantum_models import run_quantum_model
from classical_models import run_classical_benchmark

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# CONFIG
# ============================================================

with open(BASE_DIR / "config" / "config.yaml") as file:
    config = yaml.safe_load(file)


# ============================================================
# PATHS
# ============================================================

data_dir = BASE_DIR / config["data"]["directory"]
results_dir = BASE_DIR / config["results"]["directory"]

results_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

train = pd.read_csv(
    data_dir / config["data"]["train"]
)

val = pd.read_csv(
    data_dir / config["data"]["validation"]
)

test = pd.read_csv(
    data_dir / config["data"]["test"]
)


# ============================================================
# EMBEDDING FEATURES
# ============================================================

embedding_columns = [
    column
    for column in train.columns
    if column.startswith("embedding_")
]

if not embedding_columns:
    raise ValueError(
        "No embedding columns found."
    )


print("=" * 50)
print("QML-BDAS")
print("=" * 50)

print(f"Train: {len(train)}")
print(f"Validation: {len(val)}")
print(f"Test: {len(test)}")
print(f"Embedding dimensions: {len(embedding_columns)}")


# ============================================================
# QUANTUM
# ============================================================

print("\nRunning quantum model...")

run_quantum_model(
    train=train,
    val=val,
    test=test,
    embedding_columns=embedding_columns,
    config=config,
    results_dir=results_dir,
)


# ============================================================
# CLASSICAL
# ============================================================

print("\nRunning classical benchmark...")

run_classical_benchmark(
    train=train,
    val=val,
    test=test,
    embedding_columns=embedding_columns,
    config=config,
    results_dir=results_dir,
)


print("\nDone.")