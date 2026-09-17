from pathlib import Path
import sys
import warnings

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


# ============================================================
# CONFIG
# ============================================================

with open(BASE_DIR / "configs" / "experiment.yaml") as file:
    config = yaml.safe_load(file)


# ============================================================
# PATHS
# ============================================================

results_dir = BASE_DIR / "results"
results_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD & NORMALISE EMBEDDINGS
# ============================================================

data_cfg = config["data"]
source = data_cfg["source"]          # "hact" or "full"

if source == "hact":

    data_dir = BASE_DIR / "data" / "hact_embeddings"

    train = pd.read_csv(data_dir / "train.csv")
    val   = pd.read_csv(data_dir / "val.csv")
    test  = pd.read_csv(data_dir / "test.csv")

    # already has "label" and "embedding_*" columns — nothing to rename
    embedding_columns = [c for c in train.columns if c.startswith("embedding_")]

elif source == "full":

    dims = data_cfg["dimensions"]          # 10, 32, or 64
    data_dir = BASE_DIR / "data" / "full_embeddings" / str(dims)

    train = pd.read_csv(data_dir / "features_train.csv", index_col=0)
    val   = pd.read_csv(data_dir / "features_val.csv",   index_col=0)
    test  = pd.read_csv(data_dir / "features_test.csv",  index_col=0)

    # rename columns to match the expected schema
    for df in [train, val, test]:
        df.rename(columns={"image_label": "label"}, inplace=True)
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        df.rename(columns={c: c.replace("feat_", "embedding_") for c in feat_cols}, inplace=True)

    embedding_columns = [c for c in train.columns if c.startswith("embedding_")]

else:
    raise ValueError(f"data.source must be 'hact' or 'full', got: '{source}'")

if not embedding_columns:
    raise ValueError("No embedding columns found.")


warnings.filterwarnings("ignore")

print("=" * 50)
print("Quantum Machine Learning")
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
