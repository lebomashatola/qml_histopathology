# Histopathology Cell and Tissue Classification

This project implements a hybrid **Quantum Machine Learning (QML) and Classical Machine Learning benchmark** for histopathology image classification. It builds cell- and tissue-level graphs using the Histocartography library and evaluates classification performance across classical and quantum models (including local simulators and actual IBM Quantum hardware).

<p align="center">
  <img src="./1.png" alt="Project Workflow Overview" width="700">
</p>


<p align="center">
  <img src="./2.png" alt="Project Workflow Overview" width="700">
</p>

## Getting Started

This repository contains two distinct pipelines. **Keep these in separate Conda environments** to avoid dependency conflicts.

### 1. Preprocessing Pipeline Setup
The preprocessing tutorial notebook handles nuclear detection, tissue graphing, and data preparation.

```bash
# Create and activate environment
conda create -n histopathology python=3.10 -y
conda activate histopathology

# Install dependencies from the repository root
pip install -r notebooks/requirements.txt
```

**Model Requirement:**
1. Download the pre-trained `pannuke.pt` model (135 MB) from [Hugging Face](https://huggingface.co/datasets/ankandebnath/histocartography-checkpoints/blob/main/pannuke.pt).
2. Save it locally.
3. Open the preprocessing notebook and update the model path variable:
   ```python
   PANNUKE_MODEL = "/path/to/your/saved/pannuke.pt"
   ```

### 2. QML & Classical Benchmark Setup
The main benchmarking framework processes generated embedding CSV files.

```bash
# Create and activate environment
conda create -n qml-benchmark python=3.10 -y
conda activate qml-benchmark

# Install dependencies from the repository root
pip install -r requirements.txt
```

## Usage

### Running Experiments
Ensure your `qml-benchmark` environment is active, then execute the main entry point:
```bash
python main.py
```

### Configuration
Experiment hyperparameters are declared inside `config.yaml`. Modify this file to alter your setup:
```yaml
quantum:
  mode: ideal         # Available options: ideal, noisy, hardware
  ansatz: efficient_su2
  optimizer: COBYLA
  maxiter: 30
  shots: 1024
```

### IBM Quantum Hardware Integration
To run workloads on a physical quantum computer, provide your IBM Quantum API credentials inside your local configuration file. 
> **Security Warning:** Never commit your plain text API keys or tokens to GitHub.

The pipeline automatically polls available backends, selects an operational QPU matching your configuration, and queues the execution.

## Results & Project Structure

All data, metrics, and figures are isolated and saved dynamically under the following directory hierarchy:
```text
results/
├── pca/                # Dimensionality reduction visualisations
├── loss/               # Model training curves
├── circuits/           # Exported quantum circuit architectures
└── classification/     # F1 scores and precision/recall data sheets
```

