# Credit Card Fraud Detection

PyTorch-based credit card fraud detection model focused on highly imbalanced transaction data. The project trains a multi-layer perceptron (MLP) using a weighted loss to improve sensitivity to rare fraudulent transactions and provides a minimal, testable training pipeline.

## Table of Contents

- [Project Goals](#project-goals)
- [Key Features](#key-features)
- [Dataset Expectations](#dataset-expectations)
- [Model Architecture](#model-architecture)
- [Training Workflow](#training-workflow)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Repository Structure](#repository-structure)
- [Testing](#testing)
- [Notes](#notes)
- [License](#license)

## Project Goals

- Detect fraudulent transactions in highly imbalanced datasets
- Use a weighted loss function to counter class imbalance
- Provide a reproducible, testable training loop in PyTorch

## Key Features

- MLP classifier built with PyTorch
- Weighted `BCEWithLogitsLoss` using `pos_weight` computed from training labels
- Train/validation/test split with a fixed random seed
- Optional `torch.compile` acceleration when available
- Minimal, unit-tested utilities for training and evaluation

## Dataset Expectations

The training script expects a CSV file (default: `creditcard.csv`) with:

- **All numeric feature columns**
- **A binary label column as the last column** (0 = legitimate, 1 = fraud)

The model input dimension is derived from the number of feature columns. If your dataset has a different number of features than the default, pass the correct `input_dim` value when running training.

> The dataset is not included in this repository. You must provide it locally.

## Model Architecture

The model defined in `src/model.py` is a simple MLP:

- Linear → ReLU → Linear → ReLU → Linear (single logit output)
- Default hidden size: 128
- Output is trained with `BCEWithLogitsLoss`

## Training Workflow

1. Load the CSV into a Pandas dataframe
2. Convert to `TensorDataset`, separating features and labels
3. Split into train/validation/test sets (default 80/10/10)
4. Compute `pos_weight` from training labels to counter imbalance
5. Train with Adam optimizer and evaluate every `eval_every` epochs
6. Save the best model (lowest validation loss)

## Setup

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> The optional `script.sh` helper installs dependencies and runs the model using Python 3.10.

## Usage

Run training with the default settings:

```bash
python src/model.py
```

To point at a different dataset path, update the `data_path` parameter in `main()` or call the function from a small wrapper script.

## Configuration

The training entrypoint is `main()` in `src/model.py` and accepts:

- `data_path` (default: `creditcard.csv`)
- `epochs` (default: 500)
- `eval_every` (default: 100)
- `patience` (default: 500)
- `batch_size` (default: 1024)
- `num_workers` (default: 4)
- `input_dim` (default: 28)
- `hidden_dim` (default: 128)
- `enable_compile` (default: True)
- `compile_backend` (default: `inductor`)

If the dataset has a different number of features, set `input_dim` to match `(num_feature_columns)`.

## Outputs

- `best_model.pth` saved in the repository root
- Console logs with training/validation loss checkpoints and final test loss

## Repository Structure

```
.
├── src/
│   └── model.py          # Model, training, evaluation logic
├── test/
│   └── test_model.py     # Unit tests for model helpers
├── reports/
│   └── report.pdf        # Project report
├── requirements.txt
├── script.sh             # Helper script for setup + run
└── README.md
```

## Testing

```bash
pytest -v
```

## Notes

- Data preprocessing/feature engineering is not performed in the training script; ensure your CSV is already cleaned and ready for modeling.
- For best performance on GPU, install a CUDA-enabled PyTorch build (see the [PyTorch installation guide](https://pytorch.org/get-started/locally/)).

## License

[MIT](LICENSE)
