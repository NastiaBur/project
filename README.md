# Time-MoE-Mamba for Corporate Risk Forecasting

A research-oriented extension of [Time-MoE](https://github.com/Time-MoE/Time-MoE) for financial time-series forecasting with macroeconomic covariates and a Mamba-based temporal mixer.

This repository studies neural approaches to corporate risk forecasting based on:
- firm-level return series,
- market return information,
- macroeconomic and financial state variables,
- downstream risk diagnostics derived from model forecasts.

The implementation keeps the general Time-MoE training pipeline while extending it for:
- **Mamba-based temporal mixing** as an alternative to attention,
- **covariate-aware inputs**,
- **multi-horizon forecasting**,
- **macro feature fusion**,
- downstream **risk analysis**, including VaR, CoVaR, and tail-risk-oriented comparisons.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Risk Analysis](#risk-analysis)
- [Input Format](#input-format)
- [Data](#data)
- [Notes](#notes)
- [Acknowledgment](#acknowledgment)
- [Citation](#citation)
- [License](#license)

---

## Overview

The goal of this project is to develop a deep learning framework for forecasting corporate financial stress signals from open-source time-series data.

Compared with the original Time-MoE setup, this repository adds support for:
- **Mamba temporal mixer** (`--temporal_mixer mamba`)
- **main feature blocks** and **macro feature blocks**
- **macro covariate fusion**
- **covariate-aware training and evaluation**
- financial forecasting experiments on crisis periods and macro regimes

The framework is designed for experiments where the model predicts future firm returns and these predictions are later used for risk-oriented analysis.

---

## Features

- **Mixture-of-Experts forecasting backbone**
- **Attention or Mamba temporal mixer**
- **Covariate-aware input pipeline**
- **Support for macroeconomic variables**
- **Multi-horizon forecasting**
- **Evaluation on stress periods**
- **Downstream risk analysis tools**
- **Baseline comparison scripts**

---

## Repository Structure

```text
.
├── main.py
├── run_eval.py
├── sbatch.sh
├── time_moe/
│   ├── models/
│   ├── datasets/
│   ├── trainer/
│   ├── runner.py
│   └── ...
├── compare/
│   ├── baselines.py
│   ├── download_predictions.py
│   ├── compute_metrics.py
│   ├── plot_graphics.py
│   ├── compute_var_and_bands.py
│   ├── quantile_regression_covar.py
│   └── ...
└── ...
````

---

## Installation

Clone the repository and install the required dependencies in your preferred environment.

```bash
git clone <your_repository_url>
cd <repository_name>
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Depending on your setup, you may also need:

* `torch`
* `transformers`
* `pandas`
* `numpy`
* `scikit-learn`
* `statsmodels`
* `matplotlib`
* `arch`
* `yfinance`
* `pandas_datareader`

If you plan to use the Mamba temporal mixer, make sure the corresponding Mamba-related components in your environment are available and correctly configured.

---

## Quick Start

### Train a model

A typical training command has the following structure:

```bash
python main.py \
  -d <dataset_path> \
  --output_path <output_dir> \
  --train_steps <num_steps> \
  --temporal_mixer <attn_or_mamba> \
  --use_covariates \
  --main_input_size <num_main_features> \
  --macro_input_size <num_macro_features> \
  --macro_fusion <fusion_type>
```

To initialize training from scratch:

```bash
python main.py \
  -d <dataset_path> \
  --from_scratch \
  --output_path <output_dir> \
  ...
```

### Run evaluation

A typical evaluation command has the following structure:

```bash
python run_eval.py \
  -m <model_path> \
  -d <eval_dataset_path> \
  --context_length <context_length> \
  --prediction_length <prediction_length> \
  --batch_size <batch_size> \
  --use_covariates \
  --predictions_path <output_predictions_file>
```

### Cluster usage

For a ready-to-use example of cluster submission, see:

```bash
sbatch.sh
```

---

## Training

Training is launched through `main.py`.

The training pipeline supports:

* training from a pretrained checkpoint,
* training from scratch,
* attention-based temporal mixing,
* Mamba-based temporal mixing,
* covariate-aware inputs,
* configurable main and macro feature sizes.

### General training pattern

```bash
python main.py \
  -d <dataset_path> \
  --output_path <output_dir> \
  --temporal_mixer <attn_or_mamba> \
  --use_covariates \
  ...
```

### Main configurable ideas

Typical training settings include:

* dataset path
* output directory
* number of training steps or epochs
* temporal mixer type
* use of covariates
* main feature dimension
* macro feature dimension
* fusion mode
* batch size
* precision mode

### Notes

* `--temporal_mixer mamba` switches the temporal block to the Mamba version.
* `--use_covariates` enables the covariate-aware data flow.
* `--main_input_size` and `--macro_input_size` should match the dataset format.
* `--macro_fusion` controls how macro features are merged into the hidden representation.

---

## Evaluation

Evaluation is launched through `run_eval.py`.

The script supports:

* forecasting evaluation,
* covariate-aware evaluation,
* multi-horizon evaluation,
* saving predictions for downstream analysis,
* crisis-period experiments.

### General evaluation pattern

```bash
python run_eval.py \
  -m <model_path> \
  -d <eval_dataset_path> \
  --context_length <context_length> \
  --prediction_length <prediction_length> \
  --batch_size <batch_size> \
  --use_covariates \
  --predictions_path <output_predictions_file>
```

### Typical outputs

Evaluation can produce:

* direct prediction metrics,
* saved prediction files,
* plots of validation metrics,
* model outputs for risk-analysis scripts.

---

## Risk Analysis

The repository also includes utilities for risk-oriented analysis based on predicted returns.

These scripts are intended for comparing models not only in terms of forecasting accuracy, but also in terms of their implied downside-risk behavior.

### Included tools

The `compare/` directory contains utilities for:

* downloading and standardizing model predictions,
* computing forecasting metrics,
* building baseline models,
* plotting forecasts and risk quantities,
* computing downside bands and VaR-style thresholds,
* estimating quantile-regression-based CoVaR and ΔCoVaR.

### Typical analyses

Examples include:

* true returns vs predicted returns,
* mean prediction plus lower-tail band,
* VaR curves,
* CoVaR and ΔCoVaR comparisons,
* comparison against classical baselines such as GARCH, linear models, and tree-based models.

### Research use

These tools are especially useful for:

* crisis-period diagnostics,
* macro-regime comparison,
* tail-risk tracking,
* comparing models with and without macro covariates.

---

## Input Format

The project supports covariate-aware inputs in the following format:

```python
{
    "main_features": ...,
    "macro_features": ...,
    "labels": ...,
    "loss_masks": ...
}
```

Typical interpretation:

* `main_features`: firm returns and related market-level observable inputs
* `macro_features`: macroeconomic and financial state variables
* `labels`: future target values
* `loss_masks`: valid target mask

This structure is used in the covariate-aware dataset and model flow.

---

## Data

This project is designed for open financial and macroeconomic time-series data.

Typical sources include:

* equity price data
* market index data
* macroeconomic series
* financial state variables

The pipeline supports:

* training datasets,
* evaluation datasets,
* covariate-aware JSONL datasets,
* crisis-period evaluation splits.

Examples of variables that may be used include:

* firm returns,
* market returns,
* yield-curve information,
* inflation and activity indicators,
* volatility and credit-spread variables.

---

## Notes

* For cluster usage, check the example submission script in `sbatch.sh`.
* Evaluation supports covariate-aware datasets.
* Multi-horizon predictions can be used both for direct forecasting metrics and for downstream risk analysis.
* The repository is intended primarily for research and thesis experiments.
* Some components extend the original Time-MoE code path and may require careful consistency between:

  * dataset format,
  * model config,
  * evaluation setup.

---

## Acknowledgment

This project builds on the [Time-MoE](https://github.com/Time-MoE/Time-MoE) framework and extends it for financial forecasting experiments with covariates and Mamba-based temporal mixing.

