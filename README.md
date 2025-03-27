# Temporal Fusion Transformer (TFT) Implementation

This repository contains an end-to-end pipeline for time series forecasting using the Temporal Fusion Transformer (TFT) model, including dataset handling, training experiments, and a Streamlit-based visualization app.

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `download_electricity_data.ipynb` | Notebook for downloading and preprocessing the electricity dataset used for training the TFT model. |
| `exp_variations_cat_id_21_single.py` | Trains a TFT model on electricity data using categorical ID= 21 (means for 20 customers) and a single set of hyperparameters. |
| `exp_variations_cat_id_50_iterations.py` | Runs multiple training experiments on categorical ID= 50 (means for 49 customers) using different hyperparameter combinations and logs results. |
| `display_app.py` | A Streamlit app for interactively exploring hyperparameter combinations and visualizing corresponding loss plots. |
| `simulation_and_pruning.ipynb` | Notebook for simulating model performance and experimenting with model pruning techniques (add more if needed based on content). |
