"""
GSoC 2025 ContinuousML4DQM

This script trains the DepthwiseCrossViTAE model on a specified set of
runs, calculates its baseline error statistics, and determines the optimal
Z-score thresholds for anomaly detection. The resulting model, stats, and
thresholds are saved for use in the evaluation script.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader ,ConcatDataset
try:
    from .model_datasets import *
    from .models_spatial import *
    from .utils import *
except ImportError:
    from model_datasets import *
    from models_spatial import *
    from utils import *
import torch.optim as optim
import os
import json

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "tools", "dqm", "test_files")
BASE = os.path.join(PROJECT_ROOT, "sandbox")
RUNS = [323997] #first parameter and only for now i assume
NUM_EPOCHS = 2
LR = 3e-3
BATCH_SIZE = 320
MASK_PATH = os.path.join(DATA_DIR, "dataset", "he_segmentation_config_mask.npy")
BASE_DIR = os.path.join(BASE, "test")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model.pth")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
ANOMALY_STRENGTHS = [0.0,0.2, 0.4, 0.6, 0.8, 2.0]
ANOMALY_DIR = os.path.join(BASE_DIR, "anomalies")

# Model configuration
feature_dim = 1
latent_dim = 56
target_dim = 1
spatial_dims = (64, 72, 7)
memorysize = 1
image_size = [64, 72]
k_factor = 8
num_layers = 5
mask_ratio = 0.5
patch_size = 12
isvariational = True

from orchestral.tools import define_tool


def _resolve_train_data_path(dataset_dir, run):
    run_path = os.path.join(dataset_dir, f"he_train_dataset_Run{run}", "train_data.npy")
    if os.path.exists(run_path):
        return run_path
    fallback_path = os.path.join(dataset_dir, "train_data.npy")
    if os.path.exists(fallback_path):
        return fallback_path
    return run_path


def _resolve_test_data_path(dataset_dir, run):
    run_path = os.path.join(dataset_dir, f"he_test_dataset_Run{run}", "test_data.npy")
    if os.path.exists(run_path):
        return run_path
    fallback_path = os.path.join(dataset_dir, "test_data.npy")
    if os.path.exists(fallback_path):
        return fallback_path
    return run_path


def _train_model_on_runs_impl(RUNS=None, NUM_EPOCHS=None, LR=None, BATCH_SIZE=None, MODEL_SAVE_PATH=None, ANOMALY_STRENGTHS=None, SEED=None, ANOMALY_DIR=None, BASE_DIR=None, DATASET_DIR=None, MASK_PATH=None, isvariational=None):
    # Use module-level defaults if not provided
    if RUNS is None:
        RUNS = globals()['RUNS']
    if NUM_EPOCHS is None:
        NUM_EPOCHS = globals()['NUM_EPOCHS']
    if LR is None:
        LR = globals()['LR']
    if BATCH_SIZE is None:
        BATCH_SIZE = globals()['BATCH_SIZE']
    if MODEL_SAVE_PATH is None:
        MODEL_SAVE_PATH = globals()['MODEL_SAVE_PATH']
    if ANOMALY_STRENGTHS is None:
        ANOMALY_STRENGTHS = globals()['ANOMALY_STRENGTHS']
    if SEED is None:
        SEED = globals()['SEED']
    if ANOMALY_DIR is None:
        ANOMALY_DIR = globals()['ANOMALY_DIR']
    if BASE_DIR is None:
        BASE_DIR = globals()['BASE_DIR']
    if DATASET_DIR is None:
        DATASET_DIR = globals()['DATASET_DIR']
    if MASK_PATH is None:
        MASK_PATH = globals()['MASK_PATH']
    if isvariational is None:
        isvariational = globals()['isvariational']
    
    # Convert RUNS to list if it's an integer
    if isinstance(RUNS, int):
        RUNS = [RUNS]

    if not ANOMALY_STRENGTHS:
        ANOMALY_STRENGTHS = list(globals()['ANOMALY_STRENGTHS'])
    
    # Validate that required data paths exist
    print(f"Using DATASET_DIR: {DATASET_DIR}")
    print(f"Using RUNS: {RUNS}")
    
    for run in RUNS:
        train_path = _resolve_train_data_path(DATASET_DIR, run)
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Training data not found at {train_path}\n"
                f"Make sure DATASET_DIR is correct and contains either:\n"
                f"  {DATASET_DIR}/he_train_dataset_Run{run}/train_data.npy\n"
                f"or fallback file:\n"
                f"  {DATASET_DIR}/train_data.npy"
            )
    
    model = DepthwiseCrossViTAE_MultiDim_SPATIAL(
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        target_dim=target_dim,
        spatial_dims=spatial_dims,
        memorysize=memorysize,
        image_size=image_size,
        k_factor=k_factor,
        num_layers=num_layers,
        mask_ratio=mask_ratio,
        patch_size=patch_size,
        isvariational=isvariational
    ).to(device)

    # Load datasets
    train_datasets = []
    validation_datasets = []

    for run in RUNS:
        train_data_path = _resolve_train_data_path(DATASET_DIR, run)
        test_data_path = _resolve_test_data_path(DATASET_DIR, run)
        train_datasets.append(CustomDataset(train_data_path, MASK_PATH))
        validation_datasets.append(CustomDataset(test_data_path, MASK_PATH))

    # Combine datasets
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_validation_dataset = ConcatDataset(validation_datasets)

    # Create DataLoaders
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(combined_train_dataset, batch_size=int(BATCH_SIZE), shuffle=True, generator=g, worker_init_fn=lambda x: np.random.seed(42))
    validation_loader = DataLoader(combined_validation_dataset, batch_size=int(BATCH_SIZE), shuffle=False)

    print(f"Combined training samples: {len(combined_train_dataset)}")
    print(f"Combined validation samples: {len(combined_validation_dataset)}")

    # Train the model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    train_losses, validation_losses = train_model(
        model,
        train_loader,
        validation_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LR,
        weight_decay=1e-7,
        vae_reg_beta=1e-6,
        patience=2,
        model_save_path=MODEL_SAVE_PATH,
        isvariational=isvariational,
    )
    model.eval()
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Parent model saved to {MODEL_SAVE_PATH}")

    # Calculate and save base model statistics

    os.makedirs(BASE_DIR, exist_ok=True)

    subdetector_mask = np.load(MASK_PATH)
    mask_bool = subdetector_mask.astype(bool)



    datasets_to_combine = []
    for run_id in RUNS:
        train_path = _resolve_train_data_path(DATASET_DIR, run_id)
        if os.path.exists(train_path):
            datasets_to_combine.append(CustomDataset(train_path, MASK_PATH))

    if datasets_to_combine:
        combined_dataset = ConcatDataset(datasets_to_combine)
        a_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False)
    else:
        a_loader = None

    model.eval()

    with torch.no_grad():
        all_targets = []
        all_preds = []

        for data in a_loader:
            data = data.to(device)
            pred, _, _ = model(data)
            pred = pred.squeeze(axis=-1)
            all_targets.append(data.cpu())
            all_preds.append(pred.cpu())

        targets_tensor = torch.cat(all_targets, dim=0).numpy()
        preds_tensor = torch.cat(all_preds, dim=0).numpy()

        targets_tensor = np.squeeze(targets_tensor, axis=(1))
        preds_tensor = np.squeeze(preds_tensor, axis=(1))

        masked_targets_tensor = np.where(mask_bool, targets_tensor, np.nan)
        masked_preds_tensor = np.where(mask_bool, preds_tensor, np.nan)

        error = np.abs(masked_preds_tensor - masked_targets_tensor)
        mean_error = np.nanmean(error, axis=0)
        std_error = np.nanstd(error, axis=0)

        # Convert to torch tensors and handle near-zero std for later use
        mean_err_array = torch.tensor(mean_error).to(device).squeeze(-1)
        std_err_array = torch.tensor(std_error).to(device).squeeze(-1)
        std_err_array = torch.where(std_err_array.abs() < 1e-6, mean_err_array, std_err_array)

        np.save(f"{BASE_DIR}/error_mean.npy", mean_error)
        np.save(f"{BASE_DIR}/error_std.npy", std_error)
        print(f"Base model statistics saved to {BASE_DIR}")

    # Calculate and save optimal thresholds for anomaly detection
    print("Calculating optimal thresholds for anomaly detection...")

    # Get test data for threshold calculation
    test_datasets_for_threshold = []
    for run in RUNS:
        test_data_path = _resolve_test_data_path(DATASET_DIR, run)
        if os.path.exists(test_data_path):
            test_datasets_for_threshold.append(CustomDataset(test_data_path, MASK_PATH))

    combined_test_dataset = ConcatDataset(test_datasets_for_threshold)
    test_loader_threshold = DataLoader(combined_test_dataset, batch_size=32, shuffle=False)

    # Calculate normal z-scores (baseline)
    predictions_normal, targets_normal = predict(model, test_loader_threshold)
    normal_z_scores = []
    for pred, tgt in zip(predictions_normal, targets_normal):
        z_score = calc_max_zscore(pred, tgt, mean_err_array, std_err_array, subdetector_mask)
        normal_z_scores.append(z_score)
    normal_z_scores = np.array(normal_z_scores)

    # Generate anomalies and calculate thresholds for different strengths

    os.makedirs(ANOMALY_DIR, exist_ok=True)

    # Load original test data for anomaly generation
    test_data_for_anomalies = []
    for run in RUNS:
        test_data_path = _resolve_test_data_path(DATASET_DIR, run)
        if os.path.exists(test_data_path):
            test_data_for_anomalies.append(np.load(test_data_path))

    combined_test_data = np.concatenate(test_data_for_anomalies, axis=0)


    anomaly_zscore_files = {}
    anomaly_info_dict = {}

    for strength in ANOMALY_STRENGTHS:
        print(f"Processing anomaly strength {strength}...")

        # Generate anomalies
        time_window = 4
        spatial_spread = 1
        LS_distribution = lambda: np.random.randint(1, time_window)
        spread_distribution = lambda: np.random.randint(0, spatial_spread)

        modified_data, anomalies_info = generate_anomalies(
            combined_test_data, subdetector_mask, LS_distribution,
            spread_distribution, anomaly_strength=strength, seed=SEED,
            anomaly_probability=0.5
        )

        # Save anomaly data
        anomaly_file = f"{ANOMALY_DIR}/anomalous_data_{strength}.npy"
        np.save(anomaly_file, modified_data)

        # Calculate z-scores for anomalous data
        anomaly_dataset = CustomDataset(anomaly_file, MASK_PATH)
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=32, shuffle=False)

        predictions_anom, targets_anom = predict(model, anomaly_loader)
        z_scores_anom = []
        for pred, tgt in zip(predictions_anom, targets_anom):
            z_score = calc_max_zscore(pred, tgt, mean_err_array, std_err_array, subdetector_mask)
            z_scores_anom.append(z_score)
        z_scores_anom = np.array(z_scores_anom)

        # Save z-scores
        zscore_file = f"{ANOMALY_DIR}/zscore_str_{strength}.npy"
        np.save(zscore_file, z_scores_anom)
        anomaly_zscore_files[zscore_file] = str(strength)
        anomaly_info_dict[str(strength)] = anomalies_info

    # Calculate optimal thresholds
    optimal_thresholds = calculate_optimal_thresholds(normal_z_scores, anomaly_zscore_files, anomaly_info_dict)

    # Save optimal thresholds
    thresholds_file = f"{BASE_DIR}/optimal_thresholds.json"
    with open(thresholds_file, 'w') as f:
        json.dump(optimal_thresholds, f, indent=2)

    print(f"Optimal thresholds saved to: {thresholds_file}")
    print("Optimal thresholds:")
    for strength, threshold in optimal_thresholds.items():
        print(f"  Strength {strength}: {threshold:.4f}")

    return {
        "status": "ok",
        "runs": [int(r) for r in RUNS],
        "model_path": str(MODEL_SAVE_PATH),
        "base_dir": str(BASE_DIR),
        "anomaly_dir": str(ANOMALY_DIR),
        "thresholds_file": str(thresholds_file),
        "optimal_thresholds": {str(k): float(v) for k, v in optimal_thresholds.items()},
        "num_epochs": int(NUM_EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "learning_rate": float(LR),
        "anomaly_strengths": [float(x) for x in ANOMALY_STRENGTHS],
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        "final_val_loss": float(validation_losses[-1]) if validation_losses else None,
        "train_loss_curve": [float(x) for x in train_losses] if train_losses else [],
        "val_loss_curve": [float(x) for x in validation_losses] if validation_losses else [],
    }

