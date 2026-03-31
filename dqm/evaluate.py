"""
GSoC 2025 ContinuousML4DQM

This script evaluates a pre-trained model using the artifacts from the
training script. It benchmarks the model's anomaly detection performance
across multiple datasets, generating detailed summary tables and plots to
visualize the results.
"""

import os
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

def compute_model_error_stats(model,a_loader,mask_path, batch_size=128, device='cpu'):
    # Load mask
    subdetector_mask = np.load(mask_path)
    mask_bool = subdetector_mask.astype(bool)
    model.eval()
    all_targets, all_preds = [], []

    with torch.no_grad():
        for data in a_loader:
            data = data.to(device)
            pred, _, _ = model(data)
            pred = pred.squeeze(-1)
            all_targets.append(data.cpu())
            all_preds.append(pred.cpu())

    targets_tensor = torch.cat(all_targets, dim=0).numpy()
    preds_tensor = torch.cat(all_preds, dim=0).numpy()

    targets_tensor = np.squeeze(targets_tensor, axis=1)
    preds_tensor = np.squeeze(preds_tensor, axis=1)

    masked_targets_tensor = np.where(mask_bool, targets_tensor, np.nan)
    masked_preds_tensor = np.where(mask_bool, preds_tensor, np.nan)

    error = np.abs(masked_preds_tensor - masked_targets_tensor)
    mean_error = np.nanmean(error, axis=0)
    std_error = np.nanstd(error, axis=0)

    # Convert to torch tensors and handle near-zero std
    mean_err_array = torch.tensor(mean_error).to(device).squeeze(-1)
    std_err_array = torch.tensor(std_error).to(device).squeeze(-1)
    std_err_array = torch.where(std_err_array.abs() < 1e-6, mean_err_array, std_err_array)

    return mean_err_array, std_err_array

#Imports and Seed Initialization
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader,ConcatDataset,Subset
from model_datasets import *
from models_spatial import *
import torch.optim as optim
import os
import json
import matplotlib.pyplot as plt
from utils import *

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MODEL_RUN = "test_models/parent_model_exp1_test"
eval_runs = [323940, 323997, 324021, 324022, 325117, 325170,355456, 355769, 357081, 379154, 383512, 383631,386694, 386885]
# ^ some 2018 , some 2022 , some 2024 runs for eval orginially the model was trained on first three run
EXP = 1
ANOMALY_STRENGTHS = [0.0, 0.2, 0.4, 0.6, 0.8, 2.0]
mask_path = "../data/he_segmentation_config_mask.npy"
model_path = f"{MODEL_RUN}.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def _evaluate_model_on_run_impl(eval_run, model_path, mask_path, dataset_dir, anomaly_strengths, seed, results_dir):
    """
    Internal implementation of model evaluation.
    
    Args:
        eval_run: Run ID to evaluate on
        model_path: Path to the trained model
        mask_path: Path to the segmentation mask
        dataset_dir: Path to the dataset directory
        anomaly_strengths: List of anomaly strengths to test
        seed: Random seed
        results_dir: Directory to save results
    """
    
    global model
    raise FileNotFoundError(f"Model not found at {model_path} for run {MODEL_RUN}")

print(f"Loading Model from Run {MODEL_RUN}")
model = load_model(model, model_path)
model = model.eval()
final_res = []
all_results = []
for EVAL_RUN in eval_runs:
    # Step 5: Load Datasets and Generate/Load Statistics
    test_dataset = CustomDataset(f"../data/dataset/he_test_dataset_Run{EVAL_RUN}/test_data.npy", mask_path)
    test_dataset2 = CustomDataset(f"../data/dataset/he_train_dataset_Run{EVAL_RUN}/train_data.npy", mask_path)

    # Combine
    test_dataset = ConcatDataset([test_dataset, test_dataset2])

    # Loader with same name
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    stats_path = f"{MODEL_RUN}"
    print(f"Loading statistics for run {MODEL_RUN}")
    import os
    # uncomment below if you wanna use stored mean and std_dev error from training
    # mean_err_array, std_err_array = get_mean_std_dev_arrays(f"{stats_path}/error_mean.npy", f"{stats_path}/error_std.npy")
    mean_err_array, std_err_array = compute_model_error_stats(model,test_loader, mask_path, batch_size=128, device=device)
    # using the computed error statistics from the same eval run
    subdetector_mask = np.load(mask_path)

    # Load optimal thresholds
    thresholds_file = f"{stats_path}/optimal_thresholds.json"
    optimal_thresholds = {}
    if os.path.exists(thresholds_file):
        with open(thresholds_file, 'r') as f:
            optimal_thresholds = json.load(f)
        print(f"Loaded optimal thresholds from: {thresholds_file}")
        for strength, threshold in optimal_thresholds.items():
            print(f"  Strength {strength}: {threshold:.4f}")
    else:
        print(f"Optimal thresholds file not found at: {thresholds_file}")
        print("Using default threshold of 5.5 for all strengths")

    if hasattr(model, 'mask_ratio'):
        model.mask_ratio = 0.0

    # Generate Anomaly Data if it does not exist
    train_data = np.load(f"../data/dataset/he_train_dataset_Run{EVAL_RUN}/train_data.npy")
    test_data = np.load(f"../data/dataset/he_test_dataset_Run{EVAL_RUN}/test_data.npy")
    combined_data = np.concatenate((train_data, test_data), axis=0)

    anomaly_dir = f'../data/anomaly_generated/he_combined_dataset_Run{EVAL_RUN}/'
    os.makedirs(anomaly_dir, exist_ok=True)
    mask = np.load(mask_path)

    # Dictionary to store anomaly info for each strength
    anomaly_info_dict = {}

    for strength in ANOMALY_STRENGTHS:
        anomaly_file = f"{anomaly_dir}/anomalous_data_{strength}.npy"
        anomaly_info_file = f"{anomaly_dir}/anomaly_info_{strength}.json"

        if not os.path.exists(anomaly_file):
            print(f"Generating anomalies with strength {strength}...")
            time_window = 4
            spatial_spread = 1
            LS_distribution = lambda: np.random.randint(1, time_window)
            spread_distribution = lambda: np.random.randint(0, spatial_spread)
            modified_data, anomalies_info = generate_anomalies(combined_data, mask, LS_distribution, spread_distribution, anomaly_strength=strength, seed=SEED, anomaly_probability=0.3)
            # print(anomalies_info)
            np.save(anomaly_file, modified_data)

            # Save anomaly info as JSON
            with open(anomaly_info_file, 'w') as f:
                json.dump(anomalies_info, f, indent=2)

            print(f"Anomalous dataset saved to: {anomaly_file}")
            print(f"Anomaly info saved to: {anomaly_info_file}")

        # Load anomaly info (whether newly created or existing)
        if os.path.exists(anomaly_info_file):
            with open(anomaly_info_file, 'r') as f:
                anomaly_info_dict[str(strength)] = json.load(f)
        else:
            anomaly_info_dict[str(strength)] = []
    zscore_dir = "../data/zscores"
    os.makedirs(zscore_dir, exist_ok=True)
    anomaly_zscore_files = {}
    for strength in ANOMALY_STRENGTHS:
        print(f"Calculating Z-scores for anomaly strength {strength}...")
        anomaly_file = f"{anomaly_dir}/anomalous_data_{strength}.npy"
        anomaly_dataset = CustomDataset(anomaly_file, mask_path)
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=32, shuffle=False, worker_init_fn=lambda x: np.random.seed(SEED))

        predictions_anom, target_anom = predict(model, anomaly_loader)

        z_scores_anom = []
        for pred, tgt in zip(predictions_anom, target_anom):
            z_score = calc_max_zscore(pred, tgt, mean_err_array, std_err_array, subdetector_mask)
            z_scores_anom.append(z_score)

        z_scores_anom = np.array(z_scores_anom)

        zscore_file = f"{zscore_dir}/anomaly_str_{strength}.npy"
        np.save(zscore_file, z_scores_anom)
        anomaly_zscore_files[zscore_file] = str(strength)
        print(f"Saved anomalous Z-scores (strength {strength}) to {zscore_file}")

    # Generate and Print Final Evaluation Table
    print(f"\n{'='*65}")
    print("SINGLE MODEL EVALUATION RESULTS")
    print(f"{'='*65}")
    print(f"Model: Run {MODEL_RUN}")
    print(f"Test Data: Run {EVAL_RUN}")
    print(f"{'='*65}")

    results = generate_anomaly_table_with_anomaly_info(anomaly_zscore_files, anomaly_info_dict, EVAL_RUN, optimal_thresholds, 5.5)
    final_res.append(results)
    all_results.extend(results)
    print_table(results)

plot_evaluation_results(all_results, "plots")
