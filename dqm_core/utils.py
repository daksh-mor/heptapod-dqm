"""
GSoC 2025 ContinuousML4DQM

This utility script contains shared functions and classes for data handling,
model training, anomaly generation, and metric calculation. It provides the
core logic imported by the main training and evaluation scripts to ensure
code modularity and reusability.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import json
import os
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import torch
from torch.utils.data import Dataset

import numpy as np

import numpy as np

class QuantileScalerCustom:
    def __init__(self, ignore_zeros=False, iqr_eps=1e-6):
        """
        Scales data using (x - Q1) / IQR
        - Falls back to z-score if IQR < iqr_eps
        - Ignores zeros if ignore_zeros=True
        """
        self.ignore_zeros = ignore_zeros
        self.iqr_eps = iqr_eps

    def transform(self, data, mask):
        data = np.array(data)
        data = np.squeeze(data, axis=-1)

        # Apply mask
        if self.ignore_zeros:
            masked_data = np.where((mask == 1) & (data != 0), data, np.nan)
        else:
            masked_data = np.where(mask == 1, data, np.nan)

        # Compute Q1, Q3, IQR
        q1 = np.nanpercentile(masked_data, 25, axis=(1, 2, 3))
        q3 = np.nanpercentile(masked_data, 75, axis=(1, 2, 3))
        iqr = q3 - q1

        # For fallback: compute mean/std
        mean = np.nanmean(masked_data, axis=(1, 2, 3))
        std = np.nanstd(masked_data, axis=(1, 2, 3)) + 1e-10

        # Decide per-sample whether to use IQR or std
        use_zscore = iqr < self.iqr_eps

        # Expand shapes for broadcasting
        q1 = q1[:, None, None, None]
        iqr = iqr[:, None, None, None]
        mean = mean[:, None, None, None]
        std = std[:, None, None, None]

        scaled_iqr = (masked_data - q1) / (iqr + 1e-10)
        scaled_std = (masked_data - mean) / std

        # Choose per batch element
        use_zscore = use_zscore[:, None, None, None]
        scaled = np.where(use_zscore, scaled_std, scaled_iqr)

        return np.nan_to_num(scaled, nan=0.0)
class ArrayDataset(Dataset):
    """A simple PyTorch Dataset that wraps an in-memory NumPy array."""
    def __init__(self, data_array):
        self.data = torch.tensor(data_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CustomDataset(Dataset):
    def __init__(self, path, mask_path):
        data = np.load(path)
        mask = np.load(mask_path)

        scaler = QuantileScalerCustom()
        scaled_data = scaler.transform(data, mask)

        # Add channel dimension
        scaled_data = np.expand_dims(scaled_data, axis=1)

        self.data = torch.tensor(scaled_data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def shape(self):
        return self.data.shape

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

import torch
from torch import nn, optim
from tqdm import tqdm

def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=1,
    learning_rate=1e-3,
    weight_decay=1e-7,
    vae_reg_beta=1e-6,
    patience=10,
    model_save_path=None,
    isvariational=True
):
    device = next(model.parameters()).device
    criterion = nn.MSELoss()

    # Check if model is trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)

    if not trainable_params:
        print("WARNING: All parameters are frozen! Training aborted.")
        return

    print(f"Training with {trainable_count:,} / {total_params:,} trainable parameters")

    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay, amsgrad=True)

    train_losses, test_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as t:
            for batch in t:
                batch = batch.to(device)
                optimizer.zero_grad()

                if isvariational:
                    output, mu, logvar = model(batch)
                    output = output.squeeze(-1)
                    recon_loss = criterion(output, batch)
                    kl_loss = kl_divergence(mu, logvar).mean()
                    loss = recon_loss + vae_reg_beta * kl_loss
                else:
                    output = model(batch)
                    loss = criterion(output, batch)

                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()
                t.set_postfix(train_loss=loss.item())

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                if isvariational:
                    output, mu, logvar = model(batch)
                    output = output.squeeze(-1)
                    recon_loss = criterion(output, batch)
                    kl_loss = kl_divergence(mu, logvar).mean()
                    loss = recon_loss + vae_reg_beta * kl_loss
                else:
                    output = model(batch)
                    loss = criterion(output, batch)

                running_test_loss += loss.item()

        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"[{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        # Early stopping and checkpointing
        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            epochs_no_improve = 0
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
                print(f"✅ Model saved at: {model_save_path}")
        else:
            epochs_no_improve += 1
            print(f"⚠️  No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping after {patience} no-improve epochs.")
            break

    return train_losses, test_losses

def generate_anomalies(data, mask, LS_distribution, spread_distribution, anomaly_strength, seed=None, anomaly_probability=0.3, tol=1e-8):

    if seed is not None:
        np.random.seed(seed)
    data_copy = data.astype(np.float64)

    nonzero_entries = ~np.all(np.isclose(data_copy, 0.0), axis=(1, 2, 3, 4))

    data_copy = data_copy[nonzero_entries]

    N = data_copy.shape[0]  # Length of the dataset
    idx = 0  # Start index for processing

    # Store info for each anomaly
    anomalies_info = []
    anomaly_count = 0

    # Iterate over the entire dataset in time-window LS chunks
    while idx < N:
        L = LS_distribution()  # Generate a random chunk size L
        end_idx = min(idx + L, N)  # Ensure we don't go out of bounds

        # Check if LS are zero valued in DigiOccupancy
        skip_chunk = False
        for i in range(idx, end_idx):
            #Check if data is zero valued. This shouldn't occur as the data is already removed of zero valued entries.
            if np.all(np.isclose(data_copy[i, :, :, :, 0], 0.0, atol=tol)):
                print(f"Skipping index {i} (zero filled LS).")
                skip_chunk = True
                break

        if skip_chunk:
            idx += 1
            continue

        # Randomly decide whether to inject anomaly into this chunk
        if np.random.random() < anomaly_probability:
            # Find random location for the anomaly within the HCAL mask
            valid_indices = np.argwhere(np.isclose(mask, 1.0, atol=tol))

            # Make sure valid location in mask doesnt contain zero valued DigiOccupancy
            anomaly_location = None
            max_attempts = 100
            attempts = 0
            while anomaly_location is None and attempts < max_attempts:
                candidate_location = valid_indices[np.random.choice(len(valid_indices))]
                eta, phi, depth = candidate_location

                # Check if the selected location is non-zero across the whole chunk
                if not np.any(np.isclose(data_copy[idx:end_idx, eta, phi, depth, 0], 0.0, atol=tol)):
                    anomaly_location = (eta, phi, depth)
                attempts += 1

            if anomaly_location is not None:
                # Sample spread independently for each dimension
                spread_eta = spread_distribution()
                spread_phi = spread_distribution()
                spread_depth = spread_distribution()

                # Apply anomaly with mask-aware spread
                for ieta in range(max(0, eta - spread_eta), min(64, eta + spread_eta + 1)):
                    for iphi in range(max(0, phi - spread_phi), min(72, phi + spread_phi + 1)):
                        for idepth in range(max(0, depth - spread_depth), min(7, depth + spread_depth + 1)):
                            # Only modify locations where the HCAL mask value is 1.0
                            if np.isclose(mask[ieta, iphi, idepth], 1.0, atol=tol):
                                data_copy[idx:end_idx, ieta, iphi, idepth, 0] *= anomaly_strength

                # Store information about the anomaly in JSON file
                anomalies_info.append({
                    "start_index": int(idx),
                    "L_chunk_size": str(L),
                    "anomaly_location (ieta, iphi, depth)": (str(eta), str(phi), str(depth)),
                    "x_spread": str(spread_eta),
                    "y_spread": str(spread_phi),
                    "z_spread": str(spread_depth)
                })
                anomaly_count += 1

        idx = end_idx  # Move to the next chunk

    print(f"Generated {anomaly_count} anomalies out of {idx // np.mean([LS_distribution() for _ in range(100)])} total time windows (probability: {anomaly_probability})")

    return data_copy, anomalies_info

def load_model(model, model_path):
    print(f"loading model: {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, data_loader):
    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            if hasattr(model, 'mask_ratio'):
                model.mask_ratio = 0.0

            if hasattr(model, 'isvariational') and model.isvariational:
                output, _, _ = model(data)
            else:
                output = model(data)

            predictions.append(output.cpu())
            targets.append(data.cpu())

    return torch.cat(predictions), torch.cat(targets)

def get_mean_std_dev_arrays(avg_dataset, std_dev_dataset):
    avg_dataset = np.load(avg_dataset)
    std_dev_dataset = np.load(std_dev_dataset)

    if np.isnan(avg_dataset).any():
        avg_dataset = np.nan_to_num(avg_dataset, nan=0.0)
    if np.isnan(std_dev_dataset).any():
        std_dev_dataset = np.nan_to_num(std_dev_dataset, nan=0.0)

    mean_err_array = torch.tensor(avg_dataset).to(device)
    std_err_array = torch.tensor(std_dev_dataset).to(device)

    std_err_array = torch.where(std_err_array.abs() < 1e-6, mean_err_array, std_err_array)

    mean_err_array = mean_err_array.squeeze(-1)
    std_err_array = std_err_array.squeeze(-1)

    return mean_err_array, std_err_array
def calc_max_zscore(pred, true, mean_err_array, std_err_array, subdetector_mask):
    device = pred.device
    dtype = pred.dtype

    # Convert arrays to tensors if needed
    if not torch.is_tensor(mean_err_array):
        mean_err_array = torch.tensor(mean_err_array, dtype=dtype, device=device)
    else:
        mean_err_array = mean_err_array.to(device)

    if not torch.is_tensor(std_err_array):
        std_err_array = torch.tensor(std_err_array, dtype=dtype, device=device)
    else:
        std_err_array = std_err_array.to(device)

    if not torch.is_tensor(subdetector_mask):
        subdetector_mask = torch.tensor(subdetector_mask, dtype=dtype, device=device)
    else:
        subdetector_mask = subdetector_mask.to(device)

    # Handle NaN values in statistics arrays - replace NaN with safe values
    mean_err_array = torch.nan_to_num(mean_err_array, nan=0.0)
    std_err_array = torch.nan_to_num(std_err_array, nan=1.0)

    # Align predictions and targets
    pred = pred.squeeze(-1).squeeze(1).to(device)
    true = true.squeeze(-1).squeeze(1).to(device)

    # Apply mask
    pred = pred * subdetector_mask
    true = true * subdetector_mask

    # Error
    error = torch.abs(pred - true) * subdetector_mask

    # Z-score calculation with NaN handling
    diff_mean = torch.abs(error - mean_err_array) * subdetector_mask
    z_score = diff_mean / (std_err_array + 1e-6)
    z_score = z_score * subdetector_mask

    # Replace any remaining NaN or inf values with 0
    z_score = torch.nan_to_num(z_score, nan=0.0, posinf=0.0, neginf=0.0)

    # Max per sample - only consider valid (non-zero mask) locations
    z_score_masked = z_score.view(z_score.shape[0], -1)
    mask_flat = subdetector_mask.view(-1)

    # For each sample, find max z-score only in valid detector regions
    z_score_max_per_sample = torch.zeros(z_score.shape[0], dtype=dtype, device=device)
    for i in range(z_score.shape[0]):
        valid_z_scores = z_score_masked[i][mask_flat > 0]
        if valid_z_scores.numel() > 0:
            z_score_max_per_sample[i] = torch.max(valid_z_scores)
        else:
            z_score_max_per_sample[i] = 0.0

    return z_score_max_per_sample

def generate_anomaly_table(no_anomaly_path, anomaly_files_dict,run, manual_thresholds=None):
    results = []
    for file_path, strength in anomaly_files_dict.items():
        no_anomaly_scores = np.load(no_anomaly_path)
        anomaly_scores = np.load(file_path)

        all_scores = np.concatenate([no_anomaly_scores, anomaly_scores])
        labels = np.concatenate([np.zeros_like(no_anomaly_scores), np.ones_like(anomaly_scores)])

        thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 100)
        f1_list = []

        for threshold in thresholds:
            predictions = (all_scores > threshold).astype(int)
            TP = np.sum((predictions == 1) & (labels == 1))
            TN = np.sum((predictions == 0) & (labels == 0))
            FP = np.sum((predictions == 1) & (labels == 0))
            FN = np.sum((predictions == 0) & (labels == 1))
            TPR = TP / (TP + FN + 1e-10)   # Recall
            Precision = TP / (TP + FP + 1e-10)
            F1 = 2 * Precision * TPR / (Precision + TPR + 1e-10)
            f1_list.append(F1)

        if manual_thresholds and strength in manual_thresholds:
            best_threshold = manual_thresholds[strength]
            best_idx = np.argmin(np.abs(thresholds - best_threshold))
        else:
            best_idx = np.argmax(f1_list)
            best_threshold = thresholds[best_idx]

        predictions = (all_scores > best_threshold).astype(int)
        TP = np.sum((predictions == 1) & (labels == 1))
        TN = np.sum((predictions == 0) & (labels == 0))
        FP = np.sum((predictions == 1) & (labels == 0))
        FN = np.sum((predictions == 0) & (labels == 1))
        TPR = TP / (TP + FN + 1e-10)   # Recall
        Precision = TP / (TP + FP + 1e-10)
        F1 = 2 * Precision * TPR / (Precision + TPR + 1e-10)
        fpr = FP / (FP + TN + 1e-10)
        fnr = FN / (FN + TP + 1e-10)

        labels_flat = labels.reshape(-1)
        scores_flat = all_scores.reshape(-1)

        auc = roc_auc_score(labels_flat, scores_flat)

        results.append({
            'Run' : run,
            'Anomaly Strength': float(strength),
            'Optimal Threshold': best_threshold,
            'Precision': Precision,
            'Recall': TPR,
            'F1 Score': F1,
            'FPR': fpr,
            'FNR': fnr,
            'ROC AUC': auc
        })

    return results

def generate_anomaly_table_without_anomaly_file(anomaly_files_dict, run,fixed_threshold=5.5):
    results = []

    for file_path, strength in anomaly_files_dict.items():
        anomaly_scores = np.load(file_path)

        # All labels = 1
        labels = np.ones_like(anomaly_scores)

        predictions = (anomaly_scores > fixed_threshold).astype(int)
        TP = np.sum(predictions == 1)
        FN = np.sum(predictions == 0)

        Precision = 1.0 if TP > 0 else 0.0   # FP=0 always
        Recall = TP / (TP + FN + 1e-10)
        F1 = 2 * Precision * Recall / (Precision + Recall + 1e-10)
        Accuracy = np.mean(predictions == labels)

        # Count 1's and 0's in labels and predictions
        labels_1s = np.sum(labels == 1)
        labels_0s = np.sum(labels == 0)
        predictions_1s = np.sum(predictions == 1)
        predictions_0s = np.sum(predictions == 0)

        print(f"Strength {strength}:")
        print(f"  Labels: {labels_1s} ones, {labels_0s} zeros (total: {len(labels)})")
        print(f"  Predictions: {predictions_1s} ones, {predictions_0s} zeros (total: {len(predictions)})")
        print(f"  Threshold used: {fixed_threshold:.4f}")
        print(f"  Z-score stats: min={np.min(anomaly_scores):.4f}, max={np.max(anomaly_scores):.4f}, mean={np.mean(anomaly_scores):.4f}")

        results.append({
            'Run': run,
            'Anomaly Strength': float(strength),
            'Threshold': fixed_threshold,
            'Precision': Precision,
            'Recall': Recall,
            'F1 Score': F1,
            'Accuracy': Accuracy,
            'Labels_1s': int(labels_1s),
            'Labels_0s': int(labels_0s),
            'Predictions_1s': int(predictions_1s),
            'Predictions_0s': int(predictions_0s)
        })

    return results

def generate_anomaly_table_with_anomaly_info(anomaly_files_dict, anomaly_info_dict, run, thresholds_dict=None, fixed_threshold=5.5):
    """
    Generate anomaly detection evaluation table using anomaly info to create proper labels.

    Args:
        anomaly_files_dict (dict): Mapping of file paths to anomaly strength values
        anomaly_info_dict (dict): Mapping of anomaly strengths to lists of anomaly info dicts
        run (str/int): Run identifier for results
        thresholds_dict (dict, optional): Mapping of anomaly strengths to optimal thresholds
        fixed_threshold (float): Fallback threshold if optimal thresholds not available

    Returns:
        list: List of result dictionaries containing evaluation metrics
    """
    results = []

    for file_path, strength in anomaly_files_dict.items():
        anomaly_scores = np.load(file_path)

        # Get corresponding anomaly info
        anomaly_info = anomaly_info_dict.get(strength, [])

        # Create labels based on anomaly info
        labels = np.zeros_like(anomaly_scores)

        # Mark indices that fall within anomaly chunks as 1
        for anomaly in anomaly_info:
            start_idx = int(anomaly["start_index"])
            chunk_size = int(anomaly["L_chunk_size"])
            end_idx = start_idx + chunk_size

            # Ensure we don't go out of bounds
            end_idx = min(end_idx, len(labels))
            if start_idx < len(labels):
                labels[start_idx:end_idx] = 1

        # Use threshold from dictionary if available, otherwise use fixed threshold
        if thresholds_dict:
            threshold_to_use = thresholds_dict.get(strength, thresholds_dict.get(str(strength), fixed_threshold))
        else:
            threshold_to_use = fixed_threshold
        predictions = (anomaly_scores > threshold_to_use).astype(int)

        # Calculate metrics
        TP = np.sum((predictions == 1) & (labels == 1))
        TN = np.sum((predictions == 0) & (labels == 0))
        FP = np.sum((predictions == 1) & (labels == 0))
        FN = np.sum((predictions == 0) & (labels == 1))

        Precision = TP / (TP + FP + 1e-10)
        Recall = TP / (TP + FN + 1e-10)
        F1 = 2 * Precision * Recall / (Precision + Recall + 1e-10)
        Accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
        FPR = FP / (FP + TN + 1e-10)
        FNR = FN / (FN + TP + 1e-10)

        # Calculate ROC AUC
        try:
            ROC_AUC = roc_auc_score(labels, anomaly_scores)
        except:
            ROC_AUC = 0.0

        # Count 1's and 0's in labels and predictions
        labels_1s = np.sum(labels == 1)
        labels_0s = np.sum(labels == 0)
        predictions_1s = np.sum(predictions == 1)
        predictions_0s = np.sum(predictions == 0)

        print(f"Strength {strength}:")
        print(f"  Labels: {labels_1s} ones, {labels_0s} zeros (total: {len(labels)})")
        print(f"  Predictions: {predictions_1s} ones, {predictions_0s} zeros (total: {len(predictions)})")
        print(f"  Threshold used: {threshold_to_use:.4f}")
        print(f"  Z-score stats: min={np.min(anomaly_scores):.4f}, max={np.max(anomaly_scores):.4f}, mean={np.mean(anomaly_scores):.4f}")

        # Call detailed analysis for problematic cases
        if Accuracy > 0.99 or (predictions_1s == 0 or predictions_0s == 0):
            analyze_classification_details(anomaly_scores, labels, predictions, threshold_to_use, strength)

        results.append({
            'Run': run,
            'Anomaly Strength': float(strength),
            'Threshold': threshold_to_use,
            'Precision': Precision,
            'Recall': Recall,
            'F1 Score': F1,
            'Accuracy': Accuracy,
            'FPR': FPR,
            'FNR': FNR,
            'ROC_AUC': ROC_AUC,
            'Labels_1s': int(labels_1s),
            'Labels_0s': int(labels_0s),
            'Predictions_1s': int(predictions_1s),
            'Predictions_0s': int(predictions_0s)
        })

    return results

def calculate_optimal_thresholds(normal_scores, anomaly_files_dict, anomaly_info_dict):
    """
    Calculate optimal thresholds for each anomaly strength based on F1 score maximization.

    This function combines normal (baseline) z-scores with anomalous z-scores for each strength,
    creates proper labels based on anomaly injection information, and finds the threshold
    that maximizes the F1 score for anomaly detection.

    Args:
        normal_scores (numpy.ndarray): Z-scores for normal (non-anomalous) data
        anomaly_files_dict (dict): Mapping of file paths to anomaly strength strings
        anomaly_info_dict (dict): Mapping of anomaly strength strings to lists of anomaly info dicts
                                 Each anomaly info dict contains: start_index, L_chunk_size, etc.

    Returns:
        dict: Mapping of anomaly strength (str) to optimal threshold (float)
    """
    optimal_thresholds = {}

    for file_path, strength in anomaly_files_dict.items():

        anomaly_scores = np.load(file_path)

        # Get corresponding anomaly info
        anomaly_info = anomaly_info_dict.get(strength, [])

        # Create labels for anomalous data based on anomaly info
        anomaly_labels = np.zeros_like(anomaly_scores)

        # Mark indices that fall within anomaly chunks as 1
        for anomaly in anomaly_info:
            start_idx = int(anomaly["start_index"])
            chunk_size = int(anomaly["L_chunk_size"])
            end_idx = start_idx + chunk_size

            # Ensure we don't go out of bounds
            end_idx = min(end_idx, len(anomaly_labels))
            if start_idx < len(anomaly_labels):
                anomaly_labels[start_idx:end_idx] = 1

        # Combine normal and anomalous data
        all_scores = np.concatenate([normal_scores, anomaly_scores])
        all_labels = np.concatenate([np.zeros_like(normal_scores), anomaly_labels])

        # Find optimal threshold
        thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 100)
        f1_list = []

        for threshold in thresholds:
            predictions = (all_scores > threshold).astype(int)
            TP = np.sum((predictions == 1) & (all_labels == 1))
            TN = np.sum((predictions == 0) & (all_labels == 0))
            FP = np.sum((predictions == 1) & (all_labels == 0))
            FN = np.sum((predictions == 0) & (all_labels == 1))

            Precision = TP / (TP + FP + 1e-10)
            Recall = TP / (TP + FN + 1e-10)
            F1 = 2 * Precision * Recall / (Precision + Recall + 1e-10)
            f1_list.append(F1)

        best_idx = np.argmax(f1_list)
        best_threshold = thresholds[best_idx]
        optimal_thresholds[strength] = float(best_threshold)

        print(f"Optimal threshold for strength {strength}: {best_threshold:.4f} (F1: {f1_list[best_idx]:.4f})")

    return optimal_thresholds

def print_table(results):
    # Check if the first result has FPR and FNR keys to determine which format to use
    if results and 'FPR' in results[0] and 'FNR' in results[0]:
        print(f"{'Anomaly':<8} {'Threshold':<10} {'Precision':<10} {'Recall':<8} {'F1 Score':<8} {'Accuracy':<8} {'FPR':<8} {'FNR':<8} {'ROC AUC':<8}")
        print(f"{'Strength':<8} {'':<10} {'':<10} {'':<8} {'':<8} {'':<8} {'':<8} {'':<8} {'':<8}")
        print("-" * 94)
        for r in results:
            print(f"{r['Anomaly Strength']:<8.1f} {r['Threshold']:<10.4f} {r['Precision']:<10.4f} {r['Recall']:<8.4f} {r['F1 Score']:<8.4f} {r['Accuracy']:<8.4f} {r['FPR']:<8.4f} {r['FNR']:<8.4f} {r['ROC_AUC']:<8.4f}")
            # Print detailed counts if available
            if 'Labels_1s' in r:
                print(f"    → Labels: {r['Labels_1s']} ones, {r['Labels_0s']} zeros | Predictions: {r['Predictions_1s']} ones, {r['Predictions_0s']} zeros")
    else:
        print(f"{'Anomaly':<8} {'Threshold':<10} {'Precision':<10} {'Recall':<8} {'F1 Score':<8} {'Accuracy':<8}")
        print(f"{'Strength':<8} {'':<10} {'':<10} {'':<8} {'':<8} {'':<8}")
        print("-" * 70)
        for r in results:
            print(f"{r['Anomaly Strength']:<8.1f} {r['Threshold']:<10.4f} {r['Precision']:<10.4f} {r['Recall']:<8.4f} {r['F1 Score']:<8.4f} {r['Accuracy']:<8.4f}")
            # Print detailed counts if available
            if 'Labels_1s' in r:
                print(f"    → Labels: {r['Labels_1s']} ones, {r['Labels_0s']} zeros | Predictions: {r['Predictions_1s']} ones, {r['Predictions_0s']} zeros")

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_evaluation_results(results_data, save_dir=None):
    if not results_data:
        return
    runs = sorted(list(set(str(d['Run']) for d in results_data)))
    strengths = sorted(list(set(str(d['Anomaly Strength']) for d in results_data)), key=float)
    plot_data = {metric: {strength: [np.nan] * len(runs) for strength in strengths} for metric in ['ROC_AUC', 'F1 Score', 'FPR']}
    for item in results_data:
        run_str = str(item['Run'])
        run_idx = runs.index(run_str)
        strength = str(item['Anomaly Strength'])
        for metric in ['ROC_AUC', 'F1 Score', 'FPR']:
            if metric in item:
                plot_data[metric][strength][run_idx] = item[metric]
    
    # Save results to JSON files for LLM access
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save raw results
        raw_results_file = os.path.join(save_dir, "results.json")
        with open(raw_results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_serializable_results = []
            for item in results_data:
                json_item = {}
                for k, v in item.items():
                    if isinstance(v, (np.floating, np.integer)):
                        json_item[k] = float(v) if isinstance(v, np.floating) else int(v)
                    else:
                        json_item[k] = v
                json_serializable_results.append(json_item)
            json.dump(json_serializable_results, f, indent=2)
        
        # Save organized summary by run and strength
        summary_data = {}
        for item in results_data:
            run = str(item['Run'])
            strength = str(item['Anomaly Strength'])
            
            if run not in summary_data:
                summary_data[run] = {}
            if strength not in summary_data[run]:
                summary_data[run][strength] = {}
            
            # Store all metrics for this run-strength combination
            for k, v in item.items():
                if k not in ['Run', 'Anomaly Strength']:
                    if isinstance(v, (np.floating, np.integer)):
                        summary_data[run][strength][k] = float(v) if isinstance(v, np.floating) else int(v)
                    else:
                        summary_data[run][strength][k] = v
        
        summary_file = os.path.join(save_dir, "evaluation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Results saved to {raw_results_file}")
        print(f"Summary saved to {summary_file}")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    for strength in strengths:
        ax1.plot(runs, plot_data['ROC_AUC'][strength], marker='o', label=f'Str {strength}')
    ax1.set_title('ROC AUC across Runs')
    ax1.set_xlabel('Eval Run')
    ax1.set_ylabel('ROC AUC')
    ax1.legend()
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    for strength in strengths:
        ax2.plot(runs, plot_data['FPR'][strength], marker='o', linestyle='-', label=f'FPR Str {strength}')
        ax2.plot(runs, plot_data['F1 Score'][strength], marker='x', linestyle='--', label=f'F1 Str {strength}')
    ax2.set_title('FPR and F1 Score across Runs')
    ax2.set_xlabel('Eval Run')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "evaluation_plots.png"), dpi=300)
    plt.show()
