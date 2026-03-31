# """
# Agent Tools for ContinuousML4DQM
# Orchestral-AI HEPTAPOD Tool Definitions
# """
# from orchestral import define_tool
# import os
# import sys
# import json
# import numpy as np
# import pandas as pd
# import torch
# from pathlib import Path
# from typing import List, Dict, Tuple, Union, Optional
# from tqdm import tqdm

# # Add src to path for imports
# sys.path.insert(0, os.path.dirname(__file__))

# from utils import (
#     generate_anomalies,
#     calc_max_zscore,
#     get_mean_std_dev_arrays,
#     predict,
#     load_model,
#     generate_anomaly_table_with_anomaly_info,
#     calculate_optimal_thresholds
# )
# from utilities import save_csv, load_npdata, save_json
# def compute_model_error_stats(model,a_loader,mask_path, batch_size=128, device='cpu'):
#     # Load mask
#     subdetector_mask = np.load(mask_path)
#     mask_bool = subdetector_mask.astype(bool)
#     model.eval()
#     all_targets, all_preds = [], []

#     with torch.no_grad():
#         for data in a_loader:
#             data = data.to(device)
#             pred, _, _ = model(data)
#             pred = pred.squeeze(-1)
#             all_targets.append(data.cpu())
#             all_preds.append(pred.cpu())

#     targets_tensor = torch.cat(all_targets, dim=0).numpy()
#     preds_tensor = torch.cat(all_preds, dim=0).numpy()

#     targets_tensor = np.squeeze(targets_tensor, axis=1)
#     preds_tensor = np.squeeze(preds_tensor, axis=1)

#     masked_targets_tensor = np.where(mask_bool, targets_tensor, np.nan)
#     masked_preds_tensor = np.where(mask_bool, preds_tensor, np.nan)

#     error = np.abs(masked_preds_tensor - masked_targets_tensor)
#     mean_error = np.nanmean(error, axis=0)
#     std_error = np.nanstd(error, axis=0)

#     # Convert to torch tensors and handle near-zero std
#     mean_err_array = torch.tensor(mean_error).to(device).squeeze(-1)
#     std_err_array = torch.tensor(std_error).to(device).squeeze(-1)
#     std_err_array = torch.where(std_err_array.abs() < 1e-6, mean_err_array, std_err_array)

#     return mean_err_array, std_err_array

# from depthvit import DepthwiseCrossViTAE_MultiDim_SPATIAL
# from model_training import CustomDataset

# from torch.utils.data import DataLoader

# # Get device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# @define_tool
# def evaluate_trained_model(
#     run_id: Union[int, List[int]],
#     trained_model_path: str,
#     test_data_paths: Union[str, List[str]],
#     anomaly_strengths: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 2.0],
#     output_dir: str = "./results",
#     mask_path: str = "./data/he_segmentation_config_mask.npy",
#     batch_size: int = 32,
#     seed: int = 42
# ) -> Dict:
#     """
#     Evaluate trained DepthViT model on synthetic anomalies.
    
#     Generates the performance table with columns:
#     - Anomaly strength
#     - Optimal threshold
#     - F1 Score @ Optimal threshold
#     - FPR @ Optimal threshold
#     - AUC
    
#     Args:
#         run_id: Single run ID (int) or list of run IDs to concatenate
#         trained_model_path: Path to trained model checkpoint (.pt file)
#         test_data_paths: Path(s) to test data .npy files
#         anomaly_strengths: List of anomaly strengths to evaluate
#         output_dir: Directory to save results CSV
#         mask_path: Path to detector mask .npy file
#         batch_size: Batch size for inference
#         seed: Random seed for reproducibility
        
#     Returns:
#         Dictionary with:
#         - table: pandas DataFrame with results
#         - csv_path: Path where CSV was saved
#         - metrics: Dict of metrics per strength
#         - anomaly_info: Dict of anomaly metadata
#     """
    
#     np.random.seed(seed)
#     torch.manual_seed(seed)
    
#     # Normalize paths
#     trained_model_path = str(trained_model_path)
#     mask_path = str(mask_path)
#     output_dir = str(output_dir)
    
#     # Handle multiple runs
#     if isinstance(run_id, list):
#         run_ids = run_id
#         run_label = f"runs_{run_ids[0]}_to_{run_ids[-1]}"
#     else:
#         run_ids = [run_id]
#         run_label = str(run_id)
    
#     # Handle multiple test data paths
#     if isinstance(test_data_paths, str):
#         test_paths = [test_data_paths]
#     else:
#         test_paths = test_data_paths
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     print(f"\n{'='*80}")
#     print(f"EVALUATING MODEL: {trained_model_path}")
#     print(f"On test data: {test_paths}")
#     print(f"Anomaly strengths: {anomaly_strengths}")
#     print(f"Output directory: {output_dir}")
#     print(f"{'='*80}\n")
    
#     # ==================== LOAD DATA ====================
#     print("[1/5] Loading and preparing test data...")
    
#     # Load test data (concatenate if multiple)
#     test_data_list = []
#     for path in test_paths:
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Test data not found: {path}")
#         data = np.load(path)
#         print(f"  Loaded {path}: shape {data.shape}")
#         test_data_list.append(data)
    
#     # Concatenate if multiple
#     if len(test_data_list) > 1:
#         test_data = np.concatenate(test_data_list, axis=0)
#         print(f"  Concatenated shape: {test_data.shape}")
#     else:
#         test_data = test_data_list[0]
    
#     # Load mask
#     if not os.path.exists(mask_path):
#         raise FileNotFoundError(f"Mask not found: {mask_path}")
#     mask = np.load(mask_path)
#     print(f"  Loaded mask: shape {mask.shape}")
    
#     # Create dataset and loader
#     test_dataset = CustomDataset(test_data, mask_path)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     print(f"  Created DataLoader with batch_size={batch_size}")
    
#     # ==================== LOAD MODEL ====================
#     print("\n[2/5] Loading trained model...")
    
#     if not os.path.exists(trained_model_path):
#         raise FileNotFoundError(f"Model checkpoint not found: {trained_model_path}")
    
#     # Initialize model with standard params (matching your generate_model())
#     model = DepthwiseCrossViTAE_MultiDim_SPATIAL(
#         feature_dim=1,
#         latent_dim=56,
#         target_dim=1,
#         spatial_dims=(64, 72, 7),
#         memorysize=1,
#         image_size=[64, 72],
#         k_factor=8,
#         num_layers=5,
#         mask_ratio=0.0,
#         isvariational=True
#     ).to(device)
    
#     model = load_model(model, trained_model_path)
#     print(f"  ✓ Model loaded and moved to {device}")
    
#     # ==================== COMPUTE ERROR STATISTICS ====================
#     print("\n[3/5] Computing validation error statistics...")
    
#     mean_err_array, std_err_array = compute_model_error_stats(
#         model,
#         test_loader,
#         mask_path,
#         batch_size=batch_size,
#         device=device
#     )
#     print(f"  Mean error array shape: {mean_err_array.shape}")
#     print(f"  Std error array shape: {std_err_array.shape}")
    
#     # ==================== GENERATE PREDICTIONS ====================
#     print("\n[4/5] Running predictions on test data...")
    
#     predictions, targets = predict(model, test_loader)
#     print(f"  Predictions shape: {predictions.shape}")
#     print(f"  Targets shape: {targets.shape}")
    
#     # ==================== GENERATE ANOMALIES & COMPUTE Z-SCORES ====================
#     print("\n[5/5] Generating anomalies and computing z-scores...")
    
#     results_list = []
#     anomaly_info_dict = {}
#     anomaly_files_dict = {}
    
#     # For computing optimal thresholds - we need normal (no anomaly) scores first
#     normal_zscores = []
#     for strength in anomaly_strengths:
#         if strength == 0.0:  # 0.0 strength = normal data
#             # Compute z-scores for normal data
#             zscores_per_sample = []
#             for i in range(predictions.shape[0]):
#                 pred_sample = predictions[i:i+1]
#                 target_sample = targets[i:i+1]
#                 zscore = calc_max_zscore(
#                     pred_sample,
#                     target_sample,
#                     mean_err_array,
#                     std_err_array,
#                     torch.from_numpy(mask).float().to(device)
#                 )
#                 zscores_per_sample.append(zscore.cpu().numpy())
            
#             normal_zscores = np.concatenate(zscores_per_sample)
            
#             # Save normal z-scores
#             normal_path = os.path.join(output_dir, f"run_{run_label}_zscores_normal.npy")
#             np.save(normal_path, normal_zscores)
#             anomaly_files_dict["0.0"] = normal_path
            
#             print(f"  Strength {strength}: {len(normal_zscores)} samples, "
#                   f"mean_zscore={np.mean(normal_zscores):.4f}, "
#                   f"max_zscore={np.max(normal_zscores):.4f}")
            
#             break  # Stop after collecting normal scores
    
#     # Now generate anomalies at each strength
#     for strength in anomaly_strengths:
#         print(f"\n  Generating anomalies with strength {strength}...")
        
#         # Generate anomalied data
#         anomalied_data, anomaly_info = generate_anomalies(
#             test_data,
#             mask,
#             LS_distribution=lambda: np.random.randint(1, 10),
#             spread_distribution=lambda: np.random.randint(0, 2),
#             anomaly_strength=strength,
#             seed=seed,
#             anomaly_probability=0.3 if strength > 0 else 0.0
#         )
        
#         # Create dataset from anomalied data
#         anomaly_dataset = CustomDataset(anomalied_data, mask_path)
#         anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=False)
        
#         # Get predictions on anomalied data
#         anomaly_preds, _ = predict(model, anomaly_loader)
        
#         # Compute z-scores for anomalied data
#         zscores_per_sample = []
#         for i in range(anomaly_preds.shape[0]):
#             pred_sample = anomaly_preds[i:i+1]
#             target_sample = torch.from_numpy(anomalied_data[i:i+1]).float().to(device).unsqueeze(1)
            
#             zscore = calc_max_zscore(
#                 pred_sample,
#                 target_sample,
#                 mean_err_array,
#                 std_err_array,
#                 torch.from_numpy(mask).float().to(device)
#             )
#             zscores_per_sample.append(zscore.cpu().numpy())
        
#         anomaly_zscores = np.concatenate(zscores_per_sample)
        
#         # Save anomaly z-scores
#         anomaly_path = os.path.join(output_dir, f"run_{run_label}_zscores_strength_{strength}.npy")
#         np.save(anomaly_path, anomaly_zscores)
#         anomaly_files_dict[str(strength)] = anomaly_path
#         anomaly_info_dict[str(strength)] = anomaly_info
        
#         print(f"    Saved: {anomaly_path}")
#         print(f"    Z-score stats: mean={np.mean(anomaly_zscores):.4f}, "
#               f"max={np.max(anomaly_zscores):.4f}, std={np.std(anomaly_zscores):.4f}")
    
#     # ==================== CALCULATE OPTIMAL THRESHOLDS ====================
#     print("\n[EVAL] Computing optimal thresholds using normal + anomalous data...")
    
#     optimal_thresholds = calculate_optimal_thresholds(
#         normal_zscores,
#         anomaly_files_dict,
#         anomaly_info_dict
#     )
    
#     # ==================== GENERATE RESULTS TABLE ====================
#     print("\n[EVAL] Generating evaluation table...")
    
#     # Generate table using all strengths
#     table_results = []
#     for strength in anomaly_strengths:
#         if strength == 0.0:
#             # Skip 0.0 for now, will add if needed
#             continue
        
#         file_path = anomaly_files_dict.get(str(strength))
#         if not file_path or not os.path.exists(file_path):
#             print(f"  ⚠️  No data for strength {strength}, skipping...")
#             continue
        
#         # Combine normal and anomalous scores
#         anomaly_scores = np.load(file_path)
#         all_scores = np.concatenate([normal_zscores, anomaly_scores])
#         all_labels = np.concatenate([np.zeros_like(normal_zscores), np.ones_like(anomaly_scores)])
        
#         # Use optimal threshold
#         threshold = optimal_thresholds.get(str(strength), 5.5)
#         predictions = (all_scores > threshold).astype(int)
        
#         # Compute metrics
#         from sklearn.metrics import roc_auc_score
        
#         TP = np.sum((predictions == 1) & (all_labels == 1))
#         TN = np.sum((predictions == 0) & (all_labels == 0))
#         FP = np.sum((predictions == 1) & (all_labels == 0))
#         FN = np.sum((predictions == 0) & (all_labels == 1))
        
#         Precision = TP / (TP + FP + 1e-10)
#         Recall = TP / (TP + FN + 1e-10)
#         F1 = 2 * Precision * Recall / (Precision + Recall + 1e-10)
#         FPR = FP / (FP + TN + 1e-10)
        
#         try:
#             AUC = roc_auc_score(all_labels, all_scores)
#         except:
#             AUC = 0.0
        
#         result = {
#             'Anomaly strength': float(strength),
#             'Optimal threshold': float(threshold),
#             'F1 Score @ Optimal threshold': float(F1),
#             'FPR @ Optimal threshold': float(FPR),
#             'AUC': float(AUC)
#         }
#         table_results.append(result)
        
#         print(f"  Strength {strength}: F1={F1:.4f}, FPR={FPR:.4f}, AUC={AUC:.4f}, "
#               f"Threshold={threshold:.4f}")
    
#     # Convert to DataFrame
#     results_df = pd.DataFrame(table_results)
#     results_df = results_df.sort_values('Anomaly strength').reset_index(drop=True)
    
#     # ==================== SAVE RESULTS ====================
#     print("\n[SAVE] Saving results...")
    
#     # Save CSV
#     csv_path = os.path.join(output_dir, f"performance_table_run_{run_label}.csv")
#     save_csv(csv_path, results_df, index=False)
#     print(f"  ✓ CSV saved: {csv_path}")
    
#     # Save metadata JSON
#     metadata = {
#         "run_id": run_ids,
#         "model_path": trained_model_path,
#         "test_data_paths": test_paths,
#         "mask_path": mask_path,
#         "num_test_samples": int(test_data.shape[0]),
#         "anomaly_strengths": anomaly_strengths,
#         "seed": seed
#     }
#     metadata_path = os.path.join(output_dir, f"metadata_run_{run_label}.json")
#     save_json(metadata_path, metadata)
#     print(f"  ✓ Metadata saved: {metadata_path}")
    
#     # ==================== PRINT SUMMARY ====================
#     print(f"\n{'='*80}")
#     print("RESULTS SUMMARY")
#     print(f"{'='*80}")
#     print(results_df.to_string(index=False))
#     print(f"{'='*80}\n")
    
#     return {
#         "table": results_df,
#         "csv_path": csv_path,
#         "metrics": table_results,
#         "anomaly_info": anomaly_info_dict,
#         "optimal_thresholds": optimal_thresholds,
#         "normal_zscores": normal_zscores
#     }


# # For Orchestral-AI HEPTAPOD registration
# if __name__ == "__main__":
#     # Example usage / testing
#     result = evaluate_trained_model(
#         run_id=323940,
#         trained_model_path="./models/depthvit_run323940.pt",
#         test_data_paths="./data/he_test_dataset_323940/test_data.npy",
#         anomaly_strengths=[0.0, 0.2, 0.4, 0.6, 0.8, 2.0],
#         output_dir="./results",
#         mask_path="./data/he_segmentation_config_mask.npy"
#     )
#     print("\n✓ Tool execution complete!")
#     print(f"Results saved to: {result['csv_path']}")
