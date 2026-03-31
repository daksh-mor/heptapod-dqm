"""Run evaluation on one or more runs using artifacts saved by dqm.train."""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from orchestral.tools import define_tool

try:
    from .model_datasets import *
    from .models_spatial import *
    from .utils import *
except ImportError:
    from model_datasets import *
    from models_spatial import *
    from utils import *


def _build_evaluation_summary_table(results, runs_evaluated, anomaly_strengths, plots_dir):
    lines = [
        "DepthViT Evaluation Summary",
        "--------------------------",
        f"runs_evaluated   : {','.join(str(r) for r in runs_evaluated) if runs_evaluated else '-'}",
        f"anomaly_strengths: {', '.join(f'{float(x):.1f}' for x in anomaly_strengths) if anomaly_strengths else '-'}",
        f"plots_dir        : {plots_dir}",
        "",
        "Run | Str | Thr | Precision | Recall | F1 | Accuracy",
        "----|-----|-----|-----------|--------|----|---------",
    ]

    for row in results:
        run = row.get("Run", "-")
        strength = row.get("Anomaly Strength", "-")
        threshold = row.get("Threshold", row.get("Optimal Threshold", "-"))
        precision = row.get("Precision", "-")
        recall = row.get("Recall", "-")
        f1 = row.get("F1 Score", "-")
        accuracy = row.get("Accuracy", "-")

        def _fmt(v):
            if isinstance(v, (int, float)):
                return f"{float(v):.4f}"
            return str(v)

        lines.append(
            f"{run} | {_fmt(strength)} | {_fmt(threshold)} | {_fmt(precision)} | {_fmt(recall)} | {_fmt(f1)} | {_fmt(accuracy)}"
        )

    return "\n".join(lines)


def _evaluate_model_on_runs_impl(
    runs,
    base_dir,
    batch_size=32,
    anomaly_strengths=None,
    seed=42,
):
    """Evaluate a trained model on one or more runs using saved training artifacts."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fixture_dataset_dir = os.path.join(project_root, "tools", "dqm", "test_files", "dataset")
    workflow_sandbox_root = os.path.join(project_root, "examples", "workflows", "cml_dqm_sandbox")

    if isinstance(runs, int):
        runs = [runs]
    elif isinstance(runs, str):
        runs = [int(runs)]

    if not anomaly_strengths:
        anomaly_strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 2.0]

    base_dir_norm = os.path.normpath(str(base_dir)).lstrip(os.sep)
    if base_dir_norm in ("", "."):
        raise ValueError("base_dir must be a non-empty sandbox subdirectory")
    if base_dir_norm.startswith(".."):
        raise ValueError("base_dir must stay within sandbox")

    if base_dir_norm.startswith(os.path.join("examples", "workflows", "cml_dqm_sandbox")):
        model_base_dir = os.path.abspath(os.path.join(project_root, base_dir_norm))
    else:
        model_base_dir = os.path.abspath(os.path.join(workflow_sandbox_root, base_dir_norm))

    root_real = os.path.realpath(workflow_sandbox_root)
    base_real = os.path.realpath(model_base_dir)
    if not (base_real == root_real or base_real.startswith(root_real + os.sep)):
        raise ValueError("base_dir must stay within examples/workflows/cml_dqm_sandbox")

    eval_dir = os.path.join(model_base_dir, "eval")
    model_candidates = [
        os.path.join(model_base_dir, "model.pth"),
        os.path.join(model_base_dir, "models", "he_ae.pth"),
        os.path.join(model_base_dir, "he_ae.pth"),
    ]
    model_path = next((p for p in model_candidates if os.path.exists(p)), model_candidates[0])
    plots_dir = os.path.join(eval_dir, "plots")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    mask_path = os.path.join(fixture_dataset_dir, "he_segmentation_config_mask.npy")
    dataset_dir = fixture_dataset_dir

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DepthwiseCrossViTAE_MultiDim_SPATIAL(
        feature_dim=1,
        latent_dim=56,
        target_dim=1,
        spatial_dims=(64, 72, 7),
        memorysize=1,
        image_size=[64, 72],
        k_factor=8,
        num_layers=5,
        mask_ratio=0.5,
        patch_size=12,
        isvariational=True,
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found. Checked: {model_candidates}. Train first with the same base_dir."
        )

    model = load_model(model, model_path)
    model = model.eval()

    thresholds_file = os.path.join(model_base_dir, "optimal_thresholds.json")
    optimal_thresholds = {}
    if os.path.exists(thresholds_file):
        with open(thresholds_file, "r", encoding="utf-8") as f:
            optimal_thresholds = json.load(f)

    all_results = []
    evaluated_runs = []

    for eval_run in runs:
        test_data_path = os.path.join(dataset_dir, f"he_test_dataset_Run{eval_run}", "test_data.npy")
        train_data_path = os.path.join(dataset_dir, f"he_train_dataset_Run{eval_run}", "train_data.npy")

        if not os.path.exists(test_data_path):
            test_data_path = os.path.join(dataset_dir, "test_data.npy")
        if not os.path.exists(train_data_path):
            train_data_path = os.path.join(dataset_dir, "train_data.npy")

        if not (os.path.exists(test_data_path) and os.path.exists(train_data_path)):
            continue

        test_dataset = CustomDataset(test_data_path, mask_path)
        train_dataset = CustomDataset(train_data_path, mask_path)
        combined_test_dataset = ConcatDataset([test_dataset, train_dataset])
        test_loader = DataLoader(combined_test_dataset, batch_size=256, shuffle=False)

        subdetector_mask = np.load(mask_path)
        mean_err_array, std_err_array = _compute_model_error_stats(model, test_loader, mask_path, device)
        if hasattr(model, "mask_ratio"):
            model.mask_ratio = 0.0

        train_data = np.load(train_data_path)
        test_data = np.load(test_data_path)
        combined_data = np.concatenate((train_data, test_data), axis=0)

        anomaly_dir = os.path.join(eval_dir, f"anomalies_eval_run{eval_run}")
        os.makedirs(anomaly_dir, exist_ok=True)

        anomaly_info_dict = {}
        anomaly_zscore_files = {}

        for strength in anomaly_strengths:
            anomaly_file = os.path.join(anomaly_dir, f"anomalous_data_{strength}.npy")
            anomaly_info_file = os.path.join(anomaly_dir, f"anomaly_info_{strength}.json")

            if not os.path.exists(anomaly_file):
                time_window = 4
                spatial_spread = 1
                ls_distribution = lambda: np.random.randint(1, time_window)
                spread_distribution = lambda: np.random.randint(0, spatial_spread)
                modified_data, anomalies_info = generate_anomalies(
                    combined_data,
                    subdetector_mask,
                    ls_distribution,
                    spread_distribution,
                    anomaly_strength=float(strength),
                    seed=int(seed),
                    anomaly_probability=0.3,
                )
                np.save(anomaly_file, modified_data)
                with open(anomaly_info_file, "w", encoding="utf-8") as f:
                    json.dump(anomalies_info, f, indent=2)

            if os.path.exists(anomaly_info_file):
                with open(anomaly_info_file, "r", encoding="utf-8") as f:
                    anomaly_info_dict[str(strength)] = json.load(f)
            else:
                anomaly_info_dict[str(strength)] = []

            anomaly_dataset = CustomDataset(anomaly_file, mask_path)
            anomaly_loader = DataLoader(anomaly_dataset, batch_size=int(batch_size), shuffle=False)
            predictions_anom, targets_anom = predict(model, anomaly_loader)

            z_scores_anom = []
            for pred, tgt in zip(predictions_anom, targets_anom):
                z_scores_anom.append(calc_max_zscore(pred, tgt, mean_err_array, std_err_array, subdetector_mask))
            z_scores_anom = np.array(z_scores_anom)

            zscore_file = os.path.join(anomaly_dir, f"zscore_str_{strength}.npy")
            np.save(zscore_file, z_scores_anom)
            anomaly_zscore_files[zscore_file] = str(strength)

        results = generate_anomaly_table_with_anomaly_info(
            anomaly_zscore_files, anomaly_info_dict, eval_run, optimal_thresholds, 5.5
        )
        print(f"\n{'='*65}")
        print("SINGLE MODEL EVALUATION RESULTS")
        print(f"{'='*65}")
        print(f"Model path: {model_path}")
        print(f"Test Data Run: {eval_run}")
        print(f"{'='*65}")
        print_table(results)

        all_results.extend(results)
        evaluated_runs.append(int(eval_run))

    if all_results:
        plot_evaluation_results(all_results, plots_dir)

    plot_files = []
    evaluation_plot_path = os.path.join(plots_dir, "evaluation_plots.png")
    if os.path.exists(evaluation_plot_path):
        plot_files.append(os.path.relpath(evaluation_plot_path, project_root))

    summary_table = _build_evaluation_summary_table(
        results=all_results,
        runs_evaluated=evaluated_runs,
        anomaly_strengths=anomaly_strengths,
        plots_dir=os.path.relpath(plots_dir, project_root),
    )

    return {
        "status": "ok",
        "runs_requested": [int(r) for r in runs],
        "runs_evaluated": evaluated_runs,
        "base_dir": os.path.relpath(model_base_dir, project_root),
        "plots_dir": os.path.relpath(plots_dir, project_root),
        "anomaly_strengths": [float(x) for x in anomaly_strengths],
        "results": all_results,
        "summary_table": summary_table,
        "plot_files": plot_files,
    }


@define_tool()
def evaluate_model_on_runs(
    runs,
    base_dir,
    batch_size=32,
    anomaly_strengths=None,
    seed=42,
):
    """Evaluate a trained model on one or more runs using saved training artifacts."""
    return _evaluate_model_on_runs_impl(
        runs=runs,
        base_dir=base_dir,
        batch_size=batch_size,
        anomaly_strengths=anomaly_strengths,
        seed=seed,
    )


def _compute_model_error_stats(model, a_loader, mask_path, device):
    """Compute per-channel error statistics used for z-score thresholds."""
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
    
    mean_err_array = torch.tensor(mean_error).to(device).squeeze(-1)
    std_err_array = torch.tensor(std_error).to(device).squeeze(-1)
    std_err_array = torch.where(std_err_array.abs() < 1e-6, mean_err_array, std_err_array)
    
    return mean_err_array, std_err_array


@define_tool()
def evaluate_model_on_run(runs, base_dir="1", batch_size=32):
    """Backward-compatible alias for evaluate_model_on_runs."""
    return _evaluate_model_on_runs_impl(runs=runs, base_dir=base_dir, batch_size=batch_size)