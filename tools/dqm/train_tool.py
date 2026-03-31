"""
# train_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

import os
import json

from orchestral.tools import define_tool
from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

SCHEMA_VERSION = "dqm-train-1.0"


def _build_training_summary_table(result: dict) -> str:
    if not isinstance(result, dict):
        return ""

    runs = result.get("runs", [])
    runs_text = ",".join(str(r) for r in runs) if runs else "-"
    strengths = result.get("anomaly_strengths", [])
    strengths_text = ", ".join(f"{float(x):.1f}" for x in strengths) if strengths else "-"

    lines = [
        "DepthViT Training Summary",
        "------------------------",
        f"runs              : {runs_text}",
        f"epochs            : {result.get('num_epochs', '-')}",
        f"batch_size        : {result.get('batch_size', '-')}",
        f"learning_rate     : {result.get('learning_rate', '-')}",
        f"final_train_loss  : {result.get('final_train_loss', '-')}",
        f"final_val_loss    : {result.get('final_val_loss', '-')}",
        f"anomaly_strengths : {strengths_text}",
        f"model_path        : {result.get('model_path', '-')}",
        f"thresholds_file   : {result.get('thresholds_file', '-')}",
    ]

    thresholds = result.get("optimal_thresholds", {})
    if isinstance(thresholds, dict) and thresholds:
        lines.append("thresholds         :")
        for strength in sorted(thresholds.keys(), key=lambda k: float(k)):
            lines.append(f"  - strength {strength}: {thresholds[strength]:.4f}")

    return "\n".join(lines)


def _train_model_on_runs_impl(
    RUNS,
    base_dir,
    epochs=None,
    batch_size=None,
    learning_rate=None,
    anomaly_strengths=None,
):
    """
    Train model on specified runs using default configuration from dqm.train.

    Args:
        RUNS: Single run ID (int) or list of run IDs
        base_dir: Sandbox subdirectory where model and results are saved
    """
    from dqm_core.train import (
        _train_model_on_runs_impl as _core_train_impl,
        SEED,
        isvariational,
        NUM_EPOCHS as DEFAULT_NUM_EPOCHS,
        BATCH_SIZE as DEFAULT_BATCH_SIZE,
        LR as DEFAULT_LR,
        ANOMALY_STRENGTHS as DEFAULT_ANOMALY_STRENGTHS,
    )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    fixture_dataset_dir = os.path.join(project_root, "tools", "dqm", "test_files", "dataset")
    workflow_sandbox_root = os.path.join(project_root, "examples", "workflows", "cml_dqm_sandbox")

    if isinstance(RUNS, int):
        RUNS = [RUNS]

    if epochs is None:
        epochs = DEFAULT_NUM_EPOCHS
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    if learning_rate is None:
        learning_rate = DEFAULT_LR
    if not anomaly_strengths:
        anomaly_strengths = list(DEFAULT_ANOMALY_STRENGTHS)

    # Force writes to stay under sandbox.
    base_dir_norm = os.path.normpath(str(base_dir)).lstrip(os.sep)
    if base_dir_norm in ("", "."):
        raise ValueError("base_dir must be a non-empty sandbox subdirectory")
    if base_dir_norm.startswith(".."):
        raise ValueError("base_dir must stay within sandbox")

    if base_dir_norm.startswith(os.path.join("examples", "workflows", "cml_dqm_sandbox")):
        base_output_dir = os.path.abspath(os.path.join(project_root, base_dir_norm))
    else:
        base_output_dir = os.path.abspath(os.path.join(workflow_sandbox_root, base_dir_norm))

    root_real = os.path.realpath(workflow_sandbox_root)
    out_real = os.path.realpath(base_output_dir)
    if not (out_real == root_real or out_real.startswith(root_real + os.sep)):
        raise ValueError("base_dir must stay within examples/workflows/cml_dqm_sandbox")

    MASK_PATH = os.path.join(fixture_dataset_dir, "he_segmentation_config_mask.npy")
    BASE_DIR = base_output_dir
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model.pth")
    DATASET_DIR = fixture_dataset_dir
    ANOMALY_DIR = os.path.join(BASE_DIR, "anomalies")

    return _core_train_impl(
        RUNS=RUNS,
        NUM_EPOCHS=epochs,
        LR=learning_rate,
        BATCH_SIZE=batch_size,
        MODEL_SAVE_PATH=MODEL_SAVE_PATH,
        ANOMALY_STRENGTHS=anomaly_strengths,
        SEED=SEED,
        ANOMALY_DIR=ANOMALY_DIR,
        BASE_DIR=BASE_DIR,
        DATASET_DIR=DATASET_DIR,
        MASK_PATH=MASK_PATH,
        isvariational=isvariational,
    )


@define_tool()
def train_model_on_runs(
    RUNS,
    base_dir,
    epochs=None,
    batch_size=None,
    learning_rate=None,
    anomaly_strengths=None,
):
    return _train_model_on_runs_impl(
        RUNS=RUNS,
        base_dir=base_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        anomaly_strengths=anomaly_strengths,
    )


class DQMTrainDepthvitTool(BaseTool):
    """DepthViT training wrapper that trains a DQM model from run IDs."""

    RUNS: list = RuntimeField(description="Single run ID or list of run IDs")
    base_dir: str = RuntimeField(description="Sandbox subdirectory for outputs")
    epochs: int = RuntimeField(default=2, description="Epoch count (defaults to DepthViT training default)")
    batch_size: int = RuntimeField(default=320, description="Batch size (defaults to DepthViT training default)")
    learning_rate: float = RuntimeField(default=3e-3, description="Learning rate (defaults to DepthViT training default)")
    anomaly_strengths: list = RuntimeField(default=None, description="Optional anomaly strengths override")

    base_directory: str = StateField(default=".", description="Unused compatibility state field")

    def _run(self) -> str:
        try:
            result = _train_model_on_runs_impl(
                self.RUNS,
                self.base_dir,
                self.epochs,
                self.batch_size,
                self.learning_rate,
                self.anomaly_strengths,
            )
            if isinstance(result, dict) and result.get("status") == "ok":
                result["summary_table"] = _build_training_summary_table(result)
            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        except Exception as e:
            return self.format_error(
                error="Training Error",
                reason=str(e),
                suggestion="Provide RUNS and a sandbox-relative base_dir",
            )


class DQMTrainTool(DQMTrainDepthvitTool):
    """Short alias for training tool."""
