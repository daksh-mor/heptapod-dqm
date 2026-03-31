"""Lightweight training wrapper around dqm.train defaults."""

import os

from orchestral.tools import define_tool


@define_tool()
def train_model_on_runs(
    RUNS,
    base_dir,
    epochs=None,
    batch_size=None,
    learning_rate=None,
    anomaly_strengths=None,
):
    """Train on one or more runs and save artifacts inside sandbox/base_dir."""
    try:
        from .train import (
            _train_model_on_runs_impl,
            SEED,
            isvariational,
            NUM_EPOCHS as DEFAULT_NUM_EPOCHS,
            BATCH_SIZE as DEFAULT_BATCH_SIZE,
            LR as DEFAULT_LR,
            ANOMALY_STRENGTHS as DEFAULT_ANOMALY_STRENGTHS,
        )
    except ImportError:
        from train import (
            _train_model_on_runs_impl,
            SEED,
            isvariational,
            NUM_EPOCHS as DEFAULT_NUM_EPOCHS,
            BATCH_SIZE as DEFAULT_BATCH_SIZE,
            LR as DEFAULT_LR,
            ANOMALY_STRENGTHS as DEFAULT_ANOMALY_STRENGTHS,
        )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fixture_dataset_dir = os.path.join(project_root, "tools", "dqm", "test_files", "dataset")
    workflow_sandbox_root = os.path.join(project_root, "examples", "workflows", "cml_dqm_sandbox")

    if isinstance(RUNS, int):
        RUNS = [RUNS]
    elif isinstance(RUNS, str):
        RUNS = [int(RUNS)]

    if epochs is None:
        epochs = DEFAULT_NUM_EPOCHS
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    if learning_rate is None:
        learning_rate = DEFAULT_LR
    if not anomaly_strengths:
        anomaly_strengths = list(DEFAULT_ANOMALY_STRENGTHS)

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

    model_save_path = os.path.join(base_output_dir, "model.pth")
    dataset_dir = fixture_dataset_dir
    mask_path = os.path.join(fixture_dataset_dir, "he_segmentation_config_mask.npy")
    anomaly_dir = os.path.join(base_output_dir, "anomalies")

    return _train_model_on_runs_impl(
        RUNS=RUNS,
        NUM_EPOCHS=epochs,
        LR=learning_rate,
        BATCH_SIZE=batch_size,
        MODEL_SAVE_PATH=model_save_path,
        ANOMALY_STRENGTHS=anomaly_strengths,
        SEED=SEED,
        ANOMALY_DIR=anomaly_dir,
        BASE_DIR=base_output_dir,
        DATASET_DIR=dataset_dir,
        MASK_PATH=mask_path,
        isvariational=isvariational,
    )
