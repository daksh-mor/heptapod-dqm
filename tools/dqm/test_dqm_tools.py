#!/usr/bin/env python3
"""
# test_dqm_tools.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
DQM_DIR = SCRIPT_PATH.parent
TOOLS_DIR = DQM_DIR.parent
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

SKIP_REASON = None

try:
    from tools.dqm import (
        DQMEDATool,
        DQMTrainTool,
        DQMEvaluateTool,
        DQMDeploymentTool,
    )
except Exception as e:
    SKIP_REASON = f"Optional dependencies unavailable for DQM tools: {e}"

BASE_DIR = DQM_DIR / "test_files" / "sandbox"


def _write_toy_dataset(path: Path, n: int = 24) -> None:
    rng = np.random.default_rng(7)
    arr = rng.normal(0, 1, size=(n, 8, 6, 2)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def test_path_safety() -> None:
    print(">> Testing path traversal prevention...\n")

    tool = DQMEDATool(
        base_directory=str(BASE_DIR),
        dataset_path="../../../etc/passwd",
        output_dir="eda",
    )
    out = tool._run().lower()
    assert "denied" in out or "error" in out
    print("[✓] EDA path traversal rejected")

    tool = DQMTrainTool(RUNS=[323997], base_dir="../escape")
    out = tool._run().lower()
    assert "error" in out
    print("[✓] Train base_dir traversal rejected")

    tool = DQMEvaluateTool(runs=[323997], base_dir="../escape")
    out = tool._run().lower()
    assert "error" in out
    print("[✓] Evaluate base_dir traversal rejected\n")


def test_eda_and_deploy() -> None:
    print(">> Testing DQM EDA + deployment tool path handling...\n")

    toy_path = BASE_DIR / "toy.npy"
    _write_toy_dataset(toy_path)

    eda_tool = DQMEDATool(
        base_directory=str(BASE_DIR),
        dataset_path="toy.npy",
        output_dir="artifacts/eda",
        max_hist_bins=40,
    )
    ed_raw = eda_tool._run()
    ed = json.loads(ed_raw)
    assert ed.get("status") == "ok", ed_raw
    assert (BASE_DIR / ed["summary_path"]).exists()
    assert (BASE_DIR / ed["histogram_path"]).exists()
    assert (BASE_DIR / ed["projection_path"]).exists()
    print("[✓] EDA tool passed")

    model_dir = BASE_DIR / "artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "dummy_model.pth"
    model_file.write_bytes(b"placeholder")

    deploy_tool = DQMDeploymentTool(
        base_directory=str(BASE_DIR),
        model_path="artifacts/dummy_model.pth",
        endpoint_name="he-dqm-depthvit-v1",
        runtime_target="cms-dqm-stream",
    )
    dp_raw = deploy_tool._run()
    dp = json.loads(dp_raw)
    assert dp.get("status") == "ok", dp_raw
    assert dp.get("deployed") is False
    print("[✓] Deployment tool passed\n")


def cleanup() -> None:
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
        print("[✓] Cleaned test sandbox")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DQM tool tests")
    parser.add_argument("--keep-files", action="store_true", help="Keep generated test files")
    args = parser.parse_args()

    if SKIP_REASON is not None:
        print(f"[i] Skipping DQM tests: {SKIP_REASON}")
        sys.exit(0)

    ok = True
    try:
        test_path_safety()
        test_eda_and_deploy()
    except Exception as e:
        print(f"[✗] Test failure: {e}")
        ok = False

    if not args.keep_files:
        cleanup()

    if ok:
        print("\n[✓] All DQM tool tests passed\n")
        sys.exit(0)

    print("\n[✗] DQM tool tests failed\n")
    sys.exit(1)
