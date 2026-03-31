"""
# eda_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import json
import os
from typing import Optional

import numpy as np

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

SCHEMA_VERSION = "dqm-eda-1.0"


class DQMDataEDATool(BaseTool):
    """
    Run quick exploratory data analysis on a NumPy DQM dataset.

    The tool computes summary statistics and generates lightweight plots that are
    useful for initial data sanity checks before model training.

    Inputs (runtime):
      - dataset_path: Relative path to input .npy dataset.
      - output_dir: Relative directory to save summary and plots.
      - max_hist_bins: Histogram bins for value distribution (default: 60).

    State:
      - base_directory: Base sandbox directory for file operations.

    Output (JSON):
      {
        "schema": "dqm-eda-1.0",
        "status": "ok",
        "summary_path": "...",
        "histogram_path": "...",
        "projection_path": "...",
        "shape": [...]
      }
    """

    dataset_path: str = RuntimeField(description="Relative path to input .npy dataset")
    output_dir: str = RuntimeField(description="Relative output directory for summary and plots")
    max_hist_bins: int = RuntimeField(default=60, description="Number of bins for flattened-value histogram")

    base_directory: str = StateField(default=".", description="Base sandbox root for file operations")

    def _setup(self):
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.isdir(self.base_directory):
            raise ValueError(f"Base directory does not exist: {self.base_directory}")

    def _safe_path(self, rel_or_abs: str) -> Optional[str]:
        if not rel_or_abs:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel_or_abs))
        if not (full.startswith(self.base_directory + os.sep) or full == self.base_directory):
            return None
        return full

    def _run(self) -> str:
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Setup Error", reason=str(e))

        src = self._safe_path(self.dataset_path)
        out_dir = self._safe_path(self.output_dir)

        if not src or not out_dir:
            return self.format_error(
                error="Access Denied",
                reason="dataset_path or output_dir escapes base_directory",
                suggestion="Use relative paths inside base_directory"
            )

        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason="Input dataset does not exist",
                context=f"dataset_path={self.dataset_path}"
            )

        if int(self.max_hist_bins) < 5:
            return self.format_error(
                error="Invalid Input",
                reason="max_hist_bins must be >= 5",
                context=f"max_hist_bins={self.max_hist_bins}"
            )

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            return self.format_error(
                error="Dependency Missing",
                reason="matplotlib is required for EDA plots",
                suggestion="Install with: pip install matplotlib",
                context=str(e)
            )

        try:
            arr = np.load(src)
            if arr.size == 0:
                return self.format_error(error="Invalid Dataset", reason="Dataset is empty")

            os.makedirs(out_dir, exist_ok=True)

            flat = arr.reshape(-1).astype(np.float64)
            summary = {
                "schema": SCHEMA_VERSION,
                "status": "ok",
                "dataset_path": self.dataset_path,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "mean": float(np.mean(flat)),
                "std": float(np.std(flat)),
                "min": float(np.min(flat)),
                "max": float(np.max(flat)),
                "q01": float(np.quantile(flat, 0.01)),
                "q25": float(np.quantile(flat, 0.25)),
                "median": float(np.quantile(flat, 0.50)),
                "q75": float(np.quantile(flat, 0.75)),
                "q99": float(np.quantile(flat, 0.99)),
                "n_samples": int(arr.shape[0]) if arr.ndim > 0 else 1,
            }

            summary_path = os.path.join(out_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            hist_path = os.path.join(out_dir, "histogram.png")
            plt.figure(figsize=(8, 4.5))
            plt.hist(flat, bins=int(self.max_hist_bins), color="#0EA5E9", alpha=0.9)
            plt.title("DQM Value Distribution")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(hist_path, dpi=160)
            plt.close()

            projection_path = os.path.join(out_dir, "projection.png")
            projection = arr[0]
            while projection.ndim > 2:
                projection = np.mean(projection, axis=-1)

            plt.figure(figsize=(6.5, 5.0))
            plt.imshow(projection, cmap="viridis", aspect="auto")
            plt.colorbar(label="Mean value")
            plt.title("Sample 0 Projection")
            plt.tight_layout()
            plt.savefig(projection_path, dpi=160)
            plt.close()

            result = {
                "schema": SCHEMA_VERSION,
                "status": "ok",
                "summary_path": os.path.relpath(summary_path, self.base_directory),
                "histogram_path": os.path.relpath(hist_path, self.base_directory),
                "projection_path": os.path.relpath(projection_path, self.base_directory),
                "shape": list(arr.shape),
            }
            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        except Exception as e:
            return self.format_error(
                error="EDA Error",
                reason=str(e),
                suggestion="Verify dataset path and array structure"
            )


class DQMEDATool(DQMDataEDATool):
    """Short alias for DQMDataEDATool."""
