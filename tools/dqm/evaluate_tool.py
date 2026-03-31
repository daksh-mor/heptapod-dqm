"""
# evaluate_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

import json

from orchestral.tools.base.field_utils import RuntimeField, StateField
from orchestral.tools.base.tool import BaseTool

SCHEMA_VERSION = "dqm-evaluate-1.0"


class DQMEvaluateDepthvitTool(BaseTool):
    """Evaluate a trained DepthViT DQM model on one or more runs."""

    runs: list = RuntimeField(description="Single run ID or list of run IDs")
    base_dir: str = RuntimeField(description="Sandbox subdirectory used during training")
    batch_size: int = RuntimeField(default=32, description="Batch size for evaluation")
    anomaly_strengths: list = RuntimeField(default=None, description="Optional anomaly strengths override")
    seed: int = RuntimeField(default=42, description="Random seed")

    base_directory: str = StateField(default=".", description="Unused compatibility state field")

    def _run(self) -> str:
        try:
            from dqm.evaluate_tool import _evaluate_model_on_runs_impl

            result = _evaluate_model_on_runs_impl(
                runs=self.runs,
                base_dir=self.base_dir,
                batch_size=self.batch_size,
                anomaly_strengths=self.anomaly_strengths,
                seed=self.seed,
            )
            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        except Exception as e:
            return self.format_error(
                error="Evaluation Error",
                reason=str(e),
                suggestion="Provide runs and a sandbox-relative base_dir with model artifacts",
            )


class DQMEvaluateTool(DQMEvaluateDepthvitTool):
    """Short alias for evaluation tool."""


class DQMEvaluateAutoencoderTool(DQMEvaluateDepthvitTool):
    """Backward-compatible alias for older workflows."""
