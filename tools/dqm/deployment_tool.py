"""
# deployment_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import json
import os
from typing import Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

SCHEMA_VERSION = "dqm-deploy-1.0"


class DQMRealtimeDeploymentStubTool(BaseTool):
    """
    Provide a skeleton interface for real-time DQM model deployment workflows.

    This is a planning/stub tool. It validates model path safety and returns
    structured metadata for future online deployment integrations.

    Inputs (runtime):
      - model_path: Relative path to a trained model artifact.
      - endpoint_name: Logical deployment endpoint name (e.g. "he-dqm-ae-v1").
      - runtime_target: Target runtime platform label (default: "cms-dqm-stream").

    State:
      - base_directory: Base sandbox directory for file operations.

    Output (JSON):
      {
        "schema": "dqm-deploy-1.0",
        "status": "ok",
        "deployed": false,
        "model_path": "...",
        "endpoint_name": "...",
        "runtime_target": "...",
        "next_steps": [...]
      }
    """

    model_path: str = RuntimeField(description="Relative path to a trained model artifact")
    endpoint_name: str = RuntimeField(description="Logical name for deployment endpoint")
    runtime_target: str = RuntimeField(default="cms-dqm-stream", description="Target deployment runtime label")

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

        model_src = self._safe_path(self.model_path)
        if not model_src:
            return self.format_error(
                error="Access Denied",
                reason="model_path escapes base_directory",
                suggestion="Use a relative path inside base_directory"
            )

        if not os.path.exists(model_src):
            return self.format_error(
                error="File Not Found",
                reason="Model artifact does not exist",
                context=f"model_path={self.model_path}"
            )

        result = {
            "schema": SCHEMA_VERSION,
            "status": "ok",
            "deployed": False,
            "model_path": os.path.relpath(model_src, self.base_directory),
            "endpoint_name": self.endpoint_name,
            "runtime_target": self.runtime_target,
            "next_steps": [
                "Convert the checkpoint to an online-serving format (e.g. TorchScript or ONNX).",
                "Wrap model inference in a low-latency service endpoint.",
                "Attach stream ingestion from DQM online sources.",
                "Enable monitoring for drift and threshold alarms.",
            ],
        }
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)


class DQMRealtimeDeploymentTool(DQMRealtimeDeploymentStubTool):
    """Preferred deployment planning tool name."""


class DQMDeploymentTool(DQMRealtimeDeploymentStubTool):
    """Short alias for deployment planning tool."""
