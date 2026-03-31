"""
Alarming tool stub for ML4DQM anomaly alerting.

Demonstrates severity-based anomaly alerting with routing recommendations.
Integration with actual CMS alarm systems is deferred to future work.
"""

import json
from pathlib import Path
from typing import Any, Literal

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

SCHEMA_VERSION = "dqm-alarming-1.0"


class DQMAlarmingTool(BaseTool):
    """Anomaly alerting and threshold management stub.
    
    Creates and manages alarms when anomaly detection triggers in production.
    """

    severity: str = RuntimeField(description="Alert level (info, warning, critical, fatal)")
    anomaly_rate: float = RuntimeField(description="Observed anomaly rate from model")
    threshold: float = RuntimeField(description="Expected anomaly rate for normal operation")
    detector_region: str = RuntimeField(default="HE", description="CMS detector region (HE, EB, EE)")
    run_number: int = RuntimeField(default=0, description="CMS run number for context")
    base_directory: str = StateField(description="Sandbox directory for alarm logs")

    def _run(self) -> str:
        """Create or update an alarm for anomaly detection."""
        try:
            severity_levels = ["info", "warning", "critical", "fatal"]
            
            if self.severity not in severity_levels:
                result = {
                    "status": "error",
                    "message": f"Invalid severity: {self.severity}. Must be one of {severity_levels}",
                }
                return json.dumps(result, separators=(",", ":"))

            # Validate anomaly rate
            if not (0 <= self.anomaly_rate <= 1):
                result = {
                    "status": "error",
                    "message": f"Anomaly rate must be in [0, 1], got {self.anomaly_rate}",
                }
                return json.dumps(result, separators=(",", ":"))

            # Compute alarm reason
            deviation = (self.anomaly_rate - self.threshold) / max(self.threshold, 1e-6)

            alarm_record = {
                "status": "ok",
                "alarm_id": f"ML4DQM-{self.detector_region}-{self.run_number}",
                "severity": self.severity,
                "detector_region": self.detector_region,
                "run_number": self.run_number,
                "anomaly_rate": self.anomaly_rate,
                "threshold": self.threshold,
                "deviation_percent": round(deviation * 100, 2),
                "alarm_reason": (
                    f"Anomaly rate ({self.anomaly_rate:.2%}) exceeds threshold "
                    f"({self.threshold:.2%}) by {abs(deviation):.1%}"
                ),
                "recommended_actions": self._get_recommended_actions(),
                "alert_routing": {
                    "info": "Log to run report summary",
                    "warning": "Notify shifter via Slack; log to database",
                    "critical": "Page physicist; create JIRA; notify DQM lead",
                    "fatal": "Trigger shutdown; page all on-call staff",
                }[self.severity],
                "next_steps": [
                    "1. Review anomaly details in DQM GUI",
                    "2. Run DQMMonitoringTool to diagnose drift",
                    "3. If drift detected, run DQMRollbackTool",
                    "4. If genuine anomaly, investigate detector",
                ],
                "deployment_notes": (
                    "Stub for alert routing. See heptapod documentation "
                    "for integration templates with Slack, JIRA, PagerDuty."
                ),
            }

            return json.dumps(alarm_record, separators=(",", ":"))

        except Exception as e:
            result = {
                "status": "error",
                "message": f"Alarming tool error: {e}",
            }
            return json.dumps(result, separators=(",", ":"))

    def _get_recommended_actions(self) -> list[str]:
        """Return context-specific recommended actions."""
        base_actions = [
            "Review event logs for corresponding time window",
            "Check detector status (HV, cooling, readout)",
            "Compare anomaly distribution to reference",
        ]

        if self.severity in ["critical", "fatal"]:
            base_actions.extend([
                "Contact detector subsystem expert",
                "Consider temporary disabling of bad channels",
            ])

        if self.detector_region == "HE":
            base_actions.append("Check HE front-end electronics")
        elif self.detector_region in ["EB", "EE"]:
            base_actions.append("Check ECAL readout timing")

        return base_actions
