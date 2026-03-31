"""
Alarming tool stub for ML4DQM anomaly alerting.

This tool demonstrates how to raise and manage alarms when anomaly detection
triggers in real-time CMS workflows. Integration with actual DQM alarm systems
is left as a task for future contributors.

Run with:
    DQMAlarmingTool(base_directory="/path/to/sandbox").run(
        severity="critical",
        anomaly_rate=0.15,
        threshold=0.10
    )
"""

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "dqm-alarming-1.0"


class DQMAlarmingTool:
    """Stub for anomaly alerting and threshold management.
    
    Intended for integration with CMS DQM alarm systems (JIRA, Slack, PagerDuty).
    This stub demonstrates the alarm interface; routing to actual systems deferred.
    
    Attributes:
        base_directory (str): Safe base directory for alarm logs.
    """

    def __init__(self, base_directory: str):
        self.base_directory = Path(base_directory).resolve()
        self.name = "DQMAlarmingTool"
        self.severity_levels = ["info", "warning", "critical", "fatal"]

    def run(
        self,
        severity: str,
        anomaly_rate: float,
        threshold: float,
        detector_region: str = "HE",
        run_number: int = 0,
    ) -> dict[str, Any]:
        """Create or update an alarm for anomaly detection.
        
        Args:
            severity: Alert level (info, warning, critical, fatal).
            anomaly_rate: Observed anomaly rate from model.
            threshold: Expected anomaly rate for normal operation.
            detector_region: CMS detector region (e.g., "HE", "EB", "EE").
            run_number: CMS run number for context.
        
        Returns:
            dict with alarm status, routing recommendations, and next actions.
        """
        try:
            if severity not in self.severity_levels:
                return {
                    "status": "error",
                    "message": f"Invalid severity: {severity}. Must be one of {self.severity_levels}",
                }

            # Validate anomaly rate
            if not (0 <= anomaly_rate <= 1):
                return {
                    "status": "error",
                    "message": f"Anomaly rate must be in [0, 1], got {anomaly_rate}",
                }

            # Compute alarm reason
            deviation = (anomaly_rate - threshold) / max(threshold, 1e-6)

            alarm_record = {
                "status": "ok",
                "alarm_id": f"ML4DQM-{detector_region}-{run_number}",
                "severity": severity,
                "detector_region": detector_region,
                "run_number": run_number,
                "anomaly_rate": anomaly_rate,
                "threshold": threshold,
                "deviation_percent": round(deviation * 100, 2),
                "alarm_reason": (
                    f"Anomaly rate ({anomaly_rate:.2%}) exceeds threshold ({threshold:.2%}) "
                    f"by {abs(deviation):.1%}"
                ),
                "recommended_actions": _get_recommended_actions(severity, detector_region),
                "alert_routing": {
                    "info": "Log to run report summary",
                    "warning": "Notify shifter via Slack; log to run database",
                    "critical": "Page on-call physicist; create JIRA ticket; notify DQM lead",
                    "fatal": "Trigger automatic detector shutdown; page all on-call staff",
                }[severity],
                "next_steps": [
                    "1. Operator reviews anomaly details in DQM GUI.",
                    "2. Run DQMMonitoringTool to diagnose drift vs. genuine anomalies.",
                    "3. If drift detected, run DQMRollbackTool to revert model.",
                    "4. If genuine anomaly, investigate detector/event quality.",
                ],
                "deployment_notes": (
                    "This tool is a planning stub. Alert routing requires integration with "
                    "existing CMS DQM systems (Slack workspace, JIRA project, PagerDuty). "
                    "See heptapod documentation for integration templates."
                ),
            }

            return alarm_record

        except Exception as e:
            return {
                "status": "error",
                "message": f"Alarming tool error: {e}",
            }


def _get_recommended_actions(severity: str, detector_region: str) -> list[str]:
    """Return context-specific recommended actions for an alarm."""
    base_actions = [
        "Review event logs for corresponding time window.",
        "Check detector status (HV, cooling, readout).",
        "Compare anomaly score distribution to reference.",
    ]

    if severity in ["critical", "fatal"]:
        base_actions.extend([
            "Contact detector subsystem expert.",
            "Consider temporary disabling of problematic channels.",
        ])

    if detector_region == "HE":
        base_actions.append("Check HE front-end electronics status.")
    elif detector_region in ["EB", "EE"]:
        base_actions.append("Check ECAL readout chain for timing issues.")

    return base_actions
