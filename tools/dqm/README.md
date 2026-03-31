# DQM Stub Tools for Real-Time Integration

This directory contains the core ML-facing tools (train, evaluate, EDA, deployment) and planning stubs for real-time CMS DQM operations.

## Tool Inventory

### Production-Ready Tools

- **`train_tool.py` — DQMTrainTool**: DepthViT model training with configurable hyperparameters, metric tracking, and model checkpointing.
- **`evaluate_tool.py` — DQMEvaluateTool**: Model evaluation, anomaly scoring, ROC-based threshold computation, and plot generation.
- **`eda_tool.py` — DQMEDATool**: Exploratory data analysis, summary statistics, and visualization generation.

### Planning/Stub Tools for Real-Time Deployment

These stubs demonstrate a complete operational story for real-time integration. Actual CMS infrastructure wiring is deferred to operational teams or future contributors.

- **`deployment_tool.py` — DQMDeploymentTool**: Deployment manifest generation, model integrity validation, and orchestration hints.
- **`monitoring_tool.py` — DQMMonitoringTool**: Performance and data drift monitoring interface. Returns monitoring configuration and next steps for stream integration.
- **`alarming_tool.py` — DQMAlarmingTool**: Anomaly detection alerting, severity classification, and recommended actions for operators.
- **`rollback_tool.py` — DQMRollbackTool**: Model version management, fast rollback to known-good versions, and deployment history tracking.

## Why Stubs?

The monitoring, alarming, and rollback tools are stubs because:

1. **Scope Boundary**: Actual CMS integration requires access to live DQM infrastructure (MQ brokers, stream processors, alarm systems, model repositories).
2. **Operational Approval**: Changes to production DQM workflows require coordination with CMS operations and detector experts.
3. **Design Clarity**: Stubs define the interface clearly, making it obvious what a future contributor would need to implement.
4. **GSoC Task Fit**: The proposal demonstrates full deployment vision while being honest about feasibility constraints.

## Example Usage

### Training a Model

```python
from tools.dqm import DQMTrainTool

tool = DQMTrainTool(base_directory="examples/workflows/cml_dqm_sandbox/sandbox000")
result = tool.run(
    RUNS=[323997],
    base_dir="examples/workflows/cml_dqm_sandbox/sandbox000",
    epochs=10,
    batch_size=64,
    learning_rate=0.003
)
print(result)  # Returns final loss, curves, thresholds, model path
```

### Monitoring Performance

```python
from tools.dqm import DQMMonitoringTool

tool = DQMMonitoringTool(base_directory="examples/workflows/cml_dqm_sandbox/sandbox000")
result = tool.run(
    model_path="model.pth",
    reference_data="reference.npy",
    window_size=100,
    drift_threshold=0.1
)
print(result)  # Returns monitoring readiness and next steps
```

### Creating an Alarm

```python
from tools.dqm import DQMAlarmingTool

tool = DQMAlarmingTool(base_directory="examples/workflows/cml_dqm_sandbox/sandbox000")
result = tool.run(
    severity="critical",
    anomaly_rate=0.18,
    threshold=0.10,
    detector_region="HE",
    run_number=376543
)
print(result)  # Returns alarm details and routing recommendations
```

### Rolling Back a Version

```python
from tools.dqm import DQMRollbackTool

tool = DQMRollbackTool(base_directory="examples/workflows/cml_dqm_sandbox/sandbox000")

# List available versions
versions = tool.run(action="list_versions", detector_region="HE")

# Rollback to a previous version
result = tool.run(
    action="rollback",
    version="v1.2",
    detector_region="HE",
    reason="Data drift detected"
)
print(result)  # Returns rollback confirmation and next steps
```

## Testing

Run the test suite to validate all tools:

```bash
python tools/dqm/test_dqm_tools.py
```

Expected output:
```
[✓] EDA path traversal rejected
[✓] Train base_dir traversal rejected
[✓] Evaluate base_dir traversal rejected
[✓] EDA tool passed
[✓] Deployment tool passed
[✓] All DQM tool tests passed
```

## Integration with Orchestral Agent

All tools are compatible with the Orchestral AI framework. Example:

```python
from orchestral import Agent
from orchestral.llm import GPT
from tools.dqm import DQMTrainTool, DQMMonitoringTool, DQMRollbackTool

agent = Agent(
    llm=GPT(),
    tools=[
        DQMTrainTool(base_directory="..."),
        DQMMonitoringTool(base_directory="..."),
        DQMRollbackTool(base_directory="..."),
    ]
)

response = agent.run("Train a model and monitor for drift")
```

## Next Steps for Contributors

To complete real-time integration:

1. **Monitoring Tool**: Connect tool to CMS DQM producer stream; implement rolling window statistics and KL-divergence computation.
2. **Alarming Tool**: Integrate with CMS Slack workspace, JIRA project, and PagerDuty for alert routing.
3. **Rollback Tool**: Connect to model repository (S3, artifact store) and CMS deployment automation.
4. **Deployment Tool**: Wire to CMS DQM MQ brokers and real-time processing framework.

See [`docs/ml4dqm_gsoc_2026_submission.md`](docs/ml4dqm_gsoc_2026_submission.md) for architectural context and [`docs/ml4dqm_proposed_structure.md`](docs/ml4dqm_proposed_structure.md) for long-term design.

---

**Framework:** HEPTAPOD (High-Energy Physics Toolkit for Agentic Planning, Orchestration, and Deployment)  
**Related Paper:** https://arxiv.org/abs/2512.15867  
**Repository:** https://github.com/tonymenzo/heptapod
