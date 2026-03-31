# HEP DQM: A Specialized AI Agent Following the HEPTAPOD

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.12%20|%203.13-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Orchestral--AI-green.svg)](https://orchestral-ai.com)

## Overview

This repository demonstrates an intelligent Data Quality Monitoring (DQM) system for High Energy Physics. It leverages **HEPTAPOD**, a framework built on top of `orchestral-ai`, which combines Large Language Models, specialized prompts, sandbox directories, and domain-specific tools to create AI assistants tailored for physics workflows. 

## Demo
![streamlit-cml_dqm_demo-2026-03-31-15-35-34online-video-cutter com-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/7e3b7cda-ec70-4f9e-8470-606de322b0eb)

Here is a quick look at conversational Streamlit GUI in action. The agent orchestrates natural language requests, handles the underlying complexity, and visualizes the results seamlessly.


## Tools Implemented
<img width="676" height="447" alt="dqm_tools" src="https://github.com/user-attachments/assets/7d0201bf-f608-44cf-9be7-e6bc820232ad" />


## Details of Some Important Tools
**Training Tool (`DQMTrainTool`)**  
Trains a baseline DepthViT autoencoder strictly within the sandbox to learn normal detector patterns. It outputs real-time training progress, logging final loss metrics and optimal reconstruction thresholds.  
<img width="769" height="461" alt="train_reply" src="https://github.com/user-attachments/assets/a48d80f9-86bb-417d-99e4-47524a7ec295" />

**Evaluation Tool (`DQMEvaluateTool`)**  
Evaluates the trained model against test data to produce reconstruction and anomaly scores. It computes precision, recall, and ROC-AUC for various anomaly strengths.  
<img width="566" height="334" alt="image" src="https://github.com/user-attachments/assets/e8d99b4c-9bc9-43f9-8f5a-7d51fd73b802" />


**Dynamic Plot Generation and Analysis**  
The agent can dynamically generate visualizations using libraries like Matplotlib. For instance, when asked to analyze the trade-offs in anomaly detection, the agent can write and execute code to plot the relationship between anomaly strength and ROC AUC. This provides immediate visual feedback, allowing physicists to fine-tune the system's sensitivity. 

<img width="444" height="683" alt="anomaly_vs_rocauc" src="https://github.com/user-attachments/assets/9814bc8d-488a-4cdf-8378-ddd9a9e34243" />

## How to Run

1. **Configure Environment:** Create a `.env` file in the root directory and add your LLM API keys (e.g., `OPENAI_API_KEY`, `GROQ_API_KEY`, etc.).
2. **Web GUI:**  
   Launch the user interface by running:
   ```bash
   streamlit run examples/workflows/cml_dqm_demo.py
   ```
   At startup, select your preferred LLM provider, and you're good to go.
3. **CLI Usage:**  
   For terminal-based interaction, simply run the CLI script directly:
   ```bash
   python examples/workflows/cml_dqm_cli.py
   ```
   <img width="902" height="200" alt="image" src="https://github.com/user-attachments/assets/cc8f1f51-5b6d-44ba-bc82-7414a3458d19" />


   *(Check the Jupyter notebook tutorial in `examples/workflows` for a comprehensive guide!)*

## Proposed Structure

Based on extensive iteration, we propose a clean **3-Layer Architecture** for ML-driven DQM systems:

1. **User Interface Layer:** Streamlit and CLI interfaces that handle LLM connection and chat history.
2. **Agent Wrappers Layer:** The HEPTAPOD/Orchestral-AI tools (`tools/dqm/`). These securely wrap complex ML functions, intercepting raw inputs to enforce path-safety and formatting bounds.
3. **Core ML Layer:** The pure Deep Learning logic (`dqm/`). PyTorch models, data loaders, and pure evaluation metrics devoid of any LLM or Agent dependencies.

```text
heptapod/
  dqm_core/                         # core ML package
    train.py
    evaluate.py
    evaluate_tool.py
    model_datasets.py
    models_spatial.py
    ...

  tools/
    dqm/                            # agent-facing wrappers and safe tool interfaces
      train_tool.py                 # DepthViT training wrapper
      evaluate_tool.py
      eda_tool.py
      deployment_tool.py
      monitoring_tool.py
      alarming_tool.py
      rollback_tool.py
      test_dqm_tools.py
      test_files/
        dataset/
          train_data.npy
          test_data.npy
          he_segmentation_config_mask.npy

  examples/
    workflows/
      cml_dqm_demo.py               # Streamlit UX and provider-gated interaction
      cml_dqm_cli.py                # CLI runner
      cml_dqm_tutorial.ipynb        # reproducible tutorial pipeline
      cml_dqm_sandbox/              # generated workflow outputs
```

## Why This Structure

Having worked on the ML4DQM project last year, I'm confident this architecture is the right approach for integrating ML into physics workflows. It keeps the core logic pristine and independently testable, while the agent wrapper layer enforces strict type bounds and sandboxing to ensure system safety.

---


Run all tests:
```bash
python test_runner.py --skip-slow
```

Run only DQM tests:
```bash
conda run -n tradingagents python tools/dqm/test_dqm_tools.py
```

**Expected result:**
```
✅ EDA path traversal rejected
✅ Train base_dir traversal rejected
✅ Evaluate base_dir traversal rejected
✅ All 7 tools passing
✅ Zero compilation errors
```
Written by **Daksh Mor** for GSOC 26 evaluation task of ML4DQM
