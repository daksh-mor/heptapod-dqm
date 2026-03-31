# HEP DQM: A Specialized AI Agent Following the HEPTAPOD Framework

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.12%20|%203.13-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Orchestral--AI-green.svg)](https://orchestral-ai.com)

## Overview

This repository demonstrates an intelligent Data Quality Monitoring (DQM) system for High Energy Physics. It leverages **HEPTAPOD**, a framework built on top of `orchestral-ai`, which combines Large Language Models, specialized prompts, and domain-specific tools to create AI assistants tailored for physics workflows. 

A core feature of our approach is the **sandbox concept**. To ensure safety, reproducibility, and prevent unintentional system modifications, all agent operations—including data access, model training, and output generation—are strictly confined within a designated sandbox directory.

## Demo

Here is a look at our conversational Streamlit GUI in action. The agent orchestrates natural language requests, handles the underlying complexity, and visualizes the results seamlessly.

![Streamlit GUI Demo - Agent performing EDA](images/agent_reply_for_eda_tool.png)

## Tools Implemented

![DQM Tools Architecture](images/dqm_tools.png)

## Details of Some Important Tools

Our agent relies on a suite of robust tools to handle different stages of the DQM pipeline. Here is a brief look at how they work and what they produce:

**Training Tool (`DQMTrainTool`)**  
Trains a baseline DepthViT autoencoder strictly within the sandbox to learn normal detector patterns. It outputs real-time training progress, logging final loss metrics and optimal reconstruction thresholds.  
![Training Output Example](images/train_reply.png)

**Evaluation Tool (`DQMEvaluateTool`)**  
Evaluates the trained model against test data to produce reconstruction and anomaly scores. It computes precision, recall, and ROC-AUC for various anomaly strengths.  
![Evaluation Output Example](images/eval_tool_reply.png)

**Anomaly and Trade-off Analysis**  
Visualizes the relationship between anomaly strength and detection capability, allowing physicists to fine-tune the system's sensitivity.  
![Anomaly vs ROC AUC](images/anomaly_vs_rocauc.png)

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
   ![CLI Demo](images/cli_demo.png)  
   *(Check the Jupyter notebook tutorial in `examples/workflows` for a comprehensive guide!)*

## Proposed Structure

Based on extensive iteration, we propose a clean **3-Layer Architecture** for ML-driven DQM systems:

1. **User Interface Layer:** Streamlit and CLI interfaces that handle LLM connection and chat history.
2. **Agent Wrappers Layer:** The HEPTAPOD/Orchestral-AI tools (`tools/dqm/`). These securely wrap complex ML functions, intercepting raw inputs to enforce path-safety and formatting bounds.
3. **Core ML Layer:** The pure Deep Learning logic (`dqm/`). PyTorch models, data loaders, and pure evaluation metrics devoid of any LLM or Agent dependencies.

## Why This is Most Optimal

As the sole contributor to the ML4DQM project last year, I had the privilege of studying the operational real-world DQM pipelines deeply. Building on that work (which also led to an accepted workshop paper), I implemented the foundational Deep Learning logic found in the `dqm/` directory of this repo. 

I say this humbly: organizing the system in this layered manner solves the biggest bottleneck in current ML-physics integration. 
- It keeps the core ML logic pristine and separate, making it testable and scientifically rigorous. 
- It delegates all the natural language "fuzziness" to the agent wrapper layer, ensuring strict type bounds are met before touching the PyTorch models. 
- It guarantees system safety through immediate sandboxing, meaning researchers can tinker and experiment freely without fear of breaking the deployment environment. 

---

Best regards,  
**Daksh Mor**

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

---

## Citation

If you use ML4DQM or HEPTAPOD in your research:

```bibtex
@article{Menzo:2025cim,
    author = {Menzo, Tony and Roman, Alexander and Gleyzer, Sergei and Matchev, Konstantin and Fleming, George T. and H{\"o}che, Stefan and Mrenna, Stephen and Shyamsundar, Prasanth},
    title = "{HEPTAPOD: Orchestrating High Energy Physics Workflows Towards Autonomous Agency}",
    eprint = "2512.15867",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    year = "2025"
}
```

```bibtex
@misc{roman2026orchestralai,
      title={Orchestral AI: A Framework for Agent Orchestration}, 
      author={Roman, Alexander and Roman, Jacob},
      year={2026},
      eprint={2601.02577},
      archivePrefix={arXiv}
}
```

---

## License

GPL-3.0. See [LICENSE](LICENSE) for details.

---

## Contact & Support

**Issues:** [GitHub Issues](https://github.com/tonymenzo/heptapod/issues)

**Maintainers:**
- Tony Menzo - amenzo@ua.edu

**Repository:** [github.com/tonymenzo/heptapod](https://github.com/tonymenzo/heptapod)

---

**Version**: 1.0.0 | **Status**: Production Ready
