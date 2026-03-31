"""
# cml_dqm_demo.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Streamlit GUI demo for ML4DQM workflows.

Run with:
    streamlit run examples/workflows/cml_dqm_demo.py
"""

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv  
load_dotenv()
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from orchestral import Agent
from orchestral.llm import GPT, Claude, Gemini, Groq
from orchestral.tools import (
    RunCommandTool,
    WriteFileTool,
    ReadFileTool,
    EditFileTool,
    FileSearchTool,
    FindFilesTool,
    TodoWrite,
    TodoRead,
)
from orchestral.tools.hooks import TruncateOutputHook
from hooks import ToolCallLogger

from prompts import CML_DQM_EXPLORER_PROMPT
from tools.dqm import (
    DQMEDATool,
    DQMTrainTool,
    DQMEvaluateTool,
    DQMDeploymentTool,
    DQMMonitoringTool,
    DQMAlarmingTool,
    DQMRollbackTool,
)

IMAGE_PATH_PATTERN = re.compile(r"(?:!\[.*?\]\()([^)]+\.(?:png|jpg|jpeg|gif|webp))(?:\)|(?![^)]*\)))|(?:^|\s|`)([A-Za-z0-9_./\-\s]+\.(?:png|jpg|jpeg|gif|webp))", re.IGNORECASE | re.MULTILINE)
DATASET_PATH_PATTERN = re.compile(
    r"(data/dataset/he_train_dataset_Run\d+/train_data\.npy|tools/dqm/test_files/dataset/train_data\.npy)"
)
LLM_PROVIDERS = ["GPT", "Claude", "Gemini", "Groq"]

LEGAL_DISCLAIMER = """
This assistant is intended for development and experimentation workflows.
Responses can be incomplete or incorrect and should be validated before use in
analysis or operational decisions. Do not paste secrets or sensitive data.
"""

# -----------------------------------------------------------------------------
# Agent & Configuration Functions
# -----------------------------------------------------------------------------

def _get_llm(provider: str):
    provider = (provider or "GPT").strip().upper()
    if provider == "GPT":
        return GPT()
    if provider == "CLAUDE":
        return Claude()
    if provider == "GEMINI":
        return Gemini()
    return Groq()


def _env_key_issue_for_provider(provider: str) -> str | None:
    provider = (provider or "GPT").strip().upper()
    key_map = {
        "GPT": "OPENAI_API_KEY",
        "CLAUDE": "ANTHROPIC_API_KEY",
        "GEMINI": "GOOGLE_API_KEY",
        "GROQ": "GROQ_API_KEY",
    }
    key_name = key_map.get(provider)
    if not key_name:
        return None

    value = (os.getenv(key_name) or "").strip()
    if not value:
        return f"Missing {key_name}."

    lowered = value.lower()
    placeholders = ["your_", "<", ">", "...", "replace", "example", "dummy"]
    if any(token in lowered for token in placeholders):
        return f"{key_name} looks like a placeholder value."

    if provider == "GROQ" and not value.startswith("gsk_"):
        return "GROQ_API_KEY does not look valid (expected prefix gsk_)."

    return None

def _build_agent(base_directory: str, provider: str) -> Agent:
    tools = [
        RunCommandTool(base_directory=base_directory),
        WriteFileTool(base_directory=base_directory),
        ReadFileTool(base_directory=base_directory, show_line_numbers=True),
        EditFileTool(base_directory=base_directory),
        FindFilesTool(base_directory=base_directory),
        FileSearchTool(base_directory=base_directory),
        DQMEDATool(base_directory=base_directory),
        DQMTrainTool(base_directory=base_directory),
        DQMEvaluateTool(base_directory=base_directory),
        DQMDeploymentTool(base_directory=base_directory),
        DQMMonitoringTool(base_directory=base_directory),
        DQMAlarmingTool(base_directory=base_directory),
        DQMRollbackTool(base_directory=base_directory),
        TodoRead(),
        TodoWrite(base_directory=base_directory),
    ]

    tool_logger = ToolCallLogger(verbose=True, show_results=True)
    hooks = [tool_logger, TruncateOutputHook(max_length=12000)]

    agent = Agent(
        llm=_get_llm(provider),
        tools=tools,
        tool_hooks=hooks,
        system_prompt=CML_DQM_EXPLORER_PROMPT,
        debug=False,
    )
    setattr(agent, "tool_logger", tool_logger)
    return agent


def _find_available_train_datasets(base_directory: Path) -> list[Path]:
    candidates = []
    fixture_dataset = base_directory / "tools/dqm/test_files/dataset/train_data.npy"
    if fixture_dataset.exists():
        candidates.append(fixture_dataset)
    candidates.extend(base_directory.glob("data/dataset/he_train_dataset_Run*/train_data.npy"))

    def _run_id(path: Path) -> int:
        m = re.search(r"Run(\d+)", str(path))
        return int(m.group(1)) if m else -1

    return sorted(candidates, key=_run_id)


def _rewrite_missing_dataset_reference(text: str, base_directory: Path) -> tuple[str, str | None]:
    matches = DATASET_PATH_PATTERN.findall(text or "")
    if not matches:
        return text, None

    available = _find_available_train_datasets(base_directory)
    if not available:
        return text, None

    replacement = available[-1].relative_to(base_directory).as_posix()
    updated = text
    changed = False

    for dataset_path in set(matches):
        candidate = base_directory / dataset_path
        if not candidate.exists():
            updated = updated.replace(dataset_path, replacement)
            changed = True

    if not changed:
        return text, None

    note = (
        f"Requested dataset path was missing. Auto-switched to existing training dataset: {replacement}"
    )
    return updated, note

def _extract_image_paths(text: str) -> list[str]:
    matches = IMAGE_PATH_PATTERN.findall(text or "")
    cleaned = []
    for match in matches:
        # Handle both single and multiple capture groups
        if isinstance(match, tuple):
            m = next((g for g in match if g), None)
        else:
            m = match
        
        if not m:
            continue
            
        p = m.strip().strip("`\"'.,;:)")
        if p and any(p.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
            cleaned.append(p)
    return cleaned

def _render_image_paths(image_paths: list[str], base_directory: Path, allowed_directory: Path) -> None:
    seen = set()
    base = base_directory.resolve()
    for raw_path in image_paths:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (base / candidate).resolve()

        try:
            # Allow any image within the repo root (base_directory), not just sandbox
            candidate.relative_to(base)
        except ValueError:
            continue

        if not candidate.exists() or not candidate.is_file():
            continue

        rel_path = os.path.relpath(str(candidate), str(base))
        if rel_path in seen:
            continue
        seen.add(rel_path)

        st.image(str(candidate), use_container_width=True)
        st.caption(f"📍 {rel_path}")


def _render_assistant_content(text: str, base_directory: Path, allowed_directory: Path) -> None:
    st.markdown(text)
    _render_image_paths(_extract_image_paths(text), base_directory, allowed_directory)


def _render_tool_calls(tool_events: list[dict], base_directory: Path, allowed_directory: Path) -> None:
    if not tool_events:
        return

    rows = []
    pending: dict | None = None
    for event in tool_events:
        phase = event.get("phase")
        if phase == "before":
            if pending:
                rows.append(pending)
            pending = {
                "tool_name": event.get("tool_name", "unknown_tool"),
                "arguments": event.get("arguments", "{}"),
                "result": "",
            }
        elif phase == "after":
            if pending and pending.get("tool_name") == event.get("tool_name"):
                pending["result"] = event.get("result", "")
                rows.append(pending)
                pending = None
            else:
                rows.append({
                    "tool_name": event.get("tool_name", "unknown_tool"),
                    "arguments": "{}",
                    "result": event.get("result", ""),
                })

    if pending:
        rows.append(pending)

    if not rows:
        return

    with st.expander(f"Tool calls ({len(rows)})", expanded=False):
        for idx, row in enumerate(rows, start=1):
            st.markdown(f"**{idx}. {row['tool_name']}**")
            st.code(row.get("arguments", "{}"), language="json")
            if row.get("result"):
                st.code(row["result"], language="text")
                _render_image_paths(_extract_image_paths(row["result"]), base_directory, allowed_directory)

# -----------------------------------------------------------------------------
# Prompts & UI Configuration
# -----------------------------------------------------------------------------

def _build_suggestions(run_id: int, base_subdir: str) -> dict:
    return {
        ":blue[:material/analytics:] Run Data EDA": (
            "Run DQMEDATool on dataset_path='tools/dqm/test_files/dataset/train_data.npy' "
            "and save outputs into output_dir='examples/workflows/cml_dqm_sandbox/sandbox000/eda/'."
        ),
        ":green[:material/model_training:] Train DepthViT": (
            f"Run DQMTrainTool with RUNS=[{run_id}], base_dir='{base_subdir}', epochs=3, batch_size=64, learning_rate=0.003."
        ),
        ":orange[:material/assessment:] Evaluate Model": (
            f"Run DQMEvaluateTool with runs=[{run_id}], base_dir='{base_subdir}', batch_size=32."
        ),
        ":violet[:material/rocket_launch:] Deployment Plan": (
            f"Run DQMDeploymentTool with model_path='sandbox/{base_subdir}/model.pth', "
            "endpoint_name='he-dqm-depthvit-v1', runtime_target='cms-dqm-stream'."
        ),
    }


@st.dialog("Legal disclaimer")
def _show_disclaimer_dialog() -> None:
    st.caption(LEGAL_DISCLAIMER)

# -----------------------------------------------------------------------------
# Main App Execution
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="HEPTAPOD CML-DQM Assistant",
    page_icon="⚛️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize Persistent Session State Variables
if "provider_selected" not in st.session_state:
    st.session_state.provider_selected = False
if "provider_choice" not in st.session_state:
    st.session_state.provider_choice = "Groq"
if "active_llm_provider" not in st.session_state:
    st.session_state.active_llm_provider = None
if "run_id" not in st.session_state:
    st.session_state.run_id = 323997
if "base_subdir" not in st.session_state:
    st.session_state.base_subdir = "examples/workflows/cml_dqm_sandbox/sandbox000"
if "trigger_tool" not in st.session_state:
    st.session_state.trigger_tool = None

# Provider gate: app does not render chat/tool UI until provider is explicitly chosen.
if not st.session_state.provider_selected:
    st.html("<div style='font-size: 2.5rem; line-height: 1; margin-bottom: 0.5rem;'>⚛️</div>")
    st.title("Select LLM provider", anchor=False)
    st.session_state.provider_choice = st.radio(
        "Choose provider to start this session",
        options=LLM_PROVIDERS,
        index=LLM_PROVIDERS.index(st.session_state.provider_choice) if st.session_state.provider_choice in LLM_PROVIDERS else LLM_PROVIDERS.index("Groq"),
    )

    if st.button("Continue", use_container_width=True, type="primary"):
        st.session_state.active_llm_provider = st.session_state.provider_choice
        st.session_state.provider_selected = True
        st.session_state.agent_key = None
        st.rerun()

    st.caption("Pick your provider first. The assistant UI loads after this selection.")
    st.stop()

# A clean, native logo replacement
st.html("<div style='font-size: 2.5rem; line-height: 1; margin-bottom: 0.5rem;'>⚛️</div>")

title_row = st.container(horizontal=True, vertical_alignment="bottom")

with title_row:
    st.title("HEPTAPOD CML-DQM assistant", anchor=False)
st.caption("EDA, training, evaluation, and deployment planning in one assistant.")

sandbox_dir = REPO_ROOT
sandbox_rel = Path("examples/workflows/cml_dqm_sandbox/sandbox000")
sandbox_output_dir = REPO_ROOT / sandbox_rel
sandbox_output_dir.mkdir(parents=True, exist_ok=True)

# State Management for UI Flow
user_just_asked_initial_question = "initial_question" in st.session_state and st.session_state.initial_question
user_just_clicked_suggestion = "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
user_first_interaction = user_just_asked_initial_question or user_just_clicked_suggestion
has_message_history = "messages" in st.session_state and len(st.session_state.messages) > 0

# -----------------------------------------------------------------------------
# 1. EMPTY STATE UI (No Messages)
# -----------------------------------------------------------------------------
if not user_first_interaction and not has_message_history:
    st.session_state.messages = []
    suggestions = _build_suggestions(st.session_state.run_id, st.session_state.base_subdir)

    with st.container():
        st.chat_input("Ask the DQM agent to run EDA, train, evaluate, or plan deployments...", key="initial_question")

        selected_suggestion = st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=suggestions.keys(),
            key="selected_suggestion",
        )

    links_col, disclaimer_col = st.columns([0.72, 0.28])
    with links_col:
        st.caption("Quick start: use an example above or ask for a full run from data checks to evaluation plots.")
    with disclaimer_col:
        st.button(
            "Legal disclaimer",
            icon=":material/balance:",
            use_container_width=True,
            on_click=_show_disclaimer_dialog,
        )

    st.caption("Minimal modern agent UI for DQM EDA, training, evaluation, and deployment planning.")
    st.stop()


# -----------------------------------------------------------------------------
# 2. CHAT STATE UI (Messages Exist)
# -----------------------------------------------------------------------------
user_message = st.chat_input("Ask a follow-up...")

# Capture tools triggered directly from the popover menus
if st.session_state.trigger_tool:
    user_message = st.session_state.trigger_tool
    st.session_state.trigger_tool = None  # Reset so it doesn't loop

if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    if user_just_clicked_suggestion:
        suggestions = _build_suggestions(st.session_state.run_id, st.session_state.base_subdir)
        user_message = suggestions[st.session_state.selected_suggestion]

with title_row:
    def clear_conversation():
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None
        st.session_state.trigger_tool = None
        # Force a fresh Agent object so provider-side conversational context is reset.
        st.session_state.agent = None
        st.session_state.agent_key = None

    # Render top-bar controls horizontally
    controls_container = st.container(horizontal=True)
    with controls_container:
        st.button("Restart", icon=":material/refresh:", on_click=clear_conversation)
        
        current_provider = st.session_state.get("active_llm_provider", "GPT")
        with st.popover("⚙️ Settings"):
            st.caption(f"Current provider: {current_provider}")
            if st.button("Change provider", use_container_width=True):
                st.session_state.provider_selected = False
                st.session_state.provider_choice = current_provider
                st.session_state.active_llm_provider = None
                st.session_state.agent_key = None
                st.session_state.agent = None
                st.session_state.messages = []
                st.rerun()
            st.number_input("Run ID", min_value=1, step=1, key="run_id")
            st.text_input("Sandbox subdirectory", key="base_subdir")
            st.caption("Base directory")
            st.code(str(sandbox_dir), language=None)
        
        # KEY ADDITION: Quick tools are now permanently available in the top bar
        with st.popover("🛠️ Tools"):
            st.markdown("**Run Quick Actions**")
            suggestions = _build_suggestions(st.session_state.run_id, st.session_state.base_subdir)
            for label, prompt in suggestions.items():
                clean_label = re.sub(r':[a-z]+\[(.*?)\]', r'\1', label) # Strip color formatting for standard buttons
                if st.button(clean_label, use_container_width=True):
                    st.session_state.trigger_tool = prompt
                    st.rerun()

# Initialize Agent
current_provider = st.session_state.get("active_llm_provider", "GPT")
if current_provider not in LLM_PROVIDERS:
    current_provider = "GPT"
    st.session_state.active_llm_provider = current_provider

key = f"agent::{current_provider}::{sandbox_dir}"
if "agent_key" not in st.session_state or st.session_state.agent_key != key:
    st.session_state.agent = _build_agent(str(sandbox_dir), current_provider)
    st.session_state.agent_key = key

st.caption(f"Runtime provider: {current_provider}")
st.caption("If you hit Groq TPM limits, wait for the rate window to refill. Restart now also resets agent context locally.")
if st.button("Legal disclaimer", type="tertiary"):
    _show_disclaimer_dialog()
provider_issue = _env_key_issue_for_provider(current_provider)
if provider_issue:
    st.warning(f"{provider_issue} Set the correct key in .env for provider {current_provider}.")

# Render History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.container() # Fix ghost message bug natively
            _render_assistant_content(msg["content"], sandbox_dir, sandbox_output_dir)
            _render_tool_calls(msg.get("tool_events", []), sandbox_dir, sandbox_output_dir)
        else:
            st.markdown(msg["content"])

# Handle New User Input
if user_message:
    user_message, rewrite_note = _rewrite_missing_dataset_reference(user_message, sandbox_dir)

    # Streamlit Markdown engine interprets "$" as LaTeX code; this escapes it.
    user_message = user_message.replace("$", r"\$")

    with st.chat_message("user"):
        st.text(user_message)

    with st.chat_message("assistant"):
        if rewrite_note:
            st.info(rewrite_note)
        st.caption(f"Using provider: {current_provider}")

        provider_issue = _env_key_issue_for_provider(current_provider)
        if provider_issue:
            text = f"Agent execution blocked: {provider_issue} Set the key in .env and retry."
            tool_events = []
            with st.container():
                _render_assistant_content(text, sandbox_dir, sandbox_output_dir)
                st.session_state.messages.append({"role": "user", "content": user_message})
                st.session_state.messages.append({"role": "assistant", "content": text, "tool_events": tool_events})
            st.stop()

        with st.spinner("Running agent execution..."):
            try:
                logger = getattr(st.session_state.agent, "tool_logger", None)
                if logger and hasattr(logger, "reset_events"):
                    logger.reset_events()

                msg = st.session_state.agent.run(user_message)
                text = getattr(msg, "text", str(msg))
                tool_events = logger.get_events() if logger and hasattr(logger, "get_events") else []
            except Exception as e:
                text = f"Agent execution failed: {e}"
                tool_events = []
        
        with st.container():
            _render_assistant_content(text, sandbox_dir, sandbox_output_dir)
            _render_tool_calls(tool_events, sandbox_dir, sandbox_output_dir)

            st.session_state.messages.append({"role": "user", "content": user_message})
            st.session_state.messages.append({"role": "assistant", "content": text, "tool_events": tool_events})