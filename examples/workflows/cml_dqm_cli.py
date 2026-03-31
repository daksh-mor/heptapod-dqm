"""
CLI interface for the CML DQM agent.

Run with:
    python examples/workflows/cml_dqm_cli.py --provider GPT
"""

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

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
from tools.dqm import DQMEDATool, DQMTrainTool, DQMEvaluateTool, DQMDeploymentTool


def _get_llm(provider: str):
    provider = provider.upper()
    if provider == "GPT":
        return GPT()
    if provider == "CLAUDE":
        return Claude()
    if provider == "GEMINI":
        return Gemini()
    return Groq()


def create_agent(provider: str, base_directory: str) -> Agent:
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
        TodoRead(),
        TodoWrite(base_directory=base_directory),
    ]

    hooks = [ToolCallLogger(verbose=True, show_results=True), TruncateOutputHook(max_length=12000)]

    system_prompt = (
        CML_DQM_EXPLORER_PROMPT
        + f"\n\nAll output files must be saved under sandbox-relative paths. Selected base directory: {base_directory}"
    )

    return Agent(
        llm=_get_llm(provider),
        tools=tools,
        tool_hooks=hooks,
        system_prompt=system_prompt,
        debug=False,
    )


def run_cli(agent: Agent) -> None:
    console = Console()
    console.print("[bold cyan]Agent ready[/]\n")

    while True:
        try:
            user_input = console.input("\n[bold cyan]You[/]\n[cyan]> [/").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                console.print("[bold yellow]Goodbye[/]")
                break

            response = agent.run(user_input)
            text = getattr(response, "text", str(response))
            console.print(Panel(Markdown(text), title="[bold cyan]Agent[/]", border_style="bright_cyan", expand=False))
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Interrupted. Goodbye[/]")
            break
        except Exception as e:
            console.print(Panel(f"[bold red]Error:[/] {e}", title="[bold red]Error[/]", border_style="red"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CML DQM CLI agent")
    parser.add_argument("--provider", default="GPT", choices=["GPT", "Claude", "Gemini", "Groq"])
    parser.add_argument(
        "--base-directory",
        default=str(REPO_ROOT),
        help="Agent base directory for file tools (defaults to repository root)",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    agent = create_agent(args.provider, args.base_directory)
    run_cli(agent)


if __name__ == "__main__":
    main()
