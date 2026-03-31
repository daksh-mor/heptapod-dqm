from orchestral.tools.hooks import ToolHook, ToolHookResult
import json

class ToolEventLogger(ToolHook):
    def __init__(self):
        self.events = []
    
    def before_call(self, tool_name: str, arguments: dict) -> ToolHookResult:
        args = ", ".join(f"{k}={v}" for k, v in arguments.items())
        self.events.append(f"📞 {tool_name}({args})")
        print(f"  🔧 {tool_name}({args})")
        return ToolHookResult(approved=True)
    
    def after_call(self, tool_name: str, result) -> ToolHookResult:
        result_str = str(result)[:100]
        self.events.append(f"→ {result_str}")
        return ToolHookResult(approved=True)

class ToolCallLogger(ToolHook):
    """
    Hook that prints tool calls before and after execution.

    This is useful for:
    - Debugging agent behavior
    - Understanding which tools are being called
    - Seeing the arguments passed to tools
    - Verifying tool composition/chaining

    Example output:
        >>> TOOL CALL: lorentz_gamma
            Arguments: {"velocity": 0.995}
        <<< lorentz_gamma completed

        >>> TOOL CALL: time_dilation
            Arguments: {"proper_time": 2.2, "gamma": 10.01, "velocity": 0.995}
        <<< time_dilation completed
    """

    def __init__(self, verbose: bool = True, show_results: bool = True):
        """
        Initialize the tool call logger.

        Args:
            verbose: If True, show full arguments. If False, just tool names.
            show_results: If True, also print tool results (can be verbose).
        """
        self.verbose = verbose
        self.show_results = show_results
        self.events = []

    def reset_events(self):
        """Clear stored tool call events for a fresh agent run."""
        self.events = []

    def get_events(self):
        """Return a copy of structured events captured during tool execution."""
        return list(self.events)

    def before_call(self, tool_name: str, arguments: dict) -> ToolHookResult:
        """Called before each tool execution."""
        try:
            args_compact = json.dumps(arguments, ensure_ascii=False, default=str)
        except Exception:
            args_compact = str(arguments)

        self.events.append({
            "phase": "before",
            "tool_name": tool_name,
            "arguments": args_compact,
        })

        if self.verbose:
            # Pretty print arguments (truncate long values)
            args_str = json.dumps(arguments, indent=2, default=str)
            if len(args_str) > 500:
                args_str = args_str[:500] + "\n  ... (truncated)"
            print(f"\n>>> TOOL CALL: {tool_name}")
            print(f"    Arguments: {args_str}")
        else:
            print(f">>> TOOL: {tool_name}")

        return ToolHookResult(approved=True)

    def after_call(self, tool_name: str, result) -> ToolHookResult:
        """Called after each tool execution."""
        result_full = str(result)
        result_compact = result_full if len(result_full) <= 3000 else result_full[:3000] + "... (truncated)"
        self.events.append({
            "phase": "after",
            "tool_name": tool_name,
            "result": result_compact,
        })

        if self.show_results:
            result_str = result_full
            if len(result_str) > 300:
                result_str = result_str[:300] + "... (truncated)"
            print(f"<<< RESULT from {tool_name}: {result_str}")
        else:
            print(f"<<< {tool_name} completed")

        return ToolHookResult(approved=True)