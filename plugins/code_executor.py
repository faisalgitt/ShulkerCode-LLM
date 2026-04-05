"""
=====================================
 Shulker Code — Code Executor Plugin
 Safely run generated code snippets
 Developed by @kopeedev / CyeroX
=====================================

Plugin that executes Python code in an isolated subprocess
with a timeout, so you can test generated code on the fly.
"""

import subprocess
import sys
import os
import tempfile
import textwrap
from typing import Tuple, Optional


class CodeExecutorPlugin:
    """
    Execute Python code snippets safely in a subprocess.
    Captures stdout, stderr, and return code.
    Enforces a time limit to prevent infinite loops.
    """

    def __init__(self, timeout_seconds: int = 10, max_output_chars: int = 4096):
        self.timeout = timeout_seconds
        self.max_output = max_output_chars

    def run_python(self, code: str) -> Tuple[bool, str, str]:
        """
        Execute Python code and return (success, stdout, stderr).

        Args:
            code: Python source code to execute

        Returns:
            (success, stdout, stderr)
        """
        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(textwrap.dedent(code))
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout = result.stdout[:self.max_output]
            stderr = result.stderr[:self.max_output]
            success = result.returncode == 0
            return success, stdout, stderr

        except subprocess.TimeoutExpired:
            return False, "", f"⏱️  Execution timed out after {self.timeout}s"
        except Exception as e:
            return False, "", f"❌ Execution error: {str(e)}"
        finally:
            os.unlink(temp_path)

    def format_result(self, success: bool, stdout: str, stderr: str) -> str:
        """Format execution result for display."""
        lines = []
        if success:
            lines.append("✅ Code executed successfully")
            if stdout.strip():
                lines.append("\n📤 Output:")
                lines.append(stdout)
        else:
            lines.append("❌ Execution failed")
            if stderr.strip():
                lines.append("\n🔴 Error:")
                lines.append(stderr)
            if stdout.strip():
                lines.append("\n📤 Partial output:")
                lines.append(stdout)
        return "\n".join(lines)


class SyntaxChecker:
    """
    Check Python code for syntax errors without executing it.
    Uses Python's built-in `compile()` function.
    """

    def check(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Check code syntax.

        Returns:
            (is_valid, error_message_or_None)
        """
        try:
            compile(code, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            msg = f"SyntaxError at line {e.lineno}: {e.msg}"
            if e.text:
                msg += f"\n  {e.text.rstrip()}"
                if e.offset:
                    msg += f"\n  {' ' * (e.offset - 1)}^"
            return False, msg
        except Exception as e:
            return False, str(e)


# Plugin registry
PLUGINS = {
    "code_executor": CodeExecutorPlugin,
    "syntax_checker": SyntaxChecker,
}


def load_plugin(name: str, **kwargs):
    """Load a plugin by name."""
    if name not in PLUGINS:
        raise ValueError(f"Unknown plugin: {name}. Available: {list(PLUGINS.keys())}")
    return PLUGINS[name](**kwargs)
