"""AST-based security sandbox for validating agent-generated code."""

from __future__ import annotations
import ast
import logging
from typing import List, Set

logger = logging.getLogger(__name__)

BLOCKED_MODULES: Set[str] = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "ftplib", "smtplib",
    "ctypes", "multiprocessing", "signal", "resource",
    "importlib", "builtins", "__builtin__",
}

BLOCKED_FUNCTIONS: Set[str] = {
    "exec", "eval", "compile", "open", "input",
    "__import__", "globals", "locals", "getattr", "setattr", "delattr",
    "exit", "quit",
}


class SecurityViolation(Exception):
    """Raised when agent code violates security constraints."""
    pass


class CodeSandbox:
    """Validates agent-generated code before execution."""

    def __init__(
        self,
        blocked_modules: Set[str] = BLOCKED_MODULES,
        blocked_functions: Set[str] = BLOCKED_FUNCTIONS,
        max_code_length: int = 50_000,
    ):
        self._blocked_modules = blocked_modules
        self._blocked_functions = blocked_functions
        self._max_code_length = max_code_length

    def validate(self, code: str) -> List[str]:
        """Validate code and return list of violations (empty if safe)."""
        violations = []

        if len(code) > self._max_code_length:
            violations.append(f"Code exceeds maximum length ({len(code)} > {self._max_code_length})")
            return violations

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            violations.append(f"Syntax error: {e}")
            return violations

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]
                    if root_module in self._blocked_modules:
                        violations.append(f"Blocked import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_module = node.module.split(".")[0]
                    if root_module in self._blocked_modules:
                        violations.append(f"Blocked import from: {node.module}")

            # Check function calls
            elif isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name in self._blocked_functions:
                    violations.append(f"Blocked function call: {func_name}")

        return violations

    def check_and_raise(self, code: str) -> None:
        """Validate code and raise SecurityViolation if unsafe."""
        violations = self.validate(code)
        if violations:
            raise SecurityViolation(
                f"Code security violations:\n" + "\n".join(f"  - {v}" for v in violations)
            )

    @staticmethod
    def _get_call_name(node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""
