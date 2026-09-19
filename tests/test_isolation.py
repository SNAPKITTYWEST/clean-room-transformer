"""Isolation boundary, checked statically against the source tree.

The claim "no network telemetry, no dynamic dependency resolution" is only
worth anything if something checks it. These tests parse every module under
`src/` and fail on any import that could open a socket, spawn a process, or
resolve code at runtime -- and on any third-party dependency outside the
allowlist.

What this does not do: it cannot prove the *process* is isolated. A static scan
says nothing about what a linked C extension does, and NumPy is a large C
dependency. Process-level isolation is a deployment control (network namespace,
seccomp, egress firewall); see docker/Dockerfile and README "Honest scoping".
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

FORBIDDEN_MODULES = {
    # network
    "socket", "ssl", "http", "http.client", "urllib", "urllib3", "requests",
    "httpx", "aiohttp", "ftplib", "smtplib", "poplib", "imaplib", "telnetlib",
    "xmlrpc", "webbrowser", "socketserver", "asyncio",
    # process / dynamic execution
    "subprocess", "multiprocessing", "importlib", "pkg_resources", "pip",
    "setuptools", "runpy",
    # unsafe deserialization
    "pickle", "shelve", "marshal", "dill", "joblib",
    # native escape hatches
    "ctypes", "cffi", "mmap",
}

ALLOWED_THIRD_PARTY = {"numpy", "cryptography"}

STDLIB_ALLOWLIST = {
    "__future__", "abc", "ast", "collections", "dataclasses", "enum",
    "hashlib", "json", "os", "pathlib", "queue", "sys", "threading", "time",
    "typing",
}


def source_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, local to the package
                continue
            if node.module:
                names.add(node.module)
    return names


def test_source_tree_is_not_empty() -> None:
    assert len(source_files()) >= 10


@pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
def test_no_module_imports_a_forbidden_capability(path: Path) -> None:
    offending = {
        name
        for name in imported_names(path)
        if name in FORBIDDEN_MODULES or name.split(".")[0] in FORBIDDEN_MODULES
    }
    assert not offending, f"{path.relative_to(SRC)} imports {sorted(offending)}"


@pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
def test_dependencies_stay_inside_the_allowlist(path: Path) -> None:
    local_packages = {p.name for p in SRC.iterdir() if p.is_dir()}
    for name in imported_names(path):
        root = name.split(".")[0]
        if root in local_packages or root in STDLIB_ALLOWLIST:
            continue
        assert root in ALLOWED_THIRD_PARTY, (
            f"{path.relative_to(SRC)} imports third-party module {root!r}, "
            f"which is outside the allowlist {sorted(ALLOWED_THIRD_PARTY)}"
        )


def test_no_source_file_contains_a_url_literal() -> None:
    """A hard-coded endpoint is the usual shape of accidental telemetry."""
    offenders = []
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        for marker in ("http://", "https://", "ws://", "wss://"):
            if marker in text:
                offenders.append((path.relative_to(SRC), marker))
    assert not offenders, f"URL literals found: {offenders}"


def test_no_eval_or_exec_in_source() -> None:
    offenders = []
    for path in source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec", "compile", "__import__"}:
                    offenders.append((path.relative_to(SRC), node.func.id, node.lineno))
    assert not offenders, f"dynamic execution found: {offenders}"


def test_runtime_imports_cleanly_with_no_network_available() -> None:
    """Import the whole package graph; nothing may reach out at import time.

    If any module opened a connection on import, this would hang or fail in an
    egress-blocked sandbox -- which is how this test is intended to be run.
    """
    import membrane  # noqa: F401
    import runtime  # noqa: F401
    import transformer  # noqa: F401

    assert hasattr(membrane, "IntegrityMembrane")
    assert hasattr(runtime, "CleanRoomEngine")
    assert hasattr(transformer, "CleanRoomTransformer")
