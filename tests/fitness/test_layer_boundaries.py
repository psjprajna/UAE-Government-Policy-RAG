"""Fitness function: enforce the hexagonal boundary.

Domain code (api, ingestion, retrieval, generation, evals) must depend only on
ports/ and standard / third-party libraries. It must NEVER import from adapters/
directly. Adapter selection happens in config.py only.

This is the CI-enforceable expression of ADR-0002 (Hexagonal architecture).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "uae_rag"

DOMAIN_DIRS = ("api", "ingestion", "retrieval", "generation", "evals")
FORBIDDEN_PREFIX = "uae_rag.adapters"
ALLOWED_ADAPTER_IMPORTER = "config"  # only this module is allowed to touch adapters


def _python_files_under(directory: Path) -> list[Path]:
    return [p for p in directory.rglob("*.py") if p.name != "__init__.py"]


def _module_imports(path: Path) -> list[str]:
    """Return the dotted module names imported by the given Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize("domain", DOMAIN_DIRS)
def test_domain_layer_does_not_import_adapters(domain: str) -> None:
    domain_dir = SRC_ROOT / domain
    if not domain_dir.exists():
        pytest.skip(f"{domain}/ not present yet (added in a later phase)")

    violations: list[str] = []
    for file_path in _python_files_under(domain_dir):
        for imported in _module_imports(file_path):
            if imported.startswith(FORBIDDEN_PREFIX):
                violations.append(f"{file_path.relative_to(SRC_ROOT.parent)} imports {imported}")

    assert not violations, (
        "Domain code must not import from uae_rag.adapters.* directly (ADR-0002). "
        "Route through ports/ and inject via config.py. Violations:\n  - "
        + "\n  - ".join(violations)
    )


def test_only_config_module_imports_adapters() -> None:
    """config.py is the single seam where adapters are instantiated."""
    if not SRC_ROOT.exists():
        pytest.skip("src/uae_rag/ not present yet")

    violations: list[str] = []
    for file_path in SRC_ROOT.rglob("*.py"):
        relative = file_path.relative_to(SRC_ROOT)
        # config.py is allowed; adapter packages obviously may import from siblings.
        if relative.name == "config.py":
            continue
        if relative.parts and relative.parts[0] == "adapters":
            continue
        for imported in _module_imports(file_path):
            if imported.startswith(FORBIDDEN_PREFIX):
                violations.append(f"{file_path.relative_to(SRC_ROOT.parent)} imports {imported}")

    assert not violations, (
        "Only config.py may import from uae_rag.adapters.* (ADR-0002). Violations:\n  - "
        + "\n  - ".join(violations)
    )
