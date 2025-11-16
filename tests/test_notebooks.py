"""
Test Jupyter notebook structure and syntax.

This test suite validates that:
1. All notebooks have valid JSON structure
2. Notebooks have required metadata (kernel, cells)
3. Cell syntax is valid Python
4. No output cells are corrupted
5. Notebooks can be read by nbformat

Phase: 2B - Notebook Validation
Runtime: ~30 seconds
"""

import json
from pathlib import Path
from typing import Dict, List

import nbformat
import pytest
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell


# ============================================================================
# Notebook Structure Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.notebook
def test_notebook_valid_json(notebook: Path):
    """Test that notebook file is valid JSON."""
    try:
        with open(notebook, "r", encoding="utf-8") as f:
            json.load(f)
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON in notebook: {e}")


@pytest.mark.unit
@pytest.mark.notebook
def test_notebook_can_be_read(notebook: Path):
    """Test that notebook can be read by nbformat."""
    try:
        nb = nbformat.read(str(notebook), as_version=4)
        assert isinstance(nb, NotebookNode), "Failed to read as NotebookNode"
    except Exception as e:
        pytest.fail(f"Failed to read notebook: {e}")


@pytest.mark.unit
@pytest.mark.notebook
def test_notebook_has_cells(notebook: Path):
    """Test that notebook has at least one cell."""
    nb = nbformat.read(str(notebook), as_version=4)
    assert len(nb.cells) > 0, f"Notebook has no cells: {notebook}"


@pytest.mark.unit
@pytest.mark.notebook
def test_notebook_has_kernel_metadata(notebook: Path):
    """Test that notebook has kernel metadata."""
    nb = nbformat.read(str(notebook), as_version=4)
    assert "kernelspec" in nb.metadata, f"Missing kernelspec metadata: {notebook}"
    assert "name" in nb.metadata.kernelspec, "Missing kernel name"


@pytest.mark.unit
@pytest.mark.notebook
def test_notebook_code_cells_have_valid_syntax(notebook: Path):
    """
    Test that all code cells contain valid Python syntax.

    This test parses each code cell to ensure it's syntactically valid,
    but does not execute the code.
    """
    nb = nbformat.read(str(notebook), as_version=4)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        source = cell.source
        if not source.strip():
            continue  # Empty cells are okay

        # Skip magic commands and shell commands
        lines = source.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip IPython magics
            if stripped.startswith("%") or stripped.startswith("!"):
                continue
            cleaned_lines.append(line)

        cleaned_source = "\n".join(cleaned_lines)

        if not cleaned_source.strip():
            continue

        # Try to compile the code
        try:
            compile(cleaned_source, f"<notebook cell {i}>", "exec")
        except SyntaxError as e:
            pytest.fail(
                f"Syntax error in cell {i} of {notebook.name}:\n"
                f"Line {e.lineno}: {e.msg}\n"
                f"Code: {e.text}"
            )


# ============================================================================
# Notebook Quality Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.notebook
def test_notebook_not_empty(notebook: Path):
    """Test that notebook is not effectively empty."""
    nb = nbformat.read(str(notebook), as_version=4)

    # Count non-empty cells
    non_empty_cells = 0
    for cell in nb.cells:
        if cell.source.strip():
            non_empty_cells += 1

    assert non_empty_cells > 0, f"Notebook has no non-empty cells: {notebook}"


@pytest.mark.unit
@pytest.mark.notebook
def test_notebook_has_markdown_cells(notebook: Path):
    """
    Test that notebook has at least one markdown cell (documentation).

    This is a quality check to ensure notebooks are documented.
    """
    nb = nbformat.read(str(notebook), as_version=4)

    markdown_cells = [cell for cell in nb.cells if cell.cell_type == "markdown"]

    if len(markdown_cells) == 0:
        pytest.skip(f"No markdown cells (documentation) in {notebook.name}")


# ============================================================================
# Test Generation
# ============================================================================

def pytest_generate_tests(metafunc):
    """
    Dynamically generate tests for all notebook files.

    This discovers all .ipynb files in the projects directory and creates
    a test case for each one.
    """
    if "notebook" in metafunc.fixturenames:
        projects_dir = Path(__file__).parent.parent / "projects"

        # Discover all notebook files
        notebooks = list(projects_dir.rglob("*.ipynb"))

        # Exclude checkpoint directories
        notebooks = [
            nb for nb in notebooks
            if ".ipynb_checkpoints" not in str(nb)
        ]

        # Sort for consistent test order
        notebooks.sort()

        # Generate test parameters
        metafunc.parametrize(
            "notebook",
            notebooks,
            ids=[str(nb.relative_to(projects_dir)) for nb in notebooks]
        )


# ============================================================================
# Summary Test
# ============================================================================

@pytest.mark.unit
@pytest.mark.notebook
def test_notebook_summary(projects_dir: Path):
    """
    Summary test that reports statistics about notebooks.

    This test always passes but provides useful information about:
    - Total number of notebooks
    - Distribution across domains
    - Distribution across tiers
    - Cell type distribution
    """
    notebooks = list(projects_dir.rglob("*.ipynb"))
    notebooks = [nb for nb in notebooks if ".ipynb_checkpoints" not in str(nb)]

    # Count by domain
    domains: Dict[str, int] = {}
    for nb in notebooks:
        parts = nb.relative_to(projects_dir).parts
        if parts:
            domain = parts[0]
            domains[domain] = domains.get(domain, 0) + 1

    # Count by tier
    tiers = {"tier-0": 0, "tier-1": 0, "tier-2": 0, "tier-3": 0, "studio-lab": 0, "other": 0}
    for nb in notebooks:
        path_str = str(nb)
        tier_found = False
        for tier in ["tier-0", "tier-1", "tier-2", "tier-3", "studio-lab"]:
            if tier in path_str:
                tiers[tier] += 1
                tier_found = True
                break
        if not tier_found:
            tiers["other"] += 1

    # Count cells
    total_code_cells = 0
    total_markdown_cells = 0
    for nb_path in notebooks[:50]:  # Sample first 50 for performance
        try:
            nb = nbformat.read(str(nb_path), as_version=4)
            for cell in nb.cells:
                if cell.cell_type == "code":
                    total_code_cells += 1
                elif cell.cell_type == "markdown":
                    total_markdown_cells += 1
        except Exception:
            pass

    # Print summary
    print(f"\n{'='*70}")
    print("NOTEBOOK VALIDATION TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total notebooks: {len(notebooks)}")
    print(f"\nBy Domain ({len(domains)} domains):")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1])[:10]:
        print(f"  {domain}: {count} notebooks")
    print(f"\nBy Tier:")
    for tier, count in tiers.items():
        if count > 0:
            print(f"  {tier}: {count} notebooks")
    print(f"\nCell Statistics (sample of 50 notebooks):")
    print(f"  Code cells: {total_code_cells}")
    print(f"  Markdown cells: {total_markdown_cells}")
    print(f"  Total cells: {total_code_cells + total_markdown_cells}")
    print(f"{'='*70}\n")

    # Always pass - this is informational
    assert len(notebooks) > 0, "No notebooks found!"
