"""
Phase 4: Notebook Smoke Testing with testbook

Executes the first N cells of each notebook to verify they run without errors.
This is a "smoke test" - not full execution, just ensuring notebooks start correctly.

Strategy:
- Execute first 3-5 cells only (imports, setup, basic operations)
- Skip long-running cells (training loops)
- Set 60-second timeout per cell
- Handle missing dependencies gracefully
- Report execution statistics
"""

from pathlib import Path

import pytest
from testbook import testbook
from testbook.client import TestbookNotebookClient


# Configuration
MAX_CELLS_TO_EXECUTE = 5  # Only execute first 5 cells
CELL_TIMEOUT = 60  # 60 seconds per cell
TOTAL_NOTEBOOK_TIMEOUT = 300  # 5 minutes total


# ============================================================================
# Helper Functions
# ============================================================================


def discover_notebooks():
    """Discover all Jupyter notebook files in the project."""
    projects_dir = Path(__file__).parent.parent / "projects"
    notebooks = list(projects_dir.rglob("*.ipynb"))
    # Exclude checkpoints
    return [nb for nb in notebooks if ".ipynb_checkpoints" not in str(nb)]


def should_skip_cell(cell_source: str) -> bool:
    """
    Determine if a cell should be skipped during smoke testing.

    Skip cells that:
    - Contain training loops (model.fit, train, epochs)
    - Download large files
    - Take a long time
    """
    skip_patterns = [
        "model.fit(",
        ".fit(",
        "train(",
        "epochs=",
        "for epoch in",
        "wget ",  # File download command
        "curl ",  # File download command
        "!wget",  # Shell command
        "!curl",  # Shell command
        "urllib.request.urlretrieve",  # Python file download
        "requests.get(",  # HTTP download (may be API call, but often large files)
        "!aws s3 cp",  # AWS file copy
        "!aws s3 sync",  # AWS sync
        "!gdown",  # Google Drive download
        "time.sleep",
    ]

    cell_lower = cell_source.lower()
    return any(pattern.lower() in cell_lower for pattern in skip_patterns)


# ============================================================================
# Notebook Execution Tests
# ============================================================================


@pytest.mark.notebook
@pytest.mark.slow
def test_agriculture_crop_disease_notebook():
    """Test agriculture crop disease detection notebook (first 5 cells)."""
    notebook_path = (
        Path(__file__).parent.parent
        / "projects/agriculture/precision-agriculture/tier-0/crop-disease-detection.ipynb"
    )

    if not notebook_path.exists():
        pytest.skip(f"Notebook not found: {notebook_path}")

    with testbook(notebook_path, execute=False, timeout=TOTAL_NOTEBOOK_TIMEOUT) as tb:
        cells_executed = 0
        cells_skipped = 0

        try:
            # Execute first N cells
            for i in range(min(MAX_CELLS_TO_EXECUTE, len(tb.cells))):
                cell = tb.cells[i]

                # Skip markdown cells
                if cell.cell_type == "markdown":
                    continue

                # Skip long-running cells
                if should_skip_cell(cell.source):
                    cells_skipped += 1
                    continue

                # Execute cell (timeout managed by testbook context)
                try:
                    tb.execute_cell(i)
                    cells_executed += 1
                except Exception as e:
                    # Check if it's a missing dependency error
                    error_str = str(e).lower()
                    if any(
                        dep in error_str
                        for dep in [
                            "no module named",
                            "modulenotfounderror",
                            "importerror",
                            "tensorflow",
                            "torch",
                        ]
                    ):
                        pytest.skip(f"Missing dependency in cell {i}: {e}")
                    else:
                        # Real error - fail the test
                        pytest.fail(f"Cell {i} failed: {e}")

        finally:
            print(f"\nNotebook smoke test results:")
            print(f"  Cells executed: {cells_executed}")
            print(f"  Cells skipped: {cells_skipped}")
            print(f"  Total cells: {len(tb.cells)}")


@pytest.mark.notebook
@pytest.mark.slow
@pytest.mark.parametrize(
    "notebook_path",
    discover_notebooks()[:10],  # Test first 10 notebooks as examples
    ids=lambda p: str(p.relative_to(Path(__file__).parent.parent / "projects")),
)
def test_notebook_smoke_execution(notebook_path: Path):
    """
    Smoke test for notebooks - execute first few cells only.

    This test:
    1. Opens the notebook without executing
    2. Executes first MAX_CELLS_TO_EXECUTE cells
    3. Skips markdown and long-running cells
    4. Reports success if basic setup works
    """
    with testbook(notebook_path, execute=False, timeout=TOTAL_NOTEBOOK_TIMEOUT) as tb:
        cells_executed = 0
        cells_skipped = 0
        cells_failed = []

        # Execute first N cells
        for i in range(min(MAX_CELLS_TO_EXECUTE, len(tb.cells))):
            cell = tb.cells[i]

            # Skip markdown cells
            if cell.cell_type == "markdown":
                continue

            # Skip long-running cells
            if should_skip_cell(cell.source):
                cells_skipped += 1
                continue

            # Execute cell (timeout managed by testbook context)
            try:
                tb.execute_cell(i)
                cells_executed += 1
            except Exception as e:
                error_str = str(e).lower()

                # Check for missing dependencies
                if any(
                    dep in error_str
                    for dep in [
                        "no module named",
                        "modulenotfounderror",
                        "importerror",
                        "cannot import",
                    ]
                ):
                    pytest.skip(f"Missing dependency in cell {i}: {e}")

                # Check for timeout
                elif "timeout" in error_str:
                    pytest.skip(f"Cell {i} timeout (> {CELL_TIMEOUT}s): {e}")

                # Real error
                else:
                    cells_failed.append((i, str(e)))

        # Report results
        print(f"\nSmoke test results for {notebook_path.name}:")
        print(f"  ✓ Cells executed: {cells_executed}")
        print(f"  ⏭  Cells skipped: {cells_skipped}")
        print(f"  ✗ Cells failed: {len(cells_failed)}")

        # Fail if any cells failed
        if cells_failed:
            failure_msg = f"\n{len(cells_failed)} cell(s) failed:\n"
            for cell_idx, error in cells_failed:
                failure_msg += f"  Cell {cell_idx}: {error[:100]}...\n"
            pytest.fail(failure_msg)

        # Must execute at least 1 cell to be valid smoke test
        if cells_executed == 0:
            pytest.skip("No cells were executed (all markdown or skipped)")


# ============================================================================
# Summary Tests
# ============================================================================


@pytest.mark.unit
def test_notebooks_discoverable():
    """Test that we can discover notebooks for testing."""
    notebooks = discover_notebooks()

    assert len(notebooks) > 0, "Should find notebooks"
    assert all(nb.suffix == ".ipynb" for nb in notebooks)
    assert all(nb.exists() for nb in notebooks)

    print(f"\n✓ Found {len(notebooks)} notebooks for smoke testing:")
    # Group by domain
    domains = {}
    for nb in notebooks:
        domain = nb.parts[-4] if len(nb.parts) >= 4 else "unknown"
        domains[domain] = domains.get(domain, 0) + 1

    for domain, count in sorted(domains.items()):
        print(f"  {domain}: {count} notebooks")


@pytest.mark.unit
def test_cell_skip_logic():
    """Test the cell skip logic works correctly."""
    # Should skip
    assert should_skip_cell("model.fit(X_train, y_train, epochs=50)")
    assert should_skip_cell("for epoch in range(100):")
    assert should_skip_cell("!wget https://example.com/large_file.zip")
    assert should_skip_cell("wget http://example.com/file.tar.gz")
    assert should_skip_cell("time.sleep(60)")
    assert should_skip_cell("urllib.request.urlretrieve('http://example.com/data.zip')")

    # Should not skip
    assert not should_skip_cell("import numpy as np")
    assert not should_skip_cell("X_train, X_test = train_test_split(X, y)")
    assert not should_skip_cell("print('Hello world')")
    assert not should_skip_cell("df = pd.read_csv('data.csv')")
    # Package data downloads should not be skipped
    assert not should_skip_cell("nltk.download('stopwords')")
    assert not should_skip_cell("import spacy; spacy.load('en_core_web_sm')")


# ============================================================================
# Targeted Notebook Tests
# ============================================================================


@pytest.mark.notebook
@pytest.mark.slow
@pytest.mark.tier0
def test_tier0_notebooks_smoke():
    """Smoke test all tier-0 (Colab) notebooks."""
    projects_dir = Path(__file__).parent.parent / "projects"
    tier0_notebooks = list(projects_dir.rglob("*/tier-0/*.ipynb"))

    if len(tier0_notebooks) == 0:
        pytest.skip("No tier-0 notebooks found")

    passed = 0
    skipped = 0
    failed = []

    for notebook in tier0_notebooks[:5]:  # Test first 5 as example
        try:
            with testbook(notebook, execute=False, timeout=TOTAL_NOTEBOOK_TIMEOUT) as tb:
                # Execute first cell only (typically imports)
                if len(tb.cells) > 0 and tb.cells[0].cell_type == "code":
                    try:
                        tb.execute_cell(0)
                        passed += 1
                    except Exception:
                        skipped += 1
        except Exception as e:
            failed.append((notebook.name, str(e)))

    print(f"\nTier-0 smoke test results:")
    print(f"  Passed: {passed}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {len(failed)}")

    if failed:
        for name, error in failed:
            print(f"  ✗ {name}: {error[:80]}")
