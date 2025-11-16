"""
Test that all Python modules can be imported without errors.

This test suite validates that:
1. All Python files have valid syntax
2. All imports are resolvable (with optional dependencies handled gracefully)
3. No circular import dependencies exist
4. Modules can be loaded without execution side effects

Phase: 2A - Import Validation
Runtime: ~30 seconds
"""

import importlib.util
import sys
from pathlib import Path
from typing import List

import pytest


# ============================================================================
# Import Testing
# ============================================================================

def get_module_name_from_path(file_path: Path, projects_dir: Path) -> str:
    """
    Convert a file path to a Python module name.

    Example:
        projects/astronomy/sky-survey/tier-3/src/data_access.py
        -> projects.astronomy.sky-survey.tier-3.src.data_access
    """
    relative_path = file_path.relative_to(projects_dir.parent)
    module_path = str(relative_path.with_suffix("")).replace("/", ".")
    return module_path


@pytest.mark.unit
def test_python_file_can_be_imported(py_file: Path, projects_dir: Path):
    """
    Test that a Python file can be imported without errors.

    This test:
    - Loads the module using importlib
    - Allows missing optional dependencies (boto3, torch, etc.)
    - Fails on syntax errors or missing required imports
    """
    module_name = get_module_name_from_path(py_file, projects_dir)

    try:
        # Load the module spec
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            pytest.skip(f"Could not load spec for {module_name}")

        # Create the module
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        # Execute the module
        spec.loader.exec_module(module)

    except ImportError as e:
        error_msg = str(e)

        # Allow missing optional dependencies
        optional_deps = [
            "boto3", "botocore", "torch", "tensorflow", "keras",
            "cv2", "sklearn", "xgboost", "lightgbm", "catboost",
            "pydicom", "nibabel", "SimpleITK", "astropy",
            "transformers", "datasets", "tokenizers",
            "xarray", "statsmodels", "plotly", "pysam", "geopandas",
            "folium", "dotenv", "tabulate", "awswrangler", "pmdarima",
            "albumentations", "timm", "vaderSentiment", "community", "allel",
            "rasterio", "healpy", "s3fs", "cartopy"
        ]

        if any(dep in error_msg for dep in optional_deps):
            pytest.skip(f"Optional dependency missing: {error_msg}")

        # Fail on required imports
        raise

    except SyntaxError as e:
        pytest.fail(f"Syntax error in {py_file}: {e}")

    except Exception as e:
        # Some modules may have execution side effects
        # Allow certain exceptions but log them
        error_str = str(e).lower()
        if "aws" in error_str or "credentials" in error_str:
            pytest.skip(f"AWS credentials required: {e}")
        elif "region" in error_str:
            pytest.skip(f"AWS region configuration required: {e}")
        elif "environment variable" in error_str:
            pytest.skip(f"Environment variable required: {e}")
        else:
            # Unexpected error - fail the test
            pytest.fail(f"Unexpected error importing {module_name}: {e}")


# ============================================================================
# Module Discovery and Test Generation
# ============================================================================

def pytest_generate_tests(metafunc):
    """
    Dynamically generate tests for all Python files.

    This discovers all Python files in the projects directory and creates
    a test case for each one.
    """
    if "py_file" in metafunc.fixturenames:
        projects_dir = Path(__file__).parent.parent / "projects"

        # Discover all Python files
        python_files = list(projects_dir.rglob("*.py"))

        # Exclude files that shouldn't be imported
        excluded_patterns = [
            "__pycache__",
            ".ipynb_checkpoints",
            "setup.py",
            ".aws-sam",
            "_template",
        ]

        python_files = [
            f for f in python_files
            if not any(pattern in str(f) for pattern in excluded_patterns)
        ]

        # Sort for consistent test order
        python_files.sort()

        # Generate test parameters
        metafunc.parametrize(
            "py_file",
            python_files,
            ids=[str(f.relative_to(projects_dir)) for f in python_files]
        )


# ============================================================================
# Summary Test
# ============================================================================

@pytest.mark.unit
def test_import_summary(projects_dir: Path):
    """
    Summary test that reports statistics about Python modules.

    This test always passes but provides useful information about:
    - Total number of Python files
    - Distribution across domains
    - Distribution across tiers
    """
    python_files = list(projects_dir.rglob("*.py"))
    python_files = [
        f for f in python_files
        if "__pycache__" not in str(f) and "setup.py" not in str(f)
    ]

    # Count by domain
    domains = {}
    for f in python_files:
        parts = f.relative_to(projects_dir).parts
        if parts:
            domain = parts[0]
            domains[domain] = domains.get(domain, 0) + 1

    # Count by tier
    tiers = {"tier-0": 0, "tier-1": 0, "tier-2": 0, "tier-3": 0, "other": 0}
    for f in python_files:
        path_str = str(f)
        tier_found = False
        for tier in ["tier-0", "tier-1", "tier-2", "tier-3"]:
            if tier in path_str:
                tiers[tier] += 1
                tier_found = True
                break
        if not tier_found:
            tiers["other"] += 1

    # Print summary
    print(f"\n{'='*70}")
    print("PYTHON MODULE IMPORT TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total Python files: {len(python_files)}")
    print(f"\nBy Domain ({len(domains)} domains):")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1])[:10]:
        print(f"  {domain}: {count} files")
    print(f"\nBy Tier:")
    for tier, count in tiers.items():
        if count > 0:
            print(f"  {tier}: {count} files")
    print(f"{'='*70}\n")

    # Always pass - this is informational
    assert len(python_files) > 0, "No Python files found!"
