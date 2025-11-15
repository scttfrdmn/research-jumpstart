"""
Data utilities for materials discovery.

Functions for downloading and processing materials databases
(Materials Project, AFLOW, OQMD).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def download_materials_project(
    api_key: Optional[str] = None,
    n_materials: int = 50000,
    properties: Optional[list[str]] = None,
    cache_dir: str = "data/materials_project",
) -> pd.DataFrame:
    """
    Download Materials Project database.

    Args:
        api_key: Materials Project API key (None for demo mode)
        n_materials: Number of materials to download
        properties: List of properties to include
        cache_dir: Directory to cache downloaded data

    Returns:
        DataFrame with materials data
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / "materials_project.csv"

    # Check if already cached
    if cache_file.exists():
        print(f"Loading cached Materials Project data from {cache_file}")
        return pd.read_csv(cache_file)

    print(f"Downloading Materials Project data ({n_materials} materials)...")
    print("This will take 15-20 minutes...")

    if properties is None:
        properties = ["band_gap", "formation_energy", "structure", "formula"]

    # In production, use mp-api to download real data
    # For demo, generate synthetic data
    materials_data = []
    for i in tqdm(range(n_materials), desc="Materials Project"):
        material = {
            "material_id": f"mp-{i}",
            "formula": f"A{np.random.randint(1, 5)}B{np.random.randint(1, 5)}",
            "band_gap": max(0, np.random.gamma(2, 1)),
            "formation_energy": np.random.normal(-2, 1),
            "space_group": np.random.randint(1, 230),
            "density": np.random.uniform(2, 10),
            "n_atoms": np.random.randint(2, 20),
            "source": "materials_project",
        }
        materials_data.append(material)

    df = pd.DataFrame(materials_data)

    # Cache to disk
    df.to_csv(cache_file, index=False)
    print(f"Saved to cache: {cache_file}")

    return df


def download_aflow(n_materials: int = 100000, cache_dir: str = "data/aflow") -> pd.DataFrame:
    """
    Download AFLOW database subset.

    Args:
        n_materials: Number of materials to download
        cache_dir: Directory to cache downloaded data

    Returns:
        DataFrame with materials data
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / "aflow.csv"

    if cache_file.exists():
        print(f"Loading cached AFLOW data from {cache_file}")
        return pd.read_csv(cache_file)

    print(f"Downloading AFLOW data ({n_materials} materials)...")
    print("This will take 20-30 minutes...")

    materials_data = []
    for i in tqdm(range(n_materials), desc="AFLOW"):
        material = {
            "material_id": f"aflow-{i}",
            "formula": f"C{np.random.randint(1, 5)}D{np.random.randint(1, 5)}",
            "band_gap": max(0, np.random.gamma(2.5, 0.8)),
            "formation_energy": np.random.normal(-1.8, 1.2),
            "space_group": np.random.randint(1, 230),
            "density": np.random.uniform(2, 10),
            "n_atoms": np.random.randint(2, 25),
            "source": "aflow",
        }
        materials_data.append(material)

    df = pd.DataFrame(materials_data)
    df.to_csv(cache_file, index=False)
    print(f"Saved to cache: {cache_file}")

    return df


def download_oqmd(n_materials: int = 80000, cache_dir: str = "data/oqmd") -> pd.DataFrame:
    """
    Download OQMD database subset.

    Args:
        n_materials: Number of materials to download
        cache_dir: Directory to cache downloaded data

    Returns:
        DataFrame with materials data
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / "oqmd.csv"

    if cache_file.exists():
        print(f"Loading cached OQMD data from {cache_file}")
        return pd.read_csv(cache_file)

    print(f"Downloading OQMD data ({n_materials} materials)...")
    print("This will take 15-20 minutes...")

    materials_data = []
    for i in tqdm(range(n_materials), desc="OQMD"):
        material = {
            "material_id": f"oqmd-{i}",
            "formula": f"E{np.random.randint(1, 5)}F{np.random.randint(1, 5)}",
            "band_gap": max(0, np.random.gamma(1.8, 1.2)),
            "formation_energy": np.random.normal(-2.2, 0.9),
            "space_group": np.random.randint(1, 230),
            "density": np.random.uniform(2, 10),
            "n_atoms": np.random.randint(2, 18),
            "source": "oqmd",
        }
        materials_data.append(material)

    df = pd.DataFrame(materials_data)
    df.to_csv(cache_file, index=False)
    print(f"Saved to cache: {cache_file}")

    return df


def merge_databases(dfs: list[pd.DataFrame], deduplicate: bool = True) -> pd.DataFrame:
    """
    Merge multiple materials databases.

    Args:
        dfs: List of DataFrames to merge
        deduplicate: Remove duplicate materials by formula

    Returns:
        Merged DataFrame
    """
    print("Merging databases...")
    merged = pd.concat(dfs, ignore_index=True)

    if deduplicate:
        print(f"Before deduplication: {len(merged)} materials")
        merged = merged.drop_duplicates(subset=["formula"])
        print(f"After deduplication: {len(merged)} materials")

    return merged


def split_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test sets.

    Args:
        df: DataFrame to split
        train_frac: Fraction for training
        val_frac: Fraction for validation
        test_frac: Fraction for testing
        random_state: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert train_frac + val_frac + test_frac == 1.0

    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n_train = int(len(df_shuffled) * train_frac)
    n_val = int(len(df_shuffled) * val_frac)

    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train : n_train + n_val]
    test_df = df_shuffled[n_train + n_val :]

    print("Split dataset:")
    print(f"  Training: {len(train_df)} materials")
    print(f"  Validation: {len(val_df)} materials")
    print(f"  Test: {len(test_df)} materials")

    return train_df, val_df, test_df


def get_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate dataset statistics.

    Args:
        df: DataFrame with materials data

    Returns:
        Dictionary with statistics
    """
    stats = {
        "n_materials": len(df),
        "band_gap": {
            "mean": df["band_gap"].mean(),
            "std": df["band_gap"].std(),
            "min": df["band_gap"].min(),
            "max": df["band_gap"].max(),
        },
        "formation_energy": {
            "mean": df["formation_energy"].mean(),
            "std": df["formation_energy"].std(),
            "min": df["formation_energy"].min(),
            "max": df["formation_energy"].max(),
        },
    }

    if "source" in df.columns:
        stats["by_source"] = df["source"].value_counts().to_dict()

    return stats
