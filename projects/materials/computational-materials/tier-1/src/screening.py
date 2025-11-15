"""
High-throughput screening utilities.

Functions for screening materials using ensemble models.
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader


def ensemble_predict(
    models: list[torch.nn.Module],
    data_loader: DataLoader,
    device: str = "cuda",
    property_name: str = "band_gap",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate ensemble predictions with uncertainty quantification.

    Args:
        models: List of trained models
        data_loader: DataLoader with materials
        device: Device to run on
        property_name: Property to predict

    Returns:
        Tuple of (mean_predictions, std_predictions)
    """
    all_predictions = []

    for model in models:
        model.eval()
        model.to(device)

        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)
                pred = model(batch, predict_property=property_name)
                predictions.extend(pred.cpu().numpy())

        all_predictions.append(predictions)

    # Convert to numpy array (n_models, n_materials)
    all_predictions = np.array(all_predictions)

    # Calculate ensemble statistics
    mean_pred = np.mean(all_predictions, axis=0)
    std_pred = np.std(all_predictions, axis=0)

    return mean_pred, std_pred


def screen_materials(
    models: list[torch.nn.Module],
    materials_df: pd.DataFrame,
    graphs: list,
    property_name: str = "band_gap",
    batch_size: int = 32,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Screen materials using ensemble models.

    Args:
        models: List of trained models
        materials_df: DataFrame with material metadata
        graphs: List of graph representations
        property_name: Property to predict
        batch_size: Batch size for prediction
        device: Device to run on

    Returns:
        DataFrame with predictions and uncertainty
    """
    print(f"Screening {len(materials_df)} materials...")

    # Create data loader
    data_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    # Get ensemble predictions
    mean_pred, std_pred = ensemble_predict(models, data_loader, device, property_name)

    # Add to dataframe
    results_df = materials_df.copy()
    results_df[f"predicted_{property_name}"] = mean_pred
    results_df[f"uncertainty_{property_name}"] = std_pred

    return results_df


def filter_by_property(
    df: pd.DataFrame,
    property_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    max_uncertainty: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter materials by predicted property value and uncertainty.

    Args:
        df: DataFrame with predictions
        property_name: Property to filter on
        min_value: Minimum property value
        max_value: Maximum property value
        max_uncertainty: Maximum uncertainty

    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()

    pred_col = f"predicted_{property_name}"
    unc_col = f"uncertainty_{property_name}"

    if min_value is not None:
        filtered = filtered[filtered[pred_col] >= min_value]

    if max_value is not None:
        filtered = filtered[filtered[pred_col] <= max_value]

    if max_uncertainty is not None:
        filtered = filtered[filtered[unc_col] <= max_uncertainty]

    print(f"Filtered from {len(df)} to {len(filtered)} materials")
    return filtered


def rank_materials(
    df: pd.DataFrame,
    target_property: str,
    target_value: Optional[float] = None,
    minimize: bool = False,
    consider_uncertainty: bool = True,
    uncertainty_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Rank materials by how well they match target criteria.

    Args:
        df: DataFrame with predictions
        target_property: Property to optimize
        target_value: Target value (None to minimize/maximize)
        minimize: If True, prefer smaller values
        consider_uncertainty: If True, penalize high uncertainty
        uncertainty_weight: Weight for uncertainty penalty

    Returns:
        DataFrame sorted by score
    """
    pred_col = f"predicted_{target_property}"
    unc_col = f"uncertainty_{target_property}"

    df = df.copy()

    # Calculate score
    if target_value is not None:
        # Score based on distance to target
        distance = np.abs(df[pred_col] - target_value)
        score = 1.0 / (distance + 0.1)  # Closer is better
    else:
        # Score based on value itself
        if minimize:
            score = -df[pred_col]  # More negative is better
        else:
            score = df[pred_col]  # Higher is better

    # Penalize uncertainty
    if consider_uncertainty and unc_col in df.columns:
        uncertainty_penalty = uncertainty_weight * df[unc_col]
        score = score - uncertainty_penalty

    df["score"] = score
    df = df.sort_values("score", ascending=False)

    return df


def find_pareto_optimal(
    df: pd.DataFrame, property1: str, property2: str, maximize1: bool = True, maximize2: bool = True
) -> pd.DataFrame:
    """
    Find Pareto-optimal materials for multi-objective optimization.

    Args:
        df: DataFrame with predictions
        property1: First property name (predicted_X)
        property2: Second property name (predicted_Y)
        maximize1: If True, maximize property1
        maximize2: If True, maximize property2

    Returns:
        DataFrame with Pareto-optimal materials
    """
    df = df.copy()

    # Get property values
    prop1 = df[f"predicted_{property1}"].values
    prop2 = df[f"predicted_{property2}"].values

    # Flip signs if we want to minimize
    if not maximize1:
        prop1 = -prop1
    if not maximize2:
        prop2 = -prop2

    # Find Pareto frontier
    is_pareto = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if is_pareto[i]:
            # Check if any other point dominates this one
            is_dominated = np.any(
                (prop1 >= prop1[i])
                & (prop2 >= prop2[i])
                & ((prop1 > prop1[i]) | (prop2 > prop2[i]))
            )
            if is_dominated:
                is_pareto[i] = False

    pareto_df = df[is_pareto]
    print(f"Found {len(pareto_df)} Pareto-optimal materials")

    return pareto_df


def export_candidates(
    df: pd.DataFrame,
    output_file: str,
    top_n: Optional[int] = None,
    columns: Optional[list[str]] = None,
):
    """
    Export screening candidates to file.

    Args:
        df: DataFrame with screening results
        output_file: Output file path
        top_n: Number of top candidates to export
        columns: Columns to include in export
    """
    export_df = df.copy()

    if top_n is not None:
        export_df = export_df.head(top_n)

    if columns is not None:
        export_df = export_df[columns]

    # Export based on file extension
    if output_file.endswith(".csv"):
        export_df.to_csv(output_file, index=False)
    elif output_file.endswith(".json"):
        export_df.to_json(output_file, orient="records", indent=2)
    elif output_file.endswith(".xlsx"):
        export_df.to_excel(output_file, index=False)
    else:
        raise ValueError(f"Unsupported file format: {output_file}")

    print(f"Exported {len(export_df)} candidates to {output_file}")


def calculate_screening_metrics(df: pd.DataFrame, property_name: str) -> dict:
    """
    Calculate metrics for screening results.

    Args:
        df: DataFrame with predictions
        property_name: Property to analyze

    Returns:
        Dictionary with metrics
    """
    pred_col = f"predicted_{property_name}"
    unc_col = f"uncertainty_{property_name}"

    metrics = {
        "n_materials": len(df),
        "prediction": {
            "mean": df[pred_col].mean(),
            "std": df[pred_col].std(),
            "min": df[pred_col].min(),
            "max": df[pred_col].max(),
            "median": df[pred_col].median(),
        },
    }

    if unc_col in df.columns:
        metrics["uncertainty"] = {
            "mean": df[unc_col].mean(),
            "std": df[unc_col].std(),
            "min": df[unc_col].min(),
            "max": df[unc_col].max(),
            "median": df[unc_col].median(),
        }

    return metrics
