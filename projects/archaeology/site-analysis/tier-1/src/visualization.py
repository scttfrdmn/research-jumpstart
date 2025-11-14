"""
Visualization Utilities

Functions for creating publication-quality figures and interactive
visualizations of archaeological data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap


def plot_artifact_distribution(artifact_data: pd.DataFrame,
                               save_path: Optional[Path] = None) -> None:
    """
    Plot distribution of artifact types and measurements.

    Args:
        artifact_data: DataFrame with artifact information
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Type distribution
    artifact_data['type'].value_counts().plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Artifact Type Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Artifact Type')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Length vs Width
    for artifact_type in artifact_data['type'].unique():
        data = artifact_data[artifact_data['type'] == artifact_type]
        axes[0, 1].scatter(data['length_cm'], data['width_cm'],
                          label=artifact_type, alpha=0.6, s=50)
    axes[0, 1].set_xlabel('Length (cm)', fontweight='bold')
    axes[0, 1].set_ylabel('Width (cm)', fontweight='bold')
    axes[0, 1].set_title('Artifact Dimensions', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Weight distribution
    artifact_data.boxplot(column='weight_g', by='type', ax=axes[1, 0])
    axes[1, 0].set_title('Weight Distribution by Type', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Artifact Type')
    axes[1, 0].set_ylabel('Weight (g)')
    plt.sca(axes[1, 0])
    plt.xticks(rotation=45)

    # Material composition
    material_counts = pd.crosstab(artifact_data['type'], artifact_data['material'])
    material_counts.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='Set3')
    axes[1, 1].set_title('Material Composition by Type', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Artifact Type')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend(title='Material')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_terrain_analysis(elevation: np.ndarray,
                          structures: np.ndarray,
                          resolution: float,
                          save_path: Optional[Path] = None) -> None:
    """
    Plot LiDAR terrain analysis with detected structures.

    Args:
        elevation: 2D elevation array
        structures: Binary mask of detected structures
        resolution: Spatial resolution in meters
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Elevation map
    extent = [0, elevation.shape[1] * resolution, 0, elevation.shape[0] * resolution]
    im1 = axes[0].imshow(elevation, cmap='terrain', extent=extent, aspect='auto')
    axes[0].set_title('LiDAR Elevation', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Distance (m)')
    axes[0].set_ylabel('Distance (m)')
    plt.colorbar(im1, ax=axes[0], label='Elevation (m)')

    # Structures overlay
    im2 = axes[1].imshow(elevation, cmap='gray', extent=extent, aspect='auto', alpha=0.5)
    structures_overlay = np.ma.masked_where(structures == 0, structures)
    axes[1].imshow(structures_overlay, cmap='Reds', extent=extent, aspect='auto', alpha=0.7)
    axes[1].set_title('Detected Archaeological Structures', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Distance (m)')
    axes[1].set_ylabel('Distance (m)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_geophysical_survey(gpr_data: np.ndarray,
                            mag_data: np.ndarray,
                            anomalies: Dict[str, np.ndarray],
                            save_path: Optional[Path] = None) -> None:
    """
    Plot geophysical survey data and detected anomalies.

    Args:
        gpr_data: Ground-penetrating radar array
        mag_data: Magnetometry array
        anomalies: Dictionary with 'gpr' and 'mag' anomaly masks
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # GPR profile
    im1 = axes[0, 0].imshow(gpr_data, cmap='seismic', aspect='auto')
    axes[0, 0].set_title('GPR Profile', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Distance')
    axes[0, 0].set_ylabel('Depth')
    plt.colorbar(im1, ax=axes[0, 0], label='Amplitude')

    # GPR anomalies
    im2 = axes[0, 1].imshow(anomalies['gpr'], cmap='binary', aspect='auto')
    axes[0, 1].set_title('GPR Detected Anomalies', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Distance')
    axes[0, 1].set_ylabel('Depth')

    # Magnetometry
    im3 = axes[1, 0].imshow(mag_data, cmap='RdBu_r', aspect='auto')
    axes[1, 0].set_title('Magnetometry Survey', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Y (m)')
    plt.colorbar(im3, ax=axes[1, 0], label='Magnetic Field (nT)')

    # Magnetometry anomalies
    combined_mag = anomalies['mag_positive'].astype(float) - anomalies['mag_negative'].astype(float)
    im4 = axes[1, 1].imshow(combined_mag, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_title('Magnetic Anomalies (Red: +, Blue: -)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('X (m)')
    axes[1, 1].set_ylabel('Y (m)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_site_comparison(site_features: Dict[str, pd.DataFrame],
                        save_path: Optional[Path] = None) -> None:
    """
    Create comparative visualization across multiple sites.

    Args:
        site_features: Dictionary mapping site names to feature DataFrames
        save_path: Optional path to save figure
    """
    n_sites = len(site_features)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Artifact counts by site
    artifact_counts = {site: len(df) for site, df in site_features.items()}
    axes[0, 0].bar(artifact_counts.keys(), artifact_counts.values(), color='steelblue')
    axes[0, 0].set_title('Artifact Count by Site', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Site')
    axes[0, 0].set_ylabel('Number of Artifacts')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Average artifact size comparison
    avg_sizes = {site: df['length_cm'].mean() if 'length_cm' in df else 0
                 for site, df in site_features.items()}
    axes[0, 1].bar(avg_sizes.keys(), avg_sizes.values(), color='coral')
    axes[0, 1].set_title('Average Artifact Size by Site', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Site')
    axes[0, 1].set_ylabel('Average Length (cm)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Artifact type distribution
    type_data = []
    for site, df in site_features.items():
        if 'type' in df.columns:
            for artifact_type in df['type'].unique():
                count = len(df[df['type'] == artifact_type])
                type_data.append({'Site': site, 'Type': artifact_type, 'Count': count})

    if type_data:
        type_df = pd.DataFrame(type_data)
        type_pivot = type_df.pivot(index='Site', columns='Type', values='Count').fillna(0)
        type_pivot.plot(kind='bar', stacked=True, ax=axes[1, 0], colormap='Set3')
        axes[1, 0].set_title('Artifact Type Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Site')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend(title='Artifact Type', bbox_to_anchor=(1.05, 1))
        axes[1, 0].tick_params(axis='x', rotation=45)

    # Site similarity heatmap (placeholder)
    similarity_matrix = np.random.rand(n_sites, n_sites)
    np.fill_diagonal(similarity_matrix, 1.0)
    site_names = list(site_features.keys())
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=site_names, yticklabels=site_names, ax=axes[1, 1])
    axes[1, 1].set_title('Inter-Site Similarity', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_3d_site_model(elevation: np.ndarray,
                        structures: Optional[np.ndarray] = None,
                        resolution: float = 1.0,
                        save_path: Optional[Path] = None) -> go.Figure:
    """
    Create interactive 3D visualization of archaeological site.

    Args:
        elevation: 2D elevation array
        structures: Optional binary mask of structures to highlight
        resolution: Spatial resolution in meters
        save_path: Optional path to save HTML figure

    Returns:
        Plotly figure object
    """
    # Create coordinate grids
    y_coords = np.arange(elevation.shape[0]) * resolution
    x_coords = np.arange(elevation.shape[1]) * resolution
    X, Y = np.meshgrid(x_coords, y_coords)

    # Create 3D surface
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=elevation,
        colorscale='earth',
        showscale=True,
        colorbar=dict(title='Elevation (m)')
    )])

    # Add structures if provided
    if structures is not None:
        structure_coords = np.where(structures)
        if len(structure_coords[0]) > 0:
            structure_elevations = elevation[structure_coords]
            fig.add_trace(go.Scatter3d(
                x=structure_coords[1] * resolution,
                y=structure_coords[0] * resolution,
                z=structure_elevations + 0.5,  # Elevate slightly for visibility
                mode='markers',
                marker=dict(size=3, color='red', opacity=0.8),
                name='Structures'
            ))

    # Update layout
    fig.update_layout(
        title='3D Archaeological Site Model',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Elevation (m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.3)
        ),
        width=1000,
        height=800
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_temporal_analysis(sites_by_period: Dict[str, List[str]],
                          period_dates: Dict[str, Tuple[int, int]],
                          save_path: Optional[Path] = None) -> None:
    """
    Visualize temporal distribution of archaeological sites.

    Args:
        sites_by_period: Dictionary mapping periods to list of site names
        period_dates: Dictionary mapping periods to (start_year, end_year) tuples
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Sort periods by start date
    sorted_periods = sorted(period_dates.keys(), key=lambda p: period_dates[p][0])

    y_pos = 0
    colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_periods)))

    for i, period in enumerate(sorted_periods):
        start_year, end_year = period_dates[period]
        sites = sites_by_period.get(period, [])

        # Draw period bar
        ax.barh(y_pos, end_year - start_year, left=start_year,
               height=0.8, color=colors[i], alpha=0.7, label=period)

        # Add site count annotation
        mid_year = (start_year + end_year) / 2
        ax.text(mid_year, y_pos, f'{len(sites)} sites',
               ha='center', va='center', fontweight='bold', fontsize=10)

        y_pos += 1

    ax.set_xlabel('Year (CE/BCE)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Archaeological Period', fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(sorted_periods)))
    ax.set_yticklabels(sorted_periods)
    ax.set_title('Temporal Distribution of Archaeological Sites',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='BCE/CE')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
