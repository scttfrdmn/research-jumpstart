"""Visualization functions for genomic variant data."""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


def plot_manhattan(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    p_col: str = "QUAL",
    sig_threshold: float = 30.0,
    figsize: tuple[int, int] = (16, 6),
) -> None:
    """
    Create Manhattan plot for variant quality across chromosomes.

    Args:
        df: DataFrame with variant information
        save_path: Path to save figure (optional)
        p_col: Column to use for y-axis (default: QUAL)
        sig_threshold: Significance threshold line
        figsize: Figure size tuple
    """
    logger.info("Creating Manhattan plot")

    if p_col not in df.columns:
        logger.error(f"Column {p_col} not found")
        return

    # Prepare data
    plot_df = df[["CHROM", "POS", p_col]].copy()
    plot_df = plot_df.dropna(subset=[p_col])

    # Sort chromosomes
    chrom_order = sorted(
        plot_df["CHROM"].unique(),
        key=lambda x: (int(x) if x.isdigit() else (23 if x == "X" else (24 if x == "Y" else 25))),
    )

    # Assign colors
    colors = ["#1f77b4", "#ff7f0e"]
    plot_df["color"] = [
        colors[i % 2]
        for i, chrom in enumerate(chrom_order)
        for _ in range(len(plot_df[plot_df["CHROM"] == chrom]))
    ]

    # Calculate cumulative positions
    plot_df["cum_pos"] = 0
    last_pos = 0
    chrom_centers = {}

    for chrom in chrom_order:
        chrom_df = plot_df[plot_df["CHROM"] == chrom]
        plot_df.loc[plot_df["CHROM"] == chrom, "cum_pos"] = chrom_df["POS"] + last_pos
        chrom_centers[chrom] = last_pos + chrom_df["POS"].max() / 2
        last_pos += chrom_df["POS"].max()

    # Create plot
    _fig, ax = plt.subplots(figsize=figsize)

    for chrom in chrom_order:
        chrom_df = plot_df[plot_df["CHROM"] == chrom]
        ax.scatter(
            chrom_df["cum_pos"], chrom_df[p_col], c=chrom_df["color"].iloc[0], s=10, alpha=0.6
        )

    # Add significance threshold
    ax.axhline(
        sig_threshold,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Threshold = {sig_threshold}",
    )

    # Format plot
    ax.set_xlabel("Chromosome", fontsize=12, fontweight="bold")
    ax.set_ylabel(p_col, fontsize=12, fontweight="bold")
    ax.set_title("Manhattan Plot - Variant Quality Across Genome", fontsize=14, fontweight="bold")

    # Set x-axis labels
    ax.set_xticks(list(chrom_centers.values()))
    ax.set_xticklabels(list(chrom_centers.keys()))

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Manhattan plot saved to {save_path}")

    plt.show()


def plot_variant_density(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    bin_size: int = 1000000,
    figsize: tuple[int, int] = (14, 8),
) -> None:
    """
    Plot variant density along chromosomes.

    Args:
        df: DataFrame with variant information
        save_path: Path to save figure (optional)
        bin_size: Bin size in base pairs (default: 1Mb)
        figsize: Figure size tuple
    """
    logger.info(f"Creating variant density plot (bin size: {bin_size}bp)")

    if "CHROM" not in df.columns or "POS" not in df.columns:
        logger.error("CHROM and POS columns required")
        return

    # Get chromosomes
    chroms = sorted(
        df["CHROM"].unique(),
        key=lambda x: (int(x) if x.isdigit() else (23 if x == "X" else (24 if x == "Y" else 25))),
    )

    # Create subplots
    n_chroms = len(chroms)
    n_cols = 4
    n_rows = (n_chroms + n_cols - 1) // n_cols

    _fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]

    for idx, chrom in enumerate(chroms):
        if idx >= len(axes):
            break

        chrom_df = df[df["CHROM"] == chrom]
        max_pos = chrom_df["POS"].max()

        # Create bins
        bins = np.arange(0, max_pos + bin_size, bin_size)
        counts, edges = np.histogram(chrom_df["POS"], bins=bins)

        # Plot
        axes[idx].bar(
            edges[:-1] / 1e6, counts, width=(bin_size / 1e6), edgecolor="black", alpha=0.7
        )
        axes[idx].set_title(f"Chr {chrom}", fontweight="bold")
        axes[idx].set_xlabel("Position (Mb)")
        axes[idx].set_ylabel("Variant Count")
        axes[idx].grid(True, alpha=0.3, axis="y")

    # Hide empty subplots
    for idx in range(len(chroms), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Variant Density Across Chromosomes", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Density plot saved to {save_path}")

    plt.show()


def plot_quality_distribution(
    df: pd.DataFrame, save_path: Optional[str] = None, figsize: tuple[int, int] = (12, 8)
) -> None:
    """
    Plot quality metric distributions.

    Args:
        df: DataFrame with variant information
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    logger.info("Creating quality distribution plots")

    _fig, axes = plt.subplots(2, 2, figsize=figsize)

    # QUAL distribution
    if "QUAL" in df.columns:
        qual_data = df["QUAL"].dropna()
        axes[0, 0].hist(qual_data, bins=50, edgecolor="black", alpha=0.7)
        axes[0, 0].axvline(
            qual_data.mean(), color="red", linestyle="--", label=f"Mean: {qual_data.mean():.1f}"
        )
        axes[0, 0].axvline(
            qual_data.median(),
            color="green",
            linestyle="--",
            label=f"Median: {qual_data.median():.1f}",
        )
        axes[0, 0].set_title("QUAL Score Distribution", fontweight="bold")
        axes[0, 0].set_xlabel("QUAL")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # Depth distribution
    if "DP" in df.columns:
        dp_data = df["DP"].dropna()
        axes[0, 1].hist(dp_data, bins=50, edgecolor="black", alpha=0.7, color="orange")
        axes[0, 1].axvline(
            dp_data.mean(), color="red", linestyle="--", label=f"Mean: {dp_data.mean():.1f}"
        )
        axes[0, 1].axvline(
            dp_data.median(), color="green", linestyle="--", label=f"Median: {dp_data.median():.1f}"
        )
        axes[0, 1].set_title("Read Depth Distribution", fontweight="bold")
        axes[0, 1].set_xlabel("Depth (DP)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Allele frequency distribution
    if "AF" in df.columns:
        af_data = df["AF"].dropna()
        axes[1, 0].hist(af_data, bins=50, edgecolor="black", alpha=0.7, color="green")
        axes[1, 0].axvline(0.05, color="red", linestyle="--", label="Common (AF>0.05)")
        axes[1, 0].set_title("Allele Frequency Distribution", fontweight="bold")
        axes[1, 0].set_xlabel("Allele Frequency")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Variant type distribution
    if "TYPE" in df.columns:
        type_counts = df["TYPE"].value_counts()
        axes[1, 1].bar(
            range(len(type_counts)),
            type_counts.values,
            tick_label=type_counts.index,
            edgecolor="black",
            alpha=0.7,
        )
        axes[1, 1].set_title("Variant Type Distribution", fontweight="bold")
        axes[1, 1].set_xlabel("Type")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Quality plot saved to {save_path}")

    plt.show()


def plot_allele_frequency_spectrum(
    df: pd.DataFrame, save_path: Optional[str] = None, figsize: tuple[int, int] = (10, 6)
) -> None:
    """
    Plot site frequency spectrum (SFS).

    Args:
        df: DataFrame with variant information
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    logger.info("Creating allele frequency spectrum")

    if "AF" not in df.columns:
        logger.error("AF column required")
        return

    # Create bins for frequency spectrum
    af_data = df["AF"].dropna()
    bins = np.linspace(0, 0.5, 51)  # 0 to 0.5 (MAF range)
    counts, edges = np.histogram(af_data, bins=bins)

    _fig, ax = plt.subplots(figsize=figsize)
    ax.bar(edges[:-1], counts, width=np.diff(edges), edgecolor="black", alpha=0.7, align="edge")

    ax.set_title("Site Frequency Spectrum", fontsize=14, fontweight="bold")
    ax.set_xlabel("Allele Frequency", fontsize=12)
    ax.set_ylabel("Number of Variants", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # Add annotations
    rare = (af_data < 0.01).sum()
    common = (af_data >= 0.05).sum()
    ax.text(
        0.02,
        0.95,
        f"Rare (AF<0.01): {rare:,}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
    ax.text(
        0.02,
        0.88,
        f"Common (AF≥0.05): {common:,}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.5},
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"SFS plot saved to {save_path}")

    plt.show()


def plot_chromosome_summary(
    df: pd.DataFrame, save_path: Optional[str] = None, figsize: tuple[int, int] = (14, 6)
) -> None:
    """
    Create summary bar plot of variants per chromosome.

    Args:
        df: DataFrame with variant information
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    logger.info("Creating chromosome summary")

    if "CHROM" not in df.columns:
        logger.error("CHROM column required")
        return

    # Count variants per chromosome
    chrom_counts = df["CHROM"].value_counts()

    # Sort chromosomes
    chrom_order = sorted(
        chrom_counts.index,
        key=lambda x: (int(x) if x.isdigit() else (23 if x == "X" else (24 if x == "Y" else 25))),
    )
    chrom_counts = chrom_counts[chrom_order]

    # Create plot
    _fig, ax = plt.subplots(figsize=figsize)

    colors = ["#1f77b4" if i % 2 == 0 else "#ff7f0e" for i in range(len(chrom_counts))]

    bars = ax.bar(
        range(len(chrom_counts)), chrom_counts.values, color=colors, edgecolor="black", alpha=0.7
    )

    ax.set_title("Variant Count by Chromosome", fontsize=14, fontweight="bold")
    ax.set_xlabel("Chromosome", fontsize=12)
    ax.set_ylabel("Number of Variants", fontsize=12)
    ax.set_xticks(range(len(chrom_counts)))
    ax.set_xticklabels(chrom_counts.index)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for _i, (bar, count) in enumerate(zip(bars, chrom_counts.values)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(count):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Chromosome summary saved to {save_path}")

    plt.show()
