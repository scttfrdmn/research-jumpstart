"""Variant analysis functions for genomic data."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_allele_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate minor allele frequencies (MAF) for variants.

    Args:
        df: DataFrame with variant information

    Returns:
        DataFrame with added MAF column
    """
    logger.info("Calculating allele frequencies")

    df_copy = df.copy()

    # AF column should already exist from VCF parsing
    # MAF is the minimum of AF and 1-AF
    if "AF" in df_copy.columns:
        df_copy["MAF"] = df_copy["AF"].apply(lambda x: min(x, 1 - x) if pd.notna(x) else np.nan)
    else:
        logger.warning("AF column not found, cannot calculate MAF")
        df_copy["MAF"] = np.nan

    logger.info(f"Calculated MAF for {len(df_copy)} variants")
    return df_copy


def filter_variants_by_quality(
    df: pd.DataFrame,
    min_qual: Optional[float] = 30.0,
    min_dp: Optional[int] = 10,
    max_missing: Optional[float] = 0.1,
) -> pd.DataFrame:
    """
    Filter variants based on quality metrics.

    Args:
        df: DataFrame with variant information
        min_qual: Minimum QUAL score (default: 30)
        min_dp: Minimum depth (default: 10)
        max_missing: Maximum missing rate (default: 0.1)

    Returns:
        Filtered DataFrame
    """
    logger.info(f"Filtering variants (QUAL>={min_qual}, DP>={min_dp})")

    initial_count = len(df)
    filtered_df = df.copy()

    # Filter by QUAL
    if min_qual is not None and "QUAL" in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df["QUAL"].isna()) | (filtered_df["QUAL"] >= min_qual)]
        logger.info(f"After QUAL filter: {len(filtered_df)} variants")

    # Filter by depth
    if min_dp is not None and "DP" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["DP"] >= min_dp]
        logger.info(f"After DP filter: {len(filtered_df)} variants")

    # Filter by FILTER field
    if "FILTER" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["FILTER"] == "PASS") | (filtered_df["FILTER"] == ".")
        ]
        logger.info(f"After FILTER field: {len(filtered_df)} variants")

    removed = initial_count - len(filtered_df)
    pct_removed = (removed / initial_count * 100) if initial_count > 0 else 0
    logger.info(f"Removed {removed} variants ({pct_removed:.1f}%)")

    return filtered_df


def annotate_variants(df: pd.DataFrame, add_consequences: bool = True) -> pd.DataFrame:
    """
    Add functional annotations to variants.

    Args:
        df: DataFrame with variant information
        add_consequences: Add predicted consequences based on variant type

    Returns:
        DataFrame with annotation columns
    """
    logger.info("Annotating variants")

    df_copy = df.copy()

    # Classify variant types
    def classify_variant(row):
        ref = str(row.get("REF", ""))
        alt = str(row.get("ALT", ""))

        if len(ref) == len(alt) == 1:
            return "SNV"
        elif len(ref) > len(alt):
            return "deletion"
        elif len(alt) > len(ref):
            return "insertion"
        else:
            return "complex"

    df_copy["variant_class"] = df_copy.apply(classify_variant, axis=1)

    # Add consequence predictions (simplified)
    if add_consequences:

        def predict_consequence(variant_class):
            if variant_class == "SNV":
                return "missense_variant"
            elif variant_class == "deletion" or variant_class == "insertion":
                return "frameshift_variant"
            else:
                return "sequence_alteration"

        df_copy["predicted_consequence"] = df_copy["variant_class"].apply(predict_consequence)

    # Add severity score (0-1 based on allele frequency and quality)
    if "AF" in df_copy.columns and "QUAL" in df_copy.columns:
        # Rare variants with high quality are more likely to be deleterious
        df_copy["severity_score"] = ((1 - df_copy["AF"]) * (df_copy["QUAL"] / 100)).clip(0, 1)

    logger.info(f"Annotated {len(df_copy)} variants")
    return df_copy


def summarize_variants(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for variant dataset.

    Args:
        df: DataFrame with variant information

    Returns:
        Dictionary with summary statistics
    """
    logger.info("Generating variant summary")

    summary = {
        "total_variants": len(df),
        "chromosomes": df["CHROM"].nunique() if "CHROM" in df.columns else 0,
    }

    # Variant type distribution
    if "TYPE" in df.columns:
        summary["variant_types"] = df["TYPE"].value_counts().to_dict()

    # Quality metrics
    if "QUAL" in df.columns:
        summary["mean_qual"] = float(df["QUAL"].mean())
        summary["median_qual"] = float(df["QUAL"].median())

    if "DP" in df.columns:
        summary["mean_depth"] = float(df["DP"].mean())
        summary["median_depth"] = float(df["DP"].median())

    # Allele frequency distribution
    if "AF" in df.columns:
        summary["mean_af"] = float(df["AF"].mean())
        summary["rare_variants"] = int((df["AF"] < 0.01).sum())
        summary["common_variants"] = int((df["AF"] >= 0.05).sum())

    # Chromosome distribution
    if "CHROM" in df.columns:
        summary["variants_per_chrom"] = df["CHROM"].value_counts().to_dict()

    return summary


def identify_high_impact_variants(
    df: pd.DataFrame, max_af: float = 0.01, min_qual: float = 50.0
) -> pd.DataFrame:
    """
    Identify potentially high-impact rare variants.

    Args:
        df: DataFrame with variant information
        max_af: Maximum allele frequency for rare variants
        min_qual: Minimum quality score

    Returns:
        DataFrame with high-impact variants
    """
    logger.info(f"Identifying high-impact variants (AF<{max_af}, QUAL>{min_qual})")

    high_impact = df.copy()

    # Filter for rare variants
    if "AF" in high_impact.columns:
        high_impact = high_impact[high_impact["AF"] < max_af]

    # Filter for high quality
    if "QUAL" in high_impact.columns:
        high_impact = high_impact[(high_impact["QUAL"].isna()) | (high_impact["QUAL"] >= min_qual)]

    # Filter for non-synonymous if TYPE available
    if "TYPE" in high_impact.columns:
        high_impact = high_impact[
            high_impact["TYPE"].isin(["nonsynonymous SNV", "stopgain", "stoploss"])
        ]

    logger.info(f"Found {len(high_impact)} high-impact variants")
    return high_impact


def compare_variant_sets(
    df1: pd.DataFrame, df2: pd.DataFrame, name1: str = "Set1", name2: str = "Set2"
) -> dict:
    """
    Compare two variant datasets.

    Args:
        df1: First variant DataFrame
        df2: Second variant DataFrame
        name1: Name for first set
        name2: Name for second set

    Returns:
        Dictionary with comparison statistics
    """
    logger.info(f"Comparing {name1} and {name2}")

    # Create unique identifiers
    def create_variant_id(row):
        return f"{row['CHROM']}:{row['POS']}:{row['REF']}:{row['ALT']}"

    if all(col in df1.columns for col in ["CHROM", "POS", "REF", "ALT"]):
        set1 = set(df1.apply(create_variant_id, axis=1))
    else:
        set1 = set()

    if all(col in df2.columns for col in ["CHROM", "POS", "REF", "ALT"]):
        set2 = set(df2.apply(create_variant_id, axis=1))
    else:
        set2 = set()

    comparison = {
        f"{name1}_count": len(set1),
        f"{name2}_count": len(set2),
        "shared": len(set1 & set2),
        f"{name1}_unique": len(set1 - set2),
        f"{name2}_unique": len(set2 - set1),
        "jaccard_index": len(set1 & set2) / len(set1 | set2) if set1 or set2 else 0,
    }

    logger.info(f"Comparison: {comparison['shared']} shared variants")
    return comparison
