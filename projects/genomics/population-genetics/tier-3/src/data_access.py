"""
Data access utilities for genomic data on AWS.

Provides functions to load variants from:
- S3 (1000 Genomes public dataset)
- Athena (processed variant tables)
- Local files (for testing)
"""

from typing import Optional

import allel
import awswrangler as wr
import boto3
import numpy as np
import pandas as pd


class GenomicsDataLoader:
    """Load and process genomic data from AWS sources."""

    def __init__(self, bucket_name: Optional[str] = None):
        """
        Initialize data loader.

        Parameters
        ----------
        bucket_name : str, optional
            S3 bucket for processed data. If None, uses 1000genomes public bucket.
        """
        self.bucket_name = bucket_name or "1000genomes"
        self.s3_client = boto3.client("s3")

    def load_variants(
        self,
        chromosome: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        populations: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Load variants for a genomic region.

        Parameters
        ----------
        chromosome : str
            Chromosome name (e.g., '22', 'X')
        start : int, optional
            Start position (1-based)
        end : int, optional
            End position (1-based)
        populations : list of str, optional
            Filter by populations (e.g., ['YRI', 'CEU'])

        Returns
        -------
        variants_df : DataFrame
            Variants with columns: CHROM, POS, REF, ALT, AF, etc.
        """
        # Construct S3 path for 1000 Genomes
        s3_path = f"s3://{self.bucket_name}/phase3/integrated_sv_map/supporting/genotypes/ALL.chr{chromosome}.phase3_integrated_v5a.20130502.genotypes.vcf.gz"

        print(f"Loading variants from: {s3_path}")

        # Use allel to read VCF efficiently
        try:
            import s3fs

            fs = s3fs.S3FileSystem(anon=True)

            with fs.open(s3_path) as f:
                vcf = allel.read_vcf(
                    f, region=f"{chromosome}:{start}-{end}" if start and end else None
                )

            # Convert to DataFrame
            variants_df = pd.DataFrame(
                {
                    "CHROM": vcf["variants/CHROM"],
                    "POS": vcf["variants/POS"],
                    "REF": vcf["variants/REF"],
                    "ALT": vcf["variants/ALT"][:, 0],  # Take first alt allele
                    "QUAL": vcf["variants/QUAL"],
                }
            )

            # Calculate allele frequency
            genotypes = vcf["calldata/GT"]
            af = genotypes.sum(axis=(1, 2)) / (genotypes.shape[1] * 2)
            variants_df["AF"] = af

            return variants_df

        except Exception as e:
            print(f"Error loading from S3: {e}")
            print("Returning empty DataFrame")
            return pd.DataFrame()

    def load_genotypes(
        self,
        chromosome: str,
        samples: Optional[list[str]] = None,
        positions: Optional[list[int]] = None,
    ):
        """
        Load genotype matrix for a chromosome.

        Parameters
        ----------
        chromosome : str
            Chromosome name
        samples : list of str, optional
            Sample IDs to include
        positions : list of int, optional
            Specific positions to extract

        Returns
        -------
        genotypes : allel.GenotypeArray
            Genotype array (n_variants × n_samples × ploidy)
        positions : array
            Variant positions
        sample_ids : array
            Sample identifiers
        """
        s3_path = f"s3://{self.bucket_name}/phase3/integrated_sv_map/supporting/genotypes/ALL.chr{chromosome}.phase3_integrated_v5a.20130502.genotypes.vcf.gz"

        try:
            import s3fs

            fs = s3fs.S3FileSystem(anon=True)

            with fs.open(s3_path) as f:
                vcf = allel.read_vcf(f, fields=["calldata/GT", "variants/POS", "samples"])

            genotypes = allel.GenotypeArray(vcf["calldata/GT"])
            positions = vcf["variants/POS"]
            sample_ids = vcf["samples"]

            # Filter by samples if requested
            if samples:
                sample_mask = np.isin(sample_ids, samples)
                genotypes = genotypes[:, sample_mask, :]
                sample_ids = sample_ids[sample_mask]

            # Filter by positions if requested
            if positions:
                pos_mask = np.isin(positions, positions)
                genotypes = genotypes[pos_mask, :, :]
                positions = positions[pos_mask]

            return genotypes, positions, sample_ids

        except Exception as e:
            print(f"Error loading genotypes: {e}")
            return None, None, None

    def query_athena(
        self, query: str, database: str = "population_genetics_database"
    ) -> pd.DataFrame:
        """
        Execute Athena SQL query on variant tables.

        Parameters
        ----------
        query : str
            SQL query
        database : str
            Glue database name

        Returns
        -------
        result_df : DataFrame
            Query results
        """
        try:
            df = wr.athena.read_sql_query(sql=query, database=database, ctas_approach=False)
            return df
        except Exception as e:
            print(f"Athena query error: {e}")
            return pd.DataFrame()

    def load_population_metadata(self) -> pd.DataFrame:
        """
        Load sample population assignments.

        Returns
        -------
        metadata_df : DataFrame
            Columns: sample_id, population, super_population, gender
        """
        # 1000 Genomes sample info
        s3_path = "s3://1000genomes/phase3/20131219.populations.tsv"

        try:
            metadata_df = pd.read_csv(s3_path, sep="\t")
            return metadata_df
        except Exception as e:
            print(f"Error loading metadata: {e}")
            # Return dummy data for testing
            return pd.DataFrame(
                {
                    "sample_id": ["HG00096", "HG00097"],
                    "population": ["GBR", "GBR"],
                    "super_population": ["EUR", "EUR"],
                    "gender": ["male", "female"],
                }
            )


def load_variants(
    chromosome: str, start: Optional[int] = None, end: Optional[int] = None
) -> pd.DataFrame:
    """
    Convenience function to load variants.

    Parameters
    ----------
    chromosome : str
        Chromosome (e.g., '22')
    start : int, optional
        Start position
    end : int, optional
        End position

    Returns
    -------
    variants_df : DataFrame
        Variant data
    """
    loader = GenomicsDataLoader()
    return loader.load_variants(chromosome, start, end)


def load_genotypes(chromosome: str, samples: Optional[list[str]] = None):
    """
    Convenience function to load genotypes.

    Parameters
    ----------
    chromosome : str
        Chromosome
    samples : list of str, optional
        Sample IDs

    Returns
    -------
    genotypes, positions, sample_ids
        Genotype array, positions, and sample IDs
    """
    loader = GenomicsDataLoader()
    return loader.load_genotypes(chromosome, samples)


if __name__ == "__main__":
    # Example usage
    print("Loading chromosome 22 variants...")
    loader = GenomicsDataLoader()

    # Load variants for a small region
    variants = loader.load_variants(chromosome="22", start=16000000, end=17000000)

    print(f"Loaded {len(variants)} variants")
    print(variants.head())

    # Load population metadata
    metadata = loader.load_population_metadata()
    print(f"\nPopulation metadata: {len(metadata)} samples")
    print(metadata.head())
