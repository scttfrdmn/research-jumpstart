"""
Data loading and processing utilities for genomic variant analysis.

This module provides functions to download BAM files, generate pileup tensors,
extract variant features, and manage genomic data. Data is cached in Studio Lab's
persistent storage to avoid re-downloading.
"""

import urllib.request
from pathlib import Path

import numpy as np
import pysam
from tqdm import tqdm

# Data directory (persistent in Studio Lab)
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"

# Create directories
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)


# Base encoding for pileup tensors
BASE_ENCODING = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    """
    Download file from URL to destination, with caching and progress bar.

    Args:
        url: URL to download from
        destination: Local path to save to
        force: If True, re-download even if file exists

    Returns:
        Path to downloaded file
    """
    if destination.exists() and not force:
        print(f"✓ Using cached file: {destination.name}")
        return destination

    print(f"Downloading {destination.name}...")

    # Download with progress bar
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100)
        print(
            f"\r  Progress: {percent:.1f}% ({downloaded / 1e9:.2f} GB / {total_size / 1e9:.2f} GB)",
            end="",
        )

    urllib.request.urlretrieve(url, destination, reporthook=show_progress)
    print(f"\n✓ Saved to {destination}")
    return destination


def download_bam_file(
    sample_id: str, chromosome: str = "20", force: bool = False
) -> tuple[Path, Path]:
    """
    Download BAM file and index from 1000 Genomes AWS Open Data Registry.

    Args:
        sample_id: Sample ID (e.g., "NA12878")
        chromosome: Chromosome to download (default: "20")
        force: If True, re-download even if cached

    Returns:
        Tuple of (bam_path, bai_path)
    """
    # Construct URLs (AWS Open Data Registry - no credentials needed)
    base_url = "https://s3.amazonaws.com/1000genomes/phase3/data"

    # Determine population directory (simplified - in practice would query from metadata)
    # This is a placeholder - real implementation would look up sample metadata
    population_map = {
        "NA12878": "CEU",
        "NA12891": "CEU",
        "NA12892": "CEU",
        "NA19238": "YRI",
        "NA19239": "YRI",
        "NA19240": "YRI",
        "NA18525": "CHB",
        "NA18526": "CHB",
        "NA19648": "MXL",
        "NA19649": "MXL",
    }
    population = population_map.get(sample_id, "CEU")

    bam_filename = (
        f"{sample_id}.chrom{chromosome}.ILLUMINA.bwa.{population}.low_coverage.20121211.bam"
    )
    bai_filename = f"{bam_filename}.bai"

    bam_url = f"{base_url}/{sample_id}/alignment/{bam_filename}"
    bai_url = f"{base_url}/{sample_id}/alignment/{bai_filename}"

    # Download files
    bam_path = RAW_DATA_DIR / bam_filename
    bai_path = RAW_DATA_DIR / bai_filename

    download_file(bam_url, bam_path, force=force)
    download_file(bai_url, bai_path, force=force)

    return bam_path, bai_path


def load_bam_file(bam_path: Path) -> pysam.AlignmentFile:
    """
    Load BAM file using pysam.

    Args:
        bam_path: Path to BAM file

    Returns:
        pysam.AlignmentFile object
    """
    if not bam_path.exists():
        raise FileNotFoundError(f"BAM file not found: {bam_path}")

    return pysam.AlignmentFile(str(bam_path), "rb")


def load_reference_genome(chromosome: str = "20", force_download: bool = False) -> pysam.FastaFile:
    """
    Load reference genome (hg19/GRCh37).

    Args:
        chromosome: Chromosome to load
        force_download: If True, re-download even if cached

    Returns:
        pysam.FastaFile object
    """
    ref_filename = f"chr{chromosome}.fa"
    ref_path = REFERENCE_DIR / ref_filename

    if not ref_path.exists() or force_download:
        # Download from UCSC
        url = f"https://hgdownload.cse.ucsc.edu/goldenPath/hg19/chromosomes/chr{chromosome}.fa.gz"
        gz_path = REFERENCE_DIR / f"chr{chromosome}.fa.gz"

        download_file(url, gz_path, force=force_download)

        # Uncompress
        import gzip
        import shutil

        print(f"Decompressing {gz_path.name}...")
        with gzip.open(gz_path, "rb") as f_in, open(ref_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"✓ Decompressed to {ref_path}")

        # Create FASTA index
        pysam.faidx(str(ref_path))

    return pysam.FastaFile(str(ref_path))


def generate_pileup_tensor(
    bamfile: pysam.AlignmentFile,
    reference: str,
    chrom: str,
    start: int,
    end: int,
    max_depth: int = 100,
) -> np.ndarray:
    """
    Create pileup tensor for a genomic region.

    Converts aligned reads to image-like tensor representation for CNN input.

    Args:
        bamfile: pysam.AlignmentFile object
        reference: Reference sequence (string)
        chrom: Chromosome name
        start, end: Genomic coordinates (0-based)
        max_depth: Maximum read depth to consider

    Returns:
        tensor: Shape (length, max_depth, n_channels)
            Channels: [base, base_qual, map_qual, strand, is_match, is_del, is_ins]
    """
    length = end - start
    n_channels = 7  # base, base_qual, map_qual, strand, is_match, is_del, is_ins

    tensor = np.zeros((length, max_depth, n_channels), dtype=np.float32)

    # For each position, collect overlapping reads
    for pos_idx, pos in enumerate(range(start, end)):
        pileup_column = bamfile.pileup(chrom, pos, pos + 1, truncate=True, max_depth=max_depth)

        read_idx = 0
        for pileup_col in pileup_column:
            for pileup_read in pileup_col.pileups:
                if read_idx >= max_depth:
                    break

                read = pileup_read.alignment

                # Channel 0: Base encoding
                if not pileup_read.is_del and not pileup_read.is_refskip:
                    base = read.query_sequence[pileup_read.query_position]
                    tensor[pos_idx, read_idx, 0] = BASE_ENCODING.get(base, 4) / 4.0

                    # Channel 1: Base quality
                    base_qual = read.query_qualities[pileup_read.query_position]
                    tensor[pos_idx, read_idx, 1] = base_qual / 40.0  # Normalize

                # Channel 2: Mapping quality
                tensor[pos_idx, read_idx, 2] = read.mapping_quality / 60.0

                # Channel 3: Strand
                tensor[pos_idx, read_idx, 3] = 0 if read.is_reverse else 1

                # Channel 4: Is match to reference
                ref_base = reference[pos - start] if pos - start < len(reference) else "N"
                if not pileup_read.is_del and not pileup_read.is_refskip:
                    read_base = read.query_sequence[pileup_read.query_position]
                    tensor[pos_idx, read_idx, 4] = 1.0 if read_base == ref_base else 0.0

                # Channel 5: Is deletion
                tensor[pos_idx, read_idx, 5] = 1.0 if pileup_read.is_del else 0.0

                # Channel 6: Is insertion
                tensor[pos_idx, read_idx, 6] = 1.0 if pileup_read.indel > 0 else 0.0

                read_idx += 1

    return tensor


def extract_variant_features(
    bamfile: pysam.AlignmentFile,
    reference_file: pysam.FastaFile,
    chrom: str,
    position: int,
    window: int = 10,
) -> dict[str, float]:
    """
    Extract hand-crafted features for a potential variant position.

    These features complement CNN-based calling for ensemble models.

    Args:
        bamfile: pysam.AlignmentFile object
        reference_file: pysam.FastaFile object
        chrom: Chromosome name
        position: Genomic position (0-based)
        window: Window size around position

    Returns:
        Dictionary of features
    """
    features = {}

    # Reference base
    ref_base = reference_file.fetch(chrom, position, position + 1)
    features["ref_base"] = BASE_ENCODING.get(ref_base, 4)

    # Collect reads at position
    reads = list(bamfile.fetch(chrom, position, position + 1))
    features["depth"] = len(reads)

    if len(reads) == 0:
        return features

    # Base counts
    base_counts = {"A": 0, "C": 0, "G": 0, "T": 0}
    qualities = []
    map_qualities = []
    strand_counts = {"+": 0, "-": 0}

    for read in reads:
        if read.reference_start <= position < read.reference_end:
            read_pos = position - read.reference_start

            # Base
            if read_pos < len(read.query_sequence):
                base = read.query_sequence[read_pos]
                if base in base_counts:
                    base_counts[base] += 1

                # Quality
                if read_pos < len(read.query_qualities):
                    qualities.append(read.query_qualities[read_pos])

            # Mapping quality
            map_qualities.append(read.mapping_quality)

            # Strand
            strand_counts["-" if read.is_reverse else "+"] += 1

    # Allele frequencies
    total_bases = sum(base_counts.values())
    if total_bases > 0:
        for base in base_counts:
            features[f"af_{base}"] = base_counts[base] / total_bases

        # Alternate allele frequency (highest non-reference)
        alt_bases = {b: c for b, c in base_counts.items() if b != ref_base}
        if alt_bases:
            features["alt_af"] = max(alt_bases.values()) / total_bases
        else:
            features["alt_af"] = 0.0

    # Quality statistics
    if qualities:
        features["mean_base_qual"] = np.mean(qualities)
        features["min_base_qual"] = np.min(qualities)
    else:
        features["mean_base_qual"] = 0.0
        features["min_base_qual"] = 0.0

    if map_qualities:
        features["mean_map_qual"] = np.mean(map_qualities)
    else:
        features["mean_map_qual"] = 0.0

    # Strand bias
    total_strands = sum(strand_counts.values())
    if total_strands > 0:
        features["strand_bias"] = abs(strand_counts["+"] - strand_counts["-"]) / total_strands
    else:
        features["strand_bias"] = 0.0

    return features


def cache_pileup_tensors(
    sample_ids: list[str], chromosome: str, regions: list[tuple[int, int]], window_size: int = 221
) -> Path:
    """
    Generate and cache pileup tensors for multiple samples.

    Args:
        sample_ids: List of sample IDs
        chromosome: Chromosome to process
        regions: List of (start, end) tuples
        window_size: Pileup window size

    Returns:
        Path to cached HDF5 file
    """
    import h5py

    cache_file = PROCESSED_DATA_DIR / f"pileup_tensors_{chromosome}.h5"

    if cache_file.exists():
        print(f"✓ Using cached pileup tensors: {cache_file}")
        return cache_file

    print(f"Generating pileup tensors for {len(sample_ids)} samples...")

    # Load reference genome
    reference_file = load_reference_genome(chromosome)

    with h5py.File(cache_file, "w") as h5f:
        for sample_id in tqdm(sample_ids, desc="Processing samples"):
            # Load BAM file
            bam_path, _ = download_bam_file(sample_id, chromosome)
            bamfile = load_bam_file(bam_path)

            sample_tensors = []

            for start, end in regions:
                for pos in range(start, end - window_size, window_size // 2):
                    window_end = pos + window_size

                    # Get reference sequence
                    ref_seq = reference_file.fetch(chromosome, pos, window_end)

                    # Generate pileup tensor
                    tensor = generate_pileup_tensor(bamfile, ref_seq, chromosome, pos, window_end)
                    sample_tensors.append(tensor)

            # Save to HDF5
            h5f.create_dataset(sample_id, data=np.array(sample_tensors), compression="gzip")

            bamfile.close()

    print(f"✓ Cached pileup tensors to {cache_file}")
    return cache_file


def load_cached_pileup_tensors(sample_id: str, cache_file: Path) -> np.ndarray:
    """
    Load cached pileup tensors from HDF5 file.

    Args:
        sample_id: Sample ID
        cache_file: Path to HDF5 cache file

    Returns:
        Array of pileup tensors
    """
    import h5py

    with h5py.File(cache_file, "r") as h5f:
        return h5f[sample_id][:]
