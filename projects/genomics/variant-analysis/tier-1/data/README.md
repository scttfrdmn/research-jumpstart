# Genomic Data Storage

This directory stores downloaded BAM files, reference genomes, and processed features. Data persists between SageMaker Studio Lab sessions!

## Directory Structure

```
data/
├── raw/                    # Original BAM files and indices
│   ├── NA12878.chrom20.bam
│   ├── NA12878.chrom20.bam.bai
│   ├── NA19238.chrom20.bam
│   ├── NA19238.chrom20.bam.bai
│   └── ... (8-10 BAM files @ ~1.2GB each)
│
├── processed/              # Cached pileup tensors and features
│   ├── pileup_tensors_20.h5
│   ├── variant_features_20.csv
│   └── training_labels_20.npy
│
└── reference/              # Reference genome and truth sets
    ├── chr20.fa
    ├── chr20.fa.fai
    └── GIAB_truth_NA12878.vcf.gz
```

## Datasets

### 1000 Genomes BAM Files

**Source:** AWS Open Data Registry (1000 Genomes Project Phase 3)
**Access:** Public, no credentials required
**URL:** https://registry.opendata.aws/1000-genomes/

**Samples:**
- **NA12878** (CEU - European) - GIAB reference sample
- **NA12891, NA12892** (CEU - European)
- **NA19238, NA19239, NA19240** (YRI - African)
- **NA18525, NA18526** (CHB - East Asian)
- **NA19648, NA19649** (MXL - Admixed American)

**Details:**
- **Chromosome:** 20 (full or large subset)
- **Coverage:** ~30x whole-genome sequencing
- **Platform:** Illumina
- **Aligner:** BWA
- **Size per sample:** ~1.0-1.2GB
- **Total size:** ~10GB (8-10 samples)

### Reference Genome

**Source:** UCSC Genome Browser
**Build:** GRCh37/hg19
**Chromosome:** chr20
**Size:** ~60MB (uncompressed)
**URL:** https://hgdownload.cse.ucsc.edu/goldenPath/hg19/chromosomes/

### Truth Sets

**GIAB (Genome in a Bottle) - NA12878**
- **Source:** NIST Genome in a Bottle Consortium
- **Version:** v4.2.1
- **Coverage:** High-confidence variant calls
- **Size:** ~50MB (compressed VCF)
- **URL:** https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/

## Usage

Data is automatically downloaded and cached on first use:

```python
from src.data_utils import download_bam_file, load_reference_genome

# First run: downloads ~1.2GB BAM file
bam_path, bai_path = download_bam_file("NA12878", chromosome="20")

# Subsequent runs: uses cache
bam_path, bai_path = download_bam_file("NA12878", chromosome="20")  # Instant!

# Load reference genome (downloads once, ~60MB)
reference = load_reference_genome(chromosome="20")

# Force re-download if needed
bam_path, bai_path = download_bam_file("NA12878", chromosome="20", force=True)
```

## Caching Pileup Tensors

Generating pileup tensors is computationally expensive. Cache them to HDF5:

```python
from src.data_utils import cache_pileup_tensors, load_cached_pileup_tensors

# Generate and cache tensors (run once, takes ~60 min)
sample_ids = ["NA12878", "NA19238", "NA18525"]
regions = [(10000000, 15000000)]

cache_file = cache_pileup_tensors(
    sample_ids=sample_ids,
    chromosome="20",
    regions=regions,
    window_size=221
)

# Load cached tensors (instant!)
tensors = load_cached_pileup_tensors("NA12878", cache_file)
```

## Storage Management

Check current usage:
```bash
# Total data directory size
du -sh data/

# Breakdown by subdirectory
du -sh data/raw/ data/processed/ data/reference/
```

Clean old files:
```bash
# Remove old cached features
rm -rf data/processed/*.old

# Remove backup files
rm -rf data/processed/*.backup
```

Check available space:
```bash
# Studio Lab provides 15GB persistent storage
df -h ~
```

## Persistence

✅ **Persistent:** This directory survives Studio Lab session restarts
✅ **15GB Limit:** Studio Lab provides 15GB persistent storage total
✅ **Shared:** All notebooks in this project share this data directory
✅ **Git-ignored:** .gitignore excludes data/ from version control

## Download Times

Estimates for first download (with caching):

| Component | Size | Time (Est.) |
|-----------|------|-------------|
| Single BAM file | ~1.2GB | 15-20 min |
| 10 BAM files | ~10GB | 45-60 min |
| Reference genome | ~60MB | 2-3 min |
| GIAB truth set | ~50MB | 2-3 min |
| **Total (first run)** | **~10GB** | **50-70 min** |

**Subsequent runs:** Instant (data cached)

## Data Access Pattern

```
Session 1 (First Run):
├── Download BAM files (45-60 min)
├── Download reference genome (2-3 min)
├── Generate pileup tensors (60-90 min)
└── Save to data/ directory

Session 2+ (Subsequent Runs):
├── Load cached BAM files (instant)
├── Load cached reference (instant)
├── Load cached pileup tensors (instant)
└── Start analysis immediately!
```

## Population Codes

| Code | Population | Description |
|------|------------|-------------|
| CEU | Utah residents (CEPH) with Northern and Western European ancestry |
| YRI | Yoruba in Ibadan, Nigeria |
| CHB | Han Chinese in Beijing, China |
| JPT | Japanese in Tokyo, Japan |
| MXL | Mexican ancestry in Los Angeles, California |

## Sample Quality

All samples are from 1000 Genomes Phase 3 (final release):
- High-quality whole-genome sequencing
- Thorough QC and alignment
- Multiple validation studies
- Widely used in genomics research

## Troubleshooting

### BAM File Corruption
```bash
# Re-index BAM file
samtools index data/raw/NA12878.chrom20.bam
```

### Out of Space
```bash
# Remove cached pileup tensors (can regenerate)
rm -rf data/processed/pileup_tensors_*.h5

# Remove old BAM files
rm -rf data/raw/*.old
```

### Download Failure
```python
# Force re-download specific file
from src.data_utils import download_bam_file

bam_path, bai_path = download_bam_file(
    "NA12878",
    chromosome="20",
    force=True  # Re-download
)
```

### Slow Access
- BAM files must be indexed (.bai files)
- Keep BAM and BAI in same directory
- Use pysam for efficient random access

## Resources

- [1000 Genomes Project](https://www.internationalgenome.org/)
- [AWS Open Data Registry](https://registry.opendata.aws/1000-genomes/)
- [GIAB Consortium](https://www.nist.gov/programs-projects/genome-bottle)
- [pysam Documentation](https://pysam.readthedocs.io/)
- [SAM/BAM Format Specification](https://samtools.github.io/hts-specs/)

## Notes

- BAM files are binary (not human-readable)
- Use samtools or pysam to inspect BAM contents
- VCF files contain variant calls (text format, can be compressed)
- Reference genome must match BAM alignment (GRCh37/hg19)
- GIAB truth sets are high-confidence but not comprehensive

---

**Storage Tip:** 10GB may seem large, but this is real research data! Compare to climate science projects that use similar amounts. The persistence is what makes Studio Lab powerful - download once, analyze forever.
