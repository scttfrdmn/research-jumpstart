# Data Directory

This directory stores local copies of FITS images and test data for the Tier 2 project.

## Structure

```
data/
├── raw/                 # Downloaded FITS images
│   ├── sdss_test_frame_g.fits
│   ├── sdss_test_frame_r.fits
│   ├── sdss_test_frame_i.fits
│   └── metadata.json
└── README.md            # This file
```

## Getting Data

### Download Sample Images

```bash
python scripts/download_sample_fits.py
```

This downloads synthetic FITS test images (~50 MB).

For production SDSS data, see: https://www.sdss.org/

### Upload to S3

```bash
python scripts/upload_to_s3.py
```

## FITS Format

FITS (Flexible Image Transport System) is the standard format for astronomical data.

**Structure:**
- Primary HDU (Header + Data)
- Header: Metadata (instrument, filter, coordinates, etc.)
- Data: 2D array of pixel values (typically 512x512 to 4096x4096)

## .gitignore

This directory is ignored by git to avoid storing large binary files:

```
data/raw/*.fits
data/raw/*.fits.bz2
data/*.parquet
data/*.json
```

To download data fresh, use `download_sample_fits.py` after cloning.

## Storage Limits

- Studio Lab: 15 GB total storage
- Local: Depends on available disk space
- S3: Unlimited (pay per GB stored)

## Data Lineage

1. **Source:** SDSS DR18 or test data
2. **Download:** `download_sample_fits.py`
3. **Location:** `data/raw/`
4. **Upload:** `upload_to_s3.py`
5. **S3 Bucket:** `astronomy-tier2-XXXXX-raw`
6. **Processing:** Lambda function
7. **Results:** `astronomy-tier2-XXXXX-catalog`

## Cleaning Up

To save space:

```bash
# Remove FITS files
rm -rf data/raw/*.fits

# Remove processed results
rm -rf data/*.parquet data/*.json

# But keep the directory structure
mkdir -p data/raw
```

To delete S3 data, see `cleanup_guide.md`.
