# Sky Survey Data Storage

This directory stores downloaded and cross-matched astronomical catalogs. Data persists between SageMaker Studio Lab sessions!

## Directory Structure

```
data/
├── raw/                    # Original survey catalogs
│   ├── sdss_dr18.fits
│   ├── gaia_edr3.fits
│   ├── 2mass_psc.fits
│   └── wise_allsky.fits
│
└── processed/              # Cross-matched and processed catalogs
    ├── matched_catalog_v1.fits
    ├── features_v1.fits
    └── training_set_v1.fits
```

## Datasets

### SDSS DR18 (Sloan Digital Sky Survey)
- **Source:** https://www.sdss.org/dr18/
- **Coverage:** 35% of sky, 500 million objects
- **Data:** Optical photometry (ugriz), morphology, spectra
- **Size:** ~3GB for target region
- **Download time:** 15-20 minutes

### Gaia EDR3 (European Space Agency)
- **Source:** https://gea.esac.esa.int/archive/
- **Coverage:** Full sky, 1.8 billion sources
- **Data:** Astrometry (positions, proper motions, parallaxes), photometry (GBP/RP)
- **Size:** ~2GB for matched sources
- **Download time:** 10-15 minutes

### 2MASS (Two Micron All-Sky Survey)
- **Source:** https://irsa.ipac.caltech.edu/Missions/2mass.html
- **Coverage:** Full sky, 500 million point sources
- **Data:** Near-IR photometry (JHK bands)
- **Size:** ~2GB for matched sources
- **Download time:** 10-15 minutes

### WISE (Wide-field Infrared Survey Explorer)
- **Source:** https://irsa.ipac.caltech.edu/Missions/wise.html
- **Coverage:** Full sky, 750 million sources
- **Data:** Mid-IR photometry (W1-W4: 3.4, 4.6, 12, 22 μm)
- **Size:** ~3GB for matched sources
- **Download time:** 15-20 minutes

**Total storage:** ~10GB

## Usage

Data is automatically downloaded and cached on first use:

```python
from src.survey_utils import query_sdss, query_gaia, save_catalog

# First run: downloads and caches
sdss = query_sdss(ra_center=180, dec_center=30, radius_deg=5)
save_catalog(sdss, 'data/raw/sdss_dr18.fits')

# Subsequent runs: load from cache
from src.survey_utils import load_catalog
sdss = load_catalog('data/raw/sdss_dr18.fits')  # Instant!
```

## Cross-Matching

Build matched catalog across surveys:

```python
from src.crossmatch import build_multi_survey_catalog
from src.survey_utils import load_catalog

# Load individual catalogs
sdss = load_catalog('data/raw/sdss_dr18.fits')
gaia = load_catalog('data/raw/gaia_edr3.fits')
tmass = load_catalog('data/raw/2mass_psc.fits')
wise = load_catalog('data/raw/wise_allsky.fits')

# Cross-match (takes 30-45 min for 1M sources)
matched = build_multi_survey_catalog(sdss, gaia, tmass, wise,
                                     max_sep_arcsec=1.0)

# Save matched catalog
save_catalog(matched, 'data/processed/matched_catalog_v1.fits')
```

## Storage Management

Check current usage:
```bash
du -sh data/
du -sh data/raw/
du -sh data/processed/
```

Clean old files:
```bash
# Remove backup files
rm -rf data/raw/*.backup
rm -rf data/processed/*_old.fits

# Clean temporary files
rm -rf data/tmp/
```

## Persistence

Studio Lab persistent storage details:
- Total available: 15GB
- This project uses: ~10GB
- Remaining: ~5GB for models and outputs
- Data survives session restarts
- Data shared across all notebooks

## File Formats

All catalogs stored as FITS (Flexible Image Transport System):
- Standard astronomy format
- Preserves metadata and data types
- Readable by astropy, TOPCAT, DS9
- Compressed for efficient storage

## Notes

- FITS files preserve all column metadata
- Masked values handled automatically
- Cross-match separations stored for QA
- Version numbers track catalog updates
- .gitignore excludes data/ from version control
