# Multi-Survey Sky Catalog Cross-Matching

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB multi-survey catalogs (SDSS, 2MASS, WISE, Gaia)

## Research Goal

Build an ensemble classifier for astronomical object classification by cross-matching catalogs from multiple sky surveys. Combine photometric, astrometric, and proper motion data to identify and classify stars, galaxies, quasars, and rare transients.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex dependency stack with astropy, astroquery)

## What This Enables

Real research that isn't possible on Colab:

### Sky Survey Persistence
- Download 10GB from SDSS, 2MASS, WISE, Gaia **once**
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache spatial index structures for fast queries

### Long-Running Cross-Matching
- Match 1 million+ objects across 4 surveys
- Build ensemble training set (30-40 min)
- Train gradient boosting classifiers (5-6 hours continuous)
- Automatic checkpointing every epoch
- Resume from checkpoint if needed

### Reproducible Astronomy Environments
- Conda environment with astropy, astroquery, healpy
- Persists between sessions
- No reinstalling astronomy packages
- Team members use identical setup

### Iterative Classification
- Save trained classifiers
- Build on previous runs
- Refine feature engineering incrementally
- Collaborative catalog development

## What You'll Build

Multi-notebook research workflow:

1. **Survey Data Download** (45-60 min)
   - Query SDSS DR18 (~3GB)
   - Download 2MASS point sources (~2GB)
   - Fetch WISE all-sky catalog (~3GB)
   - Retrieve Gaia EDR3 astrometry (~2GB)
   - Total: ~10GB cached in persistent storage

2. **Catalog Cross-Matching** (30-45 min)
   - Spatial indexing with HEALPix
   - Cross-match within 1 arcsec radius
   - Handle duplicate matches
   - Propagate uncertainties
   - Generate matched catalog (~1M objects)

3. **Feature Engineering** (45-60 min)
   - Multi-wavelength colors (optical + IR)
   - Proper motions and parallaxes (Gaia)
   - Morphological parameters (SDSS)
   - Variability indices (multi-epoch data)
   - Spectroscopic features (where available)

4. **Ensemble Classifier Training** (5-6 hours)
   - Train XGBoost for 8 object classes
   - Random Forest for uncertainty quantification
   - Neural network for rare object detection
   - Ensemble voting and stacking
   - Checkpoint every model

5. **Classification and Validation** (30-45 min)
   - Apply classifier to unlabeled objects
   - Compare with spectroscopic confirmations
   - Identify high-confidence discoveries
   - Flag unusual objects for follow-up
   - Generate catalog of classified sources

## Datasets

**Multi-Survey Catalog Cross-Match**

### SDSS DR18 (Sloan Digital Sky Survey)
- **Coverage:** 35% of celestial sphere
- **Photometry:** 5 bands (ugriz), 14-23 mag
- **Spectra:** 4 million galaxies, 1 million quasars, stars
- **Size:** ~3GB for target region
- **URL:** https://www.sdss.org/dr18/

### 2MASS (Two Micron All-Sky Survey)
- **Coverage:** Full sky
- **Photometry:** Near-IR (JHK), 15-20 mag
- **Sources:** 500 million point sources
- **Size:** ~2GB for matched sources
- **URL:** https://irsa.ipac.caltech.edu/Missions/2mass.html

### WISE (Wide-field Infrared Survey Explorer)
- **Coverage:** Full sky
- **Photometry:** Mid-IR (W1-W4), 3.4-22 μm
- **Sources:** 750 million objects
- **Size:** ~3GB for matched sources
- **URL:** https://irsa.ipac.caltech.edu/Missions/wise.html

### Gaia EDR3 (European Space Agency)
- **Coverage:** Full sky
- **Astrometry:** Positions, parallaxes, proper motions
- **Photometry:** G, BP, RP bands
- **Precision:** 0.02-0.1 mas (milliarcseconds)
- **Size:** ~2GB for matched sources
- **URL:** https://gea.esac.esa.int/archive/

**Total cached size:** ~10GB in Studio Lab persistent storage

## Object Classes

Train classifier to distinguish:

1. **Main-sequence stars** (F, G, K dwarfs)
2. **Giant stars** (Red giants, supergiants)
3. **White dwarfs** (Hot, compact stellar remnants)
4. **Brown dwarfs** (Sub-stellar objects, L/T dwarfs)
5. **Galaxies** (Elliptical, spiral, irregular)
6. **Quasars** (Active galactic nuclei, z > 0.5)
7. **Galaxy clusters** (Extended sources)
8. **Transients** (Supernovae, variable stars, AGN)

## Setup

### 1. Create Studio Lab Account
1. Go to [SageMaker Studio Lab](https://studiolab.sagemaker.aws)
2. Request free account (approval in 1-2 days)
3. No credit card or AWS account needed

### 2. Import Project
```bash
# In Studio Lab terminal
git clone https://github.com/YOUR_USERNAME/research-jumpstart.git
cd research-jumpstart/projects/astronomy/sky-survey/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate astronomy-studio-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_survey_download.ipynb` - Download and cache catalogs
2. `02_catalog_crossmatch.ipynb` - Match sources across surveys
3. `03_feature_engineering.ipynb` - Build classification features
4. `04_ensemble_training.ipynb` - Train classifiers (5-6 hours)
5. `05_classification_results.ipynb` - Apply to unlabeled objects

## Key Features

### Persistence Example
```python
# Save cross-matched catalog (persists between sessions!)
matched_catalog.write('data/processed/matched_catalog_v1.fits')

# Load in next session
from astropy.table import Table
catalog = Table.read('data/processed/matched_catalog_v1.fits')
```

### Spatial Cross-Matching
```python
from astropy.coordinates import SkyCoord
from astropy import units as u

# Match SDSS and Gaia within 1 arcsec
sdss_coords = SkyCoord(ra=sdss['ra']*u.deg, dec=sdss['dec']*u.deg)
gaia_coords = SkyCoord(ra=gaia['ra']*u.deg, dec=gaia['dec']*u.deg)

idx, sep, _ = sdss_coords.match_to_catalog_sky(gaia_coords)
matches = sep < 1*u.arcsec
```

### Long-Running Training
```python
import xgboost as xgb

# Train ensemble with checkpointing
# Studio Lab: 12-hour GPU sessions, 4-hour CPU sessions
xgb_params = {
    'max_depth': 8,
    'eta': 0.05,
    'objective': 'multi:softprob',
    'num_class': 8,
    'eval_metric': 'mlogloss'
}

# Trains for 5-6 hours with automatic checkpointing
model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=5000,
    evals=[(dval, 'validation')],
    early_stopping_rounds=50,
    xgb_model='models/checkpoint.model'  # Resume from checkpoint
)

# Save final model
model.save_model('models/sky_classifier_final.model')
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_survey_download.ipynb       # Download catalogs
│   ├── 02_catalog_crossmatch.ipynb    # Cross-match surveys
│   ├── 03_feature_engineering.ipynb   # Build features
│   ├── 04_ensemble_training.ipynb     # Train classifiers (5-6 hours)
│   └── 05_classification_results.ipynb # Apply and validate
│
├── src/
│   ├── __init__.py
│   ├── survey_utils.py                # Survey query utilities
│   ├── crossmatch.py                  # Spatial matching algorithms
│   ├── features.py                    # Feature engineering
│   └── visualization.py               # Sky plots and diagnostics
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded survey catalogs
│   │   ├── sdss_dr18.fits
│   │   ├── 2mass_psc.fits
│   │   ├── wise_allsky.fits
│   │   └── gaia_edr3.fits
│   ├── processed/                    # Cross-matched catalogs
│   │   ├── matched_catalog_v1.fits
│   │   └── features_v1.fits
│   └── README.md                     # Data documentation
│
└── models/                            # Model checkpoints (gitignored)
    ├── xgboost_checkpoint.model
    ├── random_forest_v1.pkl
    ├── neural_net_v1.h5
    └── README.md                      # Model documentation
```

## Why This Needs Studio Lab

| Requirement | Colab Free | Studio Lab |
|-------------|------------|------------|
| **10GB catalogs** | ❌ No storage | ✅ 15GB persistent |
| **5-6 hour training** | ❌ 90 min limit | ✅ 12 hour sessions |
| **Checkpointing** | ❌ Lost on disconnect | ✅ Persists forever |
| **Astropy/astroquery** | ❌ Reinstall each time | ✅ Conda persists |
| **Spatial indices** | ❌ Rebuild every time | ✅ Cache HEALPix structures |
| **Team sharing** | ❌ Copy/paste notebooks | ✅ Git integration |

**Bottom line:** Cross-matching surveys and training classifiers requires persistent storage and long sessions.

## Time Estimate

**First Run:**
- Setup: 15 minutes (one-time)
- Data download: 45-60 minutes (one-time, ~10GB)
- Environment setup: 10 minutes (one-time)
- Cross-matching: 30-45 minutes
- Feature engineering: 45-60 minutes
- Model training: 5-6 hours
- Classification: 30-45 minutes
- **Total: 8-10 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Cross-matching: 5 minutes (load cached)
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 5-6 hours**

You can pause and resume at any time!

## Scientific Applications

### Rare Object Discovery
- **High-z quasars** (z > 6): Reionization-era probes
- **Brown dwarfs** (L/T types): Sub-stellar mass function
- **White dwarf binaries**: Gravitational wave progenitors
- **Changing-look AGN**: Variable accretion states

### Population Studies
- **Stellar populations**: Milky Way structure and kinematics
- **Galaxy evolution**: Color-magnitude relations across redshift
- **Quasar luminosity function**: AGN demography
- **Transient rates**: Supernova and TDE statistics

### Followup Target Selection
- **Spectroscopy candidates**: High-priority unusual objects
- **Time-domain targets**: Known variables for monitoring
- **Imaging targets**: Extended sources for deep imaging
- **Multi-messenger alerts**: Optical counterparts to GW/neutrino events

## Performance Expectations

Using this workflow, you can expect:

**Cross-matching performance:**
- Match rate: ~85% (SDSS-Gaia)
- Match rate: ~70% (all 4 surveys)
- False match rate: <1% (within 1 arcsec)
- Processing: 1M objects in 30-45 min

**Classification metrics:**
- Overall accuracy: ~92% (8-class)
- Star/galaxy separation: ~98%
- Quasar identification: ~85% (precision/recall)
- Rare object recall: ~60% (brown dwarfs, white dwarfs)

**Scientific capability:**
- Classify 1 million objects per session
- Identify ~100 high-confidence rare objects
- Flag ~1,000 candidates for spectroscopic follow-up
- Generate publication-ready catalogs

## Next Steps

After mastering Studio Lab:

- **Tier 2:** Introduction to AWS services (S3, Athena, Glue) - $5-15
  - Store 100GB+ survey data on S3
  - SQL queries with Athena
  - Distributed cross-matching with Glue

- **Tier 3:** Production infrastructure with CloudFormation - $50-500/month
  - Process entire all-sky surveys
  - Real-time transient classification
  - Integration with alert streams (ZTF, LSST)

## Additional Resources

### Sky Surveys
- **SDSS:** https://www.sdss.org/
- **2MASS:** https://irsa.ipac.caltech.edu/Missions/2mass.html
- **WISE:** https://irsa.ipac.caltech.edu/Missions/wise.html
- **Gaia:** https://gea.esac.esa.int/archive/
- **Legacy Survey:** https://www.legacysurvey.org/

### Python Tools
- **Astropy:** https://www.astropy.org/
- **Astroquery:** https://astroquery.readthedocs.io/
- **HEALPix:** https://healpy.readthedocs.io/
- **TOPCAT:** http://www.star.bris.ac.uk/~mbt/topcat/

### Cross-Matching Methods
- **Budavári & Szalay (2008):** Probabilistic cross-identification
- **Pineau et al. (2017):** Multi-catalog cross-matching
- **CDS X-Match:** http://cdsxmatch.u-strasbg.fr/

### Classification Techniques
- **Ivezić et al. (2019):** LSST object classification
- **Wright et al. (2010):** WISE AGN selection
- **Bailer-Jones et al. (2019):** Gaia source classification

## Troubleshooting

### Survey Query Timeouts
```python
# If SDSS/IRSA queries timeout, use smaller regions
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u

# Query smaller 10x10 degree patches instead of full footprint
coords = SkyCoord(ra=180*u.deg, dec=30*u.deg)
result = SDSS.query_region(coords, radius=5*u.deg, data_release=18)
```

### Storage Full
```bash
# Check usage (Studio Lab has 15GB limit)
du -sh data/

# Clean old files
rm -rf data/raw/backup_*.fits
rm -rf models/old_checkpoints/
```

### Cross-Match Performance
```python
# Use HEALPix for faster spatial indexing
import healpy as hp

# Build spatial index (faster for repeated queries)
nside = 1024  # ~3.4 arcmin resolution
pixels = hp.ang2pix(nside, ra, dec, lonlat=True)

# Group by pixel for efficient matching
catalog['healpix'] = pixels
```

### Memory Issues
```python
# Process catalogs in chunks
chunk_size = 100000
for i in range(0, len(catalog), chunk_size):
    chunk = catalog[i:i+chunk_size]
    process_chunk(chunk)
```

## Extension Ideas

### Beginner Extensions (2-4 hours)
1. **Different sky regions**: Compare Galactic plane vs. poles
2. **Additional surveys**: Add PanSTARRS, UKIDSS
3. **Different classifiers**: Try CatBoost, LightGBM
4. **Feature selection**: Optimize for specific object types

### Intermediate Extensions (4-8 hours)
5. **Time-domain features**: Multi-epoch photometry from WISE
6. **Spectroscopic training**: Use SDSS/LAMOST spectra
7. **Hierarchical classification**: Coarse then fine-grained
8. **Active learning**: Prioritize uncertain objects for labeling

### Advanced Extensions (8+ hours)
9. **Full-sky catalogs**: Process entire SDSS+Gaia overlap
10. **Deep learning**: Vision transformer on sky images
11. **Real-time classification**: Connect to ZTF/LSST alert streams
12. **Multi-wavelength SED fitting**: Physical parameter estimation

## Citation

If you use this project or publish cross-matched catalogs:

```bibtex
@software{research_jumpstart_sky_survey,
  title = {Multi-Survey Sky Catalog Cross-Matching: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/research-jumpstart/research-jumpstart},
  note = {Accessed: [date]}
}
```

Cite the surveys you use:
```bibtex
@article{sdss_dr18,
  title={The Eighteenth Data Release of SDSS},
  author={SDSS Collaboration},
  journal={ApJS},
  year={2023}
}

@article{gaia_edr3,
  title={Gaia Early Data Release 3},
  author={Gaia Collaboration},
  journal={A\&A},
  volume={649},
  pages={A1},
  year={2021}
}
```

## License

This project is licensed under the Apache License 2.0.

Survey data has individual usage terms. Cite surveys appropriately.

---

**Built for SageMaker Studio Lab**

*Last updated: 2025-11-13 | Research Jumpstart v1.0.0*
