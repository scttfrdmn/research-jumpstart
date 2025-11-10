# Regional Climate Modeling - Tier 2 Complete

**Duration:** 2-3 days | **Platform:** SageMaker/Unified Studio | **Cost:** $5-10

Comprehensive regional climate analysis using CORDEX projections with downscaling, bias correction, and impact assessment.

## Overview

This project demonstrates advanced regional climate modeling techniques used by climate scientists to understand local climate change impacts. Using CORDEX (Coordinated Regional Climate Downscaling Experiment) data, you'll analyze high-resolution climate projections for a specific region and assess impacts on various sectors.

## What You'll Learn

### Core Climate Science
- Regional climate downscaling methods
- Bias correction techniques (quantile mapping, delta method)
- Extreme event analysis (heat waves, droughts, floods)
- Climate change impact assessment
- Model ensemble analysis

### Technical Skills
- Processing NetCDF climate data
- Spatial analysis with xarray and rasterio
- Statistical downscaling
- Time series analysis for climate
- Geospatial visualization

## Dataset

**Primary Source:** CORDEX Regional Climate Models
- **Resolution:** 0.44° (~50km) or 0.11° (~12km)
- **Domain:** North America or Europe
- **Variables:** Temperature (tas), Precipitation (pr), Wind, Humidity
- **Scenarios:** RCP4.5, RCP8.5
- **Time Period:** Historical (1970-2005) + Future (2006-2100)
- **Models:** 5-10 regional models driven by global CMIP5/CMIP6

**Reference Data:**
- Observational gridded data (CRU, GPCC, ERA5)
- Station data for validation
- Historical climate normals

## Key Analyses

### 1. Data Processing & Quality Control
- Load and preprocess CORDEX NetCDF files
- Extract region of interest
- Calculate derived variables (extreme indices)
- Quality checks and missing data handling

### 2. Bias Correction
- Quantile mapping for temperature and precipitation
- Delta method for climate change signals
- Validation against observational data
- Uncertainty quantification

### 3. Climate Change Trends
- Temperature warming rates by season
- Precipitation changes (amount and intensity)
- Changes in climate extremes
- Multi-model ensemble statistics

### 4. Extreme Event Analysis
- Heat wave frequency, duration, intensity
- Drought indices (SPI, SPEI)
- Heavy precipitation events
- Growing season length changes

### 5. Sector Impact Assessment
- **Agriculture:** Growing degree days, frost dates, water stress
- **Water Resources:** Runoff changes, drought risk
- **Energy:** Heating/cooling degree days
- **Health:** Heat stress days, cold exposure

### 6. Spatial Mapping
- Regional climate change maps
- Hotspot identification
- Urban vs rural differences
- Elevation-dependent warming

## Project Structure

```
regional-climate-modeling/
├── README.md
├── notebooks/
│   ├── 01_data_loading.ipynb          # Load CORDEX data
│   ├── 02_bias_correction.ipynb       # Bias correction methods
│   ├── 03_trend_analysis.ipynb        # Climate change trends
│   ├── 04_extremes.ipynb              # Extreme event analysis
│   ├── 05_impacts.ipynb               # Sector impacts
│   └── 06_visualization.ipynb         # Maps and final reports
├── data/
│   ├── cordex_sample/                 # Sample CORDEX files
│   ├── observations/                  # Reference data
│   └── processed/                     # Processed outputs
├── scripts/
│   ├── download_cordex.py             # Data acquisition
│   ├── bias_correction.py             # BC algorithms
│   └── compute_indices.py             # Climate indices
└── environment.yml
```

## Setup Instructions

### Option 1: SageMaker Studio

```bash
# Clone repository in SageMaker Studio terminal
git clone https://github.com/your-org/research-jumpstart.git
cd research-jumpstart/projects/climate-science/regional-climate-modeling

# Create conda environment
conda env create -f environment.yml
conda activate regional-climate

# Launch Jupyter
jupyter lab
```

### Option 2: Unified Studio

This project is pre-configured for AWS Unified Studio with:
- Automatic environment provisioning
- Data lake integration
- Compute scaling for large datasets

## Data Access

### Downloading CORDEX Data

CORDEX data is available from:
- **ESGF Nodes:** https://esgf-node.llnl.gov/
- **Copernicus Climate Data Store:** https://cds.climate.copernicus.eu/
- **Direct download scripts provided**

Sample commands:
```bash
# Download using provided script
python scripts/download_cordex.py --domain NAM --scenario rcp85 --variable tas

# Or use wget with ESGF
wget --user=your_user --password=your_pass [ESGF_URL]
```

## Key Deliverables

### Analysis Outputs
1. **Bias-corrected climate projections** (NetCDF files)
2. **Climate change signal maps** (temperature, precipitation)
3. **Extreme event statistics** (frequency, intensity tables)
4. **Sector impact reports** (agriculture, water, energy)
5. **Interactive visualizations** (time series, maps, comparisons)

### Visualizations
- Spatial maps of climate change
- Time series of regional averages
- Probability density functions (PDFs)
- Multi-model ensemble spreads
- Impact scorecards by sector

## Scientific Methods

### Bias Correction Approaches

**Quantile Mapping:**
- Matches distribution of modeled values to observations
- Preserves climate change signal
- Applied separately by season

**Delta Method:**
- Applies model change signal to observations
- Simpler approach for temperature
- Good for large-scale trends

### Extreme Indices (ETCCDI)

- **TX90p:** Days with Tmax > 90th percentile
- **TN10p:** Days with Tmin < 10th percentile
- **R20mm:** Days with precip > 20mm
- **CDD:** Consecutive dry days
- **WSDI:** Warm spell duration index

## Computational Requirements

### Processing Time
- **Data download:** 1-2 hours (depends on region/models)
- **Bias correction:** 2-4 hours (per model)
- **Analysis:** 4-8 hours (all notebooks)
- **Total:** 1-2 days with SageMaker ml.m5.2xlarge

### Storage
- **Raw CORDEX data:** 5-20 GB
- **Processed data:** 2-10 GB
- **Outputs:** 1-2 GB

### Recommended Instance
- **SageMaker:** ml.m5.2xlarge or ml.m5.4xlarge
- **Memory:** 16-32 GB recommended
- **Storage:** 50 GB EBS volume

## Cost Estimate

**Total: $5-10 for complete project**

- Data download: Free (public data)
- Compute (ml.m5.2xlarge @ $0.461/hr): ~$4-6
- Storage (50 GB @ $0.10/GB-month): ~$1-2
- Data transfer: Minimal

## Extensions & Advanced Topics

### Model Weighting
- Performance-based weighting
- Independence weighting
- Bayesian approaches

### High-Resolution Downscaling
- Statistical downscaling (SDSM, LARS-WG)
- Dynamical downscaling (WRF)
- Machine learning approaches

### Compound Events
- Concurrent heat and drought
- Precipitation intensity during warm periods
- Multi-variable extremes

### Attribution Science
- Detection and attribution
- Fraction of attributable risk (FAR)
- Return period changes

## References

### Key Papers
- Giorgi et al. (2009): RegCM3 regional climate modeling
- Teutschbein & Seibert (2012): Bias correction methods
- Jacob et al. (2014): EURO-CORDEX assessment
- Mearns et al. (2017): NA-CORDEX results

### Data Documentation
- CORDEX: https://cordex.org/
- ESGF: https://esgf.llnl.gov/
- Climate indices: https://www.climdex.org/

### Tools
- **xarray:** NetCDF processing
- **xclim:** Climate indicators
- **bias-correction:** Statistical corrections
- **cartopy:** Geospatial plotting

## Support

For issues or questions:
- Check CORDEX documentation
- Review sample notebooks
- Contact research-jumpstart maintainers

## License

Data: Check CORDEX terms of use (typically CC BY 4.0)
Code: MIT License

---

*Last updated: 2025-11-10*
*Part of AWS Research Jumpstart initiative*
