# Ocean Monitoring Analysis - Studio Lab (Free Tier)

**Start here!** This is the free, no-AWS-account-required version perfect for learning cloud-based ocean science.

## What You'll Build

Analyze multi-sensor ocean data for comprehensive marine monitoring:
- 3 sensor types (satellite ocean color, Argo floats, acoustic)
- Sensor fusion and spatiotemporal alignment
- Ocean state prediction with uncertainty
- Marine anomaly detection
- Publication-quality visualizations
- Complete reproducible workflow

**Time to complete**: 4-6 hours (including setup and exploration)

---

## Prerequisites

### Required
- **SageMaker Studio Lab account** ([request here](https://studiolab.sagemaker.aws))
  - Free forever
  - No AWS account needed
  - No credit card required
  - Approval time varies (instant to several days)

### Knowledge
- **Python basics**: Variables, functions, loops
- **Marine science**: Basic understanding of ocean monitoring
- **Optional**: Experience with Jupyter notebooks, xarray, matplotlib

### Hardware Requirements
**None!** Everything runs in the cloud.

---

## Quick Start (3 Steps)

### Step 1: Get Studio Lab Access

1. Go to https://studiolab.sagemaker.aws
2. Click "Request account"
3. Fill out form with your email and use case
4. Wait for approval email (check spam folder)
5. Activate account from email link

### Step 2: Launch Environment

1. Log in to Studio Lab
2. Click "Start runtime" (CPU instance)
3. Wait for runtime to start (~2 minutes)
4. Click "Open project"

### Step 3: Run the Notebook

In the Studio Lab terminal:

```bash
# Clone repository
git clone https://github.com/research-jumpstart/research-jumpstart.git

# Navigate to project
cd research-jumpstart/projects/marine-science/ocean-analysis/studio-lab

# Create conda environment (one-time setup, ~10 minutes)
conda env create -f environment.yml

# Activate environment
conda activate ocean-monitoring

# Start Jupyter
jupyter notebook quickstart.ipynb
```

Then click **Run All** in the notebook menu!

---

## What's Included

### Files

- **quickstart.ipynb** - Main analysis notebook (600+ lines)
- **environment.yml** - Conda environment with all dependencies
- **README.md** - This file

### Analysis Workflow

The notebook walks you through:

1. **Setup & Configuration** (5 min)
   - Import packages
   - Define analysis parameters
   - Set region and time periods

2. **Multi-Sensor Data Generation** (10 min)
   - Simulate satellite ocean color
   - Simulate Argo float profiles
   - Simulate acoustic sensor data
   - Realistic patterns for education

3. **Sensor Fusion** (15 min)
   - Spatiotemporal alignment
   - Resolution harmonization
   - Quality control
   - Unified ocean state dataset

4. **Ocean State Modeling** (20 min)
   - CNN for satellite patterns
   - LSTM for Argo time series
   - Ensemble prediction
   - Uncertainty quantification

5. **Anomaly Detection** (15 min)
   - Marine heatwave identification
   - Algal bloom detection
   - Unusual acoustic patterns
   - Temporal trend analysis

6. **Visualization** (15 min)
   - Spatiotemporal ocean maps
   - Multi-sensor time series
   - Ensemble predictions with uncertainty
   - Publication-quality figures

7. **Interpretation** (10 min)
   - Summary statistics tables
   - Key marine findings
   - Regional ocean health
   - Next steps

---

## Customization Guide

### Change the Region

In the notebook, modify the `REGION` dictionary:

```python
REGION = {
    'name': 'Your Ocean Region',
    'lat_min': 20.0,   # Southern boundary
    'lat_max': 40.0,   # Northern boundary
    'lon_min': -140.0, # Western boundary
    'lon_max': -120.0  # Eastern boundary
}
```

**Example regions**:
- California Current: `lat: 32-42, lon: -130 to -115`
- Gulf of Mexico: `lat: 18-31, lon: -98 to -80`
- North Atlantic: `lat: 35-55, lon: -50 to -10`
- Coral Triangle: `lat: -10-20, lon: 110-140`
- Antarctic Peninsula: `lat: -70 to -60, lon: -70 to -50`

### Change the Time Period

```python
# Analysis period
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'

# Baseline for anomaly detection
BASELINE_START = '2020-01-01'
BASELINE_END = '2022-12-31'
```

### Adjust Sensor Parameters

Each sensor type has tunable parameters:

```python
# Satellite ocean color
chlorophyll_baseline = 0.5  # mg/m³
sst_baseline = 15.0         # °C

# Argo float sampling
profile_depth = 2000        # meters
sampling_frequency = 10     # days

# Acoustic detection
frequency_range = (100, 2000)  # Hz
detection_threshold = 0.3   # relative
```

---

## Understanding the Output

### Key Metrics Explained

**Ensemble Ocean State**
- Fusion of multiple sensor types
- Best estimate of true ocean conditions
- Captures spatial and temporal patterns

**Sensor Uncertainty**
- Confidence in each sensor measurement
- Larger uncertainty = less reliable
- Ensemble reduces overall uncertainty

**Anomaly Score**
- Deviation from baseline conditions
- Marine heatwaves: SST anomaly > +2°C
- Algal blooms: Chlorophyll anomaly > 2x baseline
- Acoustic events: Unusual sound patterns

### Example Results

For California Current (2020-2024):
- **SST trend**: +0.15°C/year warming
- **Marine heatwave events**: 3 detected (2023-2024)
- **Chlorophyll variability**: 30-40% seasonal
- **Acoustic diversity**: 8 distinct sound clusters
- **Ensemble accuracy**: 85% validation

---

## Limitations (Studio Lab Version)

**What's different from production**:

| Feature | Studio Lab | Unified Studio |
|---------|------------|----------------|
| Data source | Simulated | Real multi-sensor |
| Sensors | 3 types | 10+ sensor types |
| Region | Pre-configured | Any ocean region |
| Variables | 5 basic | 50+ variables |
| Temporal res | Monthly | Daily/hourly |
| Compute | 4GB RAM | 32GB+ RAM |
| Storage | 15GB, session | Unlimited S3 |
| AI features | None | Bedrock (Claude 3) |
| Cost | $0 | $20-40 per analysis |

**Studio Lab is perfect for**:
- Learning ocean data fusion
- Teaching marine monitoring
- Testing analysis workflows
- Prototyping research ideas
- Understanding ensemble methods
- Educational demonstrations

**Transition to production when you need**:
- Real multi-sensor ocean data
- More than 3 sensor types
- Global or high-resolution analysis
- Long time series (decades)
- Collaboration with team
- AI-assisted ocean insights

---

## Troubleshooting

### Environment Creation Fails

**Problem**: `conda env create -f environment.yml` errors

**Solutions**:
1. Check available disk space:
   ```bash
   df -h
   ```
   Need at least 2GB free.

2. Clean conda cache:
   ```bash
   conda clean --all -y
   ```

3. Try again:
   ```bash
   conda env create -f environment.yml
   ```

4. If still failing, install packages individually:
   ```bash
   conda create -n ocean-monitoring python=3.10 -y
   conda activate ocean-monitoring
   pip install numpy pandas xarray matplotlib cartopy torch
   ```

### Kernel Dies During Execution

**Problem**: Kernel crashes when running cells

**Cause**: Out of memory (4GB limit)

**Solutions**:
1. Restart kernel: Kernel → Restart
2. Clear outputs: Cell → All Output → Clear
3. Reduce data size:
   ```python
   # Change in notebook
   START_DATE = '2022-01-01'  # Shorter period
   REGION_SIZE = 'small'      # Smaller region
   ```

### Cartopy Import Error

**Problem**: `ImportError: cannot import name 'cartopy'`

**Solution**:
```bash
conda activate ocean-monitoring
conda install -c conda-forge cartopy=0.22.0 -y
```

### Session Expires

**Problem**: 12-hour session limit reached

**Solution**:
1. Save your work first!
2. Stop runtime from Studio Lab dashboard
3. Start runtime again
4. Environment persists, just activate:
   ```bash
   conda activate ocean-monitoring
   cd research-jumpstart/projects/marine-science/ocean-analysis/studio-lab
   jupyter notebook quickstart.ipynb
   ```

---

## Next Steps

### After Completing This Notebook

**Experiment**:
1. Change the region to your area of interest
2. Modify time periods (different seasons)
3. Adjust sensor parameters
4. Add your own analysis methods

**Learn More**:
1. Read the code carefully - it's well-documented
2. Modify functions to understand sensor fusion
3. Try additional ocean variables
4. Read about multi-sensor oceanography

**Scale Up**:
When ready for production with real data:
1. Review the main [Project README](../README.md)
2. Set up AWS account (if needed)
3. Follow [Unified Studio setup](../unified-studio/README.md)
4. Port your analysis code (minimal changes needed)

### Related Resources

**Ocean Science**:
- [NOAA Ocean Data](https://www.ncei.noaa.gov/products/ocean)
- [Argo Float Program](https://argo.ucsd.edu/)
- [NASA Ocean Color](https://oceancolor.gsfc.nasa.gov/)

**Tools & Methods**:
- [Xarray Tutorial](https://tutorial.xarray.dev/)
- [Pangeo Ocean](https://pangeo.io/)
- [OBIS](https://obis.org/)

**Research Jumpstart**:
- [Browse other projects](../../../../docs/projects/index.md)
- [Platform comparison](../../../../docs/getting-started/platform-comparison.md)
- [FAQ](../../../../docs/resources/faq.md)

---

## Getting Help

### Project-Specific Questions

- **GitHub Issues**: https://github.com/research-jumpstart/research-jumpstart/issues
- **Discussions**: https://github.com/research-jumpstart/research-jumpstart/discussions
- Tag: `marine-science`, `studio-lab`

### General Questions

- **Studio Lab support**: studiolab-support@amazon.com
- **Pangeo Discourse**: https://discourse.pangeo.io/
- **Stack Overflow**: Tags `xarray`, `oceanography`, `marine-science`

---

## Acknowledgments

This educational version uses simulated data for demonstration purposes. Production version accesses real multi-sensor ocean data.

**Thanks to**:
- NOAA for ocean monitoring data
- Argo Program for float data
- NASA for satellite observations
- Research Jumpstart contributors

---

## License

Apache License 2.0 - see [LICENSE](../../../../LICENSE)

---

*Last updated: 2025-11-13 | Studio Lab Free Tier Version*
