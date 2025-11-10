# Climate Ensemble Analysis - Studio Lab (Free Tier)

**Start here!** This is the free, no-AWS-account-required version perfect for learning cloud-based climate analysis.

## What You'll Build

Analyze climate model projections for any region using:
- 3 representative CMIP6 models (CESM2, GFDL-CM4, UKESM1-0-LL)
- Ensemble statistics (mean, spread, model agreement)
- Temperature anomaly analysis
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
- **Climate science**: Basic understanding of climate models
- **Optional**: Experience with Jupyter notebooks, pandas, matplotlib

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
cd research-jumpstart/projects/climate-science/ensemble-analysis/studio-lab

# Create conda environment (one-time setup, ~10 minutes)
conda env create -f environment.yml

# Activate environment
conda activate climate-analysis

# Start Jupyter
jupyter notebook quickstart.ipynb
```

Then click **Run All** in the notebook menu!

---

## What's Included

### Files

- **quickstart.ipynb** - Main analysis notebook (500+ lines)
- **environment.yml** - Conda environment with all dependencies
- **README.md** - This file

### Analysis Workflow

The notebook walks you through:

1. **Setup & Configuration** (5 min)
   - Import packages
   - Define analysis parameters
   - Set region and time periods

2. **Data Generation** (2 min)
   - Simulate 3 CMIP6 models
   - Realistic temperature patterns
   - Educational data that runs fast

3. **Regional Mean Calculation** (5 min)
   - Area-weighted spatial averaging
   - Handle different latitude grid spacing
   - Extract regional time series

4. **Temperature Anomaly** (5 min)
   - Calculate relative to baseline (1995-2014)
   - Annual averaging
   - Prepare for ensemble analysis

5. **Ensemble Statistics** (10 min)
   - Multi-model mean
   - Standard deviation (uncertainty)
   - Min/max range
   - Model agreement metrics

6. **Visualization** (15 min)
   - Time series with uncertainty bands
   - Model agreement box plots
   - Publication-quality figures
   - Customizable styles

7. **Interpretation** (10 min)
   - Summary statistics tables
   - Key findings
   - Confidence assessment
   - Next steps

---

## Customization Guide

### Change the Region

In the notebook, modify the `REGION` dictionary:

```python
REGION = {
    'name': 'Your Region Name',
    'lat_min': 35.0,  # Southern boundary
    'lat_max': 45.0,  # Northern boundary
    'lon_min': -125.0,  # Western boundary (negative = West)
    'lon_max': -115.0   # Eastern boundary
}
```

**Example regions**:
- US Southwest: `lat: 31-37, lon: -114 to -109`
- Pacific Northwest: `lat: 42-49, lon: -125 to -116`
- Southeast Asia: `lat: -10-20, lon: 95-140`
- Mediterranean: `lat: 30-45, lon: -10 to 40`
- Amazon Basin: `lat: -15-5, lon: -75 to -50`

### Change the Time Period

```python
# Analysis period
START_YEAR = 1995
END_YEAR = 2050  # Or 2100 for longer projection

# Baseline for anomalies
BASELINE_START = 1995
BASELINE_END = 2014
```

### Change the Scenario

```python
SCENARIO = 'ssp245'  # Middle-of-the-road emissions
# Other options: 'ssp126' (low), 'ssp370' (high), 'ssp585' (very high)
```

### Adjust Model Characteristics

Each model has tunable parameters for educational purposes:

```python
# In generate_sample_climate_data() function
base_temp = 15.0  # Base temperature (°C)
warming_rate = 0.03  # Warming per year (°C/year)
seasonal_amplitude = 8.0  # Seasonal variation (°C)
interannual_variability = 0.5  # Year-to-year noise (°C)
```

---

## Understanding the Output

### Key Metrics Explained

**Ensemble Mean**
- Average across all models
- Best estimate of projected change
- Smooths out individual model biases

**Ensemble Standard Deviation (±1σ)**
- Spread of model projections
- Measure of uncertainty
- Large spread = low confidence
- Small spread = high confidence

**Model Agreement**
- Percentage of models agreeing on change direction
- >80% agreement = robust signal
- 50-80% = moderate confidence
- <50% = low confidence

### Example Results

For US Southwest (SSP2-4.5, 2015-2050):
- **Projected warming**: 2.0-3.0°C above baseline
- **Ensemble mean**: 2.5°C
- **Uncertainty (±1σ)**: ±0.6°C
- **Model agreement**: 100% (all models show warming)
- **Warming rate**: 0.05°C/year
- **Confidence**: HIGH (tight ensemble, all models agree)

---

## Limitations (Studio Lab Version)

**What's different from production**:

| Feature | Studio Lab | Unified Studio |
|---------|------------|----------------|
| Data source | Simulated | Real CMIP6 from S3 |
| Models | 3 models | 20+ models |
| Scenarios | Single (SSP2-4.5) | All SSPs |
| Region | Pre-configured | Any global region |
| Variables | Temperature only | 100+ variables |
| Compute | 4GB RAM, local | 16GB+ RAM, distributed |
| Storage | 15GB, session | Unlimited S3 |
| AI features | None | Bedrock (Claude 3) |
| Cost | $0 | $20-30 per analysis |

**Studio Lab is perfect for**:
- ✅ Learning the workflow
- ✅ Teaching climate analysis
- ✅ Testing code before scaling
- ✅ Prototyping research ideas
- ✅ Understanding ensemble methods
- ✅ Creating educational materials

**Transition to production when you need**:
- Real CMIP6 data (not simulated)
- More than 3 models
- Multiple scenarios or variables
- Large-scale analysis
- Collaboration with team
- AI-assisted interpretation

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
   conda create -n climate-analysis python=3.10 -y
   conda activate climate-analysis
   pip install numpy pandas xarray matplotlib cartopy
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
   START_YEAR = 2015  # Instead of 1995
   END_YEAR = 2040    # Instead of 2050
   ```

### Cartopy Import Error

**Problem**: `ImportError: cannot import name 'cartopy'`

**Solution**:
```bash
conda activate climate-analysis
conda install -c conda-forge cartopy=0.22.0 -y
```

Cartopy has complex dependencies, must use exact version.

### Session Expires

**Problem**: 12-hour session limit reached

**Solution**:
1. Save your work first!
2. Stop runtime from Studio Lab dashboard
3. Start runtime again
4. Environment persists, just activate:
   ```bash
   conda activate climate-analysis
   cd research-jumpstart/projects/climate-science/ensemble-analysis/studio-lab
   jupyter notebook quickstart.ipynb
   ```

### Plots Don't Display

**Problem**: Figures not showing inline

**Solution**:
1. Check notebook is using `%matplotlib inline`:
   ```python
   %matplotlib inline
   import matplotlib.pyplot as plt
   ```

2. Or switch to:
   ```python
   %matplotlib notebook
   ```

### Git Clone Fails

**Problem**: "fatal: destination path exists"

**Solution**:
```bash
# Remove old clone
rm -rf research-jumpstart

# Clone fresh
git clone https://github.com/research-jumpstart/research-jumpstart.git
```

---

## Next Steps

### After Completing This Notebook

**Experiment**:
1. Change the region to your area of interest
2. Modify time periods (try 2015-2100)
3. Adjust model parameters to see effects
4. Export figures for presentations

**Learn More**:
1. Read the code carefully - it's well-documented
2. Modify functions to understand how they work
3. Try adding your own analysis (seasonal, trends, etc.)
4. Read about CMIP6 models and scenarios

**Scale Up**:
When ready for production with real data:
1. Review the main [Project README](../README.md)
2. Set up AWS account (if needed)
3. Follow [Unified Studio setup](../unified-studio/README.md)
4. Port your analysis code (minimal changes needed)

### Related Resources

**Climate Science**:
- [IPCC Reports](https://www.ipcc.ch/reports/)
- [CMIP6 Guide](https://pcmdi.llnl.gov/CMIP6/)
- [NASA Climate](https://climate.nasa.gov/)

**Tools & Methods**:
- [Xarray Tutorial](https://tutorial.xarray.dev/)
- [Pangeo](https://pangeo.io/)
- [Climate Data Guide](https://climatedataguide.ucar.edu/)

**Research Jumpstart**:
- [Browse other projects](../../../../docs/projects/index.md)
- [Platform comparison](../../../../docs/getting-started/platform-comparison.md)
- [FAQ](../../../../docs/resources/faq.md)

---

## Getting Help

### Project-Specific Questions

- **GitHub Issues**: https://github.com/research-jumpstart/research-jumpstart/issues
- **Discussions**: https://github.com/research-jumpstart/research-jumpstart/discussions
- Tag: `climate-science`, `studio-lab`

### General Questions

- **Studio Lab support**: studiolab-support@amazon.com
- **Pangeo Discourse**: https://discourse.pangeo.io/
- **Stack Overflow**: Tags `xarray`, `climate-data`, `cmip6`

### Report Bugs

Found an issue? Please report it!

1. Go to https://github.com/research-jumpstart/research-jumpstart/issues
2. Click "New Issue"
3. Describe:
   - What you were trying to do
   - What happened
   - Error messages
   - Your environment (Studio Lab, conda environment)

---

## Contributing

Improvements welcome!

- Fix typos or improve documentation
- Add example regions or use cases
- Enhance visualizations
- Create exercises for students

See [CONTRIBUTING.md](../../../../CONTRIBUTING.md) for guidelines.

---

## Acknowledgments

This educational version uses simulated data for demonstration purposes. Production version accesses real CMIP6 data from AWS Open Data.

**Thanks to**:
- CMIP6 modeling groups for making data available
- AWS for hosting and Studio Lab platform
- Pangeo community for cloud-native tools
- Research Jumpstart contributors

---

## License

Apache License 2.0 - see [LICENSE](../../../../LICENSE)

---

*Last updated: 2025-11-09 | Studio Lab Free Tier Version*
