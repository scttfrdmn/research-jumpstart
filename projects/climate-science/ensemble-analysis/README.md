# Climate Model Ensemble Analysis

**Flagship Project** ⭐ | **Difficulty**: 🟢 Beginner | **Time**: ⏱️⏱️ 4-6 hours (Studio Lab)

Analyze climate model ensembles from CMIP6 without downloading terabytes of data. Perfect introduction to cloud-based climate science research.

---

## What Problem Does This Solve?

Climate scientists routinely need to analyze multiple climate models to:
- Quantify uncertainty across different models
- Calculate ensemble means and spreads
- Assess model agreement on regional projections
- Compare scenarios and time periods

**Traditional approach problems**:
- CMIP6 models = **petabytes** of data on distributed servers
- Downloading even one model for one scenario = days and hundreds of GB
- Multi-model analysis requires institutional storage infrastructure
- Updating analysis when new models released = start over

**This project shows you how to**:
- Access CMIP6 data directly from AWS S3 (no downloads!)
- Process multiple models in parallel using cloud compute
- Calculate ensemble statistics efficiently with xarray/dask
- Generate publication-quality figures
- Scale from 3 models (free) to 20+ models (production)

---

## What You'll Learn

### Climate Science Skills
- Multi-model ensemble analysis techniques
- Uncertainty quantification across models
- Regional climate projection methods
- Temperature anomaly calculation
- Model agreement assessment

### Cloud Computing Skills
- Direct S3 data access (no local storage needed)
- Working with xarray and dask for large datasets
- Distributed processing patterns
- Cost-effective data analysis strategies
- Transitioning from free tier to production

### Technical Skills
- Jupyter notebook workflows
- Conda environment management
- NetCDF/Zarr data formats
- Publication-quality visualization with matplotlib/cartopy
- Git version control for research

---

## Prerequisites

### Required Knowledge
- **Climate science**: Basic understanding of climate models and projections
- **Python**: Familiarity with NumPy, pandas, matplotlib
- **None required**: No cloud experience needed!

### Optional (Helpful)
- Experience with xarray for multidimensional data
- Basic command line skills
- Git basics

### Technical Requirements

**Studio Lab (Free Tier)**
- SageMaker Studio Lab account ([request here](https://studiolab.sagemaker.aws))
- No AWS account needed
- No credit card required

**Unified Studio (Production)**
- AWS account with billing enabled
- Estimated cost: $20-30 per analysis (see Cost Estimates section)
- SageMaker Unified Studio access

---

## Quick Start

### Option 1: Studio Lab (Free - Start Here!)

Perfect for learning, testing, and small-scale analysis.

**Launch in 3 steps**:

1. **Request Studio Lab account** (if you don't have one)
   - Visit https://studiolab.sagemaker.aws
   - Create account with email
   - Approval time varies (can be instant to several days)

2. **Clone this repository**
   ```bash
   git clone https://github.com/research-jumpstart/research-jumpstart.git
   cd research-jumpstart/projects/climate-science/ensemble-analysis/studio-lab
   ```

3. **Set up environment and run**
   ```bash
   # Create conda environment (one time)
   conda env create -f environment.yml
   conda activate climate-analysis

   # Launch notebook
   jupyter notebook quickstart.ipynb
   ```

**What's included in Studio Lab version**:
- ✅ Complete workflow demonstration
- ✅ 3 representative CMIP6 models (CESM2, GFDL-CM4, UKESM1-0-LL)
- ✅ Sample data generation (simulated for educational purposes)
- ✅ All analysis techniques: regional means, anomalies, ensemble stats
- ✅ Publication-quality figures
- ✅ Comprehensive documentation

**Limitations**:
- ⚠️ Uses simulated data (not real CMIP6 from S3)
- ⚠️ Limited to 3 models (vs 20+ in production)
- ⚠️ Single region analysis
- ⚠️ 15GB storage, 12-hour sessions

**Time to complete**: 4-6 hours (including environment setup and exploring code)

---

### Option 2: Unified Studio (Production)

Full-scale climate model analysis with real CMIP6 data from S3.

**Prerequisites**:
- AWS account with billing enabled
- SageMaker Unified Studio domain set up
- Familiarity with Studio Lab version (complete it first!)

**Quick launch**:

1. **Deploy infrastructure** (one-time setup)
   ```bash
   cd unified-studio/cloudformation
   aws cloudformation create-stack \
     --stack-name climate-ensemble-analysis \
     --template-body file://climate-analysis-stack.yml \
     --parameters file://parameters.json \
     --capabilities CAPABILITY_IAM
   ```

2. **Launch Unified Studio**
   - Open SageMaker Unified Studio
   - Navigate to climate-ensemble-analysis domain
   - Launch JupyterLab environment

3. **Run analysis notebooks**
   ```bash
   cd unified-studio/notebooks
   # Follow notebooks in order:
   # 01_data_access.ipynb       - Direct S3/CMIP6 access
   # 02_analysis.ipynb          - Multi-model ensemble processing
   # 03_visualization.ipynb     - Publication figures
   # 04_bedrock_integration.ipynb - AI-assisted interpretation
   ```

**What's included in Unified Studio version**:
- ✅ Real CMIP6 data access from AWS S3 (aws-open-data)
- ✅ 20+ climate models
- ✅ Multiple scenarios (SSP1-2.6, SSP2-4.5, SSP5-8.5)
- ✅ Any region/variable combination
- ✅ Distributed processing with EMR
- ✅ AI-assisted analysis via Amazon Bedrock
- ✅ Automated report generation
- ✅ Production-ready code modules

**Cost estimate**: $20-30 per analysis (see detailed breakdown below)

**Time to complete**:
- First time setup: 2-3 hours
- Each subsequent analysis: 30-60 minutes

---

## Architecture Overview

### Studio Lab Architecture

```
┌─────────────────────────────────────────────────┐
│  SageMaker Studio Lab (Free Tier)              │
│  ┌───────────────────────────────────────────┐ │
│  │  Jupyter Notebook Environment             │ │
│  │  • Python 3.10                           │ │
│  │  • xarray, matplotlib, cartopy           │ │
│  │  • 15GB persistent storage               │ │
│  │  • 12-hour session limit                 │ │
│  └───────────────────────────────────────────┘ │
│                     │                           │
│                     ▼                           │
│  ┌───────────────────────────────────────────┐ │
│  │  Analysis Workflow                        │ │
│  │  1. Generate sample data (simulated)     │ │
│  │  2. Calculate regional means             │ │
│  │  3. Compute temperature anomalies        │ │
│  │  4. Ensemble statistics                  │ │
│  │  5. Visualization                        │ │
│  └───────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### Unified Studio Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  SageMaker Unified Studio                                       │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  JupyterLab Environment                                   │ │
│  │  • ml.t3.xlarge (4 vCPU, 16GB RAM)                       │ │
│  │  • Custom conda environment                               │ │
│  │  • Git integration                                        │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Data Access Layer                                        │ │
│  │  • S3 access to CMIP6 archive (s3://cmip6-pds)           │ │
│  │  • No egress charges (same region)                       │ │
│  │  • Zarr optimized format                                 │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Processing Layer                                         │ │
│  │  • xarray + dask for distributed processing              │ │
│  │  • Optional: EMR cluster for heavy compute               │ │
│  │  • Parallel model processing                             │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Analysis & Visualization                                 │ │
│  │  • Ensemble statistics (mean, std, percentiles)          │ │
│  │  • Model agreement analysis                              │ │
│  │  • Publication-quality figures (matplotlib/cartopy)      │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  AI-Assisted Interpretation (Bedrock)                    │ │
│  │  • Claude 3 for result interpretation                    │ │
│  │  • Automated report generation                           │ │
│  │  • Literature context integration                        │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Output Storage                                           │ │
│  │  • S3 bucket for results                                 │ │
│  │  • Figures, data files, reports                          │ │
│  │  • Version controlled outputs                            │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

See `assets/architecture-diagram.png` for full visual diagram.

---

## Cost Estimates

### Studio Lab: $0 (Always Free)

- No AWS account required
- No credit card needed
- No hidden costs
- 15GB storage, 12-hour sessions

**When Studio Lab is enough**:
- Learning cloud-based climate analysis
- Teaching/workshops
- Prototyping analysis workflows
- Small-scale regional studies

---

### Unified Studio: $20-30 per Analysis

**Realistic cost breakdown for typical analysis**:
(20 models, 2 variables, single scenario, 30-year period, one region)

| Service | Usage | Cost |
|---------|-------|------|
| **Data Access (S3)** | Read CMIP6 data (no egress) | $0 |
| **Compute (Jupyter)** | ml.t3.xlarge, 4 hours | $0.60 |
| **Storage (S3)** | 10GB results storage | $0.23/month |
| **Bedrock (Claude 3)** | Report generation | $3-5 |
| **EMR (optional)** | Heavy distributed compute | $12-18 (if needed) |
| **Total per analysis** | | **$4-6** (no EMR)<br>**$16-24** (with EMR) |

**Monthly costs if running regularly**:
- 5 analyses/month: $80-120
- 10 analyses/month: $160-240
- Storage (persistent): $2-5/month

**Cost optimization tips**:
1. Use spot instances for EMR (save 60-80%)
2. Delete intermediate results (keep only final outputs)
3. Process multiple regions in single run
4. Cache frequently-used model subsets
5. Use ml.t3.medium for lighter analyses ($0.30/hr vs $0.60/hr)

**When Unified Studio is worth it**:
- Need real CMIP6 data (not simulated)
- Analyzing 10+ models for publication
- Multiple scenarios or variables
- Regular analysis updates
- Collaboration with team (shared environment)

---

### When NOT to Use Cloud

Be honest with yourself about these scenarios:

**Stick with local/HPC if**:
- ❌ You already have CMIP6 data downloaded locally
- ❌ Your institution has free HPC with pre-staged data
- ❌ One-time analysis that works on your laptop
- ❌ Budget constraints (no AWS account available)

**Consider hybrid approach**:
- Use HPC for heavy processing
- Use cloud for visualization/collaboration
- See "HPC Hybrid" version (coming soon)

---

## Project Structure

```
ensemble-analysis/
├── README.md                          # This file
├── studio-lab/                        # Free tier version
│   ├── quickstart.ipynb              # Main analysis notebook
│   ├── environment.yml               # Conda dependencies
│   └── README.md                     # Studio Lab specific docs
├── unified-studio/                    # Production version
│   ├── notebooks/
│   │   ├── 01_data_access.ipynb     # S3/CMIP6 data access
│   │   ├── 02_analysis.ipynb        # Ensemble processing
│   │   ├── 03_visualization.ipynb   # Publication figures
│   │   └── 04_bedrock_integration.ipynb  # AI-assisted analysis
│   ├── src/
│   │   ├── data_access.py           # S3 utilities
│   │   ├── climate_analysis.py      # Core analysis functions
│   │   ├── ensemble_stats.py        # Statistical methods
│   │   ├── visualization.py         # Plotting utilities
│   │   └── bedrock_client.py        # AI integration
│   ├── cloudformation/
│   │   ├── climate-analysis-stack.yml  # Infrastructure as code
│   │   └── parameters.json          # Stack parameters
│   ├── environment.yml              # Production dependencies
│   └── README.md                    # Unified Studio docs
├── workshop/                          # Half-day workshop materials
│   ├── slides.pdf
│   ├── exercises/
│   └── solutions/
└── assets/
    ├── architecture-diagram.png      # System architecture
    ├── sample-outputs/               # Example figures
    └── cost-calculator.xlsx          # Interactive cost estimator
```

---

## Transition Pathway

### From Studio Lab to Unified Studio

Once you've completed the Studio Lab version and are ready for production:

**Step 1: Complete Studio Lab version**
- Understand the full workflow
- Know what analysis you want to run
- Identify which models/scenarios you need

**Step 2: Set up AWS account**
- Follow [AWS account setup guide](../../../docs/getting-started/aws-account-setup.md)
- Enable billing alerts ($10, $50, $100 thresholds)
- Set up IAM user with appropriate permissions

**Step 3: Deploy Unified Studio infrastructure**
- Use provided CloudFormation template
- Takes 10-15 minutes to deploy
- One-time setup

**Step 4: Port your analysis**
- **Data loading**: Replace `generate_sample_climate_data()` with real S3 access
  ```python
  # Studio Lab (simulated)
  data = generate_sample_climate_data(model_name, ...)

  # Unified Studio (real CMIP6)
  import s3fs
  fs = s3fs.S3FileSystem(anon=True)
  data = xr.open_zarr(
      f's3://cmip6-pds/CMIP6/ScenarioMIP/{model_name}/...'
  )
  ```

- **Computation**: Same xarray operations work identically
- **Visualization**: Exact same matplotlib/cartopy code

**Step 5: Add production features**
- Parallel processing with dask
- Multiple models/scenarios
- AI-assisted interpretation via Bedrock
- Automated report generation

**Estimated transition time**: 2-3 hours (mostly infrastructure setup)

### What Stays the Same
✅ All analysis code (xarray operations)
✅ Visualization code (matplotlib/cartopy)
✅ File formats (NetCDF/Zarr)
✅ Workflow structure

### What Changes
🔄 Data source (simulated → S3)
🔄 Scale (3 models → 20+ models)
🔄 Compute (local → distributed)
🔄 Features (+Bedrock, +collaboration)

---

## Detailed Workflow

### 1. Data Access

**Studio Lab**:
```python
# Generate sample data (simulated for learning)
data = generate_sample_climate_data(
    model_name='CESM2',
    region={'lat_min': 31, 'lat_max': 37,
            'lon_min': -114, 'lon_max': -109},
    start_year=1995,
    end_year=2050
)
```

**Unified Studio**:
```python
# Access real CMIP6 data from S3
import s3fs
import xarray as xr

fs = s3fs.S3FileSystem(anon=True)
store = s3fs.S3Map(
    root=f's3://cmip6-pds/CMIP6/ScenarioMIP/NCAR/CESM2/ssp245/...',
    s3=fs
)
data = xr.open_zarr(store)
```

### 2. Regional Mean Calculation

```python
def calculate_regional_mean(ds, region):
    """
    Calculate area-weighted regional mean.
    Works identically in both Studio Lab and Unified Studio.
    """
    # Select spatial subset
    tas = ds['tas'].sel(
        lat=slice(region['lat_min'], region['lat_max']),
        lon=slice(region['lon_min'], region['lon_max'])
    )

    # Area weighting (cosine of latitude)
    weights = np.cos(np.deg2rad(tas.lat))
    regional_mean = tas.weighted(weights).mean(['lat', 'lon'])

    return regional_mean
```

### 3. Temperature Anomaly

```python
def calculate_anomaly(time_series, baseline_start, baseline_end):
    """
    Calculate anomaly relative to baseline period (e.g., 1995-2014).
    """
    baseline = time_series.sel(
        time=slice(f'{baseline_start}', f'{baseline_end}')
    )
    baseline_mean = baseline.mean('time')
    anomaly = time_series - baseline_mean

    return anomaly
```

### 4. Ensemble Statistics

```python
# Stack all models into ensemble
ensemble = xr.concat(model_data_list, dim='model')

# Calculate statistics
ensemble_mean = ensemble.mean('model')
ensemble_std = ensemble.std('model')
ensemble_min = ensemble.min('model')
ensemble_max = ensemble.max('model')

# Model agreement: percentage of models agreeing on sign
model_agreement = (
    (ensemble > 0).sum('model') / len(model_data_list) * 100
)
```

### 5. Visualization

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Time series with uncertainty
fig, ax = plt.subplots(figsize=(12, 6))

# Individual models (thin lines)
for model in models:
    ax.plot(years, model_data[model], alpha=0.3, linewidth=1)

# Ensemble mean (thick line)
ax.plot(years, ensemble_mean, 'k-', linewidth=2.5, label='Ensemble Mean')

# Uncertainty range (shaded)
ax.fill_between(
    years,
    ensemble_mean - ensemble_std,
    ensemble_mean + ensemble_std,
    alpha=0.3, color='gray', label='±1σ'
)

ax.set_xlabel('Year')
ax.set_ylabel('Temperature Anomaly (°C)')
ax.set_title('Multi-Model Ensemble: Regional Temperature Projection')
ax.legend()
ax.grid(True, alpha=0.3)
```

---

## Troubleshooting

### Studio Lab Issues

**Problem**: "Conda environment creation fails"
```
Solution:
1. Check available disk space: df -h
2. If <2GB free, clean up: conda clean --all
3. Retry: conda env create -f environment.yml
```

**Problem**: "Kernel dies during execution"
```
Cause: Running out of memory (4GB limit in Studio Lab)
Solution:
- Reduce number of models from 3 to 2
- Decrease time range (1995-2030 instead of 1995-2050)
- Clear variables: del large_variable
```

**Problem**: "Session expires before completion"
```
Cause: 12-hour session limit
Solution:
- Save intermediate results: data.to_netcdf('checkpoint.nc')
- Resume in next session: data = xr.open_dataset('checkpoint.nc')
- Consider breaking into smaller notebooks
```

**Problem**: "Import errors for cartopy"
```
Solution:
conda install -c conda-forge cartopy=0.22.0
# Cartopy has complex dependencies, use exact version
```

---

### Unified Studio Issues

**Problem**: "Cannot access S3 CMIP6 data"
```
Error: botocore.exceptions.NoCredentialsError

Solution:
1. Check IAM role attached to SageMaker execution role
2. Required policy: AmazonS3ReadOnlyAccess (for public data)
3. For anonymous access: s3fs.S3FileSystem(anon=True)
```

**Problem**: "Data access is slow"
```
Cause: Reading data from different AWS region

Solution:
1. Verify you're in us-east-1 (same as cmip6-pds bucket)
2. Use Zarr format (optimized for cloud): xr.open_zarr()
3. Read only variables you need: ds[['tas', 'pr']]
4. Subset spatially before downloading: ds.sel(lat=slice(...))
```

**Problem**: "Costs higher than expected"
```
Common causes:
1. Data egress charges: Use us-east-1 region
2. Compute running idle: Stop instances when not in use
3. EMR cluster not terminated: Check EMR console
4. Large result files in S3: Clean up intermediate outputs

Check costs:
- AWS Cost Explorer: https://console.aws.amazon.com/cost-management/
- Set up billing alerts at $10, $50, $100 thresholds
```

**Problem**: "Bedrock API errors"
```
Error: AccessDeniedException

Solution:
1. Enable Bedrock in your AWS region
2. Request Claude 3 model access (takes minutes)
3. Add Bedrock permissions to execution role
4. Check quota limits: https://console.aws.amazon.com/servicequotas/
```

**Problem**: "Out of memory with 20+ models"
```
Solution:
1. Use dask for lazy evaluation:
   data = xr.open_zarr(..., chunks={'time': 12})
2. Process models in batches:
   for batch in model_batches:
       process_batch(batch)
       batch_result.to_netcdf(f'batch_{i}.nc')
3. Use larger instance: ml.m5.2xlarge (32GB RAM)
4. Or spin up EMR cluster for distributed processing
```

---

## Extension Ideas

Once you've completed the base project, try these extensions:

### Beginner Extensions (2-4 hours each)

1. **Different Variables**
   - Precipitation instead of temperature
   - Sea ice extent
   - Soil moisture

2. **Different Regions**
   - Compare multiple regions (Arctic, tropics, mid-latitudes)
   - Urban areas vs rural areas
   - Coastal vs inland

3. **Different Time Periods**
   - Historical (1850-2014) vs future (2015-2100)
   - Near-term (2025-2050) vs end-century (2075-2100)
   - Different baseline periods

4. **Additional Statistics**
   - Percentiles (10th, 25th, 75th, 90th)
   - Trend analysis (linear regression)
   - Change detection (when does warming exceed 1.5°C?)

### Intermediate Extensions (4-8 hours each)

5. **Scenario Comparison**
   - SSP1-2.6 vs SSP2-4.5 vs SSP5-8.5
   - Quantify benefits of mitigation
   - "Committed warming" analysis

6. **Seasonal Analysis**
   - Summer vs winter trends
   - Seasonal cycle changes
   - Extreme temperature analysis (95th percentile of daily max)

7. **Spatial Maps**
   - Full global analysis (not just regional)
   - Pattern scaling
   - Signal-to-noise ratios

8. **Model Evaluation**
   - Compare models to observations (ERA5 reanalysis)
   - Model skill scores
   - Identify outlier models

### Advanced Extensions (8+ hours each)

9. **Climate Indices**
   - ENSO, PDO, AMO indices from models
   - Teleconnection patterns
   - Compound extremes

10. **Machine Learning**
    - Train emulator to predict ensemble mean
    - Clustering models by regional behavior
    - Downscaling coarse model output

11. **Impact Analysis**
    - Degree-day calculations (heating/cooling)
    - Agricultural growing season changes
    - Water resource implications

12. **Publication Pipeline**
    - Automated figure generation for all regions
    - LaTeX report integration
    - Version control for analysis parameters

---

## Additional Resources

### CMIP6 Data & Documentation

- **CMIP6 Data on AWS**: https://registry.opendata.aws/cmip6/
- **CMIP6 Guide**: https://pcmdi.llnl.gov/CMIP6/
- **Variable naming**: https://clipc-services.ceda.ac.uk/dreq/index.html
- **Model documentation**: https://search.es-doc.org/

### xarray & Climate Data Analysis

- **xarray tutorial**: https://tutorial.xarray.dev/
- **Dask for parallel computing**: https://tutorial.dask.org/
- **Pangeo (cloud-native climate data)**: https://pangeo.io/
- **Climate data operators (CDO)**: https://code.mpimet.mpg.de/projects/cdo/

### AWS Services

- **SageMaker Studio Lab docs**: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html
- **SageMaker Unified Studio**: https://docs.aws.amazon.com/sagemaker/latest/dg/unified-studio.html
- **Amazon Bedrock**: https://docs.aws.amazon.com/bedrock/
- **S3 optimization**: https://docs.aws.amazon.com/s3/

### Climate Analysis Examples

- **IPCC Interactive Atlas**: https://interactive-atlas.ipcc.ch/
- **NASA Earth Data**: https://earthdata.nasa.gov/
- **NCAR Climate Data Guide**: https://climatedataguide.ucar.edu/

### Research Papers Using Similar Methods

- **Multi-model ensembles**: Tebaldi & Knutti (2007), "The use of the multi-model ensemble in probabilistic climate projections"
- **Uncertainty quantification**: Hawkins & Sutton (2009), "The Potential to Narrow Uncertainty in Regional Climate Predictions"
- **Cloud-based climate analysis**: Abernathey et al. (2021), "Cloud-Native Repositories for Big Scientific Data"

---

## Getting Help

### Project-Specific Questions

- **GitHub Issues**: https://github.com/research-jumpstart/research-jumpstart/issues
- **Discussions**: https://github.com/research-jumpstart/research-jumpstart/discussions
- Tag your question with `climate-science` and `ensemble-analysis`

### General Climate Data Questions

- **Pangeo Discourse**: https://discourse.pangeo.io/
- **xarray GitHub Discussions**: https://github.com/pydata/xarray/discussions
- **Stack Overflow**: Tag with `xarray`, `cmip6`, `climate-data`

### AWS Support

- **SageMaker Studio Lab**: studiolab-support@amazon.com
- **AWS Forums**: https://repost.aws/
- **AWS Support** (for production accounts)

---

## Contributing

Found a bug? Have an improvement? Want to add an extension?

1. **Open an issue** describing the problem/enhancement
2. **Fork the repository**
3. **Create a branch**: `git checkout -b climate-ensemble-improvements`
4. **Make your changes** with clear commit messages
5. **Test thoroughly** (include example outputs)
6. **Submit a pull request**

See main [CONTRIBUTING.md](../../../CONTRIBUTING.md) for detailed guidelines.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_climate_ensemble,
  title = {Climate Model Ensemble Analysis: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/research-jumpstart/research-jumpstart},
  note = {Accessed: [date]}
}
```

And cite the CMIP6 data you use:

```bibtex
@article{eyring2016overview,
  title={Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6) experimental design and organization},
  author={Eyring, Veronika and Bony, Sandrine and Meehl, Gerald A and others},
  journal={Geoscientific Model Development},
  volume={9},
  pages={1937--1958},
  year={2016},
  doi={10.5194/gmd-9-1937-2016}
}
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../../../LICENSE) file for details.

The CMIP6 data accessed in this project has its own citation requirements and usage terms. See https://pcmdi.llnl.gov/CMIP6/TermsOfUse for details.

---

## Acknowledgments

- **CMIP6 modeling groups** for making data publicly available
- **AWS Open Data Program** for hosting CMIP6 archive
- **Pangeo community** for cloud-native climate data tools
- **xarray/dask developers** for excellent scientific Python tools
- **Research Jumpstart community** for contributions and feedback

---

## Version History

- **v1.0.0** (2025-11-09): Initial release
  - Studio Lab version with simulated data
  - Unified Studio version with S3 access
  - 3 models, single scenario demonstration
  - Comprehensive documentation

**Planned features**:
- v1.1.0: Workshop materials and exercises
- v1.2.0: HPC hybrid version
- v2.0.0: Additional scenarios and variables
- v2.1.0: Machine learning extensions

---

## Questions?

**Not sure if this project is right for you?**
- See [Platform Comparison](../../../docs/getting-started/platform-comparison.md)
- See [FAQ](../../../docs/resources/faq.md)
- Ask in [Discussions](https://github.com/research-jumpstart/research-jumpstart/discussions)

**Ready to start?**
- [Launch Studio Lab version](#option-1-studio-lab-free---start-here) (free, 10 minutes to start)
- [Set up Unified Studio](#option-2-unified-studio-production) (production, 1 hour setup)

**Want to jump to different project?**
- [Browse all projects](../../../docs/projects/index.md)
- [Climate science projects](../../../docs/projects/climate-science.md)

---

*Last updated: 2025-11-09 | Research Jumpstart v1.0.0*
