# Multi-Sensor Environmental Monitoring

**Flagship Project** | **Difficulty**: Beginner | **Time**: 4-8 hours (Studio Lab)

Analyze multi-sensor environmental data (optical, radar, LiDAR) for ecosystem change detection without downloading massive datasets. Perfect introduction to cloud-based environmental monitoring research.

---

## What Problem Does This Solve?

Environmental scientists routinely need to analyze multiple sensor types to:
- Detect land cover and ecosystem changes over time
- Monitor forest health and deforestation
- Track wetland dynamics and water resources
- Assess agricultural practices and land use
- Quantify biodiversity and habitat loss

**Traditional approach problems**:
- Multi-sensor data = **terabytes** distributed across agencies
- Downloading Landsat + Sentinel + LiDAR = days and hundreds of GB
- Change detection requires temporal persistence and reprocessing
- Updating analysis when new imagery arrives = start over
- Different sensors have incompatible formats and projections

**This project shows you how to**:
- Access satellite imagery directly from cloud archives (no downloads!)
- Process multi-sensor data in parallel using cloud compute
- Implement change detection algorithms with temporal persistence
- Generate high-quality environmental monitoring products
- Scale from single scenes (free) to continental analysis (production)

---

## What You'll Learn

### Environmental Science Skills
- Land cover classification techniques
- Multi-sensor data fusion methods
- Temporal change detection algorithms
- Ecosystem health assessment
- Environmental impact analysis

### Cloud Computing Skills
- Direct cloud data access (no local storage needed)
- Working with geospatial data formats (GeoTIFF, COG)
- Distributed processing patterns
- Cost-effective analysis strategies
- Transitioning from free tier to production

### Technical Skills
- Jupyter notebook workflows
- Conda environment management
- Raster/vector data processing
- Machine learning for remote sensing
- Git version control for research

---

## Prerequisites

### Required Knowledge
- **Environmental science**: Basic understanding of remote sensing and land cover
- **Python**: Familiarity with NumPy, pandas, matplotlib
- **None required**: No cloud experience needed!

### Optional (Helpful)
- Experience with rasterio, GDAL for geospatial data
- Basic command line skills
- Git basics

### Technical Requirements

**Studio Lab (Free Tier)**
- SageMaker Studio Lab account ([request here](https://studiolab.sagemaker.aws))
- No AWS account needed
- No credit card required

**Unified Studio (Production)**
- AWS account with billing enabled
- Estimated cost: $15-25 per analysis (see Cost Estimates section)
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
   cd research-jumpstart/projects/environmental/ecosystem-monitoring/studio-lab
   ```

3. **Set up environment and run**
   ```bash
   # Create conda environment (one time)
   conda env create -f environment.yml
   conda activate environmental-monitoring

   # Launch notebook
   jupyter notebook quickstart.ipynb
   ```

**What's included in Studio Lab version**:
- Complete workflow demonstration
- Sample Landsat imagery (simulated for educational purposes)
- All analysis techniques: classification, change detection, time series
- Publication-quality figures
- Comprehensive documentation

**Limitations**:
- Uses simulated satellite data (not real imagery from cloud archives)
- Limited to single scene analysis
- Single sensor demonstration
- 15GB storage, 12-hour sessions

**Time to complete**: 4-6 hours (including environment setup and exploring code)

---

### Option 2: Unified Studio (Production)

Full-scale multi-sensor environmental monitoring with real satellite data from cloud archives.

**Prerequisites**:
- AWS account with billing enabled
- SageMaker Unified Studio domain set up
- Familiarity with Studio Lab version (complete it first!)

**Quick launch**:

1. **Deploy infrastructure** (one-time setup)
   ```bash
   cd unified-studio/cloudformation
   aws cloudformation create-stack \
     --stack-name environmental-monitoring \
     --template-body file://environmental-stack.yml \
     --parameters file://parameters.json \
     --capabilities CAPABILITY_IAM
   ```

2. **Launch Unified Studio**
   - Open SageMaker Unified Studio
   - Navigate to environmental-monitoring domain
   - Launch JupyterLab environment

3. **Run analysis notebooks**
   ```bash
   cd unified-studio/notebooks
   # Follow notebooks in order:
   # 01_data_access.ipynb       - Cloud satellite data access
   # 02_classification.ipynb    - Land cover classification
   # 03_change_detection.ipynb  - Temporal analysis
   # 04_bedrock_integration.ipynb - AI-assisted interpretation
   ```

**What's included in Unified Studio version**:
- Real satellite data access from AWS cloud archives
- Multi-sensor fusion (Landsat, Sentinel-1/2, LiDAR)
- Any region/time period combination
- Distributed processing with EMR
- AI-assisted analysis via Amazon Bedrock
- Automated report generation
- Production-ready code modules

**Cost estimate**: $15-25 per analysis (see detailed breakdown below)

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
│  │  • rasterio, scikit-learn, tensorflow    │ │
│  │  • 15GB persistent storage               │ │
│  │  • 12-hour session limit                 │ │
│  └───────────────────────────────────────────┘ │
│                     │                           │
│                     ▼                           │
│  ┌───────────────────────────────────────────┐ │
│  │  Analysis Workflow                        │ │
│  │  1. Generate sample imagery (simulated)  │ │
│  │  2. Train land cover classifier          │ │
│  │  3. Detect temporal changes              │ │
│  │  4. Time series analysis                 │ │
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
│  │  • S3 access to Landsat/Sentinel archives                │ │
│  │  • No egress charges (same region)                       │ │
│  │  • COG optimized format                                  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Processing Layer                                         │ │
│  │  • rasterio + dask for distributed processing            │ │
│  │  • Optional: EMR cluster for heavy compute               │ │
│  │  • Parallel scene processing                             │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Analysis & Visualization                                 │ │
│  │  • Land cover classification (Random Forest, CNN)        │ │
│  │  • Change detection algorithms                           │ │
│  │  • Time series analysis                                  │ │
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
│  │  • Classification maps, change products, reports         │ │
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
- Learning cloud-based environmental monitoring
- Teaching/workshops
- Prototyping analysis workflows
- Small-scale regional studies

---

### Unified Studio: $15-25 per Analysis

**Realistic cost breakdown for typical analysis**:
(Single region, 3 sensors, 5-year time series, change detection)

| Service | Usage | Cost |
|---------|-------|------|
| **Data Access (S3)** | Read satellite imagery (no egress) | $0 |
| **Compute (Jupyter)** | ml.t3.xlarge, 3 hours | $0.45 |
| **Storage (S3)** | 5GB results storage | $0.12/month |
| **Bedrock (Claude 3)** | Report generation | $2-4 |
| **EMR (optional)** | Heavy distributed compute | $10-15 (if needed) |
| **Total per analysis** | | **$3-5** (no EMR)<br>**$13-20** (with EMR) |

**Monthly costs if running regularly**:
- 5 analyses/month: $65-100
- 10 analyses/month: $130-200
- Storage (persistent): $1-3/month

**Cost optimization tips**:
1. Use spot instances for EMR (save 60-80%)
2. Delete intermediate results (keep only final products)
3. Process multiple regions in single run
4. Cache frequently-used imagery subsets
5. Use ml.t3.medium for lighter analyses ($0.30/hr vs $0.60/hr)

**When Unified Studio is worth it**:
- Need real satellite imagery (not simulated)
- Multi-sensor fusion for publication
- Long time series analysis (5+ years)
- Regular monitoring updates
- Collaboration with team (shared environment)

---

### When NOT to Use Cloud

Be honest with yourself about these scenarios:

**Stick with local/HPC if**:
- You already have satellite data downloaded locally
- Your institution has free HPC with pre-staged data
- One-time analysis that works on your laptop
- Budget constraints (no AWS account available)

**Consider hybrid approach**:
- Use HPC for heavy processing
- Use cloud for visualization/collaboration
- See "HPC Hybrid" version (coming soon)

---

## Project Structure

```
ecosystem-monitoring/
├── README.md                          # This file
├── studio-lab/                        # Free tier version
│   ├── quickstart.ipynb              # Main analysis notebook
│   ├── environment.yml               # Conda dependencies
│   └── README.md                     # Studio Lab specific docs
├── unified-studio/                    # Production version
│   ├── notebooks/
│   │   ├── 01_data_access.ipynb     # Cloud satellite data access
│   │   ├── 02_classification.ipynb  # Land cover classification
│   │   ├── 03_change_detection.ipynb # Temporal analysis
│   │   └── 04_bedrock_integration.ipynb  # AI-assisted analysis
│   ├── src/
│   │   ├── data_access.py           # S3 utilities
│   │   ├── classification.py        # Classification functions
│   │   ├── change_detection.py      # Change detection methods
│   │   ├── visualization.py         # Plotting utilities
│   │   └── bedrock_client.py        # AI integration
│   ├── cloudformation/
│   │   ├── environmental-stack.yml  # Infrastructure as code
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
- Identify which sensors/time periods you need

**Step 2: Set up AWS account**
- Follow [AWS account setup guide](../../../docs/getting-started/aws-account-setup.md)
- Enable billing alerts ($10, $50, $100 thresholds)
- Set up IAM user with appropriate permissions

**Step 3: Deploy Unified Studio infrastructure**
- Use provided CloudFormation template
- Takes 10-15 minutes to deploy
- One-time setup

**Step 4: Port your analysis**
- **Data loading**: Replace `generate_sample_imagery()` with real S3 access
  ```python
  # Studio Lab (simulated)
  data = generate_sample_imagery(region, date_range)

  # Unified Studio (real satellite data)
  import rasterio
  from rasterio.session import AWSSession
  with rasterio.Env(AWSSession()):
      with rasterio.open(
          f's3://landsat-pds/...'
      ) as src:
          data = src.read()
  ```

- **Computation**: Same rasterio/scikit-learn operations work identically
- **Visualization**: Exact same matplotlib code

**Step 5: Add production features**
- Parallel processing with dask
- Multiple sensors/time periods
- AI-assisted interpretation via Bedrock
- Automated report generation

**Estimated transition time**: 2-3 hours (mostly infrastructure setup)

### What Stays the Same
- All analysis code (rasterio operations)
- Visualization code (matplotlib)
- File formats (GeoTIFF, NetCDF)
- Workflow structure

### What Changes
- Data source (simulated → S3)
- Scale (1 scene → hundreds of scenes)
- Compute (local → distributed)
- Features (+Bedrock, +collaboration)

---

## Detailed Workflow

### 1. Data Access

**Studio Lab**:
```python
# Generate sample imagery (simulated for learning)
data = generate_sample_imagery(
    region={'lon_min': -110, 'lat_min': 32,
            'lon_max': -109, 'lat_max': 33},
    date_range=('2020-01-01', '2023-12-31'),
    sensor='landsat8'
)
```

**Unified Studio**:
```python
# Access real satellite data from S3
import rasterio
from rasterio.session import AWSSession

with rasterio.Env(AWSSession()):
    with rasterio.open(
        's3://landsat-pds/c1/L8/037/037/LC08_L1TP_037037_20200101_20200113_01_T1/...'
    ) as src:
        data = src.read()
        metadata = src.meta
```

### 2. Land Cover Classification

```python
from sklearn.ensemble import RandomForestClassifier

def classify_land_cover(imagery, training_data):
    """
    Classify land cover using Random Forest.
    Works identically in both Studio Lab and Unified Studio.
    """
    # Extract features (bands, indices, textures)
    features = extract_features(imagery)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(training_data['features'], training_data['labels'])

    # Predict classification
    classification = clf.predict(features)

    return classification.reshape(imagery.shape[1:])
```

### 3. Change Detection

```python
def detect_changes(image_t1, image_t2, threshold=0.3):
    """
    Detect changes between two time periods.
    """
    # Calculate spectral indices
    ndvi_t1 = calculate_ndvi(image_t1)
    ndvi_t2 = calculate_ndvi(image_t2)

    # Calculate change magnitude
    change_magnitude = np.abs(ndvi_t2 - ndvi_t1)

    # Threshold significant changes
    change_mask = change_magnitude > threshold

    return change_mask, change_magnitude
```

### 4. Time Series Analysis

```python
def analyze_time_series(imagery_stack, dates):
    """
    Analyze temporal trends in vegetation health.
    """
    # Calculate NDVI time series
    ndvi_series = [calculate_ndvi(img) for img in imagery_stack]

    # Regional statistics
    mean_ndvi = [np.mean(ndvi) for ndvi in ndvi_series]
    std_ndvi = [np.std(ndvi) for ndvi in ndvi_series]

    # Trend analysis
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        range(len(dates)), mean_ndvi
    )

    return {
        'time_series': mean_ndvi,
        'trend_slope': slope,
        'r_squared': r_value**2
    }
```

### 5. Visualization

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Time series plot
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(dates, ndvi_time_series, 'o-', linewidth=2, markersize=6)
ax.fill_between(dates,
                ndvi_time_series - std_ndvi,
                ndvi_time_series + std_ndvi,
                alpha=0.3, label='±1σ')

ax.set_xlabel('Date')
ax.set_ylabel('NDVI')
ax.set_title('Vegetation Health Time Series')
ax.legend()
ax.grid(True, alpha=0.3)

# Classification map
fig, ax = plt.subplots(figsize=(10, 10),
                      subplot_kw={'projection': ccrs.PlateCarree()})

im = ax.imshow(classification,
               extent=[lon_min, lon_max, lat_min, lat_max],
               cmap='tab10', interpolation='nearest')

ax.coastlines()
ax.gridlines(draw_labels=True)
plt.colorbar(im, label='Land Cover Class', shrink=0.8)
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
- Reduce spatial extent (process smaller tiles)
- Decrease time range (fewer images)
- Clear variables: del large_array
```

**Problem**: "Session expires before completion"
```
Cause: 12-hour session limit
Solution:
- Save intermediate results: np.save('checkpoint.npy', data)
- Resume in next session: data = np.load('checkpoint.npy')
- Consider breaking into smaller notebooks
```

**Problem**: "Import errors for rasterio"
```
Solution:
conda install -c conda-forge rasterio=1.3.9
# GDAL dependencies can be complex, use exact version
```

---

### Unified Studio Issues

**Problem**: "Cannot access S3 satellite data"
```
Error: botocore.exceptions.NoCredentialsError

Solution:
1. Check IAM role attached to SageMaker execution role
2. Required policy: AmazonS3ReadOnlyAccess (for public data)
3. For anonymous access: use AWSSession with unsigned requests
```

**Problem**: "Data access is slow"
```
Cause: Reading data from different AWS region

Solution:
1. Verify you're in us-west-2 (same as landsat-pds bucket)
2. Use COG format (optimized for cloud): read windowed data
3. Read only bands you need: src.read([4, 3, 2])
4. Subset spatially before downloading: use windowed reads
```

**Problem**: "Costs higher than expected"
```
Common causes:
1. Data egress charges: Use correct region (us-west-2 for Landsat)
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

**Problem**: "Out of memory with large imagery"
```
Solution:
1. Use dask for lazy evaluation:
   import dask.array as da
   data = da.from_array(large_array, chunks=(1024, 1024))
2. Process in tiles:
   for tile in tiles:
       process_tile(tile)
       tile_result.save(f'tile_{i}.tif')
3. Use larger instance: ml.m5.2xlarge (32GB RAM)
4. Or spin up EMR cluster for distributed processing
```

---

## Extension Ideas

Once you've completed the base project, try these extensions:

### Beginner Extensions (2-4 hours each)

1. **Different Sensors**
   - Sentinel-2 (10m resolution) instead of Landsat
   - Sentinel-1 SAR for all-weather monitoring
   - Planet imagery for daily monitoring

2. **Different Regions**
   - Compare multiple ecosystems (forest, grassland, wetland)
   - Urban expansion analysis
   - Coastal dynamics

3. **Different Indices**
   - NDWI (water index) for wetland monitoring
   - NBR (burn ratio) for fire analysis
   - SAVI (soil-adjusted vegetation index)

4. **Additional Metrics**
   - Patch metrics (fragmentation, connectivity)
   - Edge detection
   - Object-based classification

### Intermediate Extensions (4-8 hours each)

5. **Multi-Sensor Fusion**
   - Combine optical + radar for cloud-free monitoring
   - LiDAR + multispectral for 3D structure
   - Data fusion algorithms

6. **Deep Learning**
   - U-Net for semantic segmentation
   - ResNet for classification
   - Transfer learning from pre-trained models

7. **Phenology Analysis**
   - Growing season detection
   - Seasonal patterns
   - Climate change impacts on timing

8. **Validation**
   - Accuracy assessment with ground truth
   - Cross-validation strategies
   - Uncertainty quantification

### Advanced Extensions (8+ hours each)

9. **Time Series Forecasting**
   - LSTM for vegetation predictions
   - ARIMA models for trend analysis
   - Anomaly detection

10. **Impact Assessment**
    - Deforestation quantification
    - Carbon stock estimation
    - Biodiversity indicators

11. **Operational Monitoring**
    - Automated alert systems
    - Near real-time processing
    - Dashboard development

12. **Publication Pipeline**
    - Automated figure generation for all regions
    - LaTeX report integration
    - Version control for analysis parameters

---

## Additional Resources

### Satellite Data & Documentation

- **Landsat on AWS**: https://registry.opendata.aws/landsat-8/
- **Sentinel-2 on AWS**: https://registry.opendata.aws/sentinel-2/
- **Sentinel-1 SAR**: https://registry.opendata.aws/sentinel-1/
- **Planet imagery**: https://www.planet.com/

### Remote Sensing Analysis

- **rasterio tutorial**: https://rasterio.readthedocs.io/
- **GDAL documentation**: https://gdal.org/
- **Earthdata Search**: https://search.earthdata.nasa.gov/
- **Google Earth Engine**: https://earthengine.google.com/

### AWS Services

- **SageMaker Studio Lab docs**: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html
- **SageMaker Unified Studio**: https://docs.aws.amazon.com/sagemaker/latest/dg/unified-studio.html
- **Amazon Bedrock**: https://docs.aws.amazon.com/bedrock/
- **S3 optimization**: https://docs.aws.amazon.com/s3/

### Environmental Analysis Examples

- **NASA Earth Observations**: https://neo.gsfc.nasa.gov/
- **ESA Climate Change Initiative**: https://climate.esa.int/
- **USGS Land Cover**: https://www.usgs.gov/centers/eros/science/national-land-cover-database

### Research Papers Using Similar Methods

- **Land cover classification**: Gómez et al. (2016), "Optical remotely sensed time series data for land cover classification"
- **Change detection**: Zhu (2017), "Change detection using landsat time series"
- **Cloud-based analysis**: Gorelick et al. (2017), "Google Earth Engine: Planetary-scale geospatial analysis"

---

## Getting Help

### Project-Specific Questions

- **GitHub Issues**: https://github.com/research-jumpstart/research-jumpstart/issues
- **Discussions**: https://github.com/research-jumpstart/research-jumpstart/discussions
- Tag your question with `environmental` and `ecosystem-monitoring`

### General Remote Sensing Questions

- **GIS Stack Exchange**: https://gis.stackexchange.com/
- **rasterio GitHub Discussions**: https://github.com/rasterio/rasterio/discussions
- **Stack Overflow**: Tag with `rasterio`, `remote-sensing`, `satellite-imagery`

### AWS Support

- **SageMaker Studio Lab**: studiolab-support@amazon.com
- **AWS Forums**: https://repost.aws/
- **AWS Support** (for production accounts)

---

## Contributing

Found a bug? Have an improvement? Want to add an extension?

1. **Open an issue** describing the problem/enhancement
2. **Fork the repository**
3. **Create a branch**: `git checkout -b environmental-improvements`
4. **Make your changes** with clear commit messages
5. **Test thoroughly** (include example outputs)
6. **Submit a pull request**

See main [CONTRIBUTING.md](../../../CONTRIBUTING.md) for detailed guidelines.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_environmental,
  title = {Multi-Sensor Environmental Monitoring: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/research-jumpstart/research-jumpstart},
  note = {Accessed: [date]}
}
```

And cite the satellite data you use:

```bibtex
@misc{usgs_landsat,
  title = {Landsat 8-9 Operational Land Imager / Thermal Infrared Sensor Collection 2 Level-1},
  author = {{U.S. Geological Survey}},
  year = {2020},
  url = {https://www.usgs.gov/landsat-missions/landsat-collection-2}
}
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../../../LICENSE) file for details.

The satellite data accessed in this project has its own citation requirements and usage terms. See individual data provider websites for details.

---

## Acknowledgments

- **USGS** for Landsat program and open data access
- **ESA** for Copernicus Sentinel program
- **AWS Open Data Program** for hosting satellite archives
- **rasterio/GDAL developers** for excellent geospatial tools
- **Research Jumpstart community** for contributions and feedback

---

## Version History

- **v1.0.0** (2025-11-13): Initial release
  - Studio Lab version with simulated data
  - Unified Studio version with S3 access
  - Land cover classification demonstration
  - Change detection workflow
  - Comprehensive documentation

**Planned features**:
- v1.1.0: Workshop materials and exercises
- v1.2.0: Multi-sensor fusion examples
- v2.0.0: Deep learning extensions
- v2.1.0: Operational monitoring pipeline

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
- [Environmental science projects](../../../docs/projects/environmental.md)

---

*Last updated: 2025-11-13 | Research Jumpstart v1.0.0*
