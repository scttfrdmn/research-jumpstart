# Analysis Notebooks

This directory contains Jupyter notebooks for the Climate Ensemble Analysis workflow in AWS SageMaker Unified Studio.

## Notebook Workflow

Follow notebooks in order for complete analysis:

### 00_verify_setup.ipynb
**Purpose**: Verify all components are working
**Time**: 5-10 minutes
**Prerequisites**: None

**Contents**:
1. Check Python environment and packages
2. Verify S3 access to CMIP6 data
3. Verify Bedrock access (Claude 3)
4. Test loading sample data
5. Confirm all modules import correctly

**Expected outcome**: All checks pass with ✓ marks

---

### 01_data_access.ipynb
**Purpose**: Learn to access CMIP6 data from S3
**Time**: 20-30 minutes
**Prerequisites**: Setup verified

**Contents**:
1. Introduction to CMIP6 archive on AWS
2. Understanding S3 paths and Zarr format
3. Using `CMIP6DataAccess` class
4. Loading single model data
5. Loading multi-model ensemble
6. Spatial and temporal subsetting
7. Working with dask for lazy loading
8. Best practices for cloud data access

**Key code examples**:
```python
from src.data_access import CMIP6DataAccess

# Initialize client
client = CMIP6DataAccess()

# Load data
data = client.load_model_data(
    model='CESM2',
    experiment='ssp245',
    variable='tas',
    time_slice=('2015', '2050')
)
```

**Expected outcome**: Successfully load CMIP6 data without downloading files

---

### 02_analysis.ipynb
**Purpose**: Multi-model ensemble processing
**Time**: 30-45 minutes
**Prerequisites**: Notebook 01 completed

**Contents**:
1. Define analysis configuration
   - Models to include
   - Region of interest
   - Time periods (analysis + baseline)
   - Variables
2. Load ensemble data
3. Calculate regional means (area-weighted)
4. Compute temperature anomalies
5. Annual averaging
6. Ensemble statistics
   - Mean, standard deviation
   - Percentiles (10th, 25th, 75th, 90th)
   - Model agreement
   - Signal-to-noise ratio
7. Identify outlier models
8. Calculate trends
9. Save processed results to S3

**Key code examples**:
```python
from src.climate_analysis import calculate_regional_mean, calculate_anomaly
from src.ensemble_stats import create_ensemble, ensemble_mean

# Process each model
for model_name, model_data in ensemble_dict.items():
    # Regional mean
    regional_mean = calculate_regional_mean(model_data, 'tas', region)

    # Anomaly
    anomaly = calculate_anomaly(regional_mean, baseline_period)

    # Store
    processed[model_name] = anomaly

# Create ensemble
ensemble = create_ensemble(processed)
ens_mean = ensemble_mean(ensemble)
```

**Expected outcome**: Processed ensemble data ready for visualization

---

### 03_visualization.ipynb
**Purpose**: Create publication-quality figures
**Time**: 20-30 minutes
**Prerequisites**: Notebook 02 completed

**Contents**:
1. Load processed results from previous notebook
2. Time series visualization
   - Individual models (thin lines)
   - Ensemble mean (thick line)
   - Uncertainty range (shaded ±1σ)
3. Model agreement analysis
   - Decadal box plots
   - Spread evolution over time
4. Regional map
   - Analysis domain highlighted
   - Geographic context
5. Multi-panel summary figure
6. Export high-resolution figures (300 DPI)
7. Save figures to S3 results bucket

**Key code examples**:
```python
from src.visualization import (
    plot_ensemble_timeseries,
    plot_model_agreement,
    create_summary_figure
)

# Time series
fig1 = plot_ensemble_timeseries(
    ensemble,
    title='US Southwest Temperature Projection (SSP2-4.5)',
    save_path='s3://results-bucket/figures/timeseries.png'
)

# Model agreement
fig2 = plot_model_agreement(
    ensemble,
    save_path='s3://results-bucket/figures/agreement.png'
)
```

**Expected outcome**: Professional figures ready for papers/presentations

---

### 04_bedrock_integration.ipynb
**Purpose**: AI-assisted analysis and interpretation
**Time**: 15-25 minutes
**Prerequisites**: Notebooks 02-03 completed

**Contents**:
1. Initialize Bedrock client (Claude 3)
2. Generate scientific interpretation
   - Magnitude of projected change
   - Level of model agreement
   - Confidence assessment
   - Regional implications
3. Compare to IPCC findings
4. Explain outlier models
5. Generate methods section text
6. Create figure captions
7. Suggest follow-up analyses
8. Export AI-generated content

**Key code examples**:
```python
from src.bedrock_client import BedrockClimateAssistant

# Initialize AI assistant
assistant = BedrockClimateAssistant()

# Interpret results
interpretation = assistant.interpret_projection(
    stats=ensemble_stats,
    region_name='US Southwest',
    scenario='SSP2-4.5'
)

print(interpretation)

# Generate methods section
methods = assistant.generate_methods_section({
    'models': model_list,
    'scenario': 'SSP2-4.5',
    'region': 'US Southwest',
    'variable': 'Surface air temperature'
})
```

**Expected outcome**: AI-generated scientific text ready for papers

---

## Running the Complete Workflow

### Quick run (all notebooks):
```bash
# Start JupyterLab
jupyter lab

# Or run all notebooks via command line:
jupyter nbconvert --execute --to notebook \
  --output-dir=./output \
  00_verify_setup.ipynb \
  01_data_access.ipynb \
  02_analysis.ipynb \
  03_visualization.ipynb \
  04_bedrock_integration.ipynb
```

### Customizing the Analysis

Edit these sections in **02_analysis.ipynb**:

**Models**:
```python
MODELS = [
    'CESM2',
    'GFDL-CM4',
    'UKESM1-0-LL',
    # Add more models from get_available_models()
]
```

**Region**:
```python
REGION = {
    'name': 'Custom Region',
    'lat_min': 40.0,
    'lat_max': 50.0,
    'lon_min': -120.0,
    'lon_max': -110.0
}
```

**Scenario**:
```python
SCENARIO = 'ssp126'  # or 'ssp245', 'ssp370', 'ssp585'
```

**Time periods**:
```python
ANALYSIS_PERIOD = ('2015', '2100')
BASELINE_PERIOD = ('1995', '2014')
```

---

## Expected Outputs

After running all notebooks, you should have:

### In S3 Results Bucket:

```
s3://climate-ensemble-analysis-results-{account-id}/
├── data/
│   ├── ensemble_mean_2025-11-09.nc
│   ├── ensemble_std_2025-11-09.nc
│   ├── regional_means_2025-11-09.nc
│   └── summary_statistics_2025-11-09.csv
├── figures/
│   ├── ensemble_timeseries_2025-11-09.png
│   ├── model_agreement_2025-11-09.png
│   ├── regional_map_2025-11-09.png
│   └── summary_figure_2025-11-09.png
└── reports/
    ├── interpretation_2025-11-09.txt
    ├── methods_section_2025-11-09.txt
    └── figure_captions_2025-11-09.txt
```

### Locally:

- Processed xarray Datasets in memory
- Matplotlib figure objects
- Pandas DataFrames with statistics
- Text files with AI-generated content

---

## Troubleshooting

### Kernel dies during execution

**Cause**: Out of memory

**Solutions**:
- Use dask for lazy loading: `chunks={'time': 12}`
- Process fewer models at once
- Use larger instance type: ml.m5.2xlarge
- Restart kernel and clear outputs

### S3 access errors

**Cause**: IAM permissions or region mismatch

**Solutions**:
- Verify SageMaker execution role has S3 read access
- Ensure you're in us-east-1 (same as CMIP6 bucket)
- Check `check_s3_access()` returns True

### Bedrock errors

**Cause**: Model access not enabled

**Solutions**:
- Enable Bedrock in AWS Console
- Request Claude 3 access (takes < 5 min)
- Add Bedrock permissions to execution role
- Check `assistant.check_bedrock_access()` returns True

### Slow performance

**Cause**: Large data transfers or inefficient operations

**Solutions**:
- Subset data spatially/temporally before reading
- Use Zarr format (already default)
- Enable dask parallelization
- Verify us-east-1 region (no egress charges)

---

## Next Steps

After completing these notebooks:

1. **Modify for your research**:
   - Change region to your area of interest
   - Add more models for robustness
   - Try different variables (pr, tos, etc.)
   - Compare multiple scenarios

2. **Scale up**:
   - Process full model ensemble (20+ models)
   - Analyze multiple regions
   - Add seasonal analysis
   - Compute additional metrics

3. **Automate**:
   - Convert notebooks to Python scripts
   - Set up scheduled runs
   - Create parameterized workflows
   - Build data pipelines

4. **Share results**:
   - Export to paper-ready figures
   - Generate reports automatically
   - Share S3 results with collaborators
   - Publish findings

---

## Additional Resources

- **Full documentation**: ../README.md
- **Module documentation**: ../src/
- **Example outputs**: ../assets/sample-outputs/
- **Troubleshooting**: Main project README
- **Support**: GitHub Issues

---

*Last updated: 2025-11-09*
