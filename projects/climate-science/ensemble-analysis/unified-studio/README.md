# Climate Ensemble Analysis - Unified Studio (Production)

This is the production version of the Climate Model Ensemble Analysis project, designed for AWS SageMaker Unified Studio with full S3/CMIP6 data access.

## Prerequisites

- AWS account with billing enabled
- SageMaker Unified Studio domain configured
- IAM permissions for:
  - S3 read access (CMIP6 public dataset)
  - Amazon Bedrock (Claude 3)
  - SageMaker compute instances
- Estimated cost: $20-30 per analysis

## Quick Start

### 1. Deploy Infrastructure

Use CloudFormation to set up required resources:

```bash
cd cloudformation
aws cloudformation create-stack \
  --stack-name climate-ensemble-analysis \
  --template-body file://climate-analysis-stack.yml \
  --parameters file://parameters.json \
  --capabilities CAPABILITY_IAM
```

Wait for stack creation (10-15 minutes):
```bash
aws cloudformation wait stack-create-complete \
  --stack-name climate-ensemble-analysis
```

### 2. Launch Unified Studio

1. Open AWS Console → SageMaker → Unified Studio
2. Navigate to your domain
3. Launch JupyterLab environment
4. Clone this repository:
   ```bash
   git clone https://github.com/research-jumpstart/research-jumpstart.git
   cd research-jumpstart/projects/climate-science/ensemble-analysis/unified-studio
   ```

### 3. Set Up Environment

Create conda environment from requirements:

```bash
# Create environment
conda create -n climate-analysis python=3.10 -y
conda activate climate-analysis

# Install dependencies
pip install -r requirements.txt

# Install local package
pip install -e .
```

### 4. Verify Setup

Run verification notebook to check all components:

```bash
jupyter lab
# Open: notebooks/00_verify_setup.ipynb
```

This will verify:
- ✓ S3 access to CMIP6 data
- ✓ Bedrock access (Claude 3)
- ✓ All Python packages installed
- ✓ Sample data can be loaded

### 5. Run Analysis

Follow notebooks in order:

1. **01_data_access.ipynb** - Learn to access CMIP6 from S3
2. **02_analysis.ipynb** - Multi-model ensemble processing
3. **03_visualization.ipynb** - Publication-quality figures
4. **04_bedrock_integration.ipynb** - AI-assisted interpretation

---

## Project Structure

```
unified-studio/
├── src/                          # Reusable Python modules
│   ├── __init__.py
│   ├── data_access.py           # S3/CMIP6 data access
│   ├── climate_analysis.py      # Core analysis functions
│   ├── ensemble_stats.py        # Statistical methods
│   ├── visualization.py         # Plotting utilities
│   └── bedrock_client.py        # AI integration
│
├── notebooks/                    # Analysis notebooks
│   ├── 00_verify_setup.ipynb
│   ├── 01_data_access.ipynb
│   ├── 02_analysis.ipynb
│   ├── 03_visualization.ipynb
│   └── 04_bedrock_integration.ipynb
│
├── cloudformation/               # Infrastructure as code
│   ├── climate-analysis-stack.yml
│   └── parameters.json
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
└── README.md                     # This file
```

---

## Python Modules Documentation

### data_access.py

Access CMIP6 data from S3 without local downloads:

```python
from src.data_access import CMIP6DataAccess

# Initialize client
client = CMIP6DataAccess(use_anon=True)

# Load single model
data = client.load_model_data(
    model='CESM2',
    experiment='ssp245',
    variable='tas',
    time_slice=('2015', '2050')
)

# Load ensemble
ensemble = client.load_ensemble(
    models=['CESM2', 'GFDL-CM4', 'UKESM1-0-LL'],
    experiment='ssp245',
    variable='tas',
    time_slice=('2015', '2100')
)
```

**Key features**:
- Direct S3 access (no egress charges in us-east-1)
- Zarr format for cloud-optimized reads
- Lazy loading with dask
- Automatic error handling and retry

### climate_analysis.py

Core climate analysis functions:

```python
from src.climate_analysis import (
    calculate_regional_mean,
    calculate_anomaly,
    annual_mean,
    calculate_trend
)

# Regional mean with area weighting
region = {'lat_min': 31, 'lat_max': 37, 'lon_min': -114, 'lon_max': -109}
regional_mean = calculate_regional_mean(ds, 'tas', region, weights='cosine')

# Temperature anomaly
anomaly = calculate_anomaly(regional_mean, baseline_period=('1995', '2014'))

# Annual average
annual = annual_mean(anomaly)

# Linear trend
slope, intercept = calculate_trend(annual, time_period=('2015', '2050'))
print(f"Warming rate: {slope:.3f} °C/year")
```

**Available functions**:
- `calculate_regional_mean()` - Area-weighted spatial averaging
- `calculate_anomaly()` - Relative to baseline
- `annual_mean()`, `seasonal_mean()` - Temporal aggregation
- `calculate_trend()` - Linear regression
- `running_mean()` - Smoothing
- `detrend()` - Remove trends
- `calculate_climatology()` - Seasonal cycles
- `convert_units()` - Unit conversions

### ensemble_stats.py

Ensemble statistical methods:

```python
from src.ensemble_stats import (
    create_ensemble,
    ensemble_mean,
    ensemble_std,
    model_agreement,
    signal_to_noise,
    identify_outliers
)

# Create ensemble array
ensemble = create_ensemble(model_data_dict, align_time=True)

# Basic statistics
ens_mean = ensemble_mean(ensemble)
ens_std = ensemble_std(ensemble)

# Model agreement
agreement = model_agreement(ensemble, method='sign')
print(f"Model agreement: {agreement.values[-1]:.1%}")

# Signal-to-noise ratio
snr = signal_to_noise(ensemble)
high_confidence = snr > 2  # SNR > 2 indicates strong signal

# Identify outliers
outliers = identify_outliers(ensemble, method='iqr')
```

**Available functions**:
- `create_ensemble()` - Combine models
- `ensemble_mean()`, `ensemble_std()` - Basic stats
- `ensemble_percentiles()` - Quantiles
- `model_agreement()` - Consensus metrics
- `signal_to_noise()` - Confidence indicator
- `identify_outliers()` - Anomalous models
- `ensemble_summary_stats()` - Comprehensive summary

### visualization.py

Publication-quality plotting:

```python
from src.visualization import (
    plot_ensemble_timeseries,
    plot_model_agreement,
    plot_regional_map,
    create_summary_figure
)

# Time series with uncertainty
fig = plot_ensemble_timeseries(
    ensemble,
    title='US Southwest Temperature Projection',
    ylabel='Temperature Anomaly (°C)',
    save_path='figures/ensemble_timeseries.png'
)

# Model agreement analysis
fig = plot_model_agreement(
    ensemble,
    title='Model Agreement Analysis',
    save_path='figures/model_agreement.png'
)

# Comprehensive summary figure
fig = create_summary_figure(
    ensemble,
    region=region_dict,
    spatial_data=spatial_mean,
    save_path='figures/summary.png'
)
```

**Available functions**:
- `plot_ensemble_timeseries()` - Time series with uncertainty bands
- `plot_model_agreement()` - Decadal box plots and spread
- `plot_regional_map()` - Cartopy maps with analysis domain
- `plot_scenario_comparison()` - Compare SSPs
- `create_summary_figure()` - Multi-panel overview

### bedrock_client.py

AI-assisted analysis with Amazon Bedrock:

```python
from src.bedrock_client import BedrockClimateAssistant

# Initialize assistant (requires Bedrock access)
assistant = BedrockClimateAssistant(
    model_id='anthropic.claude-3-sonnet-20240229-v1:0'
)

# Interpret results
interpretation = assistant.interpret_projection(
    stats={'mean': 2.5, 'std': 0.8, 'n_models': 15},
    region_name='US Southwest',
    scenario='SSP2-4.5'
)
print(interpretation)

# Compare to literature
comparison = assistant.compare_to_literature(
    "We find 2.5°C warming by 2050",
    "US Southwest"
)

# Generate methods section
methods = assistant.generate_methods_section(config_dict)

# Explain outliers
explanation = assistant.identify_outliers_explanation(
    outlier_models=['UKESM1-0-LL'],
    ensemble_stats=stats_dict
)
```

**Available methods**:
- `interpret_projection()` - Explain results scientifically
- `compare_to_literature()` - Context from IPCC/papers
- `generate_methods_section()` - Paper text generation
- `identify_outliers_explanation()` - Why models differ
- `generate_figure_caption()` - Publication captions
- `suggest_next_analyses()` - Follow-up ideas

---

## Configuration

### AWS Credentials

Credentials are automatically handled by SageMaker execution role. No manual configuration needed.

To verify S3 access:
```python
from src.data_access import check_s3_access
check_s3_access()  # Should print: ✓ S3 access verified
```

### Bedrock Access

Bedrock requires explicit model access. Enable in AWS Console:

1. Open AWS Console → Bedrock
2. Navigate to Model Access
3. Request access to: `Claude 3 Sonnet`
4. Wait for approval (usually < 5 minutes)

Verify:
```python
from src.bedrock_client import BedrockClimateAssistant
assistant = BedrockClimateAssistant()
assistant.check_bedrock_access()  # Should print: ✓ Bedrock access verified
```

### Environment Variables (Optional)

Create `.env` file for custom configuration:

```bash
# AWS region (default: us-east-1)
AWS_REGION=us-east-1

# Bedrock model (default: Claude 3 Sonnet)
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Logging level
LOG_LEVEL=INFO

# Output directory
OUTPUT_DIR=./output
```

---

## Common Tasks

### Analyze Different Region

```python
# Define custom region
region = {
    'name': 'Pacific Northwest',
    'lat_min': 42.0,
    'lat_max': 49.0,
    'lon_min': -125.0,
    'lon_max': -116.0
}

# Run analysis
regional_mean = calculate_regional_mean(ds, 'tas', region)
# ... continue with analysis
```

### Compare Scenarios

```python
# Load multiple scenarios
scenarios = {}
for scenario in ['ssp126', 'ssp245', 'ssp585']:
    data = client.load_model_data('CESM2', scenario, 'tas')
    scenarios[scenario.upper()] = data

# Plot comparison
fig = plot_scenario_comparison(
    scenarios,
    title='Scenario Comparison: US Southwest'
)
```

### Process Many Models in Parallel

```python
from dask.distributed import Client

# Start dask cluster
dask_client = Client()

# Load ensemble with parallel processing
models = client.get_available_models('ssp245')
ensemble = client.load_ensemble(
    models=models,
    experiment='ssp245',
    variable='tas'
)

# Dask will parallelize operations automatically
ensemble_mean = ensemble.mean('model').compute()
```

---

## Troubleshooting

### S3 Access Errors

```
Error: NoCredentialsError or AccessDenied
```

**Solution**:
1. Check IAM role attached to SageMaker execution role
2. Required policy: `AmazonS3ReadOnlyAccess`
3. CMIP6 bucket (`s3://cmip6-pds`) is public, use `use_anon=True`

### Bedrock Access Errors

```
Error: AccessDeniedException
```

**Solution**:
1. Enable Bedrock in your AWS region (us-east-1 recommended)
2. Request Claude 3 model access in Bedrock console
3. Add Bedrock permissions to SageMaker execution role:
   ```json
   {
     "Effect": "Allow",
     "Action": [
       "bedrock:InvokeModel",
       "bedrock:InvokeModelWithResponseStream"
     ],
     "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
   }
   ```

### Out of Memory

```
Error: MemoryError or killed kernel
```

**Solution**:
1. Use dask for lazy evaluation:
   ```python
   data = xr.open_zarr(..., chunks={'time': 12})
   ```
2. Process models in batches instead of all at once
3. Use larger instance: ml.m5.2xlarge (32GB RAM)
4. Clear variables: `del large_dataset`

### Slow Data Access

```
Problem: Data loading takes very long
```

**Solution**:
1. Verify you're in `us-east-1` region (same as CMIP6 bucket)
2. Use Zarr format (already default)
3. Subset spatially before reading:
   ```python
   ds = ds.sel(lat=slice(30, 40), lon=slice(-120, -110))
   ```
4. Select only needed variables:
   ```python
   ds = ds[['tas']]  # Don't load all variables
   ```

### Cost Higher Than Expected

**Check**:
1. Data egress charges → Use `us-east-1` region
2. Compute running idle → Stop instances when done
3. Large result files → Clean up S3 output bucket regularly
4. Bedrock usage → Monitor token usage in CloudWatch

**Set up billing alerts**:
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name climate-analysis-budget \
  --alarm-description "Alert at $50" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --evaluation-periods 1 \
  --threshold 50 \
  --comparison-operator GreaterThanThreshold
```

---

## Cost Optimization

### Reduce Compute Costs

- Use spot instances (60-80% savings)
- Stop instances when not in use
- Use smaller instance types for light tasks
- Process multiple analyses in single session

### Reduce Storage Costs

- Delete intermediate results regularly
- Use S3 Lifecycle policies to move old data to Glacier
- Don't duplicate CMIP6 data (access directly from public bucket)

### Reduce Bedrock Costs

- Use Haiku model for simple tasks (cheaper than Sonnet)
- Cache common interpretations
- Batch multiple questions into single prompt
- Monitor token usage in CloudWatch

---

## Getting Help

- **Issues**: https://github.com/research-jumpstart/research-jumpstart/issues
- **Discussions**: https://github.com/research-jumpstart/research-jumpstart/discussions
- **AWS Support**: For account/billing issues
- **Pangeo Discourse**: For climate data questions

---

## Contributing

See main project [CONTRIBUTING.md](../../../../CONTRIBUTING.md) for guidelines.

---

## License

Apache License 2.0 - see [LICENSE](../../../../LICENSE)

---

*Last updated: 2025-11-09*
