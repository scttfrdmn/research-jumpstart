# Sample Climate Data

This directory should contain CMIP6 climate model data files for the project.

## Getting Data

### Option 1: Use AWS Open Data (Recommended)

Access CMIP6 data directly from AWS S3:

```bash
# List available CMIP6 models
aws s3 ls s3://cmip6-pds/CMIP6/ScenarioMIP/

# Example: Download a small subset (CESM2 model)
aws s3 cp s3://cmip6-pds/CMIP6/ScenarioMIP/NCAR/CESM2/ssp245/... local_file.nc
```

Available models on AWS S3:
- CESM2
- GFDL-CM4
- GFDL-ESM4
- UKESM1-0-LL
- IPSL-CM6A-LR
- MPI-ESM1-2-HR
- And many more...

### Option 2: Download from CMIP6 Archive

```bash
# CMIP6 data portal
# https://esgf-node.llnl.gov/projects/cmip6/

# Or using ESGF CLI tools
esgf-client download --dataset CMIP6.ScenarioMIP.NCAR.CESM2.ssp245
```

### Option 3: Generate Sample Data

If you don't have CMIP6 data, you can generate synthetic test data:

```python
import numpy as np
import xarray as xr

# Create sample temperature data
lat = np.linspace(-90, 90, 180)
lon = np.linspace(-180, 180, 360)
time = np.arange(1950, 2100)  # 150 years

temperature = np.random.randn(len(time), len(lat), len(lon)) * 2 + 288  # ~288K mean

ds = xr.Dataset({
    'tas': (['time', 'lat', 'lon'], temperature)
}, coords={
    'time': time,
    'lat': lat,
    'lon': lon
})

ds['tas'].attrs['units'] = 'K'
ds['tas'].attrs['long_name'] = 'Near Surface Air Temperature'

ds.to_netcdf('sample_temperature_data.nc')
print("✓ Created sample_temperature_data.nc")
```

## Data Format

Climate data should be in netCDF format with structure:

```
Dimensions:
- time: number of time steps (variable)
- lat: 180 (full resolution) or smaller
- lon: 360 (full resolution) or smaller

Variables:
- tas: Temperature (K)
- pr: Precipitation (kg m-2 s-1)
- (optional) other climate variables

Attributes:
- units: Variable units
- long_name: Variable description
- source: Model name
```

## Expected File Names

Place downloaded files here with naming convention:

```
sample_data/
├── CESM2_temperature_ssp245.nc
├── CESM2_precipitation_ssp245.nc
├── GFDL_temperature_ssp245.nc
├── GFDL_precipitation_ssp245.nc
└── ... more model files
```

## File Size Guidelines

For Tier 2 project:
- **Minimum:** 100 MB (1-2 small regions)
- **Recommended:** 500 MB - 5 GB (multiple variables/models)
- **Maximum:** ~10 GB (limited by Lambda memory)

Larger files will take longer to upload and process.

## Upload to Project

Once you have data files, upload them using:

```bash
# Upload all files
python scripts/upload_to_s3.py --bucket climate-data-xxxx --data-dir sample_data/

# Or manually
aws s3 cp sample_data/ s3://climate-data-xxxx/raw/ --recursive
```

## No Data Yet?

If you don't have data files:

1. **Generate synthetic data** (see Option 3 above)
2. **Use tiny subset** from test_data.nc (if available)
3. **Follow setup_guide.md** which includes data acquisition steps

## Storage in Git

⚠️  Large files should NOT be committed to git.

Instead:
- Add `.gitignore` entry: `sample_data/*.nc`
- Document where to get data
- Use `git-lfs` for large files (optional)

## Questions?

See:
- [README.md](../README.md) - Project overview
- [setup_guide.md](../setup_guide.md) - AWS setup with data acquisition
- [CMIP6 documentation](https://pcmdi.llnl.gov/CMIP6/) - Climate data details
