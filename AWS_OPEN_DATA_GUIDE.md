# AWS Open Data Integration Guide

This guide explains how to access petabytes of research data from the AWS Open Data Registry across Research Jumpstart projects.

## Overview

AWS Open Data Registry provides free, public datasets for research in:
- Climate Science (CMIP6, ERA5, NOAA GFS)
- Medical Imaging (NIH Chest X-rays, TCIA, Medical Segmentation)
- Genomics (1000 Genomes, TCGA, gnomAD)
- Satellite Imagery (Sentinel, Landsat, MODIS)
- And many more domains

**Key Benefits:**
- No download costs
- No egress fees (when processing in AWS)
- Direct S3 access with boto3 or AWS CLI
- Pre-integrated into Research Jumpstart projects

## Quick Start

### 1. Install AWS Tools

```bash
# Install AWS CLI and boto3
pip install awscli boto3 s3fs

# Optional: Configure credentials (not required for public datasets)
aws configure
```

### 2. Access Data Without Credentials

Most datasets are public and require no AWS account:

```python
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Create anonymous S3 client
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# List files
response = s3.list_objects_v2(Bucket='cmip6-pds', Prefix='CMIP6/', MaxKeys=10)
for obj in response['Contents']:
    print(obj['Key'])
```

### 3. Use Project-Specific Scripts

Each project includes ready-to-use AWS data access scripts:

```python
# Climate Science
from projects.climate_science.regional_climate_modeling.scripts.aws_data_access import list_cmip6_data
files = list_cmip6_data(variable='tas', experiment='historical')

# Medical Imaging
from projects.medical.image_classification.scripts.aws_data_access import download_sample_images
download_sample_images(n_samples=100, disease_filter='Pneumonia')

# Genomics
from projects.genomics.variant_analysis.studio_lab.aws_data_access import list_1000genomes
vcf_files = list_1000genomes(chromosome='chr22')
```

## Datasets by Domain

### Climate Science

**Project:** `projects/climate-science/regional-climate-modeling/`

#### CMIP6 Climate Models
- **Bucket:** `s3://cmip6-pds`
- **Size:** 400+ TB
- **Content:** Climate projections from 49 modeling centers
- **Variables:** Temperature, precipitation, humidity, wind, sea level, etc.
- **Access:** `scripts/aws_data_access.py`

```python
from scripts.aws_data_access import list_cmip6_data, download_cmip6_file

# List CESM2 historical temperature data
files = list_cmip6_data(
    variable='tas',
    experiment='historical',
    model='CESM2',
    frequency='mon'
)

# Download file
download_cmip6_file(files[0], output_path='data/cmip6_tas.nc')
```

#### ERA5 Reanalysis
- **Bucket:** `s3://era5-pds`
- **Size:** 500+ TB
- **Content:** Hourly global reanalysis from 1940-present
- **Variables:** All atmospheric and surface variables
- **Resolution:** 0.25° (~31 km)

```python
from scripts.aws_data_access import list_era5_data, download_era5_sample

# List 2020 data
files = list_era5_data(variable='2m_temperature', year=2020, month=1)

# Download sample
download_era5_sample(variable='2m_temperature', year=2020, month=1)
```

#### NOAA GFS Forecasts
- **Bucket:** `s3://noaa-gfs-bdp-pds`
- **Content:** Operational weather forecasts
- **Update:** 4 times daily
- **Resolution:** 0.25° global

```python
from scripts.aws_data_access import list_noaa_gfs_data

# List recent forecasts
files = list_noaa_gfs_data(date='20240101', hour='00')
```

**Documentation:** [Climate AWS Data Access](projects/climate-science/regional-climate-modeling/README.md#aws-open-data-registry-recommended)

### Medical Imaging

**Project:** `projects/medical/image-classification/`

#### NIH Chest X-ray14
- **Bucket:** `s3://nih-chest-xrays`
- **Size:** 45 GB
- **Content:** 112,120 frontal-view chest X-rays
- **Labels:** 14 disease categories
- **Format:** PNG images

```python
from scripts.aws_data_access import download_sample_images, download_nih_metadata

# Download metadata
metadata = download_nih_metadata('data/nih_metadata.csv')

# Download pneumonia cases
download_sample_images(
    output_dir='data/chest_xrays',
    n_samples=100,
    disease_filter='Pneumonia'
)
```

#### The Cancer Imaging Archive (TCIA)
- **Bucket:** `s3://imaging.nci.nih.gov`
- **Size:** Multiple TB
- **Content:** Cancer imaging from clinical trials
- **Modalities:** CT, MRI, PET
- **Collections:** 33 cancer types

```python
from scripts.aws_data_access import list_tcia_collections

# List available collections
collections = list_tcia_collections()
```

#### Medical Segmentation Decathlon
- **Bucket:** `s3://medicalsegmentation`
- **Size:** 35 GB
- **Content:** 10 organ segmentation tasks
- **Organs:** Brain, heart, liver, lung, prostate, pancreas, etc.
- **Format:** NIfTI (.nii.gz)

```python
from scripts.aws_data_access import get_medical_seg_decathlon_info

# Get dataset info
info = get_medical_seg_decathlon_info()
```

**Documentation:** [Medical AWS Data Access](projects/medical/image-classification/README.md#aws-open-data-registry-recommended)

### Genomics

**Project:** `projects/genomics/variant-analysis/`

#### 1000 Genomes Project
- **Bucket:** `s3://1000genomes`
- **Size:** 200 TB
- **Content:** Genetic variation from 2,504 individuals
- **Populations:** 26 populations, 5 super-populations
- **Format:** VCF, BAM, CRAM

```python
from studio_lab.aws_data_access import list_1000genomes, download_sample_vcf

# List chromosome 22 variants
files = list_1000genomes(
    phase='phase3',
    chromosome='chr22',
    data_type='vcf'
)

# Download sample
download_sample_vcf(chromosome='chr22', output_dir='data/')
```

#### TCGA (The Cancer Genome Atlas)
- **Bucket:** `s3://tcga-2-open`
- **Size:** 2.5 PB
- **Content:** Multi-omic cancer data
- **Types:** WGS, WXS, RNA-Seq, Methylation
- **Samples:** 33 cancer types, 11,000+ patients

```python
from studio_lab.aws_data_access import list_tcga_data

# List breast cancer data
files = list_tcga_data(
    project='TCGA-BRCA',
    data_category='Transcriptome Profiling'
)
```

#### gnomAD (Genome Aggregation Database)
- **Bucket:** `s3://gnomad-public-us-east-1`
- **Size:** 20 TB
- **Content:** Population allele frequencies
- **Samples:** 125,748 exomes + 71,702 genomes
- **Format:** VCF

```python
from studio_lab.aws_data_access import list_gnomad_data

# List gnomAD v3 data
files = list_gnomad_data(version='v3', data_type='vcf')
```

**Documentation:** [Genomics AWS Data Access](projects/genomics/variant-analysis/README.md#aws-open-data-registry-for-real-analysis)

## AWS CLI Examples

### List Bucket Contents

```bash
# Climate: List CMIP6 data
aws s3 ls s3://cmip6-pds/CMIP6/ --no-sign-request

# Medical: List chest X-rays
aws s3 ls s3://nih-chest-xrays/png/ --no-sign-request

# Genomics: List 1000 Genomes
aws s3 ls s3://1000genomes/phase3/ --no-sign-request
```

### Download Files

```bash
# Download single file
aws s3 cp s3://cmip6-pds/CMIP6/path/to/file.nc ./data/ --no-sign-request

# Download directory
aws s3 sync s3://nih-chest-xrays/png/ ./chest_xrays/ --no-sign-request

# Download with filter
aws s3 sync s3://1000genomes/phase3/ ./1kg/ \
  --exclude "*" \
  --include "*chr22*" \
  --no-sign-request
```

### Stream Data

```bash
# Stream compressed file to stdout
aws s3 cp s3://cmip6-pds/path/to/file.nc.gz - --no-sign-request | gunzip | head
```

## Python Integration

### Direct S3 Access with xarray

```python
import xarray as xr
import s3fs

# Setup S3 filesystem
fs = s3fs.S3FileSystem(anon=True)

# Open NetCDF directly from S3
with fs.open('s3://cmip6-pds/CMIP6/path/to/file.nc', 'rb') as f:
    ds = xr.open_dataset(f)
    print(ds)
```

### Parallel Downloads

```python
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore.config import Config
from botocore import UNSIGNED

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def download_file(key, bucket, output_dir):
    filename = key.split('/')[-1]
    s3.download_file(bucket, key, f'{output_dir}/{filename}')
    return filename

# Download multiple files in parallel
files_to_download = ['file1.nc', 'file2.nc', 'file3.nc']

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(download_file, f, 'cmip6-pds', 'data/')
        for f in files_to_download
    ]
    for future in futures:
        print(f"Downloaded: {future.result()}")
```

### Progress Tracking

```python
from tqdm import tqdm

def download_with_progress(bucket, key, output_path):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Get file size
    response = s3.head_object(Bucket=bucket, Key=key)
    file_size = response['ContentLength']

    # Download with progress bar
    with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
        def callback(bytes_transferred):
            pbar.update(bytes_transferred)

        s3.download_file(bucket, key, output_path, Callback=callback)

# Use it
download_with_progress('cmip6-pds', 'CMIP6/path/to/file.nc', 'data/file.nc')
```

## Cost Optimization

### 1. Process Data in AWS

To avoid egress fees, run analysis in AWS:
- **SageMaker Notebooks:** Jupyter notebooks with AWS integration
- **EC2 Instances:** Flexible compute for large analyses
- **AWS Batch:** Parallel processing of many files

### 2. Use Requester-Pays Buckets

Some buckets are requester-pays (you pay for data transfer):

```python
# Enable requester-pays
s3 = boto3.client('s3')  # With credentials
response = s3.list_objects_v2(
    Bucket='bucket-name',
    RequestPayer='requester'
)
```

### 3. Sample First, Download Later

```python
# List and filter before downloading
files = list_cmip6_data(variable='tas', experiment='historical', max_results=1000)

# Filter to specific region/time
relevant_files = [f for f in files if 'NAM' in f and '2015' in f]

# Download only what you need
for f in relevant_files[:10]:
    download_cmip6_file(f, output_dir='data/')
```

## Best Practices

### 1. Check Dataset Documentation

Each dataset has specific structure and conventions:
- **CMIP6:** ESGF DRS path conventions
- **Medical:** DICOM vs PNG formats
- **Genomics:** VCF, BAM, CRAM formats

### 2. Use Appropriate Tools

- **NetCDF/HDF5:** xarray, h5py
- **DICOM:** pydicom
- **VCF:** pysam, cyvcf2
- **Images:** pillow, opencv, SimpleITK

### 3. Handle Large Files

```python
# Stream large files instead of downloading
import xarray as xr
import s3fs

fs = s3fs.S3FileSystem(anon=True)

# Open dataset
ds = xr.open_dataset(
    's3://cmip6-pds/path/to/large_file.nc',
    engine='h5netcdf',
    chunks={'time': 100}  # Chunk for efficient access
)

# Extract subset
subset = ds.sel(lat=slice(30, 50), lon=slice(-120, -80))
subset.to_netcdf('subset.nc')
```

### 4. Validate Data

```python
# Check file integrity
import hashlib

def verify_checksum(filepath, expected_md5):
    md5_hash = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest() == expected_md5
```

## Troubleshooting

### Issue: Access Denied

```
botocore.exceptions.ClientError: An error occurred (403) when calling the HeadObject operation: Forbidden
```

**Solution:** Use anonymous access for public buckets:

```python
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
```

### Issue: Slow Downloads

**Solutions:**
1. Use AWS EC2/SageMaker in same region as bucket (usually us-east-1)
2. Enable parallel downloads with ThreadPoolExecutor
3. Use AWS DataSync for large batch transfers

### Issue: Out of Memory

**Solutions:**
1. Use chunked/streaming processing:
   ```python
   ds = xr.open_dataset('file.nc', chunks={'time': 100})
   ```

2. Process files one at a time:
   ```python
   for file in file_list:
       ds = xr.open_dataset(file)
       result = ds.mean()
       ds.close()
   ```

3. Use Dask for distributed processing

## Additional Resources

### AWS Open Data Registry
- **Website:** https://registry.opendata.aws/
- **Search:** Browse 400+ datasets
- **Documentation:** Usage examples and tutorials

### Dataset-Specific Resources

**Climate:**
- CMIP6: https://pcmdi.llnl.gov/CMIP6/
- ERA5: https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
- NOAA: https://www.ncei.noaa.gov/

**Medical:**
- NIH Chest X-ray: https://nihcc.app.box.com/v/ChestXray-NIHCC
- TCIA: https://www.cancerimagingarchive.net/
- Medical Segmentation: http://medicaldecathlon.com/

**Genomics:**
- 1000 Genomes: https://www.internationalgenome.org/
- TCGA: https://www.cancer.gov/tcga
- gnomAD: https://gnomad.broadinstitute.org/

### AWS Tools Documentation
- **boto3:** https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
- **AWS CLI:** https://docs.aws.amazon.com/cli/
- **s3fs:** https://s3fs.readthedocs.io/

## Support

For issues with:
- **Research Jumpstart projects:** Open an issue on GitHub
- **AWS Open Data datasets:** Contact dataset maintainers (see Registry)
- **AWS services:** AWS Support or forums

---

**Last Updated:** 2025-11-10

For the latest information, visit the AWS Open Data Registry: https://registry.opendata.aws/
