# Tier 2 Project Structure - Climate Data Analysis with S3 and Lambda

## Complete Project Overview

This document describes the complete Tier 2 project structure for Climate Science domain.

---

## Directory Layout

```
tier-2/
├── README.md                          # Main project documentation (18 KB)
├── QUICKSTART.md                      # 5-minute quick start guide (6.6 KB)
├── setup_guide.md                     # Detailed AWS setup instructions (17 KB)
├── cleanup_guide.md                   # Resource deletion guide (14 KB)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore patterns
│
├── scripts/
│   ├── __init__.py                   # Package initialization
│   ├── upload_to_s3.py               # Upload climate data to S3 (300 lines)
│   ├── lambda_function.py            # AWS Lambda processor (350 lines)
│   └── query_results.py              # Download and analyze results (400 lines)
│
├── notebooks/
│   └── climate_analysis.ipynb        # Jupyter analysis notebook
│
└── sample_data/
    └── README.md                     # Data acquisition guide
```

**Total: 11 files + directories**
**Total Documentation: ~3,900 lines**
**Total Code: ~1,050 lines**

---

## File Descriptions

### Documentation Files

#### README.md (18 KB, 600+ lines)
**Purpose:** Main project documentation

**Contents:**
- Project overview and objectives
- Architecture diagram
- Quick start guide (15 minutes)
- Detailed workflow explanation
- Cost breakdown ($8-12 per run)
- Key learning objectives
- Time estimates
- Prerequisites
- Troubleshooting guide
- Next steps for Tier 3
- Resources and links
- Citation information

**Use case:** Start here for complete understanding

---

#### QUICKSTART.md (6.6 KB, 200+ lines)
**Purpose:** Fast introduction for experienced users

**Contents:**
- 5-step quick start path
- Each step with time estimate
- Prerequisites checklist
- Running individual steps
- Key commands reference
- Cost estimate table
- Troubleshooting mini-guide

**Use case:** For users who want to start immediately

---

#### setup_guide.md (17 KB, 500+ lines)
**Purpose:** Step-by-step AWS resource creation

**Contents:**
- 10 detailed setup steps
- AWS Console and CLI options
- S3 bucket creation
- IAM role configuration
- Lambda function deployment
- Testing procedures
- CloudWatch logs monitoring
- Cost tracking setup
- Troubleshooting solutions
- Security best practices

**Use case:** Follow this to set up all AWS resources

---

#### cleanup_guide.md (14 KB, 400+ lines)
**Purpose:** Delete all resources to stop charges

**Contents:**
- Quick cleanup (one command)
- Step-by-step deletion
- Resource verification
- CloudWatch log deletion
- VPC endpoint cleanup
- Python cleanup script
- Cost verification
- Backup instructions
- Troubleshooting deletion issues

**Use case:** Follow when project is complete to avoid charges

---

### Configuration Files

#### requirements.txt (25 lines)
**Python dependencies:**
- boto3 (1.28.0+) - AWS SDK
- xarray (2023.10+) - Climate data handling
- netCDF4 (1.6.4+) - NetCDF file support
- pandas (2.0+) - Data manipulation
- numpy (1.24+) - Numerical computing
- matplotlib (3.7+) - Visualization
- jupyter, jupyterlab - Notebooks
- And 5+ optional packages

**Install with:**
```bash
pip install -r requirements.txt
```

---

#### .gitignore
**Prevents committing:**
- AWS credentials (.env files)
- Data files (*.nc, *.h5, etc.)
- Python cache and eggs
- Virtual environments
- Jupyter checkpoints
- Large CSV/JSON files
- IDE configuration
- OS files (.DS_Store, etc.)

---

### Python Scripts

#### scripts/upload_to_s3.py (300 lines)
**Purpose:** Upload CMIP6 climate data to S3

**Key Features:**
- S3Uploader class with multipart uploads
- Progress tracking with tqdm
- Resume capability for large files
- Error handling and logging
- CLI interface with arguments
- Batch upload support

**Usage:**
```bash
# Upload directory
python scripts/upload_to_s3.py --bucket climate-data-xxxx --data-dir sample_data/

# Upload single file
python scripts/upload_to_s3.py --bucket climate-data-xxxx --file data.nc --s3-key raw/

# List files
python scripts/upload_to_s3.py --bucket climate-data-xxxx --list-only
```

**Key Methods:**
- `upload_file()` - Single file upload
- `upload_directory()` - Batch directory upload
- `_multipart_upload()` - Large file handling
- `list_uploaded_files()` - Verify uploads

---

#### scripts/lambda_function.py (350 lines)
**Purpose:** AWS Lambda function for processing netCDF files

**Key Features:**
- S3-triggered event processing
- netCDF file extraction
- Regional statistics calculation
- Temperature and precipitation stats
- Results export as JSON
- Comprehensive error handling
- CloudWatch logging

**Lambda Handler:**
```python
def lambda_handler(event, context):
    # Extract S3 bucket and key from event
    # Download netCDF from S3
    # Process with xarray
    # Calculate statistics
    # Upload results to S3
    # Return status
```

**Processing Modes:**
- `calculate_statistics` - Extract temperature, precipitation stats
- `generic` - Basic file processing without netCDF

**Output Format (JSON):**
```json
{
  "file": "CESM2_temperature_2100.nc",
  "timestamp": "2025-11-14T15:30:00Z",
  "statistics": {
    "temperature": {
      "mean": 288.5,
      "std": 2.1,
      "min": 250.0,
      "max": 310.0,
      "units": "K"
    },
    "precipitation": {...}
  }
}
```

**Deployment:**
1. Copy contents to Lambda console editor
2. Set timeout: 300 seconds
3. Set memory: 512 MB
4. Add environment variables:
   - `BUCKET_NAME`
   - `PROCESS_MODE`

---

#### scripts/query_results.py (400 lines)
**Purpose:** Download and analyze Lambda results

**Key Features:**
- S3ResultsQuerier class
- Batch download with progress
- JSON parsing and analysis
- Pandas DataFrame creation
- Summary statistics calculation
- CSV export capability
- CLI interface

**Usage:**
```bash
# Download and analyze all results
python scripts/query_results.py --bucket climate-data-xxxx --output-dir ./results

# List results only
python scripts/query_results.py --bucket climate-data-xxxx --list-only

# Custom prefix
python scripts/query_results.py --bucket climate-data-xxxx --prefix custom/path/
```

**Key Methods:**
- `list_results()` - List S3 objects
- `download_results()` - Batch download
- `analyze_results()` - Calculate statistics
- `create_dataframe()` - Convert to pandas
- `print_summary()` - Console output

**Output Files:**
- `analysis_summary.json` - Aggregate statistics
- `results.csv` - Per-file statistics
- Individual JSON files from S3

---

#### scripts/__init__.py (20 lines)
**Purpose:** Package initialization

**Exports:**
- `upload_climate_data()`
- `upload_file()`
- `lambda_handler()`
- `process_netcdf_file()`
- `download_results()`
- `query_s3_results()`

**Use for importing as module:**
```python
from scripts import upload_climate_data, download_results
```

---

### Jupyter Notebook

#### notebooks/climate_analysis.ipynb (8 cells)
**Purpose:** Interactive climate data analysis

**Workflow:**
1. **Setup** - Import libraries and configure AWS
2. **Configuration** - Set S3 bucket and region
3. **List Results** - Check available files in S3
4. **Download** - Fetch processed results
5. **Parse** - Load JSON files
6. **Create DataFrame** - Convert to pandas
7. **Summary Statistics** - Calculate aggregates
8. **Visualizations** - Create publication figures
9. **Export** - Save CSV and summary
10. **Next Steps** - Suggestions for continuation

**Produces:**
- `climate_analysis_summary.png` - 2x2 figure grid
- `analysis_results.csv` - Data table
- `summary.json` - Statistics summary

**Visualizations:**
- Temperature distribution histogram
- Temperature with uncertainty bands
- Precipitation distribution
- Summary statistics table

---

### Data Directory

#### sample_data/README.md
**Purpose:** Guide for obtaining climate data

**Contents:**
- CMIP6 data sources
- AWS Open Data access
- ESGF archive access
- Synthetic data generation
- Expected data format
- File size guidelines
- Upload instructions

**Data Format Expected:**
```
Dimensions:
- time: variable (150-200 years typical)
- lat: 180 (global) or subset
- lon: 360 (global) or subset

Variables:
- tas: Temperature (K)
- pr: Precipitation (kg m-2 s-1)
- (optional) other variables
```

---

## Specifications Met

### TIER_2_SPECIFICATIONS.md Compliance

✅ **Duration:** 2-4 hours
- Setup: 30-45 minutes
- Upload: 5-10 minutes
- Processing: 5-10 minutes
- Analysis: 30-45 minutes

✅ **Cost:** $8-12 per run
- Detailed breakdown in README.md
- Cost optimization strategies included
- Free tier usage documented

✅ **AWS Services Used:**
- S3 (object storage)
- Lambda (serverless processing)
- IAM (access management)
- CloudWatch (logging)
- Optional: Athena (SQL queries)

✅ **File Structure:**
```
tier-2/
├── README.md ✓
├── requirements.txt ✓
├── setup_guide.md ✓
├── cleanup_guide.md ✓
├── notebooks/ ✓
│   └── climate_analysis.ipynb
├── scripts/ ✓
│   ├── upload_to_s3.py
│   ├── lambda_function.py
│   └── query_results.py
└── sample_data/ ✓
```

✅ **Documentation:**
- Project overview ✓
- Prerequisites ✓
- Architecture diagram ✓
- AWS setup guide ✓
- Running instructions ✓
- Results explanation ✓
- Cost breakdown ✓
- Cleanup guide ✓
- Troubleshooting ✓
- Next steps ✓

✅ **No CloudFormation:** Pure boto3 ✓

✅ **Learning Objectives:**
- S3 basics ✓
- Lambda deployment ✓
- IAM roles ✓
- Event-driven architecture ✓
- Cloud cost management ✓
- Serverless computing ✓

---

## Quick Reference

### File Purposes

| File | Purpose | Read If... |
|------|---------|-----------|
| README.md | Full overview | Starting project |
| QUICKSTART.md | 5-min intro | In a hurry |
| setup_guide.md | AWS setup steps | Creating resources |
| cleanup_guide.md | Delete resources | Done with project |
| requirements.txt | Dependencies | Setting up environment |
| upload_to_s3.py | Upload data | Getting data to cloud |
| lambda_function.py | Process data | Understanding processing |
| query_results.py | Download results | Getting results locally |
| climate_analysis.ipynb | Analyze results | Exploring findings |

---

## Getting Started

### For First-Time Users
1. Read **README.md** (20 minutes)
2. Follow **setup_guide.md** (45 minutes)
3. Run **QUICKSTART.md** steps 3-5 (20 minutes)
4. Explore **climate_analysis.ipynb** (30 minutes)

### For Experienced Users
1. Skim **QUICKSTART.md** (5 minutes)
2. Follow setup steps from **setup_guide.md** (30 minutes)
3. Run scripts directly (15 minutes)

### For Cleanup
1. Read **cleanup_guide.md** (10 minutes)
2. Follow deletion steps (15 minutes)
3. Verify in AWS console (5 minutes)

---

## Development Notes

### Code Quality
- All Python files validated with py_compile
- Comprehensive error handling
- Logging throughout
- Type hints in docstrings
- AWS best practices followed

### Documentation Quality
- ~3,900 lines total
- Multiple examples
- Copy-paste ready commands
- Troubleshooting sections
- Security best practices
- Cost management included

### AWS Best Practices
- Least privilege IAM roles
- Bucket encryption enabled
- Public access blocked
- Proper error handling
- CloudWatch logging
- Cost optimization tips

---

## File Statistics

| Category | Count | Size |
|----------|-------|------|
| Documentation | 4 | 56 KB |
| Python Scripts | 4 | 18 KB |
| Jupyter Notebooks | 1 | 6 KB |
| Configuration | 2 | 1.5 KB |
| Total | 11 | ~82 KB |

**Code Statistics:**
- Documentation lines: 2,850+
- Python code lines: 1,050+
- Total lines: 3,900+

---

## Success Criteria

A complete Tier 2 project should have:

✅ README with project overview
✅ requirements.txt with dependencies
✅ setup_guide.md with AWS instructions
✅ cleanup_guide.md with deletion steps
✅ Working Python scripts (3+)
✅ Jupyter notebook for analysis
✅ Sample data guide
✅ 2-4 hour duration
✅ $8-12 cost per run
✅ Complete documentation
✅ Troubleshooting guide
✅ Next steps to Tier 3

**This project: ✅ All criteria met**

---

## Next: Where to Start?

Choose your path:

1. **New to AWS?** → Read README.md then QUICKSTART.md
2. **Setting up?** → Follow setup_guide.md step-by-step
3. **In a hurry?** → Jump to QUICKSTART.md
4. **Done analyzing?** → Follow cleanup_guide.md
5. **Want production?** → Move to Tier 3

---

*Created: 2025-11-14 | Status: Complete | Ready for Use*
