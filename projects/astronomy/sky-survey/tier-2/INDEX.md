# Tier 2: Astronomical Image Processing with S3 and Lambda - Complete Index

**Location:** `/projects/astronomy/sky-survey/tier-2/`

**Overview:** Build a complete astronomical image processing pipeline on AWS using serverless Lambda functions, object storage (S3), and SQL queries (Athena).

**Cost:** $7-10 | **Duration:** 2-4 hours | **Platform:** AWS

## Quick Navigation

### Getting Started (Choose One)
- **FIRST TIME?** → Start with [`README.md`](README.md) (10 minutes to understand)
- **IMPATIENT?** → Use [`QUICKSTART.md`](QUICKSTART.md) (60-minute walkthrough)
- **DETAILED?** → Follow [`setup_guide.md`](setup_guide.md) (45-minute AWS setup)

### Main Documentation

| Document | Time | Purpose |
|----------|------|---------|
| **README.md** | 10 min | Complete project guide with all information |
| **QUICKSTART.md** | 5 min | Fast reference for experienced AWS users |
| **setup_guide.md** | 45 min | Step-by-step AWS resource creation (detailed) |
| **cleanup_guide.md** | 5 min | How to delete AWS resources when done |
| **PROJECT_STRUCTURE.txt** | 3 min | Overview of all files and structure |

## Workflow

```
Step 1: Download Data (5 min)
↓
python scripts/download_sample_fits.py

Step 2: Upload to S3 (2 min)
↓
python scripts/upload_to_s3.py

Step 3: Deploy Lambda (from setup_guide.md, 15 min)
↓
aws lambda create-function ...

Step 4: Run Source Detection (5 min)
↓
python scripts/invoke_lambda.py

Step 5: Query Results (5 min)
↓
python scripts/query_with_athena.py
OR
jupyter notebook notebooks/sky_analysis.ipynb

Step 6: Analyze & Visualize (10 min)
↓
See notebook results

Step 7: Cleanup (5 min when done)
↓
bash scripts/cleanup_all.sh
```

## Files by Purpose

### Documentation
- [`README.md`](README.md) - Main guide (start here!)
- [`QUICKSTART.md`](QUICKSTART.md) - 60-minute reference
- [`setup_guide.md`](setup_guide.md) - Detailed AWS setup
- [`cleanup_guide.md`](cleanup_guide.md) - Resource deletion
- [`PROJECT_STRUCTURE.txt`](PROJECT_STRUCTURE.txt) - File overview
- [`INDEX.md`](INDEX.md) - This file

### Python Scripts (in `scripts/`)
- [`download_sample_fits.py`](scripts/download_sample_fits.py) - Download test FITS images
- [`upload_to_s3.py`](scripts/upload_to_s3.py) - Upload to S3 bucket
- [`lambda_function.py`](scripts/lambda_function.py) - Source detection code
- [`invoke_lambda.py`](scripts/invoke_lambda.py) - Trigger Lambda functions
- [`query_with_athena.py`](scripts/query_with_athena.py) - Query results with SQL

### Analysis
- [`notebooks/sky_analysis.ipynb`](notebooks/sky_analysis.ipynb) - Interactive analysis notebook

### Other
- [`requirements.txt`](requirements.txt) - Python dependencies
- [`data/README.md`](data/README.md) - Data directory documentation

## AWS Services Used

| Service | Purpose | Configuration |
|---------|---------|----------------|
| **S3** | Store images & results | Two buckets: raw + catalog |
| **Lambda** | Process FITS images | Python 3.11, 1GB RAM, 5 min timeout |
| **Athena** | Query results with SQL | Database + Parquet table |
| **IAM** | Grant permissions | Role with S3 + CloudWatch access |
| **CloudWatch** | View logs | Log group for Lambda function |

## Typical Execution

### First Time Setup (45 minutes)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure AWS (from setup_guide.md)
aws configure
# Then follow sections 1-4 in setup_guide.md for:
# - IAM role creation
# - S3 bucket creation
# - Lambda deployment
# - Athena setup

# 3. Save bucket names
export BUCKET_RAW="astronomy-tier2-XXXXX-raw"
export BUCKET_CATALOG="astronomy-tier2-XXXXX-catalog"
echo "export BUCKET_RAW=${BUCKET_RAW}" >> ~/.astronomy_env
echo "export BUCKET_CATALOG=${BUCKET_CATALOG}" >> ~/.astronomy_env
source ~/.astronomy_env
```

### Data Processing (12 minutes)
```bash
# 1. Download sample data
python scripts/download_sample_fits.py    # 5 min

# 2. Upload to S3
python scripts/upload_to_s3.py            # 2 min

# 3. Run source detection
python scripts/invoke_lambda.py           # 5 min
```

### Analysis (15 minutes)
```bash
# Option A: Command-line queries
python scripts/query_with_athena.py       # 5 min

# Option B: Interactive notebook
jupyter notebook notebooks/sky_analysis.ipynb  # 10 min
```

### Cleanup (5 minutes when done)
```bash
# Automated cleanup
bash scripts/cleanup_all.sh

# Or manual (from cleanup_guide.md)
aws lambda delete-function --function-name astronomy-source-detection
# ... (see cleanup_guide.md for full steps)
```

## Key Features

### Complete & Production-Ready
- Error handling and validation
- Progress tracking and logging
- Cost awareness built-in
- Comprehensive documentation

### Educational
- Learn AWS services (S3, Lambda, Athena, IAM)
- Learn astronomy concepts (FITS, photometry, catalogs)
- Understand serverless architecture
- Practice SQL queries

### Scalable
- Can process 10-100+ images
- Parallel Lambda invocations
- SQL queries on any dataset size
- Easy path to Tier 3 production

## Common Starting Points

### "I want to learn AWS"
1. Read README.md introduction
2. Follow setup_guide.md Part 1-4
3. Run one image through the pipeline
4. Check CloudWatch logs

### "I want to analyze data"
1. Read QUICKSTART.md
2. Run through steps 1-5
3. Use notebook for analysis
4. Export results

### "I want to scale up"
1. Understand this Tier 2 project
2. Modify scripts for more images
3. Monitor costs in AWS Cost Explorer
4. Plan upgrade to Tier 3

### "I want quick results"
1. Use QUICKSTART.md
2. Follow 60-minute timeline
3. Skip detailed explanations
4. See results immediately

## Troubleshooting Quick Links

### Problem? Solution?
- AWS setup issues → See setup_guide.md "Troubleshooting"
- Lambda errors → Check CloudWatch logs section
- Athena queries → See cleanup_guide.md "Verification"
- General issues → See README.md "Troubleshooting"

### Common Commands

```bash
# Check AWS setup
aws sts get-caller-identity

# View Lambda logs
aws logs tail /aws/lambda/astronomy-source-detection --follow

# List S3 objects
aws s3 ls s3://${BUCKET_RAW}/images/ --recursive

# Run Athena query
aws athena start-query-execution \
  --query-string "SELECT COUNT(*) FROM astronomy.sources" \
  --result-configuration OutputLocation=s3://${BUCKET_CATALOG}/athena-results/
```

## File Summary

| File | Lines | Time to Read | Purpose |
|------|-------|--------------|---------|
| README.md | 634 | 10 min | Complete guide |
| setup_guide.md | 544 | 20 min | AWS setup instructions |
| cleanup_guide.md | 363 | 5 min | Resource deletion |
| lambda_function.py | 352 | 10 min | Source detection algorithm |
| invoke_lambda.py | 267 | 5 min | Trigger function |
| query_with_athena.py | 252 | 5 min | SQL queries |
| notebook | 477 | 15 min | Analysis |
| QUICKSTART.md | 205 | 5 min | Fast reference |
| upload_to_s3.py | 193 | 5 min | S3 upload |
| download_sample_fits.py | 191 | 5 min | Data download |

## Learning Path

### Beginner → Intermediate → Advanced

1. **Beginner (Day 1)**
   - Read README.md
   - Run QUICKSTART.md
   - See your first results
   - Time: 2 hours

2. **Intermediate (Day 2-3)**
   - Understand setup_guide.md deeply
   - Modify scripts for custom data
   - Write custom Athena queries
   - Time: 4-6 hours

3. **Advanced (Week 2)**
   - Scale to hundreds of images
   - Integrate with other surveys
   - Build automated pipeline
   - Plan Tier 3 upgrade
   - Time: 8+ hours

## Cost Tracking

**Expected cost:** $7-10 per run

Monitor with:
```bash
# AWS Cost Explorer
https://console.aws.amazon.com/cost-management/

# Or CLI
aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-01-31 ...
```

Delete resources immediately after testing to avoid ongoing charges.

## Next Steps

After completing Tier 2:

1. **Explore** - Try different sky regions, datasets
2. **Extend** - Add cross-matching, classification
3. **Automate** - Build event-triggered pipeline
4. **Scale** - Move to Tier 3 with CloudFormation
5. **Publish** - Share your findings!

## Getting Help

| Question | Resource |
|----------|----------|
| AWS documentation | https://docs.aws.amazon.com/ |
| Astronomy help | https://www.astropy.org/ |
| AWS CLI reference | `aws help` or online docs |
| Project issues | See troubleshooting in README.md |
| Error codes | Check cloudformation service logs |

## Version & Date

- **Version:** 1.0.0
- **Created:** 2025-11-14
- **Status:** Production-ready
- **Tested:** AWS Lambda + Athena
- **Compatibility:** Python 3.8+

---

## Start Here: Three Options

```
OPTION 1: Read README.md first
├─ Understand the full picture (10 min)
├─ Then follow setup_guide.md (45 min)
└─ Expected outcome: Full understanding + working system

OPTION 2: Use QUICKSTART.md
├─ Quick 60-minute walkthrough
├─ For experienced AWS users
└─ Expected outcome: Running system quickly

OPTION 3: Dive into setup_guide.md
├─ Step-by-step AWS setup
├─ Most detailed instructions
└─ Expected outcome: Professional AWS infrastructure
```

**All paths lead to the same result: A working astronomical source detection pipeline!**

Ready to begin? Choose your path above or start with [README.md](README.md)!
