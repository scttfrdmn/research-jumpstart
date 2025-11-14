# fMRI Data Processing with S3 and Lambda

## Project Overview

This Tier 2 project demonstrates how to process functional MRI (fMRI) data using AWS S3 and Lambda services. You'll upload NIfTI-format fMRI images, perform automated preprocessing (motion correction and spatial smoothing) using Lambda functions, and analyze brain connectivity patterns in a Jupyter notebook.

**Key Learning Outcomes:**
- Upload and manage large neuroimaging datasets in S3
- Deploy serverless data processing functions with Lambda
- Implement IAM roles with least-privilege access
- Build reproducible neuroimaging analysis pipelines
- Monitor costs and manage AWS resources

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   fMRI Analysis Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Raw Data Upload          2. Lambda Processing           │
│     ↓                           ↓                            │
│  [Local fMRI NIfTI Files] → [S3 Input Bucket]            │
│                               ↓                            │
│                          [Lambda Function]                 │
│                          - Motion correction               │
│                          - Spatial smoothing               │
│                               ↓                            │
│                       [S3 Output Bucket]                   │
│                               ↓                            │
│  4. Analysis & Visualization                               │
│     ↑                      3. Download Results             │
│  [Jupyter Notebook]  ←─────────────────────────          │
│  - Connectivity analysis                                   │
│  - Network visualization                                   │
│  - Statistical summaries                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Account & Tools
- AWS account with billing enabled
- Python 3.8+
- pip package manager
- Jupyter Notebook or JupyterLab
- AWS CLI (optional, for troubleshooting)

### Required Knowledge
- Basic Python programming
- Familiarity with AWS S3 and Lambda concepts
- Basic understanding of fMRI and neuroimaging

### Setup Time
- AWS account setup: ~10 minutes (one-time)
- S3 bucket creation: ~2 minutes
- IAM role creation: ~5 minutes
- Lambda deployment: ~10 minutes
- Data upload: ~10-20 minutes
- **Total setup: 45-60 minutes**

## Project Components

### Files Structure
```
tier-2/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup_guide.md                     # Step-by-step AWS setup
├── cleanup_guide.md                   # How to delete resources
├── notebooks/
│   └── fmri_analysis.ipynb           # Interactive analysis notebook
└── scripts/
    ├── upload_to_s3.py               # Upload sample fMRI data to S3
    ├── lambda_function.py            # Lambda preprocessing code
    ├── query_results.py              # Download and query results
    └── test_connectivity.py          # Local connectivity analysis
```

### AWS Services Used
- **S3:** Store raw fMRI data, processed results, and analysis outputs
- **Lambda:** Execute preprocessing workflows (motion correction, smoothing)
- **IAM:** Define least-privilege access policies for Lambda

### Optional Services
- **CloudWatch:** Monitor Lambda execution logs and performance
- **S3 Lifecycle:** Automatically transition old data to cheaper storage tiers

## Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: AWS Setup (Detailed guide in setup_guide.md)
1. Create S3 buckets for input and output data
2. Create IAM role for Lambda execution
3. Deploy Lambda function with preprocessing code

### Step 3: Upload Sample Data
```bash
python scripts/upload_to_s3.py \
  --bucket your-bucket-name \
  --local-path sample_data/
```

### Step 4: Run Analysis
- Open `notebooks/fmri_analysis.ipynb` in Jupyter
- Follow cells to download processed data
- Explore fMRI connectivity patterns

### Step 5: Cleanup
```bash
python scripts/cleanup.py --bucket your-bucket-name
```
See `cleanup_guide.md` for manual cleanup instructions.

## Cost Breakdown

### Estimated Costs (Single Run)
| Service | Usage | Cost |
|---------|-------|------|
| S3 Storage | 10GB for 1 week | $0.23 |
| S3 PUT requests | 1000 uploads | $0.005 |
| S3 GET requests | 2000 downloads | $0.01 |
| Lambda Compute | 50 invocations × 2min @ 256MB | $0.50 |
| Lambda Requests | 50 invocations | $0.001 |
| Data Transfer | 5GB egress | $0.90 |
| **Total** | | **$10-14** |

### Cost Optimization Tips
- Use S3 Intelligent-Tiering to automatically move old data to cheaper storage
- Set Lambda timeout to 5 minutes maximum
- Delete data immediately after analysis (see cleanup_guide.md)
- Use AWS Free Tier if eligible
- Monitor costs with AWS Cost Explorer

## Duration & Difficulty

| Metric | Value |
|--------|-------|
| **Total Duration** | 2-4 hours |
| **Difficulty** | Intermediate |
| **AWS Knowledge Required** | Basic |
| **Programming Skills** | Intermediate Python |
| **Neuroimaging Knowledge** | Basic |

## Data Requirements

### Sample Data
The project includes sample fMRI data:
- **Format:** NIfTI (.nii.gz)
- **Size:** ~100MB (4D volume)
- **Dimensions:** 64 × 64 × 32 × 150 (x, y, z, time)
- **Source:** Simulated fMRI-like data

### Your Own Data
To use your own fMRI data:
1. Prepare NIfTI files (.nii or .nii.gz format)
2. Place in `sample_data/` directory
3. Update upload script with file paths
4. Run upload and analysis scripts

## Workflow Steps

### Phase 1: Data Upload (15 min)
1. Prepare S3 buckets (input and output)
2. Create IAM role with S3 permissions
3. Upload raw fMRI files to input bucket

### Phase 2: Lambda Processing (20 min)
1. Review lambda_function.py
2. Create Lambda function via AWS Console
3. Assign IAM role to function
4. Test with sample file
5. Verify outputs in S3

### Phase 3: Analysis (30 min)
1. Download processed data from S3
2. Load data in Jupyter notebook
3. Calculate functional connectivity
4. Create network visualizations
5. Generate statistical summaries

### Phase 4: Cleanup (5 min)
1. Delete S3 objects
2. Delete S3 buckets
3. Delete Lambda function
4. Delete IAM role

## Learning Objectives

### AWS Concepts
- S3 bucket creation and object management
- Lambda function deployment and testing
- IAM roles and policies
- Event-driven architecture basics
- Pay-per-use pricing models

### Data Processing
- Neuroimaging file formats (NIfTI)
- Motion correction algorithms
- Spatial smoothing techniques
- BOLD signal analysis

### Research Applications
- Cloud-based data management
- Reproducible analysis pipelines
- Scalable processing workflows
- Cost-effective large-scale analysis

## Success Criteria

You've successfully completed this project when:
1. ✅ Sample fMRI data uploads to S3
2. ✅ Lambda function processes data without errors
3. ✅ Processed data appears in output bucket
4. ✅ Jupyter notebook downloads and analyzes results
5. ✅ Connectivity network visualizations display correctly
6. ✅ All AWS resources deleted after completion
7. ✅ Total cost is under $15

## Troubleshooting

### Lambda Timeout Issues
**Problem:** Lambda execution times out
**Solutions:**
- Check input file size (should be <100MB)
- Increase Lambda timeout in AWS Console to 5 minutes
- Check CloudWatch logs for processing bottlenecks

### S3 Access Denied Errors
**Problem:** Permission errors when accessing S3
**Solutions:**
- Verify IAM role has S3 read/write permissions
- Check bucket policy allows Lambda execution
- Review setup_guide.md IAM section

### Data Upload Failures
**Problem:** Files fail to upload to S3
**Solutions:**
- Check internet connection
- Verify S3 bucket name is correct
- Ensure IAM user credentials have S3 permissions
- Check file size (max 5GB per file recommended)

### Notebook Connection Issues
**Problem:** Jupyter notebook can't connect to AWS
**Solutions:**
- Verify AWS credentials configured locally
- Check ~/.aws/credentials file exists
- Run `aws sts get-caller-identity` to verify credentials
- Review boto3 documentation for credential configuration

## Next Steps

### Extend This Project
- Add more neuroimaging preprocessing steps (registration, atlas-based ROI extraction)
- Implement automated quality control checks
- Add SNS notifications for pipeline completion
- Use Athena for querying connectivity metrics from S3 CSV files
- Integrate with DynamoDB for storing subject-level metadata

### Move to Tier 3 (Production)
- Implement complete pipeline with CloudFormation
- Add containerization with Docker/ECR
- Deploy as production-ready service
- Implement automated testing and monitoring
- Add cost monitoring and optimization

### Related Projects
- **Tier 1:** Basic fMRI analysis in Studio Lab
- **Tier 3:** Production fMRI pipeline with CloudFormation
- **Genomics Tier 2:** DNA sequencing with S3 and Lambda
- **Medical Imaging Tier 2:** CT scan processing with S3 and Lambda

## Additional Resources

### AWS Documentation
- [AWS S3 Bucket Setup](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)
- [Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/welcome.html)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [boto3 S3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)

### Neuroimaging Resources
- [NIfTI File Format](https://nifti.nimh.nih.gov/)
- [FSL User Guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)
- [Nipype Tutorial](https://nipype.readthedocs.io/)
- [Brain Connectivity Toolbox](https://sites.google.com/site/bctnet/)

### Cost Management
- [AWS Pricing Calculator](https://calculator.aws/)
- [S3 Pricing](https://aws.amazon.com/s3/pricing/)
- [Lambda Pricing](https://aws.amazon.com/lambda/pricing/)
- [AWS Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/)

## Support & Community

For questions or issues:
1. Check Troubleshooting section above
2. Review setup_guide.md for detailed steps
3. Search AWS documentation
4. Consult boto3 examples

## License

This project is part of Research Jumpstart and follows the repository license.

---

**Last Updated:** November 2024
**Version:** 1.0
**Maintainer:** Research Jumpstart Team
