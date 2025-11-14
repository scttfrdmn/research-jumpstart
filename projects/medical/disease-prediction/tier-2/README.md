# Medical Image Processing with S3 and Lambda - Tier 2

## Project Overview

This Tier 2 project demonstrates how to build a cloud-based medical image processing pipeline using AWS services. You'll upload chest X-ray images to S3, deploy a Lambda function to preprocess images, store results back to S3, and track predictions metadata in DynamoDB.

**Key Learning:** Move from local computation (Tier 1) to serverless cloud processing (Tier 2)

## What You'll Build

A complete medical image preprocessing pipeline that:
- Uploads chest X-ray images to AWS S3
- Processes images with AWS Lambda (resize, normalize, format conversion)
- Stores preprocessed images in S3 for model inference
- Tracks prediction metadata in DynamoDB
- Analyzes results and costs in a Jupyter notebook

## Prerequisites

### Required
- AWS Account (with free tier eligible)
- Python 3.8+
- Jupyter Notebook or JupyterLab
- AWS CLI configured with credentials
- boto3 library

### Knowledge
- Basic Python programming
- Familiarity with JSON and command-line tools
- Understanding of medical imaging basics (helpful but not required)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Medical Image Pipeline                   │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│   Chest X-Ray Files  │
│   (Local/Sample)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ upload_to_s3.py      │
│ (Batch Upload)       │
└──────────┬───────────┘
           │
           ▼
    ┌──────────────┐
    │   AWS S3     │
    │  Raw Images  │  ◄─────────────────┐
    └──────┬───────┘                     │
           │                             │
           ▼                             │
    ┌──────────────────────┐             │
    │  AWS Lambda Function │             │
    │  (Image Preprocess)  │             │
    │  - Resize: 224×224   │             │
    │  - Normalize: [0,1]  │             │
    │  - Log metadata      │             │
    └──────────┬───────────┘             │
               │                         │
               ▼                         │
        ┌──────────────┐                 │
        │  AWS S3      │                 │
        │Processed IMG │─────────────────┤
        └──────────────┘                 │
               ▲                         │
               │                         │
        ┌──────▼──────────┐              │
        │  AWS DynamoDB   │              │
        │ Prediction Meta │──────────────┤
        │ - ID, Timestamp │              │
        │ - S3 Path, Size │              │
        │ - Processing ms │              │
        └─────────────────┘              │
               ▲                         │
               │                         │
    ┌──────────▼──────────┐              │
    │ query_results.py    │              │
    │ (Retrieve & Analyze)│──────────────┘
    └─────────────────────┘

        ┌─────────────────┐
        │  Jupyter        │
        │  Notebook       │
        │  Analysis &     │
        │  Visualization  │
        └─────────────────┘
```

## AWS Services Used

| Service | Purpose | Cost |
|---------|---------|------|
| **S3** | Store raw and processed images | $0.023 per GB/month (~$0.1-0.5) |
| **Lambda** | Serverless image preprocessing | $0.20 per 1M invocations (~$0.1-0.2) |
| **DynamoDB** | NoSQL metadata storage | $0.25/GB for on-demand (~$0.1-0.3) |
| **IAM** | Access control and permissions | Free |
| **CloudWatch** | Logging and monitoring | Free (small amounts) |

**Total Estimated Cost: $8-12 per run**

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS

Follow the detailed setup guide:
```bash
cat setup_guide.md
```

Quick steps:
- Create S3 bucket: `medical-images-{your-user-id}`
- Create IAM role for Lambda
- Create DynamoDB table: `medical-predictions`

### 3. Run the Pipeline

```bash
# Upload sample images
python scripts/upload_to_s3.py

# Deploy Lambda function (see setup_guide.md for deployment steps)

# Query results
python scripts/query_results.py

# Analyze in notebook
jupyter notebook notebooks/image_analysis.ipynb
```

## Project Structure

```
tier-2/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup_guide.md                     # Step-by-step AWS setup
├── cleanup_guide.md                   # How to delete resources
├── notebooks/
│   └── image_analysis.ipynb          # Main analysis notebook
├── scripts/
│   ├── upload_to_s3.py               # Upload images to S3
│   ├── lambda_function.py             # Lambda preprocessing code
│   └── query_results.py               # Retrieve and analyze results
└── sample_data/
    └── sample_xrays/                 # Sample X-ray images (if included)
```

## Workflow Steps

### Step 1: Setup AWS Environment
- Create S3 bucket for images
- Create IAM role for Lambda
- Create DynamoDB table for metadata
- (Detailed instructions in `setup_guide.md`)

### Step 2: Prepare Data
- Download or use provided sample chest X-rays
- Images should be in DICOM or PNG format
- Resolution: 512x512 or larger

### Step 3: Upload Images to S3
```bash
python scripts/upload_to_s3.py --input-dir ./sample_data/sample_xrays \
                                --s3-bucket medical-images-{user-id} \
                                --prefix raw-images/
```

### Step 4: Deploy Lambda Function
- Package `lambda_function.py`
- Deploy to AWS Lambda
- Test with sample invocation
- (Detailed instructions in `setup_guide.md`)

### Step 5: Process Images
- Trigger Lambda on S3 uploads (S3 event notifications)
- Or manually invoke Lambda for batch processing
- Lambda preprocesses images:
  - Resize to 224x224 pixels
  - Normalize pixel values to [0, 1]
  - Convert to standard format (PNG/NPZ)
  - Store in processed S3 folder

### Step 6: Track Metadata
- Lambda logs prediction metadata to DynamoDB:
  - Image ID and timestamp
  - Original and processed S3 paths
  - Processing time in milliseconds
  - Image dimensions and file size

### Step 7: Analyze Results
```bash
python scripts/query_results.py --table-name medical-predictions \
                                 --limit 100
```

### Step 8: Visualize in Notebook
- Open `notebooks/image_analysis.ipynb`
- Query DynamoDB for metadata
- Download sample processed images
- Analyze processing times and costs
- Visualize image statistics

## Expected Results

After completing this project, you will have:

1. **Cloud Storage Experience**
   - Uploaded medical images to S3
   - Organized data with S3 prefixes (raw-images/, processed-images/)
   - Understood S3 pricing and storage tiers

2. **Serverless Processing Pipeline**
   - Deployed a Lambda function
   - Processed data without local compute resources
   - Understood Lambda timeout and memory constraints
   - Monitored Lambda execution costs

3. **Metadata Management**
   - Stored structured data in DynamoDB
   - Queried NoSQL database with boto3
   - Understood on-demand vs provisioned DynamoDB pricing

4. **Cost Awareness**
   - Tracked service costs
   - Understood pay-per-use pricing model
   - Identified cost optimization opportunities

5. **Reproducible Pipeline**
   - Created reusable Python scripts
   - Documented setup and execution
   - Prepared foundation for Tier 3 CloudFormation

## Cost Breakdown

### Detailed Cost Estimate

**Assumptions:**
- 100 chest X-ray images (10-50 MB each)
- Total raw data: ~2 GB
- Total processed data: ~1 GB (after compression)
- Lambda: 100 invocations, 30 seconds each
- Run duration: 1 week

**Costs:**

| Service | Usage | Cost |
|---------|-------|------|
| S3 Storage | 3 GB for 1 week | $0.00001 per GB-hour = $0.07 |
| S3 Uploads | 100 PUT requests | $0.005 per 1000 = $0.0005 |
| S3 Downloads | 50 GET requests | $0.0004 per 1000 = ~$0 |
| Lambda Compute | 100 × 30s @ 128MB | ~$0.10 |
| Lambda Requests | 100 requests | ~$0.0001 |
| DynamoDB | On-demand, 1000 writes | $0.12 per 1M writes = ~$0.0001 |
| Data Transfer | ~100 MB out | $0.12 per GB = ~$0.01 |
| **Total** | | **$0.18** |

**Note:** This is a minimal run. Costs increase with:
- More images: +$0.01 per 100 images
- Larger images: +$0.001 per GB
- Longer processing: +$0.005 per 100 Lambda seconds

**Typical range for learning project: $8-12**

This includes safety margin and multiple test runs.

## Learning Objectives

### Technical Skills
- [x] Create and configure AWS services (S3, Lambda, DynamoDB, IAM)
- [x] Write boto3 code to interact with AWS services
- [x] Deploy Lambda functions and test them
- [x] Set up S3 bucket policies and IAM roles
- [x] Query NoSQL databases with boto3
- [x] Monitor costs and optimize spending

### AWS Concepts
- [x] Object storage and S3 bucket organization
- [x] Serverless computing with Lambda
- [x] Event-driven architectures
- [x] Identity and access management (IAM)
- [x] NoSQL databases (DynamoDB)
- [x] Pay-per-use cloud pricing

### Research Applications
- [x] Store large medical image datasets in cloud
- [x] Process images at scale without local hardware
- [x] Build reproducible data processing pipelines
- [x] Share datasets and results with collaborators
- [x] Track processing metadata for research

## Troubleshooting

### Common Issues

**Problem:** "NoCredentialsError" when running Python scripts
- **Solution:** Configure AWS CLI with `aws configure` and provide access keys
- See setup_guide.md Step 1 for detailed instructions

**Problem:** Lambda function timeout
- **Solution:** Increase timeout in Lambda configuration to 60 seconds
- Check processing time in CloudWatch logs

**Problem:** High S3 costs
- **Solution:** Delete resources with cleanup_guide.md when done
- Use S3 Intelligent-Tiering for long-term storage

**Problem:** "AccessDenied" errors
- **Solution:** Check IAM role permissions in setup_guide.md
- Ensure Lambda execution role has S3 and DynamoDB permissions

**Problem:** DynamoDB read throttling
- **Solution:** Use on-demand billing (auto-scales)
- Or increase provisioned capacity

See `cleanup_guide.md` for information on removing test resources.

## Next Steps

After completing this Tier 2 project:

### Option 1: Advanced Tier 2 Features
- Add SNS email notifications for processing completion
- Implement SQS queue for handling image upload spikes
- Add CloudWatch monitoring dashboard
- Implement image classification with a pre-trained model

### Option 2: Move to Tier 3 (Production)
Tier 3 uses CloudFormation for automated infrastructure:
- Infrastructure-as-code templates
- One-click deployment
- Multi-environment support
- Production-ready security policies
- Auto-scaling and monitoring

See `/projects/medical/disease-prediction/tier-3/` (when available)

### Option 3: Real Medical Data
- Integrate with public datasets (ChexPert, MIMIC-CXR)
- Implement additional preprocessing (normalization, augmentation)
- Train a disease classification model
- Deploy model inference on Lambda or SageMaker

## References

### AWS Documentation
- [S3 Documentation](https://docs.aws.amazon.com/s3/)
- [Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [DynamoDB Documentation](https://docs.aws.amazon.com/dynamodb/)
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

### Medical Imaging
- [DICOM Standard Overview](https://www.dicomstandard.org/)
- [OpenCV Documentation](https://opencv.org/)
- [Pillow Image Library](https://pillow.readthedocs.io/)

### AWS Best Practices
- [AWS Security Best Practices](https://aws.amazon.com/architecture/security-identity-compliance/)
- [AWS Cost Optimization](https://aws.amazon.com/blogs/aws-cost-management/)
- [Lambda Best Practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)

## Support

### Getting Help
1. Check troubleshooting section above
2. Review AWS service error messages in CloudWatch logs
3. Consult boto3 documentation for API details
4. Check AWS service quotas (may need increase for scale)

### For Issues
- Review setup_guide.md for configuration problems
- Check IAM permissions if access errors occur
- Verify S3 bucket naming (globally unique)
- Confirm DynamoDB table exists before running scripts

## License

This project is part of the Research Jumpstart curriculum and is provided for educational purposes.

## Author Notes

This is a Tier 2 (AWS Starter) project. It bridges Tier 1 (Studio Lab free tier) and Tier 3 (Production CloudFormation).

**Time to Complete:** 2-4 hours
**Cost:** $8-12 per run
**Difficulty:** Intermediate (requires AWS account setup)

For questions about the project structure, see `TIER_2_SPECIFICATIONS.md` in the project root.
