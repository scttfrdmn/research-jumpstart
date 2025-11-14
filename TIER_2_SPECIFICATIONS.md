# Tier 2: AWS Starter Projects Specification

**Purpose:** Bridge the gap between free Studio Lab (Tier 1) and production CloudFormation (Tier 3)

## Overview

Tier 2 introduces researchers to AWS services without the complexity of CloudFormation. These projects use boto3 (AWS SDK) with manual setup guides.

## Characteristics

- **Duration:** 2-4 hours
- **Cost:** $5-15 per project run
- **Platform:** AWS account required
- **Infrastructure:** Manual setup or simple Python/Bash scripts
- **No CloudFormation:** Direct AWS SDK usage (boto3)

## AWS Services to Use

### Core Services (All Projects)
- **S3:** Store datasets, results, intermediate files
- **Lambda:** Serverless data processing functions
- **IAM:** Manual role creation for Lambda

### Optional Services (Domain-Specific)
- **Athena:** SQL queries on S3 data (structured data domains)
- **DynamoDB:** NoSQL storage (real-time analytics)
- **SNS:** Email/SMS notifications (monitoring/alerts)
- **SQS:** Message queues (async processing)
- **CloudWatch:** Basic logging and monitoring

## File Structure

```
projects/domain/project-name/tier-2/
├── README.md                      # Project overview
├── requirements.txt               # Python dependencies (boto3, etc.)
├── setup_guide.md                 # Step-by-step AWS setup
├── notebooks/
│   └── analysis.ipynb            # Main analysis notebook
├── scripts/
│   ├── upload_to_s3.py           # Upload dataset to S3
│   ├── lambda_function.py        # Lambda function code
│   └── query_results.py          # Retrieve and analyze results
└── cleanup_guide.md              # How to delete resources
```

## Example: Climate Science Tier 2

**Project:** Climate Data Analysis with S3 and Lambda

**Workflow:**
1. Upload CMIP6 subset to S3 (~5GB)
2. Deploy Lambda function to process netCDF files
3. Store results in S3
4. Query results with Athena or download locally
5. Visualize results in notebook

**AWS Setup:**
- S3 bucket: `climate-data-{user-id}`
- Lambda function: `process-climate-data`
- IAM role: `lambda-climate-processor` (S3 read/write)
- Cost: ~$8 (S3 storage + Lambda invocations)

## Key Learning Objectives

### Technical Skills
- Create and configure AWS services manually
- Use boto3 SDK for AWS interactions
- Understand S3 bucket policies and IAM roles
- Deploy and test Lambda functions
- Monitor costs with AWS Cost Explorer

### AWS Concepts
- Object storage (S3)
- Serverless computing (Lambda)
- Identity and access management (IAM)
- Event-driven architectures
- Pay-per-use pricing

### Research Applications
- Store large datasets in cloud
- Process data without local compute limits
- Share data and results with collaborators
- Run reproducible analysis pipelines

## Differences from Other Tiers

### vs. Tier 1 (Studio Lab)
- **Added:** AWS account, boto3, cloud storage
- **Cost:** $5-15 (vs free)
- **Scale:** Can process larger datasets (100GB+)
- **Persistence:** Data persists beyond Studio Lab limits

### vs. Tier 3 (Production)
- **Simpler:** No CloudFormation templates
- **Manual:** Step-by-step setup vs automated deployment
- **Limited:** Fewer AWS services, smaller scale
- **Learning:** Focus on understanding basics

## Cost Management

### Keep Costs Low
- Use S3 Intelligent-Tiering
- Set Lambda timeout to 5 minutes max
- Delete resources after completion
- Use AWS Free Tier when possible

### Typical Costs (per project)
- S3 storage (10GB, 1 week): $0.25
- Lambda invocations (1000 runs): $0.20
- Data transfer: $1-2
- Athena queries (10GB scanned): $0.50
- **Total:** $5-15 per project

## Setup Complexity

**Time to First Result:**
- AWS account setup: 10 minutes (one-time)
- S3 bucket creation: 2 minutes
- IAM role creation: 5 minutes
- Lambda deployment: 10 minutes
- Data upload: 10-30 minutes
- **Total:** 45-60 minutes setup + 1-2 hours analysis

## Success Criteria

A good Tier 2 project should:
1. ✅ Use at least 2 AWS services (S3 + Lambda minimum)
2. ✅ Cost less than $15 to run
3. ✅ Complete in 2-4 hours total time
4. ✅ Have clear, step-by-step setup guide
5. ✅ Include cleanup instructions
6. ✅ Demonstrate clear advantage over Tier 1
7. ✅ Provide foundation for understanding Tier 3

## Anti-Patterns to Avoid

- ❌ Don't use CloudFormation (save for Tier 3)
- ❌ Don't replicate Tier 1 functionality
- ❌ Don't use expensive services (SageMaker training, EMR)
- ❌ Don't create complex architectures (save for Tier 3)
- ❌ Don't leave resources running (include cleanup)

## Example Projects by Domain

### Climate Science
- Upload CMIP6 data to S3
- Lambda functions for netCDF processing
- Store results in S3, query with Athena

### Genomics
- Upload BAM files to S3
- Lambda for variant calling on genomic regions
- DynamoDB for variant metadata

### Medical
- Upload medical images to S3
- Lambda for image preprocessing
- Store predictions in DynamoDB

### Astronomy
- Upload FITS images to S3
- Lambda for source detection
- Athena for catalog queries

### Machine Learning Domains
- Upload training data to S3
- Lambda for data preprocessing
- Store processed data back to S3
- Train model locally (or in Studio Lab) on processed data

## Documentation Template

Each Tier 2 README should include:

1. **Project Overview** - What you'll build
2. **Prerequisites** - AWS account, Python, boto3
3. **Architecture Diagram** - Simple text/ASCII diagram
4. **AWS Setup Guide** - Step-by-step with screenshots references
5. **Running the Project** - Jupyter notebook or Python scripts
6. **Results** - What you'll learn/discover
7. **Cost Breakdown** - Detailed cost estimate
8. **Cleanup** - How to delete all resources
9. **Troubleshooting** - Common issues
10. **Next Steps** - Link to Tier 3

## Implementation Notes

- All Tier 2 projects should work independently
- No dependencies on other tiers
- Can run on Studio Lab or local machine (with AWS credentials)
- Include sample data or instructions to get data
- Clear documentation for AWS beginners
- Security best practices (least privilege IAM roles)
