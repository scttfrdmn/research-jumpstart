# CloudFormation Infrastructure

This directory contains AWS CloudFormation templates for deploying the Social Media Analysis infrastructure.

## Template Overview

**`social-media-infrastructure.yaml`** - Main infrastructure template that creates:

- **S3 Buckets**: Data storage and results
- **IAM Role**: Permissions for SageMaker and Lambda
- **CloudWatch**: Logging and monitoring
- **SNS Topic**: Cost and error alerts
- **Cost Alarms**: Budget monitoring

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Basic understanding of AWS services

## Quick Deployment

### Option 1: AWS Console

1. Navigate to [CloudFormation Console](https://console.aws.amazon.com/cloudformation)
2. Click **Create Stack** → **With new resources**
3. Choose **Upload a template file**
4. Select `social-media-infrastructure.yaml`
5. Configure parameters (see below)
6. Review and create stack

### Option 2: AWS CLI

```bash
# Deploy with default parameters
aws cloudformation create-stack \
  --stack-name social-media-analysis-dev \
  --template-body file://social-media-infrastructure.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters \
    ParameterKey=Environment,ParameterValue=dev \
    ParameterKey=EnableComprehend,ParameterValue=true

# Monitor deployment
aws cloudformation wait stack-create-complete \
  --stack-name social-media-analysis-dev

# Get outputs
aws cloudformation describe-stacks \
  --stack-name social-media-analysis-dev \
  --query 'Stacks[0].Outputs'
```

### Option 3: Using Parameter File

Create `parameters.json`:

```json
[
  {
    "ParameterKey": "ProjectName",
    "ParameterValue": "social-media-analysis"
  },
  {
    "ParameterKey": "Environment",
    "ParameterValue": "dev"
  },
  {
    "ParameterKey": "DataRetentionDays",
    "ParameterValue": "90"
  },
  {
    "ParameterKey": "EnableComprehend",
    "ParameterValue": "true"
  },
  {
    "ParameterKey": "MonthlyBudgetLimit",
    "ParameterValue": "100"
  }
]
```

Deploy:

```bash
aws cloudformation create-stack \
  --stack-name social-media-analysis-dev \
  --template-body file://social-media-infrastructure.yaml \
  --parameters file://parameters.json \
  --capabilities CAPABILITY_NAMED_IAM
```

## Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `ProjectName` | Project identifier | `social-media-analysis` | Used in resource naming |
| `Environment` | Deployment environment | `dev` | Options: dev, staging, prod |
| `DataRetentionDays` | S3 data retention | `90` | 1-3653 days |
| `EnableComprehend` | Enable AWS Comprehend | `true` | Set to false to reduce costs |
| `MonthlyBudgetLimit` | Cost alert threshold | `100` | USD, minimum 10 |

## Stack Outputs

After deployment, the stack provides these outputs:

```bash
# Get all outputs
aws cloudformation describe-stacks \
  --stack-name social-media-analysis-dev \
  --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
  --output table
```

Key outputs:
- **DataBucketName**: S3 bucket for input data
- **ResultsBucketName**: S3 bucket for analysis results
- **AnalysisRoleArn**: IAM role ARN for notebooks
- **LogGroupName**: CloudWatch log group
- **AlertTopicArn**: SNS topic for notifications

## Post-Deployment Setup

### 1. Configure Environment Variables

```bash
# Get stack outputs
export DATA_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name social-media-analysis-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`DataBucketName`].OutputValue' \
  --output text)

export RESULTS_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name social-media-analysis-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`ResultsBucketName`].OutputValue' \
  --output text)

export ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name social-media-analysis-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`AnalysisRoleArn`].OutputValue' \
  --output text)

# Save to .env file
cat > ../.env << EOF
DATA_BUCKET=$DATA_BUCKET
RESULTS_BUCKET=$RESULTS_BUCKET
ROLE_ARN=$ROLE_ARN
AWS_REGION=us-east-1
EOF
```

### 2. Subscribe to Alerts (Optional)

```bash
# Get SNS topic ARN
TOPIC_ARN=$(aws cloudformation describe-stacks \
  --stack-name social-media-analysis-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`AlertTopicArn`].OutputValue' \
  --output text)

# Subscribe email
aws sns subscribe \
  --topic-arn $TOPIC_ARN \
  --protocol email \
  --notification-endpoint your-email@example.com

# Confirm subscription via email
```

### 3. Upload Sample Data (Optional)

```bash
# Upload sample dataset
aws s3 cp ../studio-lab/sample_data.csv \
  s3://$DATA_BUCKET/sample/sample_data.csv

# Verify upload
aws s3 ls s3://$DATA_BUCKET/sample/
```

## Cost Estimation

### Base Infrastructure
- **S3 Storage**: ~$0.023/GB/month (Standard)
- **CloudWatch Logs**: ~$0.50/GB ingested, $0.03/GB stored
- **SNS**: ~$0.50/million notifications

### Usage-Based Costs
- **Comprehend Sentiment**: $0.0001 per unit (100 chars)
- **Comprehend Entity Detection**: $0.0001 per unit
- **Comprehend Key Phrases**: $0.0001 per unit
- **S3 Data Transfer Out**: $0.09/GB (after 100GB free tier)

### Example Monthly Costs

**Light Usage** (testing, development):
- Storage: 10GB → $0.23
- CloudWatch: 1GB logs → $0.53
- Comprehend: 1M units → $100
- **Total**: ~$101/month

**Medium Usage** (active research):
- Storage: 100GB → $2.30
- CloudWatch: 5GB logs → $2.65
- Comprehend: 5M units → $500
- **Total**: ~$505/month

**Heavy Usage** (production analysis):
- Storage: 1TB → $23.00
- CloudWatch: 20GB logs → $10.60
- Comprehend: 20M units → $2,000
- **Total**: ~$2,034/month

**Cost Optimization Tips**:
- Set `EnableComprehend=false` to use local VADER sentiment (saves ~95% on NLP costs)
- Reduce `DataRetentionDays` to delete old data automatically
- Use S3 Intelligent-Tiering (included in template)
- Process data in batches to minimize API calls
- Monitor CloudWatch billing dashboard regularly

## Updating the Stack

```bash
# Update with new parameters
aws cloudformation update-stack \
  --stack-name social-media-analysis-dev \
  --template-body file://social-media-infrastructure.yaml \
  --parameters \
    ParameterKey=MonthlyBudgetLimit,ParameterValue=200 \
    ParameterKey=DataRetentionDays,ParameterValue=60 \
  --capabilities CAPABILITY_NAMED_IAM

# Monitor update
aws cloudformation wait stack-update-complete \
  --stack-name social-media-analysis-dev
```

## Deleting the Stack

**WARNING**: This will delete all data in S3 buckets!

```bash
# Backup data first (optional)
aws s3 sync s3://$DATA_BUCKET ./backup/data/
aws s3 sync s3://$RESULTS_BUCKET ./backup/results/

# Empty S3 buckets (required before deletion)
aws s3 rm s3://$DATA_BUCKET --recursive
aws s3 rm s3://$RESULTS_BUCKET --recursive

# Delete stack
aws cloudformation delete-stack \
  --stack-name social-media-analysis-dev

# Monitor deletion
aws cloudformation wait stack-delete-complete \
  --stack-name social-media-analysis-dev
```

## Multi-Environment Deployment

Deploy separate stacks for dev/staging/prod:

```bash
# Development
aws cloudformation create-stack \
  --stack-name social-media-analysis-dev \
  --template-body file://social-media-infrastructure.yaml \
  --parameters ParameterKey=Environment,ParameterValue=dev \
  --capabilities CAPABILITY_NAMED_IAM

# Staging
aws cloudformation create-stack \
  --stack-name social-media-analysis-staging \
  --template-body file://social-media-infrastructure.yaml \
  --parameters ParameterKey=Environment,ParameterValue=staging \
  --capabilities CAPABILITY_NAMED_IAM

# Production
aws cloudformation create-stack \
  --stack-name social-media-analysis-prod \
  --template-body file://social-media-infrastructure.yaml \
  --parameters \
    ParameterKey=Environment,ParameterValue=prod \
    ParameterKey=DataRetentionDays,ParameterValue=365 \
    ParameterKey=MonthlyBudgetLimit,ParameterValue=500 \
  --capabilities CAPABILITY_NAMED_IAM
```

## Troubleshooting

### Stack Creation Fails

```bash
# Check stack events
aws cloudformation describe-stack-events \
  --stack-name social-media-analysis-dev \
  --max-items 20

# Common issues:
# - Insufficient IAM permissions → Add AdministratorAccess temporarily
# - Bucket name already exists → Change ProjectName parameter
# - Region not supported → Use us-east-1, us-west-2, or eu-west-1
```

### Permission Denied Errors

```bash
# Verify IAM role exists
aws iam get-role --role-name social-media-analysis-analysis-role-dev

# Test S3 access
aws s3 ls s3://$DATA_BUCKET --profile your-profile

# Verify Comprehend permissions (if enabled)
aws comprehend detect-sentiment \
  --text "Test sentiment" \
  --language-code en
```

### High Costs

```bash
# Check Comprehend usage
aws cloudwatch get-metric-statistics \
  --namespace AWS/Comprehend \
  --metric-name CharacterCount \
  --start-time 2025-11-01T00:00:00Z \
  --end-time 2025-11-09T23:59:59Z \
  --period 86400 \
  --statistics Sum

# Check S3 storage
aws s3 ls s3://$DATA_BUCKET --recursive --summarize

# Review cost allocation tags
aws cloudformation describe-stacks \
  --stack-name social-media-analysis-dev \
  --query 'Stacks[0].Tags'
```

## Security Best Practices

1. **Use IAM Roles**: Never hardcode credentials in notebooks
2. **Enable MFA**: Require MFA for stack modifications
3. **Restrict Access**: Use least-privilege IAM policies
4. **Encrypt Data**: Template enables S3 encryption by default
5. **Monitor Access**: Enable CloudTrail logging
6. **Regular Updates**: Review and update security groups quarterly

## Additional Resources

- [AWS CloudFormation Documentation](https://docs.aws.amazon.com/cloudformation/)
- [Amazon Comprehend Pricing](https://aws.amazon.com/comprehend/pricing/)
- [SageMaker Studio Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [S3 Cost Optimization](https://aws.amazon.com/s3/cost-optimization/)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)

## Support

For issues specific to this template:
1. Check stack events in CloudFormation console
2. Review CloudWatch logs in `/aws/sagemaker/social-media-analysis-{env}`
3. Verify IAM permissions match template requirements
4. Open an issue in the Research Jumpstart repository
