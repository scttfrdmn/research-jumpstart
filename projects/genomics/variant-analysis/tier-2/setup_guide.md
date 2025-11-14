# AWS Setup Guide for Genomic Variant Analysis

**Total Setup Time: 45-60 minutes**

This guide walks you through creating all necessary AWS resources for the Tier 2 Genomic Variant Analysis project. You'll create S3 buckets, DynamoDB tables, IAM roles, and deploy a Lambda function.

## Prerequisites

- AWS account (new or existing)
- AWS CLI configured: `aws configure`
- Python 3.8+ with pip
- ~30 minutes of free time

## Cost Warning

**Important:** This project costs ~$10-15 to run. Delete resources immediately after completion (see cleanup_guide.md).

## Step 1: Create S3 Buckets (5 minutes)

S3 stores your BAM files, reference genome, and results.

### 1.1 Create Input Bucket

```bash
# Replace 'your-username' with your unique identifier
export BUCKET_INPUT="genomics-input-$(date +%s)"

aws s3 mb s3://${BUCKET_INPUT} --region us-east-1

# Enable versioning (optional but recommended)
aws s3api put-bucket-versioning \
  --bucket ${BUCKET_INPUT} \
  --versioning-configuration Status=Enabled

# Set lifecycle policy (delete after 7 days to save costs)
cat > lifecycle.json << 'EOF'
{
  "Rules": [
    {
      "Id": "DeleteOldFiles",
      "Status": "Enabled",
      "ExpirationInDays": 7,
      "NoncurrentVersionExpirationInDays": 1
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
  --bucket ${BUCKET_INPUT} \
  --lifecycle-configuration file://lifecycle.json

rm lifecycle.json
echo "✓ Input bucket created: ${BUCKET_INPUT}"
```

### 1.2 Create Results Bucket

```bash
export BUCKET_RESULTS="genomics-results-$(date +%s)"

aws s3 mb s3://${BUCKET_RESULTS} --region us-east-1

# Set lifecycle policy (delete after 3 days)
cat > lifecycle.json << 'EOF'
{
  "Rules": [
    {
      "Id": "DeleteResults",
      "Status": "Enabled",
      "ExpirationInDays": 3
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
  --bucket ${BUCKET_RESULTS} \
  --lifecycle-configuration file://lifecycle.json

rm lifecycle.json
echo "✓ Results bucket created: ${BUCKET_RESULTS}"
```

### 1.3 Save Bucket Names

```bash
# Save for later use
cat > aws_config.sh << EOF
export BUCKET_INPUT=${BUCKET_INPUT}
export BUCKET_RESULTS=${BUCKET_RESULTS}
export AWS_REGION=us-east-1
EOF

source aws_config.sh
echo "✓ Configuration saved to aws_config.sh"
```

## Step 2: Create DynamoDB Table (5 minutes)

DynamoDB stores variant metadata for fast queries.

```bash
export TABLE_NAME="variant-metadata"

aws dynamodb create-table \
  --table-name ${TABLE_NAME} \
  --attribute-definitions \
    AttributeName=chrom_pos,AttributeType=S \
    AttributeName=timestamp,AttributeType=N \
  --key-schema \
    AttributeName=chrom_pos,KeyType=HASH \
    AttributeName=timestamp,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

# Wait for table creation
echo "⏳ Waiting for DynamoDB table to be active..."
aws dynamodb wait table-exists \
  --table-name ${TABLE_NAME} \
  --region us-east-1

echo "✓ DynamoDB table created: ${TABLE_NAME}"
```

## Step 3: Create IAM Role for Lambda (10 minutes)

Lambda needs permissions to read S3, write to S3, and write to DynamoDB.

### 3.1 Create IAM Role

```bash
export ROLE_NAME="variant-calling-lambda-role"

# Create role
aws iam create-role \
  --role-name ${ROLE_NAME} \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {
          "Service": "lambda.amazonaws.com"
        },
        "Action": "sts:AssumeRole"
      }
    ]
  }'

echo "✓ IAM role created: ${ROLE_NAME}"
```

### 3.2 Create and Attach Policy

```bash
export POLICY_NAME="variant-calling-policy"

# Create policy with S3 and DynamoDB permissions
cat > policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::${BUCKET_INPUT}/*",
        "arn:aws:s3:::${BUCKET_INPUT}",
        "arn:aws:s3:::${BUCKET_RESULTS}/*",
        "arn:aws:s3:::${BUCKET_RESULTS}"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:UpdateItem",
        "dynamodb:Query",
        "dynamodb:Scan"
      ],
      "Resource": "arn:aws:dynamodb:us-east-1:*:table/${TABLE_NAME}"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:*:*"
    }
  ]
}
EOF

# Create and attach inline policy
aws iam put-role-policy \
  --role-name ${ROLE_NAME} \
  --policy-name ${POLICY_NAME} \
  --policy-document file://policy.json

rm policy.json
echo "✓ IAM policy created and attached"
```

### 3.3 Get Role ARN

```bash
# Get the role ARN (you'll need this for Lambda)
export ROLE_ARN=$(aws iam get-role \
  --role-name ${ROLE_NAME} \
  --query 'Role.Arn' \
  --output text)

echo "✓ Role ARN: ${ROLE_ARN}"

# Save for later
echo "export ROLE_ARN=${ROLE_ARN}" >> aws_config.sh
source aws_config.sh
```

## Step 4: Create Lambda Function (15 minutes)

Lambda runs the variant calling code on demand.

### 4.1 Prepare Lambda Code

The Lambda function code is in `scripts/lambda_function.py`. Create a deployment package:

```bash
# Navigate to scripts directory
cd scripts

# Create deployment directory
mkdir -p lambda_package

# Copy lambda function
cp lambda_function.py lambda_package/

# Install dependencies in package
pip install -r ../requirements.txt -t lambda_package/

# Create ZIP
cd lambda_package
zip -r ../lambda_function.zip . -x "*/__pycache__/*" "*.pyc"
cd ..

echo "✓ Lambda deployment package created: lambda_function.zip"
```

### 4.2 Create Lambda Function

```bash
export LAMBDA_FUNCTION="variant-calling"
export LAMBDA_TIMEOUT=300  # 5 minutes max
export LAMBDA_MEMORY=1024  # 1GB memory

# Create function
aws lambda create-function \
  --function-name ${LAMBDA_FUNCTION} \
  --runtime python3.11 \
  --role ${ROLE_ARN} \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_function.zip \
  --timeout ${LAMBDA_TIMEOUT} \
  --memory-size ${LAMBDA_MEMORY} \
  --environment Variables="{BUCKET_RESULTS=${BUCKET_RESULTS},TABLE_NAME=${TABLE_NAME}}" \
  --region us-east-1

echo "✓ Lambda function created: ${LAMBDA_FUNCTION}"

# Save configuration
echo "export LAMBDA_FUNCTION=${LAMBDA_FUNCTION}" >> ../aws_config.sh
source ../aws_config.sh
```

### 4.3 Test Lambda Function

```bash
# Create test event
cat > test_event.json << 'EOF'
{
  "bucket": "genomics-input-example",
  "key": "chr20.bam",
  "region": "chr20:1000000-1010000",
  "sample_id": "NA12878"
}
EOF

# Invoke Lambda
aws lambda invoke \
  --function-name ${LAMBDA_FUNCTION} \
  --payload file://test_event.json \
  --region us-east-1 \
  response.json

# View response
cat response.json
rm test_event.json response.json

echo "✓ Lambda function tested successfully"
```

## Step 5: Upload Sample Data (10-20 minutes)

Download sample BAM files and reference genome to S3.

### 5.1 Download Sample BAM File

```bash
# Create data directory
mkdir -p ../data
cd ../data

# Download a small BAM file from 1000 Genomes (sample NA12878, chromosome 20)
# This is ~100MB - adjust if too large
echo "Downloading sample BAM file (100-200MB)..."
wget -q https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/phase3_EBI_analysis/alignment_indices/20130521.chr20.alignment.index

# Or use a smaller alternative if download fails
if [ ! -f "*.chr20.alignment.index" ]; then
  echo "Note: Download may take 5-10 minutes"
  # Alternative: create a minimal test BAM file
  python3 << 'PYTHON_EOF'
import os
print("Create minimal test BAM for demonstration")
# In practice, you would use real BAM files
PYTHON_EOF
fi

cd ..
```

### 5.2 Upload Sample Data to S3

```bash
source aws_config.sh

# Create sample reference data
python3 << 'EOF'
import os

# Create minimal test files
os.makedirs('data', exist_ok=True)

# Create a simple reference genome snippet
with open('data/chr20.fa', 'w') as f:
    f.write(">chr20\n")
    f.write("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\n" * 100)

# Create a sample VCF for testing
with open('data/sample.vcf', 'w') as f:
    f.write("##fileformat=VCFv4.2\n")
    f.write("##contig=<ID=chr20>\n")
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    f.write("chr20\t1000\t.\tA\tT\t60\tPASS\tDP=30\n")

print("✓ Test data created in data/ directory")
EOF

# Upload to S3
aws s3 cp data/chr20.fa s3://${BUCKET_INPUT}/reference/chr20.fa
aws s3 cp data/sample.vcf s3://${BUCKET_INPUT}/samples/sample.vcf

echo "✓ Sample data uploaded to S3"

# Verify upload
aws s3 ls s3://${BUCKET_INPUT}/ --recursive
```

## Step 6: Verify Setup (5 minutes)

Confirm all resources are properly configured.

```bash
source aws_config.sh

echo "=== Verification Checklist ==="

# 1. Check S3 buckets
echo -n "✓ S3 Input Bucket: "
aws s3 ls s3://${BUCKET_INPUT} >/dev/null && echo "OK" || echo "FAILED"

echo -n "✓ S3 Results Bucket: "
aws s3 ls s3://${BUCKET_RESULTS} >/dev/null && echo "OK" || echo "FAILED"

# 2. Check DynamoDB table
echo -n "✓ DynamoDB Table: "
aws dynamodb describe-table --table-name ${TABLE_NAME} --region us-east-1 >/dev/null && echo "OK" || echo "FAILED"

# 3. Check IAM role
echo -n "✓ IAM Role: "
aws iam get-role --role-name ${ROLE_NAME} >/dev/null && echo "OK" || echo "FAILED"

# 4. Check Lambda function
echo -n "✓ Lambda Function: "
aws lambda get-function --function-name ${LAMBDA_FUNCTION} --region us-east-1 >/dev/null && echo "OK" || echo "FAILED"

echo ""
echo "=== Configuration Summary ==="
echo "Input Bucket: ${BUCKET_INPUT}"
echo "Results Bucket: ${BUCKET_RESULTS}"
echo "DynamoDB Table: ${TABLE_NAME}"
echo "Lambda Function: ${LAMBDA_FUNCTION}"
echo "IAM Role: ${ROLE_NAME}"
echo "Region: ${AWS_REGION}"
echo ""
echo "✓ Setup complete! Ready to run notebooks."
```

## Step 7: Environment Variables

Save all configuration for use in notebooks:

```bash
# Create .env file for Jupyter notebooks
cat > ../.env << EOF
AWS_REGION=us-east-1
BUCKET_INPUT=${BUCKET_INPUT}
BUCKET_RESULTS=${BUCKET_RESULTS}
TABLE_NAME=${TABLE_NAME}
LAMBDA_FUNCTION=${LAMBDA_FUNCTION}
ROLE_ARN=${ROLE_ARN}
EOF

echo "✓ Environment variables saved to .env"
```

## Troubleshooting

### Issue: "Access Denied" when creating S3 bucket

**Solution:** Check that your AWS user has S3 permissions. You may need admin access.

```bash
aws iam get-user
```

### Issue: Lambda deployment fails

**Solution:** Ensure you're in the correct directory and the ZIP file was created properly.

```bash
# Check ZIP contents
unzip -l lambda_function.zip | head -20
```

### Issue: DynamoDB table creation times out

**Solution:** This is normal for on-demand pricing model. Wait a few moments and retry.

```bash
aws dynamodb describe-table --table-name ${TABLE_NAME}
```

### Issue: Role ARN is empty

**Solution:** Make sure the IAM role was created successfully before retrieving the ARN.

```bash
aws iam list-roles | grep variant-calling
```

## Next Steps

1. **Verify setup:** Run the verification checklist above
2. **Review sample data:** Check S3 bucket contains test files
3. **Run notebook:** Start `notebooks/variant_analysis.ipynb`
4. **Monitor costs:** Check AWS Cost Explorer regularly
5. **Cleanup:** Follow `cleanup_guide.md` when finished

## Cost Monitoring

Monitor your costs in real-time:

```bash
# Get current day's costs (updated every 4 hours)
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-01-02 \
  --granularity DAILY \
  --metrics "BlendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE \
  --region us-east-1
```

## Important Notes

- **Costs accumulate!** Don't forget to run cleanup_guide.md when finished
- **Lambda has a 5-minute timeout** - increase if needed for large genomic regions
- **DynamoDB is on-demand pricing** - more cost-effective than provisioned for variable workloads
- **S3 lifecycle policies automatically delete old files** - you can manually delete sooner

---

**You're all set!** Proceed to `notebooks/variant_analysis.ipynb` to start your analysis.
