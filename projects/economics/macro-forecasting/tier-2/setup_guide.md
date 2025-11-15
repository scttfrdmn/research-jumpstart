# AWS Setup Guide for Macroeconomic Forecasting

**Total Setup Time: 45-60 minutes**

This guide walks you through creating all necessary AWS resources for the Tier 2 Macroeconomic Forecasting project. You'll create S3 buckets, DynamoDB tables, IAM roles, and deploy a Lambda function with forecasting libraries.

## Prerequisites

- AWS account (new or existing)
- AWS CLI configured: `aws configure`
- Python 3.8+ with pip
- ~45 minutes of free time

## Cost Warning

**Important:** This project costs ~$6-11 to run. Delete resources immediately after completion (see cleanup_guide.md) to avoid ongoing charges.

## Step 1: Configure AWS CLI (5 minutes)

### 1.1 Install AWS CLI

```bash
# Check if AWS CLI is installed
aws --version

# If not installed:
# macOS/Linux
pip install awscli

# Windows
pip install awscli
```

### 1.2 Configure Credentials

```bash
# Configure AWS credentials
aws configure

# You'll be prompted for:
# AWS Access Key ID: [Your access key]
# AWS Secret Access Key: [Your secret key]
# Default region name: us-east-1
# Default output format: json
```

To create access keys:
1. Log into AWS Console: https://console.aws.amazon.com/
2. Click your name (top right) → Security credentials
3. Scroll to "Access keys" → Create access key
4. Download and save the credentials securely

### 1.3 Verify Configuration

```bash
# Test AWS CLI
aws sts get-caller-identity

# Expected output:
# {
#     "UserId": "...",
#     "Account": "123456789012",
#     "Arn": "arn:aws:iam::123456789012:user/yourname"
# }
```

## Step 2: Create S3 Bucket (5 minutes)

S3 stores your economic time series data.

### 2.1 Create Bucket

```bash
# Generate unique bucket name
export BUCKET_NAME="economic-data-$(date +%s)"

# Create bucket in us-east-1
aws s3 mb s3://${BUCKET_NAME} --region us-east-1

echo "✓ S3 bucket created: ${BUCKET_NAME}"
```

### 2.2 Configure Lifecycle Policy

```bash
# Automatically delete data after 7 days to save costs
cat > lifecycle.json << 'EOF'
{
  "Rules": [
    {
      "Id": "DeleteOldData",
      "Status": "Enabled",
      "ExpirationInDays": 7,
      "NoncurrentVersionExpirationInDays": 1
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
  --bucket ${BUCKET_NAME} \
  --lifecycle-configuration file://lifecycle.json

rm lifecycle.json
echo "✓ Lifecycle policy applied (data deleted after 7 days)"
```

### 2.3 Create Folder Structure

```bash
# Create folders for organization
aws s3api put-object --bucket ${BUCKET_NAME} --key raw/gdp/
aws s3api put-object --bucket ${BUCKET_NAME} --key raw/unemployment/
aws s3api put-object --bucket ${BUCKET_NAME} --key raw/inflation/

echo "✓ S3 folder structure created"
```

### 2.4 Save Configuration

```bash
# Save bucket name for later use
cat > aws_config.sh << EOF
export BUCKET_NAME=${BUCKET_NAME}
export AWS_REGION=us-east-1
export TABLE_NAME=EconomicForecasts
export FUNCTION_NAME=forecast-economic-indicators
EOF

source aws_config.sh
echo "✓ Configuration saved to aws_config.sh"
```

## Step 3: Create DynamoDB Table (5 minutes)

DynamoDB stores forecast predictions for fast queries.

### 3.1 Create Table

```bash
export TABLE_NAME="EconomicForecasts"

aws dynamodb create-table \
  --table-name ${TABLE_NAME} \
  --attribute-definitions \
    AttributeName=indicator_country,AttributeType=S \
    AttributeName=forecast_date,AttributeType=N \
  --key-schema \
    AttributeName=indicator_country,KeyType=HASH \
    AttributeName=forecast_date,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

echo "⏳ Waiting for DynamoDB table to be active..."
```

### 3.2 Wait for Table Creation

```bash
aws dynamodb wait table-exists \
  --table-name ${TABLE_NAME} \
  --region us-east-1

echo "✓ DynamoDB table created: ${TABLE_NAME}"
```

### 3.3 Verify Table

```bash
# Describe table
aws dynamodb describe-table \
  --table-name ${TABLE_NAME} \
  --query 'Table.[TableName,TableStatus,ItemCount]' \
  --output table

# Expected output:
# -------------------------------
# |      DescribeTable          |
# +-----------------------------+
# |  EconomicForecasts          |
# |  ACTIVE                     |
# |  0                          |
# +-----------------------------+
```

## Step 4: Create IAM Role for Lambda (10 minutes)

Lambda needs permissions to read S3, write to S3, write to DynamoDB, and log to CloudWatch.

### 4.1 Create IAM Role

```bash
export ROLE_NAME="lambda-economic-forecaster"

# Create role with Lambda trust policy
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

### 4.2 Create and Attach Policy

```bash
export POLICY_NAME="economic-forecasting-policy"

# Create policy with S3, DynamoDB, and CloudWatch permissions
cat > policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::${BUCKET_NAME}/*",
        "arn:aws:s3:::${BUCKET_NAME}"
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

# Create policy
POLICY_ARN=$(aws iam create-policy \
  --policy-name ${POLICY_NAME} \
  --policy-document file://policy.json \
  --query 'Policy.Arn' \
  --output text)

# Attach policy to role
aws iam attach-role-policy \
  --role-name ${ROLE_NAME} \
  --policy-arn ${POLICY_ARN}

rm policy.json
echo "✓ Policy created and attached: ${POLICY_NAME}"
```

### 4.3 Save Role ARN

```bash
# Get role ARN for Lambda deployment
export ROLE_ARN=$(aws iam get-role \
  --role-name ${ROLE_NAME} \
  --query 'Role.Arn' \
  --output text)

# Add to config file
echo "export ROLE_ARN=${ROLE_ARN}" >> aws_config.sh
source aws_config.sh

echo "✓ Role ARN saved: ${ROLE_ARN}"
```

## Step 5: Deploy Lambda Function (15 minutes)

Lambda runs forecasting models when new data is uploaded to S3.

### 5.1 Create Lambda Deployment Package

```bash
# Create deployment directory
mkdir -p lambda_deploy
cd lambda_deploy

# Copy Lambda function
cp ../scripts/lambda_function.py .

# Install dependencies
# Note: statsmodels is large, so we'll use a Lambda Layer
pip install \
  boto3 \
  pandas \
  numpy \
  -t .

echo "✓ Lambda dependencies installed"
```

### 5.2 Create Lambda Layer for statsmodels

statsmodels is too large to package with Lambda function. Use a Lambda Layer instead.

```bash
# Create layer directory
mkdir -p statsmodels-layer/python/lib/python3.9/site-packages

# Install statsmodels to layer
pip install \
  statsmodels \
  scipy \
  patsy \
  -t statsmodels-layer/python/lib/python3.9/site-packages/

# Package layer
cd statsmodels-layer
zip -r ../statsmodels-layer.zip python/
cd ..

# Publish layer
LAYER_ARN=$(aws lambda publish-layer-version \
  --layer-name statsmodels \
  --description "statsmodels for time series forecasting" \
  --zip-file fileb://statsmodels-layer.zip \
  --compatible-runtimes python3.9 python3.10 python3.11 \
  --query 'LayerVersionArn' \
  --output text)

echo "✓ Lambda layer published: ${LAYER_ARN}"
echo "export LAYER_ARN=${LAYER_ARN}" >> ../aws_config.sh
```

### 5.3 Package Lambda Function

```bash
# Package Lambda function (without statsmodels)
zip -r function.zip lambda_function.py

# Add lightweight dependencies
cd ..
zip -r lambda_deploy/function.zip boto3 pandas numpy

cd lambda_deploy
echo "✓ Lambda function packaged"
```

### 5.4 Create Lambda Function

```bash
export FUNCTION_NAME="forecast-economic-indicators"

# Wait 10 seconds for IAM role to propagate
echo "⏳ Waiting for IAM role to propagate (10 seconds)..."
sleep 10

# Create Lambda function
aws lambda create-function \
  --function-name ${FUNCTION_NAME} \
  --runtime python3.9 \
  --role ${ROLE_ARN} \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://function.zip \
  --timeout 300 \
  --memory-size 512 \
  --environment "Variables={
    BUCKET_NAME=${BUCKET_NAME},
    TABLE_NAME=${TABLE_NAME}
  }" \
  --region us-east-1

echo "✓ Lambda function created: ${FUNCTION_NAME}"
```

### 5.5 Attach Lambda Layer

```bash
# Attach statsmodels layer to Lambda function
aws lambda update-function-configuration \
  --function-name ${FUNCTION_NAME} \
  --layers ${LAYER_ARN} \
  --region us-east-1

# Wait for update to complete
echo "⏳ Waiting for Lambda update to complete..."
sleep 5

echo "✓ Lambda layer attached"
```

### 5.6 Test Lambda Function

```bash
# Create test event
cat > test_event.json << EOF
{
  "bucket": "${BUCKET_NAME}",
  "key": "raw/gdp/test_data.csv",
  "indicator": "GDP",
  "country": "USA"
}
EOF

# Invoke Lambda
aws lambda invoke \
  --function-name ${FUNCTION_NAME} \
  --payload file://test_event.json \
  --region us-east-1 \
  response.json

# Check response
cat response.json

# Clean up
rm test_event.json response.json
cd ..

echo "✓ Lambda function tested successfully"
```

## Step 6: Configure S3 Event Notifications (5 minutes)

Automatically trigger Lambda when new CSV files are uploaded to S3.

### 6.1 Grant S3 Permission to Invoke Lambda

```bash
# Add Lambda permission for S3 to invoke
aws lambda add-permission \
  --function-name ${FUNCTION_NAME} \
  --statement-id s3-trigger \
  --action lambda:InvokeFunction \
  --principal s3.amazonaws.com \
  --source-arn arn:aws:s3:::${BUCKET_NAME} \
  --region us-east-1

echo "✓ Lambda permission added for S3"
```

### 6.2 Configure S3 Event Notification

```bash
# Get Lambda function ARN
LAMBDA_ARN=$(aws lambda get-function \
  --function-name ${FUNCTION_NAME} \
  --query 'Configuration.FunctionArn' \
  --output text)

# Create notification configuration
cat > notification.json << EOF
{
  "LambdaFunctionConfigurations": [
    {
      "LambdaFunctionArn": "${LAMBDA_ARN}",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {
              "Name": "prefix",
              "Value": "raw/"
            },
            {
              "Name": "suffix",
              "Value": ".csv"
            }
          ]
        }
      }
    }
  ]
}
EOF

# Apply notification configuration
aws s3api put-bucket-notification-configuration \
  --bucket ${BUCKET_NAME} \
  --notification-configuration file://notification.json

rm notification.json
echo "✓ S3 event notification configured"
```

## Step 7: (Optional) Configure Athena (10 minutes)

Athena allows SQL queries on DynamoDB data.

### 7.1 Create Athena Query Results Bucket

```bash
export ATHENA_BUCKET="athena-results-$(date +%s)"

aws s3 mb s3://${ATHENA_BUCKET} --region us-east-1

echo "✓ Athena results bucket created: ${ATHENA_BUCKET}"
echo "export ATHENA_BUCKET=${ATHENA_BUCKET}" >> aws_config.sh
```

### 7.2 Create Athena Workgroup

```bash
# Create workgroup with query results location
aws athena create-work-group \
  --name economic-forecasting \
  --configuration "ResultConfigurationUpdates={
    OutputLocation=s3://${ATHENA_BUCKET}/
  }" \
  --region us-east-1

echo "✓ Athena workgroup created"
```

### 7.3 Create External Table for DynamoDB

Note: Athena cannot directly query DynamoDB. Instead, export DynamoDB to S3 and query with Athena, or use DynamoDB directly with boto3 (recommended).

```sql
-- This is for reference; you'll run this in Athena console
-- Not needed for this project (we'll query DynamoDB directly)

CREATE EXTERNAL TABLE economic_forecasts_athena (
  indicator_country STRING,
  forecast_date BIGINT,
  indicator STRING,
  country STRING,
  forecast_value DOUBLE,
  confidence_80_lower DOUBLE,
  confidence_80_upper DOUBLE,
  confidence_95_lower DOUBLE,
  confidence_95_upper DOUBLE,
  model_type STRING
)
STORED BY 'org.apache.hadoop.hive.dynamodb.DynamoDBStorageHandler'
TBLPROPERTIES (
  "dynamodb.table.name" = "EconomicForecasts",
  "dynamodb.column.mapping" = "indicator_country:indicator_country,forecast_date:forecast_date"
);
```

## Step 8: Verify Setup (5 minutes)

### 8.1 Check All Resources

```bash
# Source configuration
source aws_config.sh

echo "===== AWS Resources ====="
echo "S3 Bucket: ${BUCKET_NAME}"
echo "DynamoDB Table: ${TABLE_NAME}"
echo "Lambda Function: ${FUNCTION_NAME}"
echo "IAM Role: ${ROLE_NAME}"
echo "Athena Bucket: ${ATHENA_BUCKET}"
echo "======================="

# Verify S3 bucket
aws s3 ls s3://${BUCKET_NAME}

# Verify DynamoDB table
aws dynamodb describe-table \
  --table-name ${TABLE_NAME} \
  --query 'Table.TableStatus' \
  --output text

# Verify Lambda function
aws lambda get-function \
  --function-name ${FUNCTION_NAME} \
  --query 'Configuration.State' \
  --output text

echo "✓ All resources verified"
```

### 8.2 Test End-to-End Workflow

```bash
# Create sample CSV file
cat > sample.csv << 'EOF'
date,value
2020-01-01,20000
2020-04-01,19500
2020-07-01,20500
2020-10-01,21000
2021-01-01,21500
2021-04-01,22000
2021-07-01,22500
2021-10-01,23000
2022-01-01,23500
2022-04-01,24000
2022-07-01,24500
2022-10-01,25000
EOF

# Upload to S3 (triggers Lambda)
aws s3 cp sample.csv s3://${BUCKET_NAME}/raw/gdp/usa_gdp_quarterly.csv

echo "✓ Sample file uploaded, Lambda should be triggered"
echo "⏳ Wait 30 seconds for Lambda processing..."
sleep 30

# Check DynamoDB for results
aws dynamodb scan \
  --table-name ${TABLE_NAME} \
  --limit 5 \
  --region us-east-1

rm sample.csv
echo "✓ End-to-end test complete"
```

### 8.3 Monitor Lambda Execution

```bash
# Get recent Lambda logs
aws logs tail /aws/lambda/${FUNCTION_NAME} --follow --region us-east-1

# Press Ctrl+C to stop tailing logs

# Or view in AWS Console:
# https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups/log-group/$252Faws$252Flambda$252Fforecast-economic-indicators
```

## Step 9: Final Configuration

### 9.1 Save All Variables

```bash
# Ensure all variables are saved
cat > aws_config.sh << EOF
export BUCKET_NAME=${BUCKET_NAME}
export TABLE_NAME=${TABLE_NAME}
export FUNCTION_NAME=${FUNCTION_NAME}
export ROLE_NAME=${ROLE_NAME}
export ROLE_ARN=${ROLE_ARN}
export POLICY_ARN=${POLICY_ARN}
export LAYER_ARN=${LAYER_ARN}
export ATHENA_BUCKET=${ATHENA_BUCKET}
export AWS_REGION=us-east-1
EOF

source aws_config.sh
echo "✓ Configuration saved to aws_config.sh"
```

### 9.2 Print Setup Summary

```bash
echo "===== Setup Complete ====="
echo ""
echo "S3 Bucket: ${BUCKET_NAME}"
echo "DynamoDB Table: ${TABLE_NAME}"
echo "Lambda Function: ${FUNCTION_NAME}"
echo "IAM Role: ${ROLE_NAME}"
echo "Lambda Layer: statsmodels"
echo "Athena Bucket: ${ATHENA_BUCKET}"
echo ""
echo "Next Steps:"
echo "1. Run: python scripts/upload_to_s3.py"
echo "2. Monitor Lambda: aws logs tail /aws/lambda/${FUNCTION_NAME} --follow"
echo "3. Query results: python scripts/query_results.py"
echo "4. Analyze: jupyter notebook notebooks/economic_analysis.ipynb"
echo ""
echo "Cleanup: See cleanup_guide.md when finished"
echo "========================="
```

## Troubleshooting

### Issue: "Access Denied" when creating resources

**Solution:** Ensure your AWS user has necessary permissions:
- AmazonS3FullAccess
- AmazonDynamoDBFullAccess
- AWSLambda_FullAccess
- IAMFullAccess (or IAMUserChangePassword)

### Issue: Lambda Layer too large

**Solution:** Use a pre-built layer or reduce dependencies:
```bash
# Use scipy-optimized builds
pip install --platform manylinux2014_x86_64 --only-binary=:all: statsmodels
```

### Issue: Lambda timeout

**Solution:** Increase timeout and memory:
```bash
aws lambda update-function-configuration \
  --function-name ${FUNCTION_NAME} \
  --timeout 300 \
  --memory-size 1024
```

### Issue: DynamoDB write throttling

**Solution:** DynamoDB on-demand mode should auto-scale. If issues persist:
```bash
# Check throttling metrics in CloudWatch
aws cloudwatch get-metric-statistics \
  --namespace AWS/DynamoDB \
  --metric-name UserErrors \
  --dimensions Name=TableName,Value=${TABLE_NAME} \
  --start-time 2025-11-14T00:00:00Z \
  --end-time 2025-11-14T23:59:59Z \
  --period 3600 \
  --statistics Sum
```

### Issue: S3 event notifications not triggering Lambda

**Solution:** Check notification configuration:
```bash
aws s3api get-bucket-notification-configuration \
  --bucket ${BUCKET_NAME}

# Ensure Lambda has permission:
aws lambda get-policy \
  --function-name ${FUNCTION_NAME}
```

## Cost Estimates

### Setup Costs (One-time)
- IAM roles/policies: Free
- Lambda function creation: Free
- S3 bucket creation: Free
- DynamoDB table creation: Free
- **Total setup cost: $0**

### Ongoing Costs (Per Run)
- S3 storage (100 MB, 7 days): $0.02
- Lambda executions (50 @ 30s): $1.30
- DynamoDB writes (500 items): $0.65
- Athena queries (5 @ 1GB): $0.25
- **Total per run: ~$2.22**
- **With safety margin: $6-11**

### Cost Optimization
1. Delete data after analysis: Save $0.02/week
2. Use on-demand DynamoDB: Auto-scales, pay-per-use
3. Batch Lambda invocations: Reduce request costs
4. Use CloudWatch Logs Insights sparingly: $0.005/GB

## Next Steps

You've successfully set up all AWS resources! Now:

1. **Upload Data**: `python scripts/upload_to_s3.py`
2. **Query Results**: `python scripts/query_results.py`
3. **Analyze**: `jupyter notebook notebooks/economic_analysis.ipynb`
4. **Cleanup**: See `cleanup_guide.md` when finished

## Support

If you encounter issues:
1. Check AWS CloudWatch logs
2. Review IAM permissions
3. Verify all environment variables are set
4. See troubleshooting section above

**AWS Console Links:**
- S3: https://console.aws.amazon.com/s3/
- DynamoDB: https://console.aws.amazon.com/dynamodb/
- Lambda: https://console.aws.amazon.com/lambda/
- IAM: https://console.aws.amazon.com/iam/
- CloudWatch: https://console.aws.amazon.com/cloudwatch/

---

**Setup complete!** Configuration saved in `aws_config.sh`
