# AWS Cleanup Guide - Ocean Data Analysis Tier 2

This guide walks you through deleting all AWS resources to stop incurring charges.

**Important:** Once resources are deleted, all data will be permanently lost. Export any results you want to keep before proceeding.

---

## Before You Start

### Export Data to Keep

```bash
# 1. Export DynamoDB observations
python scripts/query_results.py --export ocean_observations_backup.csv

# 2. Download S3 files you want to keep
aws s3 sync s3://$BUCKET_NAME/processed/ ./local_backup/

# 3. Save any generated figures from Jupyter notebook
```

### Verify Resources to Delete

```bash
# List all resources that will be deleted
echo "Resources to delete:"
echo "1. S3 Bucket: $BUCKET_NAME"
echo "2. DynamoDB Table: OceanObservations"
echo "3. Lambda Function: analyze-ocean-data"
echo "4. SNS Topic: ocean-anomaly-alerts"
echo "5. IAM Role: lambda-ocean-processor"
```

---

## Cleanup Steps

### Step 1: Delete S3 Bucket Contents

**Important:** You must delete all objects before deleting the bucket.

#### Option A: Using AWS Console

1. Go to [S3 Console](https://console.aws.amazon.com/s3/)
2. Click on your bucket (e.g., `ocean-data-alice-12345`)
3. Select all objects (check the box at the top)
4. Click "Delete"
5. Type "permanently delete" to confirm
6. Click "Delete objects"

#### Option B: Using AWS CLI

```bash
# Set your bucket name
BUCKET_NAME="ocean-data-YOUR-NAME"  # Replace with your bucket name

# Delete all objects in bucket
aws s3 rm s3://$BUCKET_NAME --recursive

# Verify bucket is empty
aws s3 ls s3://$BUCKET_NAME --recursive

# Delete the bucket itself
aws s3 rb s3://$BUCKET_NAME

# Verify bucket deleted
aws s3 ls | grep ocean-data
```

**Expected output:** Bucket should not appear in list.

---

### Step 2: Delete DynamoDB Table

#### Option A: Using AWS Console

1. Go to [DynamoDB Console](https://console.aws.amazon.com/dynamodb/)
2. Click "Tables" in left menu
3. Select `OceanObservations` table
4. Click "Delete"
5. Confirm by typing "delete"
6. Click "Delete table"

#### Option B: Using AWS CLI

```bash
# Delete DynamoDB table
aws dynamodb delete-table --table-name OceanObservations

# Wait for deletion to complete
aws dynamodb wait table-not-exists --table-name OceanObservations

# Verify table deleted
aws dynamodb list-tables | grep OceanObservations
```

**Expected output:** Table should not appear in list.

---

### Step 3: Delete Lambda Function

#### Option A: Using AWS Console

1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Click "Functions" in left menu
3. Select `analyze-ocean-data` function
4. Click "Actions" → "Delete"
5. Type "delete" to confirm
6. Click "Delete"

#### Option B: Using AWS CLI

```bash
# Delete Lambda function
aws lambda delete-function --function-name analyze-ocean-data

# Verify function deleted
aws lambda list-functions | grep analyze-ocean-data
```

**Expected output:** Function should not appear in list.

---

### Step 4: Delete SNS Topic

#### Option A: Using AWS Console

1. Go to [SNS Console](https://console.aws.amazon.com/sns/)
2. Click "Topics" in left menu
3. Select `ocean-anomaly-alerts` topic
4. Click "Delete"
5. Type "delete me" to confirm
6. Click "Delete"

#### Option B: Using AWS CLI

```bash
# Get topic ARN
TOPIC_ARN=$(aws sns list-topics --query 'Topics[?contains(TopicArn, `ocean-anomaly-alerts`)].TopicArn' --output text)

# Delete SNS topic
aws sns delete-topic --topic-arn "$TOPIC_ARN"

# Verify topic deleted
aws sns list-topics | grep ocean-anomaly-alerts
```

**Expected output:** Topic should not appear in list.

---

### Step 5: Delete IAM Role

**Important:** Detach all policies before deleting the role.

#### Option A: Using AWS Console

1. Go to [IAM Console](https://console.aws.amazon.com/iam/)
2. Click "Roles" in left menu
3. Search for `lambda-ocean-processor`
4. Click on the role name
5. In "Permissions" tab, detach all policies:
   - Click each policy → "Detach"
6. Click "Delete role"
7. Type the role name to confirm
8. Click "Delete"

#### Option B: Using AWS CLI

```bash
# Detach all policies
aws iam detach-role-policy \
  --role-name lambda-ocean-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam detach-role-policy \
  --role-name lambda-ocean-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam detach-role-policy \
  --role-name lambda-ocean-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

aws iam detach-role-policy \
  --role-name lambda-ocean-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonSNSFullAccess

# Delete IAM role
aws iam delete-role --role-name lambda-ocean-processor

# Verify role deleted
aws iam list-roles | grep lambda-ocean-processor
```

**Expected output:** Role should not appear in list.

---

### Step 6: Delete CloudWatch Logs (Optional)

Lambda automatically creates CloudWatch log groups. These are cheap but you can delete them.

#### Option A: Using AWS Console

1. Go to [CloudWatch Console](https://console.aws.amazon.com/cloudwatch/)
2. Click "Logs" → "Log groups"
3. Search for `/aws/lambda/analyze-ocean-data`
4. Select the log group
5. Click "Actions" → "Delete log group"
6. Confirm deletion

#### Option B: Using AWS CLI

```bash
# Delete CloudWatch log group
aws logs delete-log-group --log-group-name /aws/lambda/analyze-ocean-data

# Verify log group deleted
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/analyze-ocean-data
```

---

## Verification Checklist

Run these commands to verify all resources are deleted:

```bash
echo "Verification Checklist:"
echo "======================"

# Check S3 bucket
echo -n "S3 Bucket: "
aws s3 ls | grep ocean-data && echo "❌ STILL EXISTS" || echo "✓ Deleted"

# Check DynamoDB table
echo -n "DynamoDB Table: "
aws dynamodb list-tables | grep OceanObservations && echo "❌ STILL EXISTS" || echo "✓ Deleted"

# Check Lambda function
echo -n "Lambda Function: "
aws lambda list-functions | grep analyze-ocean-data && echo "❌ STILL EXISTS" || echo "✓ Deleted"

# Check SNS topic
echo -n "SNS Topic: "
aws sns list-topics | grep ocean-anomaly-alerts && echo "❌ STILL EXISTS" || echo "✓ Deleted"

# Check IAM role
echo -n "IAM Role: "
aws iam list-roles | grep lambda-ocean-processor && echo "❌ STILL EXISTS" || echo "✓ Deleted"

echo ""
echo "If all show '✓ Deleted', cleanup is complete!"
```

---

## Cost Verification

After cleanup, verify no charges are accumulating:

### Check AWS Billing

1. Go to [AWS Billing Console](https://console.aws.amazon.com/billing/)
2. Click "Bills" in left menu
3. Select current month
4. Verify charges for:
   - S3: Should be $0 after ~24 hours
   - Lambda: Should be $0 immediately
   - DynamoDB: Should be $0 after ~24 hours
   - SNS: Should be $0 immediately

### Set Up Final Cost Alert

```bash
# Check costs for the past 7 days
aws ce get-cost-and-usage \
  --time-period Start=$(date -u -d '7 days ago' '+%Y-%m-%d'),End=$(date -u '+%Y-%m-%d') \
  --granularity DAILY \
  --metrics "BlendedCost" \
  --group-by Type=SERVICE

# Should show decreasing costs after cleanup
```

---

## Troubleshooting Cleanup

### "Bucket Not Empty" Error

```bash
# Force delete all versions and delete markers
aws s3api delete-bucket --bucket $BUCKET_NAME --force

# Or use sync with delete flag
aws s3 sync s3://$BUCKET_NAME /dev/null --delete
aws s3 rb s3://$BUCKET_NAME --force
```

### "Table Not Found" in DynamoDB

- Table may already be deleted
- Wait 60 seconds and try again (deletion in progress)

### "Role Has Attached Policies"

```bash
# List attached policies
aws iam list-attached-role-policies --role-name lambda-ocean-processor

# Detach each policy manually
aws iam detach-role-policy \
  --role-name lambda-ocean-processor \
  --policy-arn <POLICY_ARN>
```

### "SNS Topic Still Has Subscriptions"

```bash
# List subscriptions
aws sns list-subscriptions-by-topic --topic-arn $TOPIC_ARN

# Unsubscribe each
aws sns unsubscribe --subscription-arn <SUBSCRIPTION_ARN>

# Then delete topic
aws sns delete-topic --topic-arn $TOPIC_ARN
```

---

## Automated Cleanup Script

For convenience, here's a complete cleanup script:

```bash
#!/bin/bash
# cleanup.sh - Automated AWS resource cleanup

set -e

echo "AWS Ocean Data Analysis - Cleanup Script"
echo "========================================"
echo ""
echo "⚠️  WARNING: This will permanently delete all resources and data!"
echo ""
read -p "Enter your S3 bucket name (e.g., ocean-data-alice-12345): " BUCKET_NAME
echo ""
read -p "Are you sure you want to proceed? (type 'yes'): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Starting cleanup..."
echo ""

# 1. Delete S3 bucket
echo "1. Deleting S3 bucket..."
aws s3 rm s3://$BUCKET_NAME --recursive 2>/dev/null || echo "  Bucket already empty"
aws s3 rb s3://$BUCKET_NAME 2>/dev/null || echo "  Bucket already deleted"
echo "  ✓ S3 bucket deleted"

# 2. Delete DynamoDB table
echo "2. Deleting DynamoDB table..."
aws dynamodb delete-table --table-name OceanObservations 2>/dev/null || echo "  Table already deleted"
echo "  ✓ DynamoDB table deleted"

# 3. Delete Lambda function
echo "3. Deleting Lambda function..."
aws lambda delete-function --function-name analyze-ocean-data 2>/dev/null || echo "  Function already deleted"
echo "  ✓ Lambda function deleted"

# 4. Delete SNS topic
echo "4. Deleting SNS topic..."
TOPIC_ARN=$(aws sns list-topics --query 'Topics[?contains(TopicArn, `ocean-anomaly-alerts`)].TopicArn' --output text 2>/dev/null)
if [ -n "$TOPIC_ARN" ]; then
    aws sns delete-topic --topic-arn "$TOPIC_ARN"
    echo "  ✓ SNS topic deleted"
else
    echo "  Topic already deleted"
fi

# 5. Delete IAM role
echo "5. Deleting IAM role..."
aws iam detach-role-policy --role-name lambda-ocean-processor --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true
aws iam detach-role-policy --role-name lambda-ocean-processor --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess 2>/dev/null || true
aws iam detach-role-policy --role-name lambda-ocean-processor --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess 2>/dev/null || true
aws iam detach-role-policy --role-name lambda-ocean-processor --policy-arn arn:aws:iam::aws:policy/AmazonSNSFullAccess 2>/dev/null || true
aws iam delete-role --role-name lambda-ocean-processor 2>/dev/null || echo "  Role already deleted"
echo "  ✓ IAM role deleted"

# 6. Delete CloudWatch logs
echo "6. Deleting CloudWatch logs..."
aws logs delete-log-group --log-group-name /aws/lambda/analyze-ocean-data 2>/dev/null || echo "  Log group already deleted"
echo "  ✓ CloudWatch logs deleted"

echo ""
echo "=========================================="
echo "✓ Cleanup complete!"
echo ""
echo "Wait 5-10 minutes, then check AWS billing console to verify no charges."
echo "=========================================="
```

Save this as `cleanup.sh`, make it executable, and run:

```bash
chmod +x cleanup.sh
./cleanup.sh
```

---

## Final Steps

1. **Wait 24 hours** - Some charges may still appear for a day
2. **Check billing** - Verify no ongoing charges
3. **Remove local files** - Delete downloaded data and scripts if no longer needed
4. **Update documentation** - Note any lessons learned

---

**Cleanup complete!** Your AWS account should now be clean of all ocean analysis resources.

**Last updated:** 2025-11-14 | Research Jumpstart v1.0.0
