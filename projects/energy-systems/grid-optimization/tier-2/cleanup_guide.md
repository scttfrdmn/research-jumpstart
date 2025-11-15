# AWS Cleanup Guide - Smart Grid Optimization Tier 2

This guide walks you through deleting all AWS resources created for this project to avoid ongoing charges.

**IMPORTANT:** Follow these steps carefully to ensure all resources are deleted.

---

## Prerequisites

- AWS CLI configured with credentials
- List of resource names you created

---

## Quick Cleanup (Automated)

If you want to delete everything at once:

```bash
# Set your resource names
export BUCKET_NAME="energy-grid-YOUR-ID"
export DYNAMODB_TABLE="GridAnalysis"
export LAMBDA_FUNCTION="optimize-energy-grid"
export IAM_ROLE="lambda-grid-optimizer"
export SNS_TOPIC_ARN="arn:aws:sns:us-east-1:ACCOUNT:grid-anomaly-alerts"

# Run cleanup script (see below)
bash cleanup_all.sh
```

---

## Step-by-Step Cleanup

### Step 1: Delete S3 Bucket

**IMPORTANT:** This will delete all data in the bucket. Download any data you want to keep first.

#### Option A: Using AWS Console

1. Go to [S3 Console](https://console.aws.amazon.com/s3/)
2. Find your bucket: `energy-grid-xxxx`
3. Select the bucket
4. Click "Empty"
5. Confirm by typing the bucket name
6. Wait for objects to be deleted
7. Click "Delete"
8. Confirm by typing the bucket name

#### Option B: Using AWS CLI

```bash
# Set your bucket name
BUCKET_NAME="energy-grid-YOUR-ID"

# List contents (to verify)
aws s3 ls "s3://$BUCKET_NAME" --recursive

# Delete all objects
aws s3 rm "s3://$BUCKET_NAME" --recursive

# Verify empty
aws s3 ls "s3://$BUCKET_NAME"

# Delete bucket
aws s3 rb "s3://$BUCKET_NAME"

# Verify deletion
aws s3 ls | grep energy-grid
```

**Expected output:** No buckets named `energy-grid-*`

**Cost saved:** ~$0.35/week for 5GB storage

---

### Step 2: Delete Lambda Function

#### Option A: Using AWS Console

1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Find function: `optimize-energy-grid`
3. Select the function
4. Click "Actions" → "Delete"
5. Confirm deletion

#### Option B: Using AWS CLI

```bash
# Delete Lambda function
LAMBDA_FUNCTION="optimize-energy-grid"

aws lambda delete-function --function-name "$LAMBDA_FUNCTION"

# Verify deletion
aws lambda list-functions | grep optimize-energy-grid
```

**Expected output:** No function named `optimize-energy-grid`

**Cost saved:** ~$2-3/day for Lambda executions

---

### Step 3: Delete DynamoDB Table

**WARNING:** This will delete all stored grid analysis data.

#### Option A: Using AWS Console

1. Go to [DynamoDB Console](https://console.aws.amazon.com/dynamodb/)
2. Click "Tables"
3. Select `GridAnalysis`
4. Click "Delete"
5. Confirm by typing "delete"

#### Option B: Using AWS CLI

```bash
# Delete DynamoDB table
DYNAMODB_TABLE="GridAnalysis"

aws dynamodb delete-table --table-name "$DYNAMODB_TABLE"

# Wait for deletion (takes ~1 minute)
aws dynamodb wait table-not-exists --table-name "$DYNAMODB_TABLE"

# Verify deletion
aws dynamodb list-tables | grep GridAnalysis
```

**Expected output:** No table named `GridAnalysis`

**Cost saved:** ~$1.25/GB/month for storage

---

### Step 4: Delete SNS Topic and Subscriptions

#### Option A: Using AWS Console

1. Go to [SNS Console](https://console.aws.amazon.com/sns/)
2. Click "Topics"
3. Select `grid-anomaly-alerts`
4. Click "Delete"
5. Confirm deletion

Subscriptions are automatically deleted with the topic.

#### Option B: Using AWS CLI

```bash
# Get SNS topic ARN
SNS_TOPIC_ARN=$(aws sns list-topics --query "Topics[?contains(TopicArn, 'grid-anomaly-alerts')].TopicArn" --output text)

echo "Deleting SNS topic: $SNS_TOPIC_ARN"

# Delete subscriptions first (optional, will be deleted with topic)
aws sns list-subscriptions-by-topic --topic-arn "$SNS_TOPIC_ARN" \
  --query 'Subscriptions[].SubscriptionArn' \
  --output text | xargs -I {} aws sns unsubscribe --subscription-arn {}

# Delete topic
aws sns delete-topic --topic-arn "$SNS_TOPIC_ARN"

# Verify deletion
aws sns list-topics | grep grid-anomaly-alerts
```

**Expected output:** No topic named `grid-anomaly-alerts`

**Cost saved:** ~$0.05/day for notifications

---

### Step 5: Delete IAM Role

**NOTE:** Only delete if you're not using this role for other projects.

#### Option A: Using AWS Console

1. Go to [IAM Console](https://console.aws.amazon.com/iam/)
2. Click "Roles"
3. Search for `lambda-grid-optimizer`
4. Select the role
5. Click "Delete"
6. Confirm deletion

#### Option B: Using AWS CLI

```bash
# Delete IAM role
IAM_ROLE="lambda-grid-optimizer"

# Detach managed policies first
aws iam list-attached-role-policies --role-name "$IAM_ROLE" \
  --query 'AttachedPolicies[].PolicyArn' \
  --output text | xargs -I {} aws iam detach-role-policy --role-name "$IAM_ROLE" --policy-arn {}

# Delete inline policies (if any)
aws iam list-role-policies --role-name "$IAM_ROLE" \
  --query 'PolicyNames' \
  --output text | xargs -I {} aws iam delete-role-policy --role-name "$IAM_ROLE" --policy-name {}

# Delete role
aws iam delete-role --role-name "$IAM_ROLE"

# Verify deletion
aws iam get-role --role-name "$IAM_ROLE" 2>&1 | grep NoSuchEntity
```

**Expected output:** `NoSuchEntity` error (role doesn't exist)

**Cost saved:** Free (IAM roles don't incur charges)

---

### Step 6: Delete CloudWatch Logs (Optional)

Lambda creates CloudWatch log groups that persist after deletion.

#### Using AWS CLI

```bash
# List log groups
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/optimize-energy-grid"

# Delete log group
aws logs delete-log-group --log-group-name "/aws/lambda/optimize-energy-grid"

# Verify deletion
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/optimize-energy-grid"
```

**Expected output:** No log groups found

**Cost saved:** ~$0.01/GB/month for log storage

---

### Step 7: Delete Local Files (Optional)

Clean up local sample data and results:

```bash
# Delete local directories
rm -rf sample_data/
rm -rf results/

# Delete environment file
rm -f .env

echo "✓ Local cleanup complete"
```

---

## Verification Checklist

After completing all steps, verify all resources are deleted:

```bash
# Check S3 buckets
echo "S3 Buckets:"
aws s3 ls | grep energy-grid
# Expected: No output

# Check Lambda functions
echo "Lambda Functions:"
aws lambda list-functions --query "Functions[?FunctionName=='optimize-energy-grid'].FunctionName"
# Expected: []

# Check DynamoDB tables
echo "DynamoDB Tables:"
aws dynamodb list-tables --query "TableNames[?@=='GridAnalysis']"
# Expected: []

# Check SNS topics
echo "SNS Topics:"
aws sns list-topics | grep grid-anomaly-alerts
# Expected: No output

# Check IAM roles
echo "IAM Roles:"
aws iam get-role --role-name lambda-grid-optimizer 2>&1 | grep NoSuchEntity
# Expected: NoSuchEntity error

# Check CloudWatch logs
echo "CloudWatch Logs:"
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/optimize-energy-grid"
# Expected: No log groups

echo ""
echo "✓ All resources verified as deleted"
```

---

## Cost Verification

After cleanup, verify no ongoing charges:

```bash
# Check costs for last 7 days
aws ce get-cost-and-usage \
  --time-period Start=$(date -v-7d +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics "UnblendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE \
  --query 'ResultsByTime[*].[TimePeriod.Start,Groups[?Keys[0]==`Amazon Simple Storage Service` || Keys[0]==`AWS Lambda` || Keys[0]==`Amazon DynamoDB`]]' \
  --output table
```

**Expected:** Costs should drop to $0 within 24 hours of cleanup.

---

## Troubleshooting

### Problem: S3 bucket won't delete

**Cause:** Bucket still has objects or versioning enabled

**Solution:**
```bash
# List all objects including versions
aws s3api list-object-versions --bucket "$BUCKET_NAME"

# Delete all object versions
aws s3api list-object-versions --bucket "$BUCKET_NAME" \
  --query 'Versions[].{Key:Key,VersionId:VersionId}' \
  --output text | xargs -I {} aws s3api delete-object --bucket "$BUCKET_NAME" --key {}

# Try deleting bucket again
aws s3 rb "s3://$BUCKET_NAME"
```

### Problem: Lambda function still exists after deletion

**Cause:** Lambda might have multiple versions

**Solution:**
```bash
# List all versions
aws lambda list-versions-by-function --function-name "$LAMBDA_FUNCTION"

# Delete all versions
aws lambda list-versions-by-function --function-name "$LAMBDA_FUNCTION" \
  --query 'Versions[?Version!=`$LATEST`].Version' \
  --output text | xargs -I {} aws lambda delete-function --function-name "$LAMBDA_FUNCTION" --qualifier {}

# Delete function
aws lambda delete-function --function-name "$LAMBDA_FUNCTION"
```

### Problem: IAM role won't delete

**Cause:** Policies still attached or role in use

**Solution:**
```bash
# Force detach all policies
aws iam list-attached-role-policies --role-name "$IAM_ROLE" --query 'AttachedPolicies[].PolicyArn' --output text | while read policy; do
  aws iam detach-role-policy --role-name "$IAM_ROLE" --policy-arn "$policy"
done

# Delete inline policies
aws iam list-role-policies --role-name "$IAM_ROLE" --query 'PolicyNames' --output text | while read policy; do
  aws iam delete-role-policy --role-name "$IAM_ROLE" --policy-name "$policy"
done

# Try deleting role again
aws iam delete-role --role-name "$IAM_ROLE"
```

### Problem: Unexpected charges after cleanup

**Solution:**
1. Check AWS Cost Explorer: https://console.aws.amazon.com/cost-management/
2. Look for resources in other regions: `aws ec2 describe-regions --query 'Regions[].RegionName'`
3. Search for resources with tag: `Environment=tier-2`
4. Contact AWS Support if charges persist

---

## Automated Cleanup Script

Save this as `cleanup_all.sh`:

```bash
#!/bin/bash

# Smart Grid Optimization - Complete Cleanup Script

set -e

echo "Smart Grid Optimization - AWS Cleanup"
echo "======================================"
echo ""
echo "WARNING: This will delete all AWS resources for this project."
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Configuration
BUCKET_NAME="${BUCKET_NAME:-energy-grid-YOUR-ID}"
DYNAMODB_TABLE="${DYNAMODB_TABLE:-GridAnalysis}"
LAMBDA_FUNCTION="${LAMBDA_FUNCTION:-optimize-energy-grid}"
IAM_ROLE="${IAM_ROLE:-lambda-grid-optimizer}"
LOG_GROUP="/aws/lambda/$LAMBDA_FUNCTION"

echo ""
echo "Deleting resources..."
echo ""

# 1. S3 Bucket
echo "[1/6] Deleting S3 bucket: $BUCKET_NAME"
aws s3 rm "s3://$BUCKET_NAME" --recursive 2>/dev/null || true
aws s3 rb "s3://$BUCKET_NAME" 2>/dev/null || true
echo "  ✓ S3 bucket deleted"

# 2. Lambda Function
echo "[2/6] Deleting Lambda function: $LAMBDA_FUNCTION"
aws lambda delete-function --function-name "$LAMBDA_FUNCTION" 2>/dev/null || true
echo "  ✓ Lambda function deleted"

# 3. DynamoDB Table
echo "[3/6] Deleting DynamoDB table: $DYNAMODB_TABLE"
aws dynamodb delete-table --table-name "$DYNAMODB_TABLE" 2>/dev/null || true
echo "  ✓ DynamoDB table deleted"

# 4. SNS Topic
echo "[4/6] Deleting SNS topic: grid-anomaly-alerts"
SNS_TOPIC_ARN=$(aws sns list-topics --query "Topics[?contains(TopicArn, 'grid-anomaly-alerts')].TopicArn" --output text 2>/dev/null || true)
if [ -n "$SNS_TOPIC_ARN" ]; then
  aws sns delete-topic --topic-arn "$SNS_TOPIC_ARN" 2>/dev/null || true
  echo "  ✓ SNS topic deleted"
else
  echo "  ℹ SNS topic not found (already deleted)"
fi

# 5. IAM Role
echo "[5/6] Deleting IAM role: $IAM_ROLE"
# Detach policies
aws iam list-attached-role-policies --role-name "$IAM_ROLE" --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null | xargs -I {} aws iam detach-role-policy --role-name "$IAM_ROLE" --policy-arn {} 2>/dev/null || true
# Delete role
aws iam delete-role --role-name "$IAM_ROLE" 2>/dev/null || true
echo "  ✓ IAM role deleted"

# 6. CloudWatch Logs
echo "[6/6] Deleting CloudWatch logs: $LOG_GROUP"
aws logs delete-log-group --log-group-name "$LOG_GROUP" 2>/dev/null || true
echo "  ✓ CloudWatch logs deleted"

echo ""
echo "======================================"
echo "✓ Cleanup complete!"
echo ""
echo "All AWS resources have been deleted."
echo "Verify at: https://console.aws.amazon.com/"
echo ""
```

**Usage:**
```bash
chmod +x cleanup_all.sh
./cleanup_all.sh
```

---

## Final Steps

1. **Verify in AWS Console:**
   - Visit https://console.aws.amazon.com/
   - Check S3, Lambda, DynamoDB, SNS, IAM
   - Confirm no resources remain

2. **Check Billing:**
   - Visit https://console.aws.amazon.com/billing/
   - Review current charges
   - Set up billing alert for $1 to catch any remaining charges

3. **Document:**
   - Save any analysis results you want to keep
   - Document lessons learned
   - Note estimated costs for your records

---

## Summary

**Resources Deleted:**
- ✅ S3 bucket and all objects
- ✅ Lambda function
- ✅ DynamoDB table
- ✅ SNS topic and subscriptions
- ✅ IAM role and policies
- ✅ CloudWatch log groups

**Cost Savings:**
- ~$10-15/week avoided

**Next Steps:**
- Move to Tier 3 for production infrastructure
- Explore other Tier 2 projects
- Apply learnings to your research

---

**Questions?** See main [README.md](README.md) or open an issue.

**Last updated:** 2025-11-14
