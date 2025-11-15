# AWS Cleanup Guide - Quantum Circuit Simulation Tier 2

**IMPORTANT:** Follow this guide to delete all AWS resources and stop incurring charges.

Estimated time: 10-15 minutes

---

## Why Cleanup is Critical

AWS charges for:
- **S3 Storage**: $0.023/GB/month (even if unused)
- **DynamoDB Storage**: $0.25/GB/month for on-demand tables
- **Lambda**: No charge if not invoked, but logs accumulate
- **CloudWatch Logs**: $0.50/GB/month after free tier

**If you don't clean up, you'll continue paying ~$2-5/month for storage alone.**

---

## Pre-Cleanup Checklist

Before deleting resources, ensure you've:

- ✅ Downloaded all analysis results you want to keep
- ✅ Exported any important circuit results from DynamoDB
- ✅ Saved any custom circuits or configurations
- ✅ Documented your findings in the Jupyter notebook
- ✅ (Optional) Exported CloudWatch logs for debugging

---

## Quick Cleanup (5 minutes)

For fast cleanup, run these commands in order:

```bash
# Set your bucket name
BUCKET_NAME="quantum-circuits-xxxx"  # Replace with your actual bucket name

# 1. Delete all S3 objects
aws s3 rm "s3://$BUCKET_NAME" --recursive

# 2. Delete S3 bucket
aws s3 rb "s3://$BUCKET_NAME"

# 3. Delete Lambda function
aws lambda delete-function --function-name simulate-quantum-circuit

# 4. Delete DynamoDB table
aws dynamodb delete-table --table-name QuantumResults

# 5. Detach policies from IAM role
aws iam detach-role-policy \
  --role-name lambda-quantum-simulator \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam detach-role-policy \
  --role-name lambda-quantum-simulator \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam detach-role-policy \
  --role-name lambda-quantum-simulator \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

# 6. Delete IAM role
aws iam delete-role --role-name lambda-quantum-simulator

# 7. Delete CloudWatch log groups
aws logs delete-log-group --log-group-name /aws/lambda/simulate-quantum-circuit

echo "✅ Cleanup complete!"
```

---

## Step-by-Step Cleanup (Recommended)

For careful cleanup with verification at each step:

### Step 1: Download Important Data

Before deleting anything, save what you need:

```bash
# Create backup directory
mkdir -p backup/circuits backup/results backup/logs

# Download all circuit definitions
aws s3 sync "s3://$BUCKET_NAME/circuits/" backup/circuits/

# Download all simulation results
aws s3 sync "s3://$BUCKET_NAME/results/" backup/results/

# Export DynamoDB data
aws dynamodb scan --table-name QuantumResults > backup/dynamodb_results.json

# Download CloudWatch logs (optional)
aws logs filter-log-events \
  --log-group-name /aws/lambda/simulate-quantum-circuit \
  --output json > backup/lambda_logs.json

echo "✅ Data backed up to ./backup/"
```

### Step 2: Delete S3 Bucket

S3 buckets must be empty before deletion.

#### Option A: Using AWS Console

1. Go to [S3 Console](https://console.aws.amazon.com/s3/)
2. Find your `quantum-circuits-xxxx` bucket
3. Select the bucket (checkbox, don't click name)
4. Click "Empty"
5. Type "permanently delete" to confirm
6. Click "Empty"
7. Wait for completion
8. Select the bucket again
9. Click "Delete"
10. Type bucket name to confirm
11. Click "Delete bucket"

#### Option B: Using AWS CLI

```bash
# Delete all objects (including versions if versioning enabled)
aws s3 rm "s3://$BUCKET_NAME" --recursive

# Verify bucket is empty
aws s3 ls "s3://$BUCKET_NAME" --recursive

# Delete the bucket
aws s3 rb "s3://$BUCKET_NAME"

# Verify deletion
aws s3 ls | grep quantum-circuits
# Should return nothing
```

**Verification:**
```bash
# This should return an error (bucket doesn't exist)
aws s3 ls "s3://$BUCKET_NAME"
```

### Step 3: Delete Lambda Function

#### Option A: Using AWS Console

1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Find `simulate-quantum-circuit` function
3. Select the function (checkbox)
4. Click "Actions" → "Delete"
5. Type "delete" to confirm
6. Click "Delete"

#### Option B: Using AWS CLI

```bash
# Delete Lambda function
aws lambda delete-function --function-name simulate-quantum-circuit

# Verify deletion
aws lambda list-functions | grep simulate-quantum-circuit
# Should return nothing
```

**Verification:**
```bash
# This should return an error (function doesn't exist)
aws lambda get-function --function-name simulate-quantum-circuit
```

### Step 4: Delete DynamoDB Table

#### Option A: Using AWS Console

1. Go to [DynamoDB Console](https://console.aws.amazon.com/dynamodb/)
2. Click "Tables" in left menu
3. Find `QuantumResults` table
4. Select the table (checkbox)
5. Click "Delete"
6. Uncheck "Create a backup"
7. Type "delete" to confirm
8. Click "Delete table"

#### Option B: Using AWS CLI

```bash
# Delete DynamoDB table
aws dynamodb delete-table --table-name QuantumResults

# Wait for deletion to complete
aws dynamodb wait table-not-exists --table-name QuantumResults

# Verify deletion
aws dynamodb list-tables | grep QuantumResults
# Should return nothing
```

**Verification:**
```bash
# This should return an error (table doesn't exist)
aws dynamodb describe-table --table-name QuantumResults
```

### Step 5: Delete IAM Role

IAM roles require detaching policies before deletion.

#### Option A: Using AWS Console

1. Go to [IAM Console](https://console.aws.amazon.com/iam/)
2. Click "Roles" in left menu
3. Search for `lambda-quantum-simulator`
4. Click the role name
5. Click "Permissions" tab
6. For each attached policy:
   - Select policy
   - Click "Detach policy"
   - Confirm
7. After all policies detached, click "Delete"
8. Type role name to confirm
9. Click "Delete"

#### Option B: Using AWS CLI

```bash
# List attached policies (to verify what needs detaching)
aws iam list-attached-role-policies --role-name lambda-quantum-simulator

# Detach all policies
aws iam detach-role-policy \
  --role-name lambda-quantum-simulator \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam detach-role-policy \
  --role-name lambda-quantum-simulator \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam detach-role-policy \
  --role-name lambda-quantum-simulator \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

# Delete the role
aws iam delete-role --role-name lambda-quantum-simulator

# Verify deletion
aws iam get-role --role-name lambda-quantum-simulator
# Should return an error
```

**Verification:**
```bash
# This should return an error (role doesn't exist)
aws iam get-role --role-name lambda-quantum-simulator
```

### Step 6: Delete CloudWatch Logs

Lambda automatically creates log groups that continue to accumulate charges.

#### Option A: Using AWS Console

1. Go to [CloudWatch Console](https://console.aws.amazon.com/cloudwatch/)
2. Click "Logs" → "Log groups"
3. Find `/aws/lambda/simulate-quantum-circuit`
4. Select log group (checkbox)
5. Click "Actions" → "Delete log group(s)"
6. Click "Delete"

#### Option B: Using AWS CLI

```bash
# Delete Lambda log group
aws logs delete-log-group --log-group-name /aws/lambda/simulate-quantum-circuit

# Verify deletion
aws logs describe-log-groups | grep simulate-quantum-circuit
# Should return nothing
```

**Verification:**
```bash
# This should return an error (log group doesn't exist)
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/simulate-quantum-circuit
```

### Step 7: (Optional) Delete Athena Resources

If you set up Athena for SQL queries:

#### Delete Athena Query Results

```bash
# Athena stores query results in S3
# If you created a separate results bucket or folder
aws s3 rm "s3://$BUCKET_NAME/athena-results/" --recursive
```

#### Drop Athena Tables and Database

```bash
# Drop the table (via Athena console or CLI)
aws athena start-query-execution \
  --query-string "DROP TABLE IF EXISTS quantum_circuits.results;" \
  --result-configuration "OutputLocation=s3://$BUCKET_NAME/athena-results/"

# Drop the database
aws athena start-query-execution \
  --query-string "DROP DATABASE IF EXISTS quantum_circuits;" \
  --result-configuration "OutputLocation=s3://$BUCKET_NAME/athena-results/"
```

### Step 8: Verify All Resources Deleted

Run these commands to ensure nothing remains:

```bash
# Check S3 buckets
aws s3 ls | grep quantum-circuits
# Expected: No output

# Check Lambda functions
aws lambda list-functions --query 'Functions[?contains(FunctionName, `quantum`)]'
# Expected: Empty list []

# Check DynamoDB tables
aws dynamodb list-tables --query 'TableNames[?contains(@, `Quantum`)]'
# Expected: Empty list []

# Check IAM roles
aws iam list-roles --query 'Roles[?contains(RoleName, `quantum`)]'
# Expected: Empty list []

# Check CloudWatch log groups
aws logs describe-log-groups --query 'logGroups[?contains(logGroupName, `quantum`)]'
# Expected: Empty list []
```

### Step 9: Delete Local Files (Optional)

Clean up local project files if no longer needed:

```bash
# Remove virtual environment
rm -rf venv/

# Remove downloaded data (keep backup/)
rm -rf results/ circuits/

# Remove environment config
rm .env

# Remove test files
rm test_*.qasm response.json
```

---

## Cost Verification

After cleanup, verify charges have stopped:

### Check AWS Bill

```bash
# View costs for current month
aws ce get-cost-and-usage \
  --time-period Start=$(date -d 'this month' +%Y-%m-01),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE

# Look for these services (should show $0 after cleanup):
# - Amazon Simple Storage Service (S3)
# - AWS Lambda
# - Amazon DynamoDB
# - Amazon CloudWatch
```

### Set Up Billing Alert (If Not Already Done)

```bash
# Create a $5 budget to catch any remaining charges
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget '{
    "BudgetName": "post-cleanup-alert",
    "BudgetLimit": {"Amount": "5", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }' \
  --notifications-with-subscribers '[{
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 50,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [{
      "SubscriptionType": "EMAIL",
      "Address": "your-email@example.com"
    }]
  }]'
```

### Monitor for 1 Week

Check your AWS bill daily for the next week to ensure no charges:

1. Go to [AWS Cost Explorer](https://console.aws.amazon.com/cost-management/)
2. Select "Daily" granularity
3. Group by "Service"
4. Look for any charges related to:
   - S3
   - Lambda
   - DynamoDB
   - CloudWatch

**Expected:** $0 for all these services after cleanup

---

## Troubleshooting Cleanup Issues

### Problem: Can't delete S3 bucket (not empty)

**Solution:**
```bash
# Force delete all objects including versions
aws s3api delete-objects \
  --bucket "$BUCKET_NAME" \
  --delete "$(aws s3api list-object-versions \
    --bucket "$BUCKET_NAME" \
    --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' \
    --max-items 1000)"

# Then try deleting bucket again
aws s3 rb "s3://$BUCKET_NAME"
```

### Problem: Can't delete IAM role (still attached to resources)

**Solution:**
```bash
# Check what's attached
aws iam list-attached-role-policies --role-name lambda-quantum-simulator
aws iam list-role-policies --role-name lambda-quantum-simulator

# Detach managed policies
aws iam detach-role-policy \
  --role-name lambda-quantum-simulator \
  --policy-arn [policy-arn-from-above]

# Delete inline policies
aws iam delete-role-policy \
  --role-name lambda-quantum-simulator \
  --policy-name [policy-name-from-above]

# Then delete role
aws iam delete-role --role-name lambda-quantum-simulator
```

### Problem: Lambda permission still exists for S3

**Solution:**
```bash
# Remove Lambda permission
aws lambda remove-permission \
  --function-name simulate-quantum-circuit \
  --statement-id s3-trigger-permission

# If function already deleted, this error is safe to ignore
```

### Problem: DynamoDB table stuck in "DELETING" status

**Solution:**
```bash
# Wait for deletion to complete (can take 1-2 minutes)
aws dynamodb wait table-not-exists --table-name QuantumResults

# Check status
aws dynamodb describe-table --table-name QuantumResults
# Should eventually return error: "Table not found"
```

### Problem: Still seeing charges after cleanup

**Solution:**
```bash
# Check for orphaned resources
aws resourcegroupstaggingapi get-resources \
  --tag-filters Key=Project,Values=quantum-tier-2

# List all S3 buckets (check for any missed)
aws s3 ls

# List all Lambda functions
aws lambda list-functions

# List all DynamoDB tables
aws dynamodb list-tables

# Check CloudWatch Logs (these accumulate charges)
aws logs describe-log-groups --query 'logGroups[*].logGroupName'
```

---

## Post-Cleanup Checklist

After running cleanup:

- ✅ S3 bucket deleted (verify in console)
- ✅ Lambda function deleted (verify in console)
- ✅ DynamoDB table deleted (verify in console)
- ✅ IAM role deleted (verify in console)
- ✅ CloudWatch logs deleted (verify in console)
- ✅ Athena resources deleted (if used)
- ✅ Billing alerts set up
- ✅ AWS bill checked (should show $0 for quantum project)
- ✅ Backup created of important results
- ✅ Local files cleaned up (optional)

---

## Final Verification Script

Run this script to verify all resources are deleted:

```bash
#!/bin/bash

echo "🔍 Verifying AWS cleanup..."
echo ""

# Check S3
echo "Checking S3 buckets..."
S3_COUNT=$(aws s3 ls | grep -c quantum-circuits || true)
if [ "$S3_COUNT" -eq 0 ]; then
  echo "✅ No quantum S3 buckets found"
else
  echo "❌ Found $S3_COUNT quantum S3 buckets - delete them!"
  aws s3 ls | grep quantum-circuits
fi
echo ""

# Check Lambda
echo "Checking Lambda functions..."
LAMBDA_COUNT=$(aws lambda list-functions --query 'Functions[?contains(FunctionName, `quantum`)]' --output json | grep -c FunctionName || true)
if [ "$LAMBDA_COUNT" -eq 0 ]; then
  echo "✅ No quantum Lambda functions found"
else
  echo "❌ Found quantum Lambda functions - delete them!"
  aws lambda list-functions --query 'Functions[?contains(FunctionName, `quantum`)].FunctionName'
fi
echo ""

# Check DynamoDB
echo "Checking DynamoDB tables..."
DYNAMO_COUNT=$(aws dynamodb list-tables --query 'TableNames[?contains(@, `Quantum`)]' --output json | grep -c Quantum || true)
if [ "$DYNAMO_COUNT" -eq 0 ]; then
  echo "✅ No quantum DynamoDB tables found"
else
  echo "❌ Found quantum DynamoDB tables - delete them!"
  aws dynamodb list-tables --query 'TableNames[?contains(@, `Quantum`)]'
fi
echo ""

# Check IAM
echo "Checking IAM roles..."
IAM_COUNT=$(aws iam list-roles --query 'Roles[?contains(RoleName, `quantum`)]' --output json | grep -c RoleName || true)
if [ "$IAM_COUNT" -eq 0 ]; then
  echo "✅ No quantum IAM roles found"
else
  echo "❌ Found quantum IAM roles - delete them!"
  aws iam list-roles --query 'Roles[?contains(RoleName, `quantum`)].RoleName'
fi
echo ""

# Check CloudWatch
echo "Checking CloudWatch log groups..."
CW_COUNT=$(aws logs describe-log-groups --query 'logGroups[?contains(logGroupName, `quantum`)]' --output json | grep -c logGroupName || true)
if [ "$CW_COUNT" -eq 0 ]; then
  echo "✅ No quantum CloudWatch log groups found"
else
  echo "❌ Found quantum CloudWatch log groups - delete them!"
  aws logs describe-log-groups --query 'logGroups[?contains(logGroupName, `quantum`)].logGroupName'
fi
echo ""

echo "🎉 Cleanup verification complete!"
```

Save this as `verify_cleanup.sh` and run:
```bash
chmod +x verify_cleanup.sh
./verify_cleanup.sh
```

---

## What to Keep

You may want to keep:
- **Jupyter notebooks** - Your analysis and results
- **backup/** folder - Downloaded circuits and results
- **Scripts** - For future use or learning
- **Documentation** - README.md, setup_guide.md

You should delete:
- **AWS resources** - Everything in AWS (costs money)
- **.env file** - Contains AWS-specific configuration
- **venv/** - Python virtual environment (recreate anytime)

---

## Re-Running the Project

If you want to run the project again later:

1. Keep the project files (scripts, notebooks)
2. Delete all AWS resources (follow this guide)
3. When ready to re-run:
   - Follow `setup_guide.md` again
   - Create new S3 bucket, Lambda, DynamoDB
   - Upload circuits and analyze

**Cost:** Same as initial run ($8-14)

---

## Support

If you encounter issues during cleanup:
- Check AWS Console manually for any remaining resources
- Review AWS Cost Explorer for unexpected charges
- Contact AWS Support if charges persist after cleanup
- Open GitHub issue for project-specific questions

---

**Last updated:** 2025-11-14 | Research Jumpstart v1.0.0
