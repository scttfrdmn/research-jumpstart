# AWS Cleanup Guide - Behavioral Data Analysis

**Time Required:** 5-10 minutes
**Purpose:** Delete all AWS resources to stop incurring charges

This guide walks you through deleting all resources created for the behavioral analysis project.

---

## Why Cleanup is Important

AWS charges for resources even when they're not actively used:
- **S3**: Storage charges per GB per month
- **Lambda**: No charges when not invoked (only execution charges)
- **DynamoDB**: On-demand billing charges per storage
- **CloudWatch Logs**: Storage charges for logs

**Estimated cost if not cleaned up:** $2-5 per month for unused resources

---

## Cleanup Checklist

- [ ] Delete DynamoDB table
- [ ] Delete S3 bucket and all contents
- [ ] Delete Lambda function
- [ ] Delete Lambda layers (if custom)
- [ ] Delete IAM role and policies
- [ ] Delete CloudWatch log groups
- [ ] (Optional) Delete Athena workgroup
- [ ] Verify all resources deleted
- [ ] Check AWS billing for remaining charges

---

## Step 1: Delete DynamoDB Table

### Option A: AWS Console

1. Go to DynamoDB Console: https://console.aws.amazon.com/dynamodb/
2. Click "Tables" in left sidebar
3. Select "BehavioralAnalysis" table
4. Click "Delete table"
5. Confirm by typing "delete" and clicking "Delete"

### Option B: AWS CLI

```bash
aws dynamodb delete-table --table-name BehavioralAnalysis --region us-east-1

# Wait for deletion to complete
aws dynamodb wait table-not-exists --table-name BehavioralAnalysis --region us-east-1

echo "DynamoDB table deleted"
```

### Verify Deletion

```bash
aws dynamodb list-tables --region us-east-1 | grep BehavioralAnalysis
# Should return no results
```

---

## Step 2: Delete S3 Bucket

**Important:** You must delete all objects before deleting the bucket.

### Option A: AWS Console

1. Go to S3 Console: https://console.aws.amazon.com/s3/
2. Find your bucket: "behavioral-data-xxxxx"
3. Click on the bucket name
4. Click "Empty bucket"
5. Confirm by typing "permanently delete" and clicking "Empty"
6. Go back to bucket list
7. Select the bucket (checkbox)
8. Click "Delete bucket"
9. Confirm by typing the bucket name and clicking "Delete bucket"

### Option B: AWS CLI

```bash
# Set your bucket name
BUCKET_NAME="behavioral-data-xxxxx"  # Replace with your bucket name

# Delete all objects in the bucket
aws s3 rm s3://$BUCKET_NAME --recursive

# Delete all versions (if versioning was enabled)
aws s3api list-object-versions --bucket $BUCKET_NAME | \
  jq -r '.Versions[] | "--key \"\(.Key)\" --version-id \(.VersionId)"' | \
  xargs -L1 aws s3api delete-object --bucket $BUCKET_NAME

# Delete the bucket
aws s3 rb s3://$BUCKET_NAME

echo "S3 bucket deleted"
```

### Verify Deletion

```bash
aws s3 ls | grep behavioral-data
# Should return no results
```

---

## Step 3: Delete Lambda Function

### Option A: AWS Console

1. Go to Lambda Console: https://console.aws.amazon.com/lambda/
2. Click "Functions" in left sidebar
3. Select "analyze-behavioral-data"
4. Click "Actions" → "Delete"
5. Confirm deletion

### Option B: AWS CLI

```bash
aws lambda delete-function \
  --function-name analyze-behavioral-data \
  --region us-east-1

echo "Lambda function deleted"
```

### Verify Deletion

```bash
aws lambda list-functions --region us-east-1 | grep analyze-behavioral-data
# Should return no results
```

---

## Step 4: Delete Lambda Layers (If Custom)

If you created a custom Lambda layer:

### Option A: AWS Console

1. Go to Lambda Console: https://console.aws.amazon.com/lambda/
2. Click "Layers" in left sidebar
3. Select your custom layer (e.g., "scipy-numpy-pandas")
4. Click "Delete"
5. Confirm deletion

### Option B: AWS CLI

```bash
# List your layers
aws lambda list-layers --region us-east-1

# Delete custom layer (if you created one)
# aws lambda delete-layer-version \
#   --layer-name scipy-numpy-pandas \
#   --version-number 1 \
#   --region us-east-1

# Note: AWS-provided layers don't need to be deleted
```

---

## Step 5: Delete IAM Role and Policies

### Option A: AWS Console

1. Go to IAM Console: https://console.aws.amazon.com/iam/
2. Click "Roles" in left sidebar
3. Search for "lambda-behavioral-processor"
4. Click on the role name
5. Click "Delete role"
6. Confirm deletion

### Option B: AWS CLI

```bash
# Delete inline policy
aws iam delete-role-policy \
  --role-name lambda-behavioral-processor \
  --policy-name BehavioralAnalysisPolicy

# Delete the role
aws iam delete-role --role-name lambda-behavioral-processor

echo "IAM role deleted"
```

### Verify Deletion

```bash
aws iam get-role --role-name lambda-behavioral-processor 2>&1 | grep NoSuchEntity
# Should show "NoSuchEntity" error
```

---

## Step 6: Delete CloudWatch Log Groups

Lambda automatically creates CloudWatch log groups. Clean them up:

### Option A: AWS Console

1. Go to CloudWatch Console: https://console.aws.amazon.com/cloudwatch/
2. Click "Log groups" in left sidebar
3. Search for "/aws/lambda/analyze-behavioral-data"
4. Select the log group (checkbox)
5. Click "Actions" → "Delete log group(s)"
6. Confirm deletion

### Option B: AWS CLI

```bash
# Delete Lambda log group
aws logs delete-log-group \
  --log-group-name /aws/lambda/analyze-behavioral-data \
  --region us-east-1

echo "CloudWatch logs deleted"
```

### Verify Deletion

```bash
aws logs describe-log-groups --region us-east-1 | grep analyze-behavioral-data
# Should return no results
```

---

## Step 7: Delete Athena Resources (If Created)

If you created Athena workgroup:

### Option A: AWS Console

1. Go to Athena Console: https://console.aws.amazon.com/athena/
2. Click "Workgroups" in left sidebar
3. Select "behavioral-analysis"
4. Click "Delete"
5. Confirm deletion

### Option B: AWS CLI

```bash
# Delete workgroup
aws athena delete-work-group \
  --work-group behavioral-analysis \
  --recursive-delete-option \
  --region us-east-1

# Delete Athena results bucket (if created)
ATHENA_BUCKET="athena-results-xxxxx"  # Replace with your bucket name
aws s3 rm s3://$ATHENA_BUCKET --recursive
aws s3 rb s3://$ATHENA_BUCKET

echo "Athena resources deleted"
```

---

## Step 8: Verify All Resources Deleted

Run these commands to ensure everything is cleaned up:

```bash
echo "Checking for remaining resources..."

# Check S3 buckets
echo -n "S3 buckets: "
aws s3 ls | grep behavioral-data && echo "⚠️  Found" || echo "✓ Clean"

# Check Lambda functions
echo -n "Lambda functions: "
aws lambda list-functions --region us-east-1 | grep analyze-behavioral-data && echo "⚠️  Found" || echo "✓ Clean"

# Check DynamoDB tables
echo -n "DynamoDB tables: "
aws dynamodb list-tables --region us-east-1 | grep BehavioralAnalysis && echo "⚠️  Found" || echo "✓ Clean"

# Check IAM roles
echo -n "IAM roles: "
aws iam get-role --role-name lambda-behavioral-processor 2>&1 | grep -q NoSuchEntity && echo "✓ Clean" || echo "⚠️  Found"

# Check CloudWatch logs
echo -n "CloudWatch logs: "
aws logs describe-log-groups --region us-east-1 | grep analyze-behavioral-data && echo "⚠️  Found" || echo "✓ Clean"

echo ""
echo "Verification complete!"
```

---

## Step 9: Check AWS Billing

### Verify No Ongoing Charges

1. Go to AWS Billing Console: https://console.aws.amazon.com/billing/
2. Click "Bills" in left sidebar
3. Check current month charges
4. Verify charges are minimal or zero after cleanup

### Set Up Cost Alerts (Recommended)

To avoid unexpected charges in the future:

1. Go to AWS Budgets: https://console.aws.amazon.com/billing/home#/budgets
2. Click "Create budget"
3. Select "Cost budget"
4. Set threshold: $5
5. Add email notification
6. Create budget

---

## Automated Cleanup Script

For convenience, here's an automated cleanup script:

```bash
#!/bin/bash
# cleanup.sh - Automated AWS resource cleanup

set -e

echo "==========================================="
echo "AWS Behavioral Analysis Cleanup"
echo "==========================================="
echo ""

# Configuration
BUCKET_NAME="behavioral-data-xxxxx"  # REPLACE THIS
LAMBDA_FUNCTION="analyze-behavioral-data"
DYNAMODB_TABLE="BehavioralAnalysis"
IAM_ROLE="lambda-behavioral-processor"
AWS_REGION="us-east-1"

echo "This will delete the following resources:"
echo "  - S3 bucket: $BUCKET_NAME"
echo "  - Lambda function: $LAMBDA_FUNCTION"
echo "  - DynamoDB table: $DYNAMODB_TABLE"
echo "  - IAM role: $IAM_ROLE"
echo ""
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled"
    exit 0
fi

echo ""
echo "Starting cleanup..."

# Delete DynamoDB table
echo "1. Deleting DynamoDB table..."
aws dynamodb delete-table --table-name $DYNAMODB_TABLE --region $AWS_REGION 2>/dev/null || echo "  (already deleted)"

# Delete S3 bucket
echo "2. Deleting S3 bucket..."
aws s3 rm s3://$BUCKET_NAME --recursive 2>/dev/null || echo "  (already empty)"
aws s3 rb s3://$BUCKET_NAME 2>/dev/null || echo "  (already deleted)"

# Delete Lambda function
echo "3. Deleting Lambda function..."
aws lambda delete-function --function-name $LAMBDA_FUNCTION --region $AWS_REGION 2>/dev/null || echo "  (already deleted)"

# Delete CloudWatch logs
echo "4. Deleting CloudWatch logs..."
aws logs delete-log-group --log-group-name /aws/lambda/$LAMBDA_FUNCTION --region $AWS_REGION 2>/dev/null || echo "  (already deleted)"

# Delete IAM role
echo "5. Deleting IAM role..."
aws iam delete-role-policy --role-name $IAM_ROLE --policy-name BehavioralAnalysisPolicy 2>/dev/null || echo "  (policy already deleted)"
aws iam delete-role --role-name $IAM_ROLE 2>/dev/null || echo "  (role already deleted)"

echo ""
echo "==========================================="
echo "Cleanup complete!"
echo "==========================================="
echo ""
echo "Please verify in AWS Console that all resources are deleted."
echo "Check AWS billing in 24 hours to confirm no ongoing charges."
```

To use this script:

```bash
# Make it executable
chmod +x cleanup.sh

# Edit the script to set your BUCKET_NAME
nano cleanup.sh

# Run it
./cleanup.sh
```

---

## Troubleshooting Cleanup

### Problem: "Cannot delete bucket - bucket not empty"

**Solution:**
```bash
# Force delete all objects including versions
aws s3api delete-objects \
  --bucket behavioral-data-xxxxx \
  --delete "$(aws s3api list-object-versions \
    --bucket behavioral-data-xxxxx \
    --output=json \
    --query='{Objects: Versions[].{Key:Key,VersionId:VersionId}}')"
```

### Problem: "Cannot delete role - role has policies attached"

**Solution:**
```bash
# List attached policies
aws iam list-attached-role-policies --role-name lambda-behavioral-processor

# Detach each policy
aws iam detach-role-policy \
  --role-name lambda-behavioral-processor \
  --policy-arn arn:aws:iam::aws:policy/...

# Then delete role
aws iam delete-role --role-name lambda-behavioral-processor
```

### Problem: "Access denied" errors

**Solution:** Ensure your AWS credentials have sufficient permissions. You need:
- `s3:DeleteBucket`, `s3:DeleteObject`
- `lambda:DeleteFunction`
- `dynamodb:DeleteTable`
- `iam:DeleteRole`, `iam:DeleteRolePolicy`
- `logs:DeleteLogGroup`

---

## Cost After Cleanup

After completing cleanup, your ongoing costs should be:
- **$0** - All resources deleted

**Note:** It may take 24-48 hours for charges to stop appearing in AWS billing.

---

## Re-running the Project

If you want to run the project again:

1. Follow `setup_guide.md` to recreate resources
2. All data will be fresh (previous data is deleted)
3. You can reuse the same scripts and notebook

---

## Questions?

If you encounter issues during cleanup:
1. Check AWS Console to see which resources remain
2. Review error messages from AWS CLI commands
3. Consult AWS documentation for specific services
4. Open an issue on GitHub if needed

---

**Cleanup complete!** Your AWS account is now clean of all behavioral analysis resources.

Return to [README.md](README.md) for project overview.
