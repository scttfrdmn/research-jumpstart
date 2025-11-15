# AWS Cleanup Guide - Historical Text Analysis Tier 2

This guide walks you through deleting all AWS resources created for this project to avoid ongoing charges.

**Important:** Follow these steps carefully to ensure all resources are deleted and billing stops.

---

## Cost Impact

**Before cleanup:**
- S3 storage: ~$0.023/GB/month
- DynamoDB storage: ~$0.25/GB/month (first 25GB free)
- Lambda: No cost when not invoked

**After cleanup:**
- All charges stop immediately
- No ongoing costs

---

## Prerequisites

- AWS CLI configured with credentials
- Access to AWS Console
- Note your bucket name and table name

---

## Step 1: Delete S3 Bucket Contents

S3 buckets must be empty before deletion.

### Option A: Using AWS Console

1. Go to [S3 Console](https://console.aws.amazon.com/s3/)
2. Find your bucket: `text-corpus-{your-id}`
3. Select the bucket (checkbox, don't click name)
4. Click "Empty" button
5. Type `permanently delete` to confirm
6. Click "Empty"
7. Wait for deletion to complete (may take a few minutes)

### Option B: Using AWS CLI

```bash
# Set your bucket name
BUCKET_NAME="text-corpus-your-id"  # Replace with your actual bucket name

# Delete all objects in bucket (including versions if enabled)
aws s3 rm "s3://$BUCKET_NAME" --recursive

# Verify bucket is empty
aws s3 ls "s3://$BUCKET_NAME/" --recursive

# Should return nothing
```

### Verify S3 Deletion

```bash
# Check bucket is empty
aws s3 ls "s3://$BUCKET_NAME/" --recursive

# If any objects remain, delete them manually
aws s3 rm "s3://$BUCKET_NAME/path/to/file"
```

---

## Step 2: Delete S3 Bucket

Now that the bucket is empty, delete it.

### Option A: Using AWS Console

1. In S3 Console, select your bucket
2. Click "Delete" button
3. Type bucket name to confirm
4. Click "Delete bucket"

### Option B: Using AWS CLI

```bash
# Delete the empty bucket
aws s3 rb "s3://$BUCKET_NAME"

# Verify deletion
aws s3 ls | grep text-corpus

# Should not show your bucket
```

### Verify Bucket Deletion

```bash
# Try to access bucket (should fail)
aws s3 ls "s3://$BUCKET_NAME"

# Should return: "An error occurred (NoSuchBucket)"
```

---

## Step 3: Delete DynamoDB Table

Delete the table containing analysis results.

### Option A: Using AWS Console

1. Go to [DynamoDB Console](https://console.aws.amazon.com/dynamodb/)
2. Click "Tables" in left menu
3. Select `TextAnalysis` table
4. Click "Delete" button
5. Confirm deletion:
   - Uncheck "Create backup before deletion" (unless you want backup)
   - Type `delete` to confirm
6. Click "Delete table"
7. Wait for deletion to complete (1-2 minutes)

### Option B: Using AWS CLI

```bash
# Delete DynamoDB table
aws dynamodb delete-table --table-name TextAnalysis

# Wait for deletion to complete
aws dynamodb wait table-not-exists --table-name TextAnalysis

# Verify deletion
aws dynamodb list-tables | grep TextAnalysis

# Should not appear in list
```

### Verify Table Deletion

```bash
# Try to describe table (should fail)
aws dynamodb describe-table --table-name TextAnalysis

# Should return: "Table not found"
```

---

## Step 4: Delete Lambda Function

Remove the text processing Lambda function.

### Option A: Using AWS Console

1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Find function: `process-text-document`
3. Select the function (checkbox)
4. Click "Actions" → "Delete"
5. Type `delete` to confirm
6. Click "Delete"

### Option B: Using AWS CLI

```bash
# Delete Lambda function
aws lambda delete-function --function-name process-text-document

# Verify deletion
aws lambda list-functions | grep process-text-document

# Should not appear in list
```

### Verify Lambda Deletion

```bash
# Try to get function (should fail)
aws lambda get-function --function-name process-text-document

# Should return: "Function not found"
```

---

## Step 5: Remove S3 Trigger Configuration

If S3 trigger was configured, remove the permission.

### Option A: Using AWS Console

The trigger is automatically deleted when the Lambda function is deleted.

### Option B: Using AWS CLI

```bash
# Check if notification configuration exists
aws s3api get-bucket-notification-configuration --bucket "$BUCKET_NAME" 2>/dev/null

# If bucket still exists and has notifications, remove them
aws s3api put-bucket-notification-configuration \
  --bucket "$BUCKET_NAME" \
  --notification-configuration '{}'
```

**Note:** This step is only needed if you deleted Lambda but kept the S3 bucket.

---

## Step 6: Delete IAM Role

Delete the Lambda execution role.

### Option A: Using AWS Console

1. Go to [IAM Console](https://console.aws.amazon.com/iam/)
2. Click "Roles" in left menu
3. Search for: `lambda-text-processor`
4. Select the role (checkbox)
5. Click "Delete"
6. Type role name to confirm
7. Click "Delete"

### Option B: Using AWS CLI

```bash
# Detach all policies from role
aws iam detach-role-policy \
  --role-name lambda-text-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam detach-role-policy \
  --role-name lambda-text-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam detach-role-policy \
  --role-name lambda-text-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

# Delete the role
aws iam delete-role --role-name lambda-text-processor

# Verify deletion
aws iam get-role --role-name lambda-text-processor

# Should return: "Role not found"
```

### Verify Role Deletion

```bash
# List roles (should not include lambda-text-processor)
aws iam list-roles | grep lambda-text-processor

# Should return nothing
```

---

## Step 7: Delete CloudWatch Logs

Lambda functions create CloudWatch log groups that persist after deletion.

### Option A: Using AWS Console

1. Go to [CloudWatch Console](https://console.aws.amazon.com/cloudwatch/)
2. Click "Logs" → "Log groups" in left menu
3. Search for: `/aws/lambda/process-text-document`
4. Select the log group (checkbox)
5. Click "Actions" → "Delete log group(s)"
6. Confirm deletion

### Option B: Using AWS CLI

```bash
# Delete Lambda log group
aws logs delete-log-group --log-group-name /aws/lambda/process-text-document

# Verify deletion
aws logs describe-log-groups | grep process-text-document

# Should return nothing
```

### Find and Delete All Related Logs

```bash
# List all log groups related to this project
aws logs describe-log-groups \
  --query 'logGroups[?contains(logGroupName, `text`) || contains(logGroupName, `Text`)].logGroupName' \
  --output text

# Delete each one
aws logs delete-log-group --log-group-name /aws/lambda/process-text-document
```

---

## Step 8: Delete Athena Resources (If Used)

If you configured Athena workspace for SQL queries.

### Delete Athena Query Results

```bash
# Athena query results are stored in S3
# If you specified a results location like s3://text-corpus-xxx/athena-results/

# Delete Athena results (if they exist)
aws s3 rm "s3://$BUCKET_NAME/athena-results/" --recursive

# Or if you created a separate bucket for Athena results
ATHENA_BUCKET="aws-athena-query-results-{account-id}-{region}"
aws s3 rm "s3://$ATHENA_BUCKET/" --recursive --include "*text-corpus*"
```

### Drop Athena Database and Tables

Using AWS Console:
1. Go to [Athena Console](https://console.aws.amazon.com/athena/)
2. In Query Editor, run:
   ```sql
   DROP TABLE IF EXISTS text_corpus_db.documents;
   DROP DATABASE IF EXISTS text_corpus_db;
   ```

Using AWS CLI:
```bash
# Drop table
aws athena start-query-execution \
  --query-string "DROP TABLE IF EXISTS text_corpus_db.documents" \
  --result-configuration "OutputLocation=s3://your-athena-results-bucket/"

# Drop database
aws athena start-query-execution \
  --query-string "DROP DATABASE IF EXISTS text_corpus_db" \
  --result-configuration "OutputLocation=s3://your-athena-results-bucket/"
```

---

## Step 9: Verify All Resources Deleted

Run these commands to ensure everything is cleaned up.

```bash
# Check S3 buckets
echo "Checking S3 buckets..."
aws s3 ls | grep text-corpus
# Should return nothing

# Check DynamoDB tables
echo "Checking DynamoDB tables..."
aws dynamodb list-tables | grep TextAnalysis
# Should return nothing

# Check Lambda functions
echo "Checking Lambda functions..."
aws lambda list-functions | grep process-text-document
# Should return nothing

# Check IAM roles
echo "Checking IAM roles..."
aws iam list-roles | grep lambda-text-processor
# Should return nothing

# Check CloudWatch log groups
echo "Checking CloudWatch logs..."
aws logs describe-log-groups | grep process-text-document
# Should return nothing

echo ""
echo "✓ Cleanup verification complete!"
```

---

## Step 10: Verify Billing

Ensure no ongoing charges.

### Using AWS Console

1. Go to [AWS Billing Console](https://console.aws.amazon.com/billing/)
2. Click "Bills" in left menu
3. Review current month charges
4. Look for charges from:
   - S3
   - Lambda
   - DynamoDB
   - CloudWatch
   - Athena

### Check Cost Explorer

1. Go to [Cost Explorer](https://console.aws.amazon.com/cost-management/home#/cost-explorer)
2. View costs for last 7 days
3. Filter by service:
   - S3: Should show $0 after deletion
   - Lambda: Should show $0 (or only past invocations)
   - DynamoDB: Should show $0 after deletion

### Set Up Billing Alert (Recommended)

To avoid unexpected charges in future:

1. Go to [Budgets](https://console.aws.amazon.com/billing/home#/budgets)
2. Create budget:
   - Budget type: Cost budget
   - Budget amount: $5/month (or your preference)
   - Alert threshold: 80% ($4)
   - Email notification: Your email

---

## Cleanup Checklist

Use this checklist to ensure nothing is missed:

- [ ] S3 bucket contents deleted
- [ ] S3 bucket deleted
- [ ] DynamoDB table deleted
- [ ] Lambda function deleted
- [ ] IAM role policies detached
- [ ] IAM role deleted
- [ ] CloudWatch log groups deleted
- [ ] Athena resources deleted (if used)
- [ ] S3 trigger configuration removed
- [ ] All resources verified as deleted
- [ ] Billing dashboard checked
- [ ] No ongoing charges confirmed

---

## Automated Cleanup Script

For convenience, here's a complete cleanup script:

```bash
#!/bin/bash
# cleanup_all.sh - Automated cleanup script

set -e

# Configuration
BUCKET_NAME="text-corpus-your-id"  # CHANGE THIS!
TABLE_NAME="TextAnalysis"
LAMBDA_FUNCTION="process-text-document"
IAM_ROLE="lambda-text-processor"

echo "Starting cleanup of AWS resources..."
echo "========================================="
echo ""

# 1. Empty and delete S3 bucket
echo "1. Deleting S3 bucket: $BUCKET_NAME"
aws s3 rm "s3://$BUCKET_NAME" --recursive 2>/dev/null || echo "  Bucket already empty or doesn't exist"
aws s3 rb "s3://$BUCKET_NAME" 2>/dev/null || echo "  Bucket already deleted"
echo "  ✓ S3 bucket deleted"
echo ""

# 2. Delete DynamoDB table
echo "2. Deleting DynamoDB table: $TABLE_NAME"
aws dynamodb delete-table --table-name "$TABLE_NAME" 2>/dev/null || echo "  Table already deleted"
echo "  ✓ DynamoDB table deleted"
echo ""

# 3. Delete Lambda function
echo "3. Deleting Lambda function: $LAMBDA_FUNCTION"
aws lambda delete-function --function-name "$LAMBDA_FUNCTION" 2>/dev/null || echo "  Function already deleted"
echo "  ✓ Lambda function deleted"
echo ""

# 4. Delete IAM role
echo "4. Deleting IAM role: $IAM_ROLE"
aws iam detach-role-policy --role-name "$IAM_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true
aws iam detach-role-policy --role-name "$IAM_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess 2>/dev/null || true
aws iam detach-role-policy --role-name "$IAM_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess 2>/dev/null || true
aws iam delete-role --role-name "$IAM_ROLE" 2>/dev/null || echo "  Role already deleted"
echo "  ✓ IAM role deleted"
echo ""

# 5. Delete CloudWatch logs
echo "5. Deleting CloudWatch log groups"
aws logs delete-log-group --log-group-name "/aws/lambda/$LAMBDA_FUNCTION" 2>/dev/null || echo "  Log group already deleted"
echo "  ✓ CloudWatch logs deleted"
echo ""

echo "========================================="
echo "Cleanup complete!"
echo ""
echo "Please verify in AWS Console:"
echo "  - S3: https://console.aws.amazon.com/s3/"
echo "  - DynamoDB: https://console.aws.amazon.com/dynamodb/"
echo "  - Lambda: https://console.aws.amazon.com/lambda/"
echo "  - Billing: https://console.aws.amazon.com/billing/"
echo ""
echo "All charges should stop within 24 hours."
```

To use:
```bash
# Download the script
chmod +x cleanup_all.sh

# Edit BUCKET_NAME to your actual bucket name
nano cleanup_all.sh

# Run cleanup
./cleanup_all.sh
```

---

## Common Issues

### "Bucket is not empty" error

**Solution:**
```bash
# Force delete all objects including versions
aws s3api list-object-versions --bucket "$BUCKET_NAME" \
  --query 'Versions[].{Key:Key,VersionId:VersionId}' \
  --output json | jq -r '.[] | "\(.Key) \(.VersionId)"' | \
  while read key version; do
    aws s3api delete-object --bucket "$BUCKET_NAME" --key "$key" --version-id "$version"
  done

# Then delete bucket
aws s3 rb "s3://$BUCKET_NAME" --force
```

### "Role has attached policies" error

**Solution:**
```bash
# List attached policies
aws iam list-attached-role-policies --role-name lambda-text-processor

# Detach each one
aws iam detach-role-policy --role-name lambda-text-processor \
  --policy-arn <policy-arn-from-list>

# Then delete role
aws iam delete-role --role-name lambda-text-processor
```

### "Table is being deleted" - Cannot verify

**Solution:**
Wait 1-2 minutes and check again:
```bash
aws dynamodb wait table-not-exists --table-name TextAnalysis
echo "Table successfully deleted"
```

---

## Final Verification

After 24 hours, check your AWS bill:

1. Go to [AWS Billing Dashboard](https://console.aws.amazon.com/billing/)
2. Check "Month-to-Date Spend"
3. Verify no charges from:
   - S3
   - Lambda
   - DynamoDB
   - CloudWatch

If you see any charges after cleanup, check for:
- Athena query results still in S3
- CloudWatch logs not deleted
- Other projects using same services

---

## Need Help?

If you encounter issues during cleanup:

1. Check AWS Console to see which resources still exist
2. Review CloudWatch logs for error messages
3. Contact AWS Support if resources won't delete
4. Post issue on GitHub: https://github.com/research-jumpstart/research-jumpstart/issues

---

## Summary

You have successfully cleaned up all AWS resources for the Historical Text Analysis project. Your AWS bill should return to normal (or $0 if not using other services) within 24 hours.

**Important:** Keep your local files (corpus, scripts, notebooks) for future reference. Only AWS cloud resources have been deleted.

---

**Last updated:** 2025-11-14
