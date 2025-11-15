# AWS Cleanup Guide for Learning Analytics Platform

This guide provides step-by-step instructions to delete all AWS resources created for this project and avoid ongoing costs.

**Important:** Follow these steps carefully to ensure complete cleanup and avoid unexpected charges.

**Estimated Time:** 10-15 minutes

---

## Pre-Cleanup Checklist

Before deleting resources, you may want to:

- [ ] Download any important analysis results or reports
- [ ] Export student analytics data from DynamoDB for records
- [ ] Save CloudWatch logs if needed for audit purposes
- [ ] Review AWS Cost Explorer to confirm expected costs

---

## Step 1: Delete S3 Objects and Bucket

S3 storage incurs costs based on data size and duration. Empty the bucket first, then delete it.

### Using AWS Management Console

1. Go to **S3** > **Buckets**
2. Click on your bucket: `learning-analytics-{your-user-id}`
3. Click **Empty**
   - Type `permanently delete` to confirm
   - Click **Empty**
   - Wait for deletion to complete
4. After bucket is empty, click **Delete**
   - Type the bucket name to confirm
   - Click **Delete bucket**

### Using AWS CLI

```bash
# Set your bucket name
export BUCKET_NAME="learning-analytics-your-user-id"

# Empty bucket (delete all objects)
aws s3 rm s3://${BUCKET_NAME} --recursive

# Delete bucket
aws s3 rb s3://${BUCKET_NAME}

# Verify deletion
aws s3 ls | grep learning-analytics
# Should return nothing
```

**Expected Cost Savings:** ~$0.023 per GB per month

---

## Step 2: Delete Lambda Function

Lambda functions incur costs only when invoked, but it's good practice to delete unused functions.

### Using AWS Management Console

1. Go to **Lambda** > **Functions**
2. Select: `analyze-student-performance`
3. Click **Actions** > **Delete**
4. Type `delete` to confirm
5. Click **Delete**

### Using AWS CLI

```bash
# Delete Lambda function
aws lambda delete-function --function-name analyze-student-performance

# Verify deletion
aws lambda list-functions | grep analyze-student-performance
# Should return nothing
```

**Expected Cost Savings:** No ongoing costs (pay-per-invocation only)

---

## Step 3: Delete DynamoDB Table

DynamoDB incurs costs for storage and read/write capacity.

### Using AWS Management Console

1. Go to **DynamoDB** > **Tables**
2. Select: `StudentAnalytics`
3. Click **Delete**
4. **Important:** Uncheck "Create a backup before deleting" (unless you need it)
5. Type `confirm` to confirm deletion
6. Click **Delete**

### Using AWS CLI

```bash
# Delete DynamoDB table
aws dynamodb delete-table --table-name StudentAnalytics

# Verify deletion
aws dynamodb list-tables | grep StudentAnalytics
# Should return nothing
```

**Expected Cost Savings:** ~$0.25 per GB per month + on-demand read/write costs

---

## Step 4: Delete IAM Role

IAM roles don't incur costs, but removing unused roles improves security hygiene.

### Using AWS Management Console

1. Go to **IAM** > **Roles**
2. Search for: `lambda-learning-analytics`
3. Select the role
4. Click **Delete**
5. Type the role name to confirm
6. Click **Delete**

### Using AWS CLI

```bash
# Detach policies first
aws iam detach-role-policy \
    --role-name lambda-learning-analytics \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam detach-role-policy \
    --role-name lambda-learning-analytics \
    --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

aws iam detach-role-policy \
    --role-name lambda-learning-analytics \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

aws iam detach-role-policy \
    --role-name lambda-learning-analytics \
    --policy-arn arn:aws:iam::aws:policy/AmazonAthenaFullAccess

# Delete role
aws iam delete-role --role-name lambda-learning-analytics

# Verify deletion
aws iam get-role --role-name lambda-learning-analytics
# Should return error: "NoSuchEntity"
```

**Expected Cost Savings:** Free (no cost)

---

## Step 5: Delete Athena Resources (Optional)

Athena workgroups and databases don't incur storage costs, but you can delete them for cleanup.

### Delete Athena Workgroup

```bash
# Delete workgroup (console or CLI)
aws athena delete-work-group \
    --work-group learning-analytics-queries \
    --recursive-delete-option

# Verify deletion
aws athena list-work-groups | grep learning-analytics-queries
```

### Delete Athena Database

```bash
# Delete database (this doesn't delete S3 data, just the catalog)
aws athena start-query-execution \
    --query-string "DROP DATABASE IF EXISTS learning_analytics CASCADE" \
    --result-configuration "OutputLocation=s3://${BUCKET_NAME}/athena-results/" \
    --region us-east-1

# Note: This requires the S3 bucket to still exist. Run before Step 1 if needed.
```

**Expected Cost Savings:** No ongoing costs (pay-per-query only)

---

## Step 6: Delete CloudWatch Logs

Lambda automatically creates CloudWatch log groups. These incur small storage costs.

### Using AWS Management Console

1. Go to **CloudWatch** > **Logs** > **Log groups**
2. Search for: `/aws/lambda/analyze-student-performance`
3. Select the log group
4. Click **Actions** > **Delete log group(s)**
5. Click **Delete**

### Using AWS CLI

```bash
# Delete Lambda log group
aws logs delete-log-group --log-group-name /aws/lambda/analyze-student-performance

# Verify deletion
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/analyze-student-performance
# Should return empty
```

**Expected Cost Savings:** ~$0.50 per GB per month

---

## Step 7: Verify Complete Cleanup

Run these commands to verify all resources are deleted:

```bash
# Check S3 buckets
echo "S3 Buckets:"
aws s3 ls | grep learning-analytics
# Should return nothing

# Check Lambda functions
echo "Lambda Functions:"
aws lambda list-functions --query 'Functions[?contains(FunctionName, `student-performance`)]'
# Should return empty array

# Check DynamoDB tables
echo "DynamoDB Tables:"
aws dynamodb list-tables | grep StudentAnalytics
# Should return nothing

# Check IAM roles
echo "IAM Roles:"
aws iam get-role --role-name lambda-learning-analytics 2>&1 | grep NoSuchEntity
# Should show "NoSuchEntity" error

# Check CloudWatch logs
echo "CloudWatch Logs:"
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/analyze-student-performance
# Should return empty
```

If all commands return empty results (or expected errors), cleanup is complete!

---

## Step 8: Verify Costs in AWS Cost Explorer

Wait 24-48 hours for final costs to appear:

1. Go to **AWS Cost Management** > **Cost Explorer**
2. Set date range to include your project period
3. Filter by service:
   - S3
   - Lambda
   - DynamoDB
   - CloudWatch
4. Verify costs are as expected (should be $6-11 total)
5. Check that no ongoing costs appear after cleanup

---

## Local Cleanup

Delete local generated data:

```bash
# Navigate to project directory
cd projects/education/learning-analytics-platform/tier-2

# Delete generated data
rm -rf generated_data/
rm -f *.png *.csv *.json upload_log.json

# Delete Lambda package
rm -rf lambda-package/
rm -f lambda-analytics.zip

# Delete Python virtual environment (if created)
rm -rf venv/

echo "✓ Local cleanup complete"
```

---

## Troubleshooting Cleanup Issues

### "Bucket not empty" error when deleting S3

**Solution:** The bucket must be completely empty before deletion.

```bash
# Force delete all objects including versions
aws s3api delete-objects \
    --bucket ${BUCKET_NAME} \
    --delete "$(aws s3api list-object-versions \
        --bucket ${BUCKET_NAME} \
        --query='{Objects: Versions[].{Key:Key,VersionId:VersionId}}')"

# Then delete bucket
aws s3 rb s3://${BUCKET_NAME}
```

### "Role has attached policies" error

**Solution:** Detach all policies before deleting role.

```bash
# List attached policies
aws iam list-attached-role-policies --role-name lambda-learning-analytics

# Detach each policy
aws iam detach-role-policy --role-name lambda-learning-analytics --policy-arn <policy-arn>

# Then delete role
aws iam delete-role --role-name lambda-learning-analytics
```

### DynamoDB table stuck in "DELETING" status

**Solution:** Wait 1-2 minutes. DynamoDB deletion can take time, especially for tables with data.

```bash
# Check status
aws dynamodb describe-table --table-name StudentAnalytics

# If stuck for >5 minutes, contact AWS support
```

### CloudWatch logs still showing costs

**Solution:** Ensure log group is deleted:

```bash
# Force delete with retention set to 0
aws logs put-retention-policy \
    --log-group-name /aws/lambda/analyze-student-performance \
    --retention-in-days 1

# Wait 24 hours, then delete
aws logs delete-log-group --log-group-name /aws/lambda/analyze-student-performance
```

---

## Cost Summary After Cleanup

After following this guide:

| Resource | Before Cleanup | After Cleanup |
|----------|----------------|---------------|
| S3 Storage | $0.023/GB/month | $0.00 |
| Lambda | $0.00 (idle) | $0.00 |
| DynamoDB | $0.25/GB/month | $0.00 |
| CloudWatch | $0.50/GB/month | $0.00 |
| **Total Monthly** | **~$1-2/month** | **$0.00** |

**One-time project cost:** $6-11 (for the initial run)

---

## Automation Script (Optional)

Create a cleanup script for quick deletion:

```bash
#!/bin/bash
# cleanup.sh - Automated AWS resource cleanup

set -e

BUCKET_NAME="learning-analytics-your-user-id"
TABLE_NAME="StudentAnalytics"
FUNCTION_NAME="analyze-student-performance"
ROLE_NAME="lambda-learning-analytics"

echo "Starting cleanup for Learning Analytics Platform..."

# Delete S3
echo "Deleting S3 bucket..."
aws s3 rm s3://${BUCKET_NAME} --recursive || true
aws s3 rb s3://${BUCKET_NAME} || true

# Delete Lambda
echo "Deleting Lambda function..."
aws lambda delete-function --function-name ${FUNCTION_NAME} || true

# Delete DynamoDB
echo "Deleting DynamoDB table..."
aws dynamodb delete-table --table-name ${TABLE_NAME} || true

# Delete CloudWatch logs
echo "Deleting CloudWatch logs..."
aws logs delete-log-group --log-group-name /aws/lambda/${FUNCTION_NAME} || true

# Delete IAM role
echo "Deleting IAM role..."
for policy_arn in $(aws iam list-attached-role-policies --role-name ${ROLE_NAME} --query 'AttachedPolicies[].PolicyArn' --output text); do
    aws iam detach-role-policy --role-name ${ROLE_NAME} --policy-arn ${policy_arn} || true
done
aws iam delete-role --role-name ${ROLE_NAME} || true

echo "✓ Cleanup complete! Verify in AWS Console."
echo "Wait 24-48 hours and check Cost Explorer for final costs."
```

Make executable and run:
```bash
chmod +x cleanup.sh
./cleanup.sh
```

---

## Final Verification Checklist

After cleanup, verify:

- [ ] S3 bucket deleted
- [ ] Lambda function deleted
- [ ] DynamoDB table deleted
- [ ] IAM role deleted
- [ ] CloudWatch logs deleted
- [ ] Athena resources deleted (optional)
- [ ] No ongoing costs in Cost Explorer
- [ ] Local files cleaned up

---

## Support

If you encounter issues during cleanup:

1. Check the troubleshooting section above
2. Review AWS documentation for specific services
3. Contact AWS support if resources are stuck
4. Monitor Cost Explorer for unexpected charges

Remember: You can always recreate resources by following `setup_guide.md` again!

---

**Cleanup Complete!**

Your AWS environment is now clean, and you won't incur ongoing costs.

For questions about the project, see `README.md` or `setup_guide.md`.
