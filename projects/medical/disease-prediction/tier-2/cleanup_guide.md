# AWS Cleanup Guide - Medical Image Processing Pipeline

This guide explains how to delete all AWS resources created for the Medical Image Processing Tier 2 project.

**Important:** Incomplete cleanup can result in ongoing AWS charges. Follow these steps carefully to avoid surprise billing.

---

## Overview

Resources to delete:
1. S3 bucket and all contents
2. DynamoDB table
3. Lambda function
4. IAM role
5. CloudWatch logs
6. S3 event notifications

**Total cleanup time:** 10-15 minutes

---

## Step 1: Delete S3 Bucket

### Important: Empty Bucket First

S3 buckets must be empty before deletion.

#### Using AWS Management Console

1. Go to **S3** > **Buckets**
2. Click on your bucket: `medical-images-{your-user-id}`
3. Click **Empty** button at the top
4. Type `permanently delete` in the confirmation box
5. Click **Empty**
6. Wait for completion (shows "Empty bucket succeeded")
7. Go back to bucket list
8. Select your bucket checkbox
9. Click **Delete** button
10. Type bucket name to confirm
11. Click **Delete Bucket**

#### Using AWS CLI

```bash
# Empty the bucket (delete all objects)
aws s3 rm s3://medical-images-{your-user-id} --recursive

# Delete the bucket
aws s3 rb s3://medical-images-{your-user-id}

# Verify deletion
aws s3 ls | grep medical-images
# Should not appear in list
```

**Verification:** Bucket should no longer appear in `aws s3 ls` output.

---

## Step 2: Delete DynamoDB Table

### Using AWS Management Console

1. Go to **DynamoDB** > **Tables**
2. Select `medical-predictions` table
3. Click **Delete** button
4. Check **Delete this table and all its data**
5. Type `delete` to confirm
6. Click **Delete Table**
7. Wait for table deletion (shows "Deleting..." then disappears)

### Using AWS CLI

```bash
aws dynamodb delete-table --table-name medical-predictions --region us-east-1

# Verify deletion
aws dynamodb list-tables --region us-east-1
# medical-predictions should not appear
```

**Verification:** Table should no longer appear in DynamoDB console.

---

## Step 3: Delete Lambda Function

### Using AWS Management Console

1. Go to **Lambda** > **Functions**
2. Select `process-medical-images` function
3. Click **Delete** button at the top right
4. Type the function name to confirm
5. Click **Delete**

### Using AWS CLI

```bash
aws lambda delete-function --function-name process-medical-images

# Verify deletion
aws lambda list-functions --region us-east-1
# process-medical-images should not appear
```

**Verification:** Function should no longer appear in Lambda console.

---

## Step 4: Delete IAM Role and Permissions

### Remove Attached Policies First

#### Using AWS Management Console

1. Go to **IAM** > **Roles**
2. Search for `lambda-medical-processor`
3. Click on the role
4. Under **Permissions**, find each attached policy:
   - AmazonS3FullAccess
   - AmazonDynamoDBFullAccess
   - CloudWatchLogsFullAccess
5. Click **X** next to each policy to detach
6. Go back to Roles list
7. Select the role checkbox
8. Click **Delete** button
9. Click **Delete role**

#### Using AWS CLI

```bash
# Detach policies
aws iam detach-role-policy \
    --role-name lambda-medical-processor \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam detach-role-policy \
    --role-name lambda-medical-processor \
    --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

aws iam detach-role-policy \
    --role-name lambda-medical-processor \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

# Delete the role
aws iam delete-role --role-name lambda-medical-processor

# Verify deletion
aws iam list-roles | grep lambda-medical-processor
# Should not appear
```

**Verification:** Role should no longer appear in IAM console.

---

## Step 5: Delete CloudWatch Logs

Lambda creates log groups automatically. Clean them up to prevent ongoing costs.

### Using AWS Management Console

1. Go to **CloudWatch** > **Log Groups**
2. Search for `/aws/lambda/process-medical-images`
3. Click on the log group
4. Click **Delete log group** button
5. Click **Delete**

### Using AWS CLI

```bash
# List log groups
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/process-medical-images"

# Delete log group
aws logs delete-log-group --log-group-name "/aws/lambda/process-medical-images"

# Verify deletion
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/process-medical-images"
# Should return empty list
```

**Verification:** Log group should no longer appear in CloudWatch.

---

## Step 6: Verify All Resources Deleted

Run this verification script to ensure everything is cleaned up:

```bash
#!/bin/bash

echo "Checking for remaining AWS resources..."
echo ""

# Check S3
echo "S3 Buckets:"
aws s3 ls | grep medical-images && echo "✗ S3 bucket still exists" || echo "✓ S3 bucket deleted"

# Check DynamoDB
echo ""
echo "DynamoDB Tables:"
aws dynamodb list-tables --region us-east-1 --query 'TableNames[*]' | grep medical-predictions && echo "✗ DynamoDB table still exists" || echo "✓ DynamoDB table deleted"

# Check Lambda
echo ""
echo "Lambda Functions:"
aws lambda list-functions --region us-east-1 --query 'Functions[*].FunctionName' | grep process-medical-images && echo "✗ Lambda function still exists" || echo "✓ Lambda function deleted"

# Check IAM Role
echo ""
echo "IAM Roles:"
aws iam list-roles --query 'Roles[*].RoleName' | grep lambda-medical-processor && echo "✗ IAM role still exists" || echo "✓ IAM role deleted"

echo ""
echo "Cleanup verification complete!"
```

Save as `verify_cleanup.sh` and run:
```bash
bash verify_cleanup.sh
```

---

## Step 7: Delete Local Files (Optional)

If you want to clean up your local environment:

```bash
# Remove virtual environment
rm -rf venv/

# Remove temporary files
rm -f lambda_function.zip
rm -f test-event.json
rm -f trust-policy.json
rm -f s3-event-config.json
rm -f .env

# Remove local data
rm -rf sample_data/

# Note: Keep project files if you want to reference them later
```

---

## Common Cleanup Issues

### "The bucket you tried to delete is not empty"
- Solution: Empty the bucket first with `aws s3 rm s3://bucket-name --recursive`
- Then try deleting the bucket

### "Cannot delete role while it has attached policies"
- Solution: Detach all policies first with `detach-role-policy` commands
- Then delete the role

### "Access Denied" when deleting resources
- Solution: Verify your IAM user has appropriate permissions
- Check with `aws sts get-caller-identity`
- May need administrator to grant permissions

### Lambda still appears after deletion
- Sometimes takes a few minutes to update console
- Try refreshing browser or checking with CLI

### S3 bucket deletion stuck
- Verify all versions of all objects are deleted
- Check for lifecycle policies that might block deletion
- Try deleting with AWS CLI instead of console

---

## Cost Verification

### Check Remaining Charges

After cleanup, verify no charges are accruing:

1. Go to **Billing** > **Cost Explorer**
2. Check last 7 days of charges
3. Filter by Medical Image resources
4. Verify costs have stopped

```bash
# Check with AWS CLI
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-08 \
    --granularity DAILY \
    --metrics BlendedCost \
    --group-by Type=DIMENSION,Key=SERVICE \
    --filter file://cost-filter.json
```

---

## Partial Cleanup

If you want to keep some resources for future use:

### Keep S3, Delete Others
- Useful if you want to maintain a dataset
- Still delete DynamoDB table, Lambda, and IAM role

### Keep Lambda, Delete Data
- Useful if you want to rerun with new data
- Delete S3 objects but keep bucket
- Keep IAM role and Lambda function

### Example: Delete DynamoDB Only
```bash
# Just delete the table
aws dynamodb delete-table --table-name medical-predictions

# Query remaining resources
aws s3 ls | grep medical-images
aws lambda list-functions | grep process-medical-images
```

---

## Archiving Results Before Cleanup

If you want to keep analysis results:

```bash
# Download processed images
aws s3 sync s3://medical-images-{your-user-id}/processed-images/ ./local-backup/processed-images/

# Export DynamoDB data
aws dynamodb scan --table-name medical-predictions --output json > medical-predictions-backup.json

# Export notebook results
# (Jupyter notebook automatically saves)
```

---

## Account-Wide Cleanup

To find all Tier 2 project resources across your account:

```bash
# List all S3 buckets
aws s3 ls

# List all DynamoDB tables
aws dynamodb list-tables --region us-east-1

# List all Lambda functions
aws lambda list-functions --region us-east-1

# List all IAM roles
aws iam list-roles | grep -E 'lambda-|medical-'

# List all CloudWatch log groups
aws logs describe-log-groups | grep medical
```

---

## Cleanup Checklist

Use this checklist to ensure complete cleanup:

- [ ] S3 bucket emptied and deleted
- [ ] DynamoDB table deleted
- [ ] Lambda function deleted
- [ ] IAM role policies detached
- [ ] IAM role deleted
- [ ] CloudWatch log groups deleted
- [ ] Verified with `verify_cleanup.sh` script
- [ ] Billing shows no charges from resources
- [ ] Local .env file deleted
- [ ] Local test files removed (optional)

---

## What NOT to Delete

Do NOT delete these resources as they may be shared:

- [ ] Default VPC and subnets
- [ ] Root AWS account user
- [ ] IAM policies you use elsewhere
- [ ] CloudWatch dashboards (unless created for this project)
- [ ] AWS Config settings

---

## Troubleshooting

### Stuck in Cleanup?

If a resource won't delete:

1. Check AWS Status Page (status.aws.amazon.com)
2. Wait 5 minutes and retry
3. Try with AWS CLI instead of console
4. Check CloudTrail for error details
5. Contact AWS Support if issue persists

### Still Being Charged After Cleanup?

1. Run verification script to find remaining resources
2. Check for:
   - Partial S3 object deletion
   - DynamoDB backups
   - Lambda Reserved Concurrency settings
   - VPC endpoints
3. Use **Cost Explorer** to identify offending service

---

## Contact AWS Support

If you need help with cleanup:

1. Log in to AWS Console
2. Go to **Support** > **Create case**
3. Select **Technical support**
4. Describe the issue and resources that won't delete
5. AWS will help identify and remove resources

---

## Summary

Complete cleanup should:
- [ ] Delete all project resources
- [ ] Take 10-15 minutes
- [ ] Result in zero charges from this project
- [ ] Leave your AWS account clean

For questions during cleanup, refer to the troubleshooting section above or consult AWS documentation.

After cleanup is complete, you can start a new Tier 2 project or move on to Tier 3!
