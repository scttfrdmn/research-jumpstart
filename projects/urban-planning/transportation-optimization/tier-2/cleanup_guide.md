# AWS Cleanup Guide - Transportation Flow Analysis

This guide walks you through deleting all AWS resources created for the Transportation Optimization Tier 2 project to avoid ongoing charges.

**Estimated time:** 10-15 minutes

## Important Notes

- **Complete cleanup is crucial** to avoid unexpected charges
- Resources are listed in the order they should be deleted
- Some resources cannot be deleted until dependent resources are removed
- Double-check resource names before deletion
- Save any data you want to keep before cleanup

---

## Quick Cleanup Checklist

- [ ] Delete S3 bucket contents
- [ ] Delete S3 bucket
- [ ] Delete DynamoDB table
- [ ] Delete Lambda function
- [ ] Remove S3 event notification
- [ ] Delete IAM role and policies
- [ ] (Optional) Delete Athena database and tables
- [ ] Verify all resources are deleted
- [ ] Check final AWS bill

---

## Step 1: Delete S3 Bucket Contents

S3 buckets must be empty before they can be deleted.

### Using AWS Console

1. Go to **S3** > **Buckets**
2. Click on your bucket: `transportation-data-{your-user-id}`
3. Select all objects (check the box at the top)
4. Click **Delete**
5. Type `permanently delete` to confirm
6. Click **Delete objects**

### Using AWS CLI

```bash
# Set your bucket name
BUCKET_NAME="transportation-data-{your-user-id}"

# Delete all objects (including versioned objects if versioning was enabled)
aws s3 rm s3://${BUCKET_NAME} --recursive

# If versioning was enabled, also delete version markers
aws s3api delete-objects \
    --bucket ${BUCKET_NAME} \
    --delete "$(aws s3api list-object-versions \
        --bucket ${BUCKET_NAME} \
        --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' \
        --output json)"

# Verify bucket is empty
aws s3 ls s3://${BUCKET_NAME}/ --recursive
```

**Expected result:** No objects listed

---

## Step 2: Delete S3 Bucket

Now that the bucket is empty, delete it.

### Using AWS Console

1. In **S3** > **Buckets**, select your bucket
2. Click **Delete**
3. Type the bucket name to confirm
4. Click **Delete bucket**

### Using AWS CLI

```bash
# Delete the bucket
aws s3 rb s3://${BUCKET_NAME}

# Verify deletion
aws s3 ls | grep transportation-data
```

**Expected result:** Bucket not found in listing

---

## Step 3: Delete DynamoDB Table

Delete the DynamoDB table that stores traffic analysis results.

### Using AWS Console

1. Go to **DynamoDB** > **Tables**
2. Select the table: `TrafficAnalysis`
3. Click **Delete**
4. Confirm by typing `delete` in the confirmation box
5. Uncheck "Create a backup" (unless you want to keep a backup)
6. Click **Delete table**

### Using AWS CLI

```bash
# Delete the table
aws dynamodb delete-table --table-name TrafficAnalysis

# Wait for deletion to complete (optional)
aws dynamodb wait table-not-exists --table-name TrafficAnalysis

# Verify deletion
aws dynamodb list-tables
```

**Expected result:** `TrafficAnalysis` not in table list

**Note:** DynamoDB tables can take 1-2 minutes to fully delete.

---

## Step 4: Delete Lambda Function

Delete the Lambda function that processes traffic data.

### Using AWS Console

1. Go to **Lambda** > **Functions**
2. Select the function: `analyze-traffic-flow`
3. Click **Actions** > **Delete**
4. Type `delete` to confirm
5. Click **Delete**

### Using AWS CLI

```bash
# Delete the Lambda function
aws lambda delete-function --function-name analyze-traffic-flow

# Verify deletion
aws lambda list-functions | grep analyze-traffic-flow
```

**Expected result:** Function not found in listing

---

## Step 5: Delete IAM Role

Delete the IAM role used by Lambda.

### Using AWS Console

1. Go to **IAM** > **Roles**
2. Search for: `lambda-traffic-processor`
3. Select the role
4. Click **Delete**
5. Type the role name to confirm
6. Click **Delete**

### Using AWS CLI

```bash
# Detach all policies from the role
aws iam detach-role-policy \
    --role-name lambda-traffic-processor \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam detach-role-policy \
    --role-name lambda-traffic-processor \
    --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

aws iam detach-role-policy \
    --role-name lambda-traffic-processor \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

# Delete the role
aws iam delete-role --role-name lambda-traffic-processor

# Verify deletion
aws iam list-roles | grep lambda-traffic-processor
```

**Expected result:** Role not found in listing

---

## Step 6: Delete Athena Resources (Optional)

If you created Athena database and tables, delete them.

### Using Athena Console

1. Go to **Athena** > **Query editor**
2. Run these queries:

```sql
-- Drop the table
DROP TABLE IF EXISTS traffic_db.traffic_data;

-- Drop the database
DROP DATABASE IF EXISTS traffic_db;
```

### Using AWS CLI

```bash
# Delete the table
aws athena start-query-execution \
    --query-string "DROP TABLE IF EXISTS traffic_db.traffic_data" \
    --result-configuration "OutputLocation=s3://${BUCKET_NAME}/athena-results/"

# Delete the database
aws athena start-query-execution \
    --query-string "DROP DATABASE IF EXISTS traffic_db" \
    --result-configuration "OutputLocation=s3://${BUCKET_NAME}/athena-results/"
```

**Note:** Athena queries themselves are free to delete, but query results stored in S3 were deleted in Step 1.

---

## Step 7: Delete CloudWatch Logs (Optional)

Lambda automatically creates CloudWatch log groups. Delete them to avoid minimal storage charges.

### Using AWS Console

1. Go to **CloudWatch** > **Log groups**
2. Search for: `/aws/lambda/analyze-traffic-flow`
3. Select the log group
4. Click **Actions** > **Delete log group(s)**
5. Confirm deletion

### Using AWS CLI

```bash
# Delete Lambda log group
aws logs delete-log-group --log-group-name /aws/lambda/analyze-traffic-flow

# Verify deletion
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/analyze-traffic-flow
```

**Expected result:** Log group not found

---

## Step 8: Verify Complete Cleanup

Run these verification commands to ensure all resources are deleted:

```bash
# Check S3 buckets
echo "=== S3 Buckets ==="
aws s3 ls | grep transportation-data
echo "Expected: (empty)"

# Check DynamoDB tables
echo -e "\n=== DynamoDB Tables ==="
aws dynamodb list-tables | grep TrafficAnalysis
echo "Expected: (empty)"

# Check Lambda functions
echo -e "\n=== Lambda Functions ==="
aws lambda list-functions | grep analyze-traffic-flow
echo "Expected: (empty)"

# Check IAM roles
echo -e "\n=== IAM Roles ==="
aws iam list-roles | grep lambda-traffic-processor
echo "Expected: (empty)"

# Check CloudWatch log groups
echo -e "\n=== CloudWatch Log Groups ==="
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/analyze-traffic-flow
echo "Expected: (empty)"

echo -e "\n=== Cleanup Verification Complete ==="
```

---

## Step 9: Check AWS Billing

Verify that resources are no longer incurring charges:

### Using AWS Console

1. Go to **Billing** > **Bills**
2. Select current month
3. Expand **Service charges**
4. Check for charges from:
   - Amazon S3
   - Amazon DynamoDB
   - AWS Lambda
   - Amazon CloudWatch

### Using AWS CLI

```bash
# Get cost for current month
aws ce get-cost-and-usage \
    --time-period Start=$(date +%Y-%m-01),End=$(date +%Y-%m-%d) \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=DIMENSION,Key=SERVICE
```

**Expected:** Charges should stop accumulating after cleanup

---

## Common Cleanup Issues

### "Bucket not empty" error

**Problem:** S3 bucket still contains objects

**Solution:**
```bash
# Force delete all objects including versions
aws s3api delete-objects \
    --bucket ${BUCKET_NAME} \
    --delete "$(aws s3api list-object-versions \
        --bucket ${BUCKET_NAME} \
        --output=json \
        --query='{Objects: Versions[].{Key:Key,VersionId:VersionId}}')"

# Then delete bucket
aws s3 rb s3://${BUCKET_NAME}
```

### "Role has attached policies" error

**Problem:** IAM role still has attached policies

**Solution:**
```bash
# List all attached policies
aws iam list-attached-role-policies --role-name lambda-traffic-processor

# Detach each policy
aws iam detach-role-policy \
    --role-name lambda-traffic-processor \
    --policy-arn <POLICY_ARN>

# Then delete role
aws iam delete-role --role-name lambda-traffic-processor
```

### Lambda function still exists

**Problem:** Lambda function not fully deleted

**Solution:**
```bash
# Check function state
aws lambda get-function --function-name analyze-traffic-flow

# If it exists, force delete
aws lambda delete-function --function-name analyze-traffic-flow
```

### DynamoDB table in "DELETING" state

**Problem:** Table deletion in progress

**Solution:**
```bash
# Wait for deletion to complete
aws dynamodb wait table-not-exists --table-name TrafficAnalysis

# This may take 1-2 minutes
```

---

## Cost After Cleanup

After complete cleanup, you should see:

- **S3 charges**: $0 (bucket deleted)
- **DynamoDB charges**: $0 (table deleted)
- **Lambda charges**: $0 (function deleted)
- **Data transfer**: $0 (no ongoing transfers)

**Note:** Final charges may appear for the period before deletion. These are normal and expected.

---

## Automated Cleanup Script

For convenience, you can use this automated cleanup script:

```bash
#!/bin/bash
# cleanup_all.sh - Automated cleanup script

set -e

BUCKET_NAME="transportation-data-{your-user-id}"
TABLE_NAME="TrafficAnalysis"
FUNCTION_NAME="analyze-traffic-flow"
ROLE_NAME="lambda-traffic-processor"

echo "Starting AWS cleanup..."

# 1. Delete S3 bucket contents and bucket
echo "Deleting S3 bucket..."
aws s3 rm s3://${BUCKET_NAME} --recursive
aws s3 rb s3://${BUCKET_NAME}

# 2. Delete DynamoDB table
echo "Deleting DynamoDB table..."
aws dynamodb delete-table --table-name ${TABLE_NAME}

# 3. Delete Lambda function
echo "Deleting Lambda function..."
aws lambda delete-function --function-name ${FUNCTION_NAME}

# 4. Delete IAM role
echo "Deleting IAM role..."
aws iam detach-role-policy --role-name ${ROLE_NAME} \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam detach-role-policy --role-name ${ROLE_NAME} \
    --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess
aws iam detach-role-policy --role-name ${ROLE_NAME} \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
aws iam delete-role --role-name ${ROLE_NAME}

# 5. Delete CloudWatch logs
echo "Deleting CloudWatch logs..."
aws logs delete-log-group --log-group-name /aws/lambda/${FUNCTION_NAME}

echo "Cleanup complete!"
```

Save as `cleanup_all.sh`, make executable with `chmod +x cleanup_all.sh`, and run with `./cleanup_all.sh`.

---

## Backup Before Cleanup (Optional)

If you want to keep your data before cleanup:

### Backup DynamoDB Table

```bash
# Export table to S3
aws dynamodb export-table-to-point-in-time \
    --table-arn arn:aws:dynamodb:REGION:ACCOUNT:table/TrafficAnalysis \
    --s3-bucket transportation-data-backup \
    --s3-prefix dynamodb-export/
```

### Backup S3 Data

```bash
# Download all data to local machine
aws s3 sync s3://transportation-data-{your-user-id}/ ./backup/
```

---

## Final Verification Checklist

After cleanup, verify:

- [ ] No S3 buckets with "transportation-data" prefix
- [ ] No DynamoDB tables named "TrafficAnalysis"
- [ ] No Lambda functions named "analyze-traffic-flow"
- [ ] No IAM roles named "lambda-traffic-processor"
- [ ] No CloudWatch log groups for Lambda function
- [ ] AWS Cost Explorer shows $0 for relevant services
- [ ] Billing dashboard shows no ongoing charges

---

## Getting Help

If you encounter issues during cleanup:

1. Check AWS documentation for service-specific deletion procedures
2. Verify you have proper IAM permissions for deletion
3. Check AWS Service Health Dashboard for service issues
4. Contact AWS Support if resources cannot be deleted

---

## Cost Estimate After Cleanup

**Immediate:** $0/month
**Potential residual charges:** $0.01-0.10 for final hour of usage

---

**Cleanup complete!** All AWS resources have been deleted and charges should stop.

For questions, see the main [README.md](README.md) or open an issue on GitHub.
