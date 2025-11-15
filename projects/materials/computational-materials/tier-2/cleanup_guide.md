# Cleanup Guide - Delete AWS Resources

After completing the project, delete all AWS resources to stop incurring charges.

**Important:** Once deleted, resources cannot be recovered. Ensure you've backed up any results you want to keep.

---

## Quick Cleanup (Recommended)

```bash
# Set your bucket name
BUCKET_NAME="materials-data-xxxx"  # Replace with your bucket name

# Delete S3 bucket and all contents
aws s3 rm s3://$BUCKET_NAME --recursive
aws s3 rb s3://$BUCKET_NAME

# Delete DynamoDB table
aws dynamodb delete-table --table-name MaterialsProperties

# Delete Lambda function
aws lambda delete-function --function-name process-crystal-structure

# Delete IAM role policies
aws iam detach-role-policy \
  --role-name lambda-materials-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam detach-role-policy \
  --role-name lambda-materials-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam detach-role-policy \
  --role-name lambda-materials-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

# Delete IAM role
aws iam delete-role --role-name lambda-materials-processor

echo "All resources deleted!"
```

---

## Step-by-Step Deletion

If you prefer to delete resources individually or want to verify each step:

### Step 1: Back Up Results (Optional)

Save any results before deletion:

```bash
# Create backup directory
mkdir -p backups

# Export DynamoDB table to JSON
aws dynamodb scan \
  --table-name MaterialsProperties \
  --output json > backups/materials_properties.json

# Download all structures from S3
aws s3 cp s3://$BUCKET_NAME/structures/ \
  ./backups/structures/ --recursive

# Download logs
aws s3 cp s3://$BUCKET_NAME/logs/ \
  ./backups/logs/ --recursive

# Verify backup
ls -lah backups/
echo "Total structures backed up: $(ls backups/structures/ | wc -l)"
```

### Step 2: Delete DynamoDB Table

```bash
# Get table info before deletion
aws dynamodb describe-table --table-name MaterialsProperties

# Delete the table
aws dynamodb delete-table --table-name MaterialsProperties

# Wait for deletion to complete
aws dynamodb wait table-not-exists --table-name MaterialsProperties

# Verify deletion
aws dynamodb list-tables | grep MaterialsProperties
# Should return nothing
```

**Cost Impact:** DynamoDB charges stop immediately after deletion.

### Step 3: Delete S3 Bucket

S3 buckets must be completely empty before deletion.

```bash
# List bucket contents
aws s3 ls s3://$BUCKET_NAME --recursive

# Count objects
OBJECT_COUNT=$(aws s3 ls s3://$BUCKET_NAME --recursive | wc -l)
echo "Deleting $OBJECT_COUNT objects..."

# Delete all objects and versions
aws s3 rm s3://$BUCKET_NAME --recursive

# If bucket has versioning enabled, delete all versions
aws s3api delete-objects \
  --bucket $BUCKET_NAME \
  --delete "$(aws s3api list-object-versions \
    --bucket $BUCKET_NAME \
    --output json \
    --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}')"

# Delete bucket
aws s3 rb s3://$BUCKET_NAME

# Verify deletion
aws s3 ls | grep materials-data
# Should return nothing
```

**Cost Impact:** S3 storage charges stop immediately. Any data transfer charges already incurred remain.

### Step 4: Delete S3 Event Trigger

If you set up S3->Lambda triggers:

```bash
# Remove S3 event notification
aws s3api put-bucket-notification-configuration \
  --bucket $BUCKET_NAME \
  --notification-configuration '{}'

# Note: This may fail if bucket is already deleted (that's OK)
```

### Step 5: Delete Lambda Function

```bash
# Get function info before deletion
aws lambda get-function --function-name process-crystal-structure

# Delete the Lambda function
aws lambda delete-function --function-name process-crystal-structure

# Verify deletion
aws lambda get-function --function-name process-crystal-structure 2>&1
# Should return: "Function not found"
```

**Cost Impact:** Lambda charges stop immediately.

### Step 6: Delete Lambda Permissions

```bash
# Remove Lambda S3 invocation permission
aws lambda remove-permission \
  --function-name process-crystal-structure \
  --statement-id "AllowS3Invoke" 2>&1
# May fail if function already deleted (that's OK)
```

### Step 7: Delete IAM Role

IAM roles must have all policies detached before deletion.

```bash
# List attached policies
aws iam list-attached-role-policies \
  --role-name lambda-materials-processor

# Detach managed policies
aws iam detach-role-policy \
  --role-name lambda-materials-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam detach-role-policy \
  --role-name lambda-materials-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam detach-role-policy \
  --role-name lambda-materials-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

# List inline policies (if any)
aws iam list-role-policies --role-name lambda-materials-processor

# Delete inline policies
for policy_name in $(aws iam list-role-policies \
  --role-name lambda-materials-processor \
  --query 'PolicyNames[]' --output text); do
  aws iam delete-role-policy \
    --role-name lambda-materials-processor \
    --policy-name $policy_name
done

# Finally, delete the role
aws iam delete-role --role-name lambda-materials-processor

# Verify deletion
aws iam get-role --role-name lambda-materials-processor 2>&1
# Should return: "NoSuchEntity"
```

**Cost Impact:** IAM roles are free, but removing them improves security.

### Step 8: Delete CloudWatch Logs (Optional)

CloudWatch logs persist even after Lambda deletion and incur small storage costs.

```bash
# Check log group size
aws logs describe-log-groups \
  --log-group-name-prefix /aws/lambda/process-crystal-structure

# Delete log group
aws logs delete-log-group \
  --log-group-name /aws/lambda/process-crystal-structure

# Verify deletion
aws logs describe-log-groups | grep process-crystal-structure
# Should return nothing
```

**Cost Impact:** ~$0.50/GB/month. Small log groups cost less than $0.10/month.

---

## Verify All Resources Deleted

Check that everything is cleaned up:

```bash
# Summary check
echo "=== S3 Buckets ==="
aws s3 ls | grep materials-data || echo "✓ No materials-data buckets"

echo ""
echo "=== DynamoDB Tables ==="
aws dynamodb list-tables | grep MaterialsProperties || echo "✓ No MaterialsProperties table"

echo ""
echo "=== Lambda Functions ==="
aws lambda list-functions | grep process-crystal-structure || echo "✓ No Lambda function"

echo ""
echo "=== IAM Roles ==="
aws iam list-roles | grep lambda-materials-processor || echo "✓ No IAM role"

echo ""
echo "=== CloudWatch Logs ==="
aws logs describe-log-groups | grep process-crystal-structure || echo "✓ No log groups"

echo ""
echo "=== Cleanup Complete ==="
```

---

## Cost Verification

After cleanup, verify no ongoing charges:

### Check Final Costs

```bash
# Get cost for last 7 days
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '7 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics "UnblendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE \
  --output json | python -m json.tool
```

### Expected Costs

After running this project once:
- **Total project cost:** $2-5
- **S3 storage (1 week):** ~$0.10
- **Lambda invocations:** ~$0.50
- **DynamoDB:** ~$0.50
- **Data transfer:** ~$0.20

### Verify Zero Future Charges

1. Go to [AWS Billing Console](https://console.aws.amazon.com/billing/)
2. Click "Bills"
3. Check current month charges
4. Verify no active resources in regions used

---

## Automated Cleanup Script

For convenience, you can use the automated cleanup script:

```bash
# Run cleanup script
python scripts/cleanup.py --bucket materials-data-xxxx

# Or with confirmation prompts
python scripts/cleanup.py --bucket materials-data-xxxx --interactive
```

**Cleanup script features:**
- Backs up DynamoDB data to JSON
- Downloads all S3 objects to local backup
- Deletes all AWS resources
- Verifies deletion
- Reports final costs

---

## Troubleshooting Cleanup

### Problem: "Bucket not empty" error

**Solution:**
```bash
# Delete all object versions (if versioning enabled)
aws s3api list-object-versions \
  --bucket $BUCKET_NAME \
  --output json | \
  jq -r '.Versions[],.DeleteMarkers[] | "--key \(.Key) --version-id \(.VersionId)"' | \
  xargs -n 4 aws s3api delete-object --bucket $BUCKET_NAME

# Then delete bucket
aws s3 rb s3://$BUCKET_NAME
```

### Problem: "Role has attached policies" error

**Solution:**
```bash
# Force detach all policies
for policy in $(aws iam list-attached-role-policies \
  --role-name lambda-materials-processor \
  --query 'AttachedPolicies[].PolicyArn' \
  --output text); do
  aws iam detach-role-policy \
    --role-name lambda-materials-processor \
    --policy-arn $policy
done

# Delete role
aws iam delete-role --role-name lambda-materials-processor
```

### Problem: "Table is being deleted" status

**Solution:**
```bash
# Wait for deletion to complete
aws dynamodb wait table-not-exists --table-name MaterialsProperties

# This can take 1-2 minutes
```

### Problem: Can't find bucket name

**Solution:**
```bash
# List all your S3 buckets
aws s3 ls | grep materials

# Get bucket name from .env file
cat .env | grep BUCKET_NAME

# Or check CloudFormation stacks (if any)
aws cloudformation list-stacks --stack-status-filter CREATE_COMPLETE
```

---

## Partial Cleanup

If you want to keep some resources but reduce costs:

### Keep S3, Delete Lambda and DynamoDB

```bash
# Delete compute resources only
aws lambda delete-function --function-name process-crystal-structure
aws dynamodb delete-table --table-name MaterialsProperties

# Keep S3 bucket with data
# Cost: ~$0.023/GB/month
```

### Keep DynamoDB, Delete S3 and Lambda

```bash
# Delete storage and compute
aws s3 rm s3://$BUCKET_NAME --recursive
aws s3 rb s3://$BUCKET_NAME
aws lambda delete-function --function-name process-crystal-structure

# Keep DynamoDB with results
# Cost: ~$0.25/GB/month (on-demand)
```

---

## Re-Running the Project

If you want to run the project again later:

1. All resources can be recreated using `setup_guide.md`
2. If you kept backups, restore them:

```bash
# Restore DynamoDB from backup
python scripts/restore_dynamodb.py --file backups/materials_properties.json

# Restore S3 structures
aws s3 cp backups/structures/ s3://$BUCKET_NAME/structures/ --recursive
```

---

## Final Checklist

Before closing your terminal, verify:

- [ ] S3 bucket deleted (no storage charges)
- [ ] DynamoDB table deleted (no capacity charges)
- [ ] Lambda function deleted (no invocation charges)
- [ ] IAM role deleted (security best practice)
- [ ] CloudWatch logs deleted (minimal storage charges)
- [ ] Local backups created (if needed)
- [ ] Final costs verified in AWS Console
- [ ] Billing alerts still active (for future projects)

---

## Getting Help

If you encounter issues during cleanup:

1. **Check AWS Console:** Visual verification of resources
2. **Review CloudWatch Logs:** Error messages and traces
3. **AWS Support:** Free tier includes basic support
4. **GitHub Issues:** Project-specific help

---

## Cost After Cleanup

**Expected final cost:** $2-5 (one-time)

**Ongoing costs after cleanup:** $0.00

**Future charges:** None (all resources deleted)

---

**Cleanup complete!** Your AWS account is now clean and no longer incurring charges from this project.

For questions or issues, see [README.md](README.md) or open a GitHub issue.

---

**Last updated:** 2025-01-14 | Research Jumpstart v1.0.0
