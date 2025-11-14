# Cleanup Guide - Delete AWS Resources

**Important:** Run this guide immediately after completing your analysis to avoid unexpected charges.

**Total Cleanup Time: 10 minutes**

## Cost Warning

- **Each S3 bucket:** ~$0.023/GB/month for storage
- **Each DynamoDB table:** ~$1.25/GB/month (on-demand)
- **Lambda functions:** Minimal cost but should be deleted
- **IAM roles:** Free but good to clean up

**If you leave resources running for 30 days, you'll pay $10-15 again!**

## Quick Cleanup (5 minutes)

If you want to delete everything immediately:

```bash
# Source configuration
source aws_config.sh

# Run all cleanup steps
./cleanup_all.sh
```

## Step-by-Step Cleanup

### Step 1: Delete S3 Objects (2 minutes)

**WARNING:** This permanently deletes all files in your S3 buckets.

```bash
source aws_config.sh

# List what you're about to delete
echo "Files in input bucket:"
aws s3 ls s3://${BUCKET_INPUT} --recursive

echo ""
echo "Files in results bucket:"
aws s3 ls s3://${BUCKET_RESULTS} --recursive

# Delete all versions and objects
echo "Deleting S3 objects..."
aws s3 rm s3://${BUCKET_INPUT} --recursive
aws s3 rm s3://${BUCKET_RESULTS} --recursive

echo "✓ S3 objects deleted"
```

### Step 2: Delete S3 Buckets (1 minute)

```bash
source aws_config.sh

# Remove bucket versioning
aws s3api put-bucket-versioning \
  --bucket ${BUCKET_INPUT} \
  --versioning-configuration Status=Suspended

aws s3api put-bucket-versioning \
  --bucket ${BUCKET_RESULTS} \
  --versioning-configuration Status=Suspended

# Delete buckets
aws s3 rb s3://${BUCKET_INPUT}
aws s3 rb s3://${BUCKET_RESULTS}

echo "✓ S3 buckets deleted"
```

### Step 3: Delete DynamoDB Table (2 minutes)

```bash
source aws_config.sh

# Delete table
aws dynamodb delete-table \
  --table-name ${TABLE_NAME} \
  --region us-east-1

# Wait for deletion
echo "Waiting for DynamoDB table deletion..."
sleep 30

echo "✓ DynamoDB table deleted"
```

### Step 4: Delete Lambda Function (1 minute)

```bash
source aws_config.sh

# Delete Lambda function
aws lambda delete-function \
  --function-name ${LAMBDA_FUNCTION} \
  --region us-east-1

echo "✓ Lambda function deleted"
```

### Step 5: Delete IAM Role and Policy (2 minutes)

```bash
source aws_config.sh

# Delete inline policy first
aws iam delete-role-policy \
  --role-name ${ROLE_NAME} \
  --policy-name variant-calling-policy

# Delete role
aws iam delete-role \
  --role-name ${ROLE_NAME}

echo "✓ IAM role and policy deleted"
```

### Step 6: Verify Deletion

```bash
echo "=== Deletion Verification ==="

# Check S3 buckets (should be empty or not exist)
echo -n "S3 Input Bucket: "
aws s3 ls s3://${BUCKET_INPUT} 2>/dev/null && echo "Still exists (FAILED)" || echo "Deleted (OK)"

echo -n "S3 Results Bucket: "
aws s3 ls s3://${BUCKET_RESULTS} 2>/dev/null && echo "Still exists (FAILED)" || echo "Deleted (OK)"

# Check DynamoDB table (should not exist)
echo -n "DynamoDB Table: "
aws dynamodb describe-table --table-name ${TABLE_NAME} --region us-east-1 2>/dev/null && echo "Still exists (FAILED)" || echo "Deleted (OK)"

# Check Lambda function (should not exist)
echo -n "Lambda Function: "
aws lambda get-function --function-name ${LAMBDA_FUNCTION} --region us-east-1 2>/dev/null && echo "Still exists (FAILED)" || echo "Deleted (OK)"

# Check IAM role (should not exist)
echo -n "IAM Role: "
aws iam get-role --role-name ${ROLE_NAME} 2>/dev/null && echo "Still exists (FAILED)" || echo "Deleted (OK)"

echo ""
echo "✓ Cleanup complete!"
```

## Automated Cleanup Script

Create this script for one-command cleanup:

```bash
# Save as cleanup_all.sh
cat > cleanup_all.sh << 'EOF'
#!/bin/bash

# Load configuration
source aws_config.sh

echo "=== AWS Resource Cleanup ==="
echo "This will DELETE all resources created for Genomic Variant Analysis"
echo ""
echo "Input Bucket: ${BUCKET_INPUT}"
echo "Results Bucket: ${BUCKET_RESULTS}"
echo "DynamoDB Table: ${TABLE_NAME}"
echo "Lambda Function: ${LAMBDA_FUNCTION}"
echo "IAM Role: ${ROLE_NAME}"
echo ""
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
  echo "Cleanup cancelled."
  exit 0
fi

echo ""
echo "Starting cleanup..."

# Delete S3 objects
echo "1. Deleting S3 objects..."
aws s3 rm s3://${BUCKET_INPUT} --recursive 2>/dev/null
aws s3 rm s3://${BUCKET_RESULTS} --recursive 2>/dev/null

# Delete S3 buckets
echo "2. Deleting S3 buckets..."
aws s3 rb s3://${BUCKET_INPUT} 2>/dev/null
aws s3 rb s3://${BUCKET_RESULTS} 2>/dev/null

# Delete DynamoDB table
echo "3. Deleting DynamoDB table..."
aws dynamodb delete-table \
  --table-name ${TABLE_NAME} \
  --region us-east-1 2>/dev/null

# Delete Lambda function
echo "4. Deleting Lambda function..."
aws lambda delete-function \
  --function-name ${LAMBDA_FUNCTION} \
  --region us-east-1 2>/dev/null

# Delete IAM resources
echo "5. Deleting IAM role..."
aws iam delete-role-policy \
  --role-name ${ROLE_NAME} \
  --policy-name variant-calling-policy 2>/dev/null
aws iam delete-role \
  --role-name ${ROLE_NAME} 2>/dev/null

echo ""
echo "✓ Cleanup complete!"
echo "All AWS resources have been deleted."
EOF

chmod +x cleanup_all.sh
./cleanup_all.sh
```

## Partial Cleanup

If you want to keep some resources (e.g., for future work):

### Keep Everything Except S3 Data

```bash
# Just delete S3 objects, keep infrastructure
aws s3 rm s3://${BUCKET_INPUT} --recursive
aws s3 rm s3://${BUCKET_RESULTS} --recursive
```

### Keep DynamoDB for Future Queries

```bash
# Skip the DynamoDB deletion step
# Just delete S3, Lambda, and IAM
```

## Cost Verification

Verify that charges have stopped:

```bash
# Check AWS Cost Explorer
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-01-05 \
  --granularity DAILY \
  --metrics "BlendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE
```

You should see:
- **Lambda:** ~$0 (after function deletion)
- **DynamoDB:** ~$0 (after table deletion)
- **S3:** ~$0 (after bucket deletion)
- **IAM:** Always free

## Troubleshooting

### Issue: "NoSuchBucket" error when deleting

**Solution:** The bucket may already be deleted. This is fine - proceed to next step.

```bash
aws s3 ls  # Lists all your buckets to verify
```

### Issue: "ResourceNotFoundException" for Lambda

**Solution:** The function was already deleted. Proceed to next step.

### Issue: "EntityAlreadyDeleted" for IAM role

**Solution:** The role was already deleted. This is fine.

### Issue: Can't delete S3 bucket - "BucketNotEmpty"

**Solution:** Make sure all versions are deleted:

```bash
# Delete all object versions
aws s3api list-object-versions \
  --bucket ${BUCKET_INPUT} \
  --output json | jq '.Versions[] | {Key:.Key, VersionId:.VersionId}' | \
  while read -r version; do
    aws s3api delete-object --bucket ${BUCKET_INPUT} --key "$version"
  done

# Then delete bucket
aws s3 rb s3://${BUCKET_INPUT}
```

## Cleanup Checklist

- [ ] Verified cleanup script (ran verification)
- [ ] Deleted S3 objects
- [ ] Deleted S3 buckets
- [ ] Deleted DynamoDB table
- [ ] Deleted Lambda function
- [ ] Deleted IAM role and policy
- [ ] Verified AWS Cost Explorer shows $0 charges
- [ ] Removed .env and aws_config.sh files from local machine

## Important Notes

1. **Deletion is permanent** - You cannot recover deleted S3 buckets or DynamoDB tables
2. **Versioned objects** - If you enabled versioning, you must delete all versions
3. **Lifecycle policies** - Will automatically delete old files after 7 days anyway
4. **Cost monitoring** - AWS may take 24 hours to update cost reports
5. **Lambda storage** - Lambda functions don't have storage costs, only execution

## Prevention Tips for Next Time

To avoid high costs in the future:

1. **Always run cleanup immediately** after finishing
2. **Set a CloudWatch alarm** for high costs
3. **Use lifecycle policies** (included in setup)
4. **Delete S3 buckets, not just objects** - buckets themselves cost money if empty
5. **Monitor CloudWatch logs** - don't pay for logs you don't need

---

**Cleanup complete!** You've successfully deleted all AWS resources.

If you want to run the analysis again, follow `setup_guide.md` to create new resources.
