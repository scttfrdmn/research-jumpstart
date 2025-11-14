# Cleanup Guide - Delete AWS Resources

After completing the Tier 2 project, delete all AWS resources to avoid unexpected charges.

**Important: Resources continue to incur charges until deleted!**

## Step 1: Delete Lambda Function (2 minutes)

```bash
# Delete the Lambda function
aws lambda delete-function \
  --function-name astronomy-source-detection

# Verify deletion
aws lambda list-functions | grep astronomy-source-detection
# Should return nothing
```

## Step 2: Delete IAM Role (5 minutes)

First, detach policies from the role:

```bash
# Detach S3 policy
aws iam detach-role-policy \
  --role-name lambda-astronomy-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Detach CloudWatch policy
aws iam detach-role-policy \
  --role-name lambda-astronomy-role \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

# Delete the role
aws iam delete-role \
  --role-name lambda-astronomy-role

# Verify deletion
aws iam get-role --role-name lambda-astronomy-role 2>&1 | grep NoSuchEntity
```

## Step 3: Delete S3 Buckets (5 minutes)

### Important: S3 buckets must be empty before deletion

```bash
# Load bucket names
source ~/.astronomy_env

# Empty the raw images bucket
aws s3 rm s3://${BUCKET_RAW} --recursive

# Empty the catalog bucket
aws s3 rm s3://${BUCKET_CATALOG} --recursive

# Delete the raw images bucket
aws s3 rb s3://${BUCKET_RAW}

# Delete the catalog bucket
aws s3 rb s3://${BUCKET_CATALOG}

# Verify deletion
aws s3 ls | grep astronomy
# Should return nothing
```

### Alternative: Faster deletion with sync to /dev/null

```bash
# For very large buckets, use sync to delete quickly
aws s3 sync s3://${BUCKET_RAW} /dev/null --delete
aws s3 sync s3://${BUCKET_CATALOG} /dev/null --delete

# Then delete the buckets
aws s3 rb s3://${BUCKET_RAW}
aws s3 rb s3://${BUCKET_CATALOG}
```

## Step 4: Delete Athena Workgroup (2 minutes)

```bash
# Delete the Athena workgroup
aws athena delete-work-group \
  --work-group astronomy-workgroup

# Verify deletion
aws athena list-work-groups | grep astronomy-workgroup
# Should return nothing
```

## Step 5: Delete CloudWatch Logs (2 minutes)

CloudWatch Logs are auto-deleted but you can force deletion:

```bash
# List log groups
aws logs describe-log-groups | grep astronomy

# Delete the Lambda log group
aws logs delete-log-group \
  --log-group-name /aws/lambda/astronomy-source-detection

# Verify deletion
aws logs describe-log-groups | grep astronomy
# Should return nothing
```

## Step 6: Check for Other Resources (5 minutes)

Make sure you didn't create other resources:

```bash
# Check for any remaining S3 buckets with your prefix
aws s3 ls | grep astronomy

# Check for EC2 instances
aws ec2 describe-instances | grep astronomy

# Check for RDS databases
aws rds describe-db-instances | grep astronomy

# Check for unused Elastic IPs
aws ec2 describe-addresses
```

## Verification Checklist

Confirm all resources are deleted:

```bash
# 1. Lambda function gone
aws lambda list-functions | grep -c astronomy-source-detection
# Should return 0

# 2. IAM role deleted
aws iam get-role --role-name lambda-astronomy-role 2>&1 | grep NoSuchEntity
# Should show "NoSuchEntity" error

# 3. S3 buckets deleted
aws s3 ls | grep -c astronomy
# Should return 0

# 4. Athena workgroup deleted
aws athena list-work-groups | grep -c astronomy-workgroup
# Should return 0

# 5. CloudWatch logs deleted
aws logs describe-log-groups | grep -c astronomy
# Should return 0
```

## Cost Verification

Check AWS Cost Explorer to ensure charges have stopped:

1. Go to [AWS Cost Explorer](https://console.aws.amazon.com/cost-management/home)
2. Select date range (last 30 days)
3. Filter by service
4. Look for S3, Lambda, Athena charges
5. After deletion, these should not appear in new charges

## Automated Cleanup Script

```bash
#!/bin/bash
# cleanup_all.sh - Delete all astronomy project resources

set -e

echo "Starting cleanup..."

# Load environment
source ~/.astronomy_env

# Delete Lambda
echo "Deleting Lambda function..."
aws lambda delete-function --function-name astronomy-source-detection 2>/dev/null || true

# Delete IAM role policies
echo "Detaching IAM policies..."
aws iam detach-role-policy --role-name lambda-astronomy-role --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess 2>/dev/null || true
aws iam detach-role-policy --role-name lambda-astronomy-role --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess 2>/dev/null || true

# Delete IAM role
echo "Deleting IAM role..."
aws iam delete-role --role-name lambda-astronomy-role 2>/dev/null || true

# Empty and delete S3 buckets
echo "Deleting S3 buckets..."
aws s3 rm s3://${BUCKET_RAW} --recursive 2>/dev/null || true
aws s3 rb s3://${BUCKET_RAW} 2>/dev/null || true

aws s3 rm s3://${BUCKET_CATALOG} --recursive 2>/dev/null || true
aws s3 rb s3://${BUCKET_CATALOG} 2>/dev/null || true

# Delete Athena workgroup
echo "Deleting Athena workgroup..."
aws athena delete-work-group --work-group astronomy-workgroup 2>/dev/null || true

# Delete CloudWatch logs
echo "Deleting CloudWatch logs..."
aws logs delete-log-group --log-group-name /aws/lambda/astronomy-source-detection 2>/dev/null || true

# Clean local files
echo "Cleaning local data..."
rm -rf data/raw/*
rm -rf data/*.parquet

echo "Cleanup complete!"
echo "Verify with: aws s3 ls && aws lambda list-functions && aws iam list-roles"
```

Save this as `scripts/cleanup_all.sh` and run:

```bash
chmod +x scripts/cleanup_all.sh
bash scripts/cleanup_all.sh
```

## Cost Monitoring Before Deletion

Before deleting, check your actual costs:

```bash
# Get current month's costs
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics "BlendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE \
  --filter file://cost-filter.json

# Create cost-filter.json for astronomy resources
cat > cost-filter.json << 'EOF'
{
  "Dimensions": {
    "Key": "SERVICE",
    "Values": ["Amazon Simple Storage Service", "AWS Lambda", "Amazon Athena"]
  }
}
EOF
```

## What Gets Charged?

Even after one run, you may see charges for:

| Service | Charge | When? |
|---------|--------|-------|
| **S3 Storage** | $0.023 per GB/month | While bucket exists |
| **S3 Requests** | $0.0004 per 1000 requests | With data transfer |
| **Lambda** | $0.20 per million requests | While function exists |
| **CloudWatch Logs** | $0.50 per GB ingested | While logs exist |
| **Athena** | $6.25 per TB scanned | While table exists |

**Largest charges come from:**
1. S3 storage (if you leave large files)
2. CloudWatch logs (if log retention is set)

## Keep Resources for Reuse?

If you want to keep running the project, only delete what you don't need:

**Minimal deletion (keep for reuse):**
```bash
# Just delete unused components
aws s3 rm s3://${BUCKET_CATALOG}/athena-results/ --recursive
aws logs delete-log-group --log-group-name /aws/lambda/astronomy-source-detection
```

**Full preservation:**
```bash
# Keep everything, just stop using it
# No resources to delete - just avoid running Lambda
```

**Total preservation cost: ~$0.50/month** (S3 storage for small amounts)

## Troubleshooting Deletion

### Issue: "Cannot delete bucket - not empty"

Solution:
```bash
# Remove all objects first
aws s3 rm s3://${BUCKET_NAME} --recursive --force

# Then delete bucket
aws s3 rb s3://${BUCKET_NAME}
```

### Issue: "Cannot delete IAM role - has inline policies"

Solution:
```bash
# List inline policies
aws iam list-role-policies --role-name lambda-astronomy-role

# Delete each inline policy
aws iam delete-role-policy \
  --role-name lambda-astronomy-role \
  --policy-name <POLICY_NAME>

# Then delete role
aws iam delete-role --role-name lambda-astronomy-role
```

### Issue: "Cannot delete Lambda - still has versions"

Solution:
```bash
# Lambda versions are deleted with the function, just retry:
aws lambda delete-function --function-name astronomy-source-detection

# Wait 5 seconds and try again if it fails
sleep 5
aws lambda list-functions | grep astronomy-source-detection
```

## Final Verification

After cleanup, confirm these commands return nothing:

```bash
# Should all return empty results
echo "Lambda functions:"
aws lambda list-functions --query 'Functions[?contains(FunctionName, `astronomy`)]'

echo "IAM roles:"
aws iam list-roles --query 'Roles[?contains(RoleName, `astronomy`)]'

echo "S3 buckets:"
aws s3 ls | grep astronomy

echo "Athena workgroups:"
aws athena list-work-groups --query 'WorkGroups[?Name==`astronomy-workgroup`]'

echo "CloudWatch log groups:"
aws logs describe-log-groups --query 'logGroups[?contains(logGroupName, `astronomy`)]'
```

## Summary

You've successfully cleaned up all AWS resources created for the Tier 2 project.

**What's deleted:**
- ✓ Lambda function
- ✓ IAM role
- ✓ S3 buckets and data
- ✓ Athena workgroup
- ✓ CloudWatch logs
- ✓ Local data files

**What you've kept:**
- Your AWS account (can reuse)
- The code in this repository (can rerun anytime)
- Knowledge of how to use AWS services!

To run the project again, follow setup_guide.md from scratch.

---

**Deletion complete!** Your AWS account is clean and ready for other projects.
