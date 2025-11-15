# Cleanup Guide - Delete AWS Resources

After completing the archaeological site analysis project, delete all AWS resources to stop incurring charges.

**Important:** Once deleted, resources cannot be recovered. Ensure you've backed up any results you want to keep.

---

## Quick Cleanup (Recommended)

```bash
# Set your bucket name
BUCKET_NAME="archaeology-data-xxxx"  # Replace with your bucket name

# Delete S3 bucket and all contents
aws s3 rm s3://$BUCKET_NAME --recursive
aws s3 rb s3://$BUCKET_NAME

# Delete DynamoDB table
aws dynamodb delete-table --table-name ArtifactCatalog

# Delete Lambda function
aws lambda delete-function --function-name classify-artifacts

# Detach and delete IAM role policies
aws iam detach-role-policy \
  --role-name lambda-archaeology-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam detach-role-policy \
  --role-name lambda-archaeology-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam detach-role-policy \
  --role-name lambda-archaeology-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

# Delete IAM role
aws iam delete-role --role-name lambda-archaeology-processor

echo "All resources deleted!"
```

---

## Step-by-Step Deletion

If you prefer to delete resources individually or want to verify each step:

### Step 1: Back Up Results (Optional)

Save any results before deletion:

```bash
# Download all processed results from S3
aws s3 cp s3://$BUCKET_NAME/processed/ ./backups/processed/ --recursive

# Download analysis results
aws s3 cp s3://$BUCKET_NAME/analysis/ ./backups/analysis/ --recursive

# Download artifact images (if any)
aws s3 cp s3://$BUCKET_NAME/images/ ./backups/images/ --recursive

# Verify download
ls -lah backups/
```

### Step 2: Export DynamoDB Data (Optional)

Export artifact catalog before deletion:

```bash
# Scan and export all artifacts to JSON
aws dynamodb scan \
  --table-name ArtifactCatalog \
  --output json > artifact_catalog_backup.json

# Or export to CSV using Python
python scripts/export_dynamodb.py

# Verify export
wc -l artifact_catalog_backup.json
```

### Step 3: Delete S3 Bucket

S3 buckets must be completely empty before deletion.

```bash
# List bucket contents
aws s3 ls s3://$BUCKET_NAME --recursive

# Delete all objects
aws s3 rm s3://$BUCKET_NAME --recursive

# Verify bucket is empty
aws s3 ls s3://$BUCKET_NAME --recursive
# Should return nothing

# Delete bucket
aws s3 rb s3://$BUCKET_NAME

# Verify deletion
aws s3 ls | grep archaeology-data
# Should return nothing
```

### Step 4: Delete DynamoDB Table

```bash
# Delete the DynamoDB table
aws dynamodb delete-table --table-name ArtifactCatalog

# Wait for deletion (30-60 seconds)
aws dynamodb wait table-not-exists --table-name ArtifactCatalog

# Verify deletion
aws dynamodb list-tables | grep ArtifactCatalog
# Should return nothing
```

### Step 5: Delete Lambda Function

```bash
# Delete the Lambda function
aws lambda delete-function --function-name classify-artifacts

# Verify deletion
aws lambda get-function --function-name classify-artifacts 2>&1
# Should return: "Function not found"
```

### Step 6: Delete Lambda S3 Event Trigger

If you set up S3->Lambda triggers:

```bash
# Remove S3 event notification
aws s3api delete-bucket-notification-configuration \
  --bucket $BUCKET_NAME 2>&1

# Note: This may fail if bucket is already deleted (that's OK)
```

### Step 7: Delete Lambda Permission

```bash
# Remove Lambda S3 invocation permission
aws lambda remove-permission \
  --function-name classify-artifacts \
  --statement-id AllowS3Invoke 2>&1

# May fail if function already deleted (that's OK)
```

### Step 8: Delete IAM Role

IAM roles must have all policies detached before deletion.

```bash
# List attached policies
aws iam list-attached-role-policies \
  --role-name lambda-archaeology-processor

# Detach each policy
aws iam detach-role-policy \
  --role-name lambda-archaeology-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam detach-role-policy \
  --role-name lambda-archaeology-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam detach-role-policy \
  --role-name lambda-archaeology-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

# List inline policies (if any)
aws iam list-role-policies --role-name lambda-archaeology-processor

# Delete inline policies (if any exist)
for policy_name in $(aws iam list-role-policies \
  --role-name lambda-archaeology-processor \
  --query 'PolicyNames[]' --output text); do
  aws iam delete-role-policy \
    --role-name lambda-archaeology-processor \
    --policy-name $policy_name
done

# Finally, delete the role
aws iam delete-role --role-name lambda-archaeology-processor

# Verify deletion
aws iam get-role --role-name lambda-archaeology-processor 2>&1
# Should return: "NoSuchEntity"
```

### Step 9: Check for Remaining Resources

```bash
# Verify Lambda is gone
aws lambda list-functions | grep classify-artifacts

# Verify IAM role is gone
aws iam list-roles | grep lambda-archaeology-processor

# Verify S3 bucket is gone
aws s3 ls | grep archaeology-data

# Verify DynamoDB table is gone
aws dynamodb list-tables | grep ArtifactCatalog

# Verify CloudWatch logs (optional, see next section)
aws logs describe-log-groups | grep classify-artifacts
```

---

## Delete CloudWatch Logs (Optional)

CloudWatch logs persist even after Lambda deletion and incur small storage costs (~$0.50/GB/month).

```bash
# Delete log group
aws logs delete-log-group \
  --log-group-name /aws/lambda/classify-artifacts

# Verify deletion
aws logs describe-log-groups | grep classify-artifacts
# Should return nothing
```

---

## Delete Athena Resources (If Created)

If you set up Athena for SQL queries:

```bash
# Delete Athena query results from S3
aws s3 rm s3://$BUCKET_NAME/athena-results/ --recursive

# Drop Athena table
aws athena start-query-execution \
  --query-string "DROP TABLE IF EXISTS archaeology.artifacts" \
  --query-execution-context Database=archaeology \
  --result-configuration "OutputLocation=s3://$BUCKET_NAME/athena-results/"

# Drop Athena database
aws athena start-query-execution \
  --query-string "DROP DATABASE IF EXISTS archaeology" \
  --result-configuration "OutputLocation=s3://$BUCKET_NAME/athena-results/"

# Note: Database will be dropped when S3 bucket is deleted
```

---

## Verify All Resources Deleted

Check that everything is cleaned up:

```bash
# Summary check
echo "=== S3 Buckets ==="
aws s3 ls | grep archaeology-data || echo "✓ No archaeology-data buckets"

echo "=== Lambda Functions ==="
aws lambda list-functions | grep classify-artifacts || echo "✓ No classify-artifacts functions"

echo "=== DynamoDB Tables ==="
aws dynamodb list-tables | grep ArtifactCatalog || echo "✓ No ArtifactCatalog tables"

echo "=== IAM Roles ==="
aws iam list-roles | grep lambda-archaeology-processor || echo "✓ No lambda-archaeology-processor roles"

echo "=== CloudWatch Logs ==="
aws logs describe-log-groups | grep classify-artifacts || echo "✓ No classify-artifacts logs"

echo ""
echo "All resources cleaned up!"
```

---

## Expected Remaining Charges

After cleanup, you should only see:

1. **CloudWatch logs** (if not deleted): ~$0.03 per month
2. **Athena query history**: ~$0.01
3. **Minimal S3 request charges**: ~$0.01

**If you've completed all steps above, final charges should be < $0.05**

---

## Cleanup Python Script

You can also use an automated Python cleanup script:

```bash
# Create cleanup.py
cat > cleanup.py << 'EOFPYTHON'
#!/usr/bin/env python3
"""Clean up AWS resources for Archaeological Site Analysis Tier 2 project."""

import boto3
import sys
import time
from botocore.exceptions import ClientError


def cleanup_s3(bucket_name):
    """Delete S3 bucket and all contents."""
    s3 = boto3.client('s3')

    print(f"Cleaning up S3 bucket: {bucket_name}")

    try:
        # List all objects (including versions)
        paginator = s3.get_paginator('list_object_versions')
        pages = paginator.paginate(Bucket=bucket_name)

        for page in pages:
            # Delete versions
            if 'Versions' in page:
                for obj in page['Versions']:
                    print(f"  Deleting version: {obj['Key']} (version: {obj['VersionId']})")
                    s3.delete_object(
                        Bucket=bucket_name,
                        Key=obj['Key'],
                        VersionId=obj['VersionId']
                    )

            # Delete delete markers
            if 'DeleteMarkers' in page:
                for obj in page['DeleteMarkers']:
                    print(f"  Deleting marker: {obj['Key']} (version: {obj['VersionId']})")
                    s3.delete_object(
                        Bucket=bucket_name,
                        Key=obj['Key'],
                        VersionId=obj['VersionId']
                    )

        # Delete bucket
        s3.delete_bucket(Bucket=bucket_name)
        print(f"✓ Bucket deleted: {bucket_name}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            print(f"✓ Bucket already deleted: {bucket_name}")
            return True
        else:
            print(f"✗ Error deleting bucket: {e}")
            return False


def cleanup_dynamodb():
    """Delete DynamoDB table."""
    dynamodb = boto3.client('dynamodb')

    print("Cleaning up DynamoDB table: ArtifactCatalog")

    try:
        dynamodb.delete_table(TableName='ArtifactCatalog')
        print("✓ DynamoDB table deleted")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print("✓ DynamoDB table already deleted")
            return True
        else:
            print(f"✗ Error deleting DynamoDB table: {e}")
            return False


def cleanup_lambda():
    """Delete Lambda function."""
    lam = boto3.client('lambda')

    print("Cleaning up Lambda function: classify-artifacts")

    try:
        lam.delete_function(FunctionName='classify-artifacts')
        print("✓ Lambda function deleted")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print("✓ Lambda function already deleted")
            return True
        else:
            print(f"✗ Error deleting Lambda: {e}")
            return False


def cleanup_iam_role():
    """Delete IAM role and policies."""
    iam = boto3.client('iam')

    print("Cleaning up IAM role: lambda-archaeology-processor")

    try:
        role_name = 'lambda-archaeology-processor'

        # Detach attached policies
        response = iam.list_attached_role_policies(RoleName=role_name)
        for policy in response['AttachedPolicies']:
            print(f"  Detaching policy: {policy['PolicyName']}")
            iam.detach_role_policy(
                RoleName=role_name,
                PolicyArn=policy['PolicyArn']
            )

        # Delete inline policies
        response = iam.list_role_policies(RoleName=role_name)
        for policy_name in response['PolicyNames']:
            print(f"  Deleting inline policy: {policy_name}")
            iam.delete_role_policy(
                RoleName=role_name,
                PolicyName=policy_name
            )

        # Delete role
        iam.delete_role(RoleName=role_name)
        print("✓ IAM role deleted")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            print("✓ IAM role already deleted")
            return True
        else:
            print(f"✗ Error deleting IAM role: {e}")
            return False


def cleanup_cloudwatch_logs():
    """Delete CloudWatch logs."""
    logs = boto3.client('logs')

    print("Cleaning up CloudWatch logs")

    try:
        logs.delete_log_group(
            logGroupName='/aws/lambda/classify-artifacts'
        )
        print("✓ CloudWatch logs deleted")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print("✓ CloudWatch logs already deleted")
            return True
        else:
            print(f"✗ Error deleting logs: {e}")
            return False


def main():
    """Main cleanup function."""
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()
    bucket_name = os.environ.get('BUCKET_NAME', 'archaeology-data-unknown')

    print("=" * 60)
    print("Archaeological Site Analysis Tier 2 - AWS Resource Cleanup")
    print("=" * 60)
    print()

    # Confirm deletion
    response = input(f"Delete all resources for bucket '{bucket_name}'? (yes/no): ")
    if response.lower() != 'yes':
        print("Cleanup cancelled")
        sys.exit(0)

    results = []

    # Clean up in order
    results.append(("S3 Bucket", cleanup_s3(bucket_name)))
    time.sleep(2)

    results.append(("DynamoDB Table", cleanup_dynamodb()))
    time.sleep(2)

    results.append(("Lambda Function", cleanup_lambda()))
    time.sleep(2)

    results.append(("CloudWatch Logs", cleanup_cloudwatch_logs()))
    time.sleep(2)

    results.append(("IAM Role", cleanup_iam_role()))

    print()
    print("=" * 60)
    print("Cleanup Summary")
    print("=" * 60)

    for resource, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {resource}")

    all_success = all(success for _, success in results)

    if all_success:
        print()
        print("✓ All resources successfully deleted!")
        print("✓ Future charges should be minimal (<$0.01)")
    else:
        print()
        print("✗ Some resources failed to delete")
        print("  Please check the errors above and try again")
        sys.exit(1)


if __name__ == '__main__':
    main()
EOFPYTHON

# Make executable
chmod +x cleanup.py

# Run cleanup
python cleanup.py
```

---

## Troubleshooting Cleanup

### Problem: "Bucket is not empty"

```bash
# Delete all versions (including old versions)
aws s3api list-object-versions \
  --bucket $BUCKET_NAME \
  --query 'Versions[].{Key:Key,VersionId:VersionId}' \
  --output text | while read key version; do
  aws s3api delete-object \
    --bucket $BUCKET_NAME \
    --key "$key" \
    --version-id "$version"
done

# Delete delete markers
aws s3api list-object-versions \
  --bucket $BUCKET_NAME \
  --query 'DeleteMarkers[].{Key:Key,VersionId:VersionId}' \
  --output text | while read key version; do
  aws s3api delete-object \
    --bucket $BUCKET_NAME \
    --key "$key" \
    --version-id "$version"
done

# Then delete bucket
aws s3 rb s3://$BUCKET_NAME
```

### Problem: "Cannot delete role while it has attached policies"

```bash
# List all attached policies
aws iam list-attached-role-policies \
  --role-name lambda-archaeology-processor

# Detach each one
for policy_arn in $(aws iam list-attached-role-policies \
  --role-name lambda-archaeology-processor \
  --query 'AttachedPolicies[].PolicyArn' --output text); do
  aws iam detach-role-policy \
    --role-name lambda-archaeology-processor \
    --policy-arn $policy_arn
done

# Then delete the role
aws iam delete-role --role-name lambda-archaeology-processor
```

### Problem: "Table is being deleted" for DynamoDB

```bash
# Wait for table deletion to complete
aws dynamodb wait table-not-exists --table-name ArtifactCatalog

# This can take 30-60 seconds
```

### Problem: "Lambda function still has permissions"

```bash
# Remove all Lambda permissions
aws lambda get-policy --function-name classify-artifacts 2>&1 | \
  grep -o '"Sid":"[^"]*"' | \
  cut -d'"' -f4 | \
  xargs -I {} aws lambda remove-permission \
    --function-name classify-artifacts \
    --statement-id {}

# Then delete function
aws lambda delete-function --function-name classify-artifacts
```

---

## Verify Costs Are Stopping

After cleanup, monitor that charges stop:

```bash
# Check costs for today
aws ce get-cost-and-usage \
  --time-period Start=$(date +%Y-%m-%d),End=$(date -d tomorrow +%Y-%m-%d) \
  --granularity DAILY \
  --metrics "UnblendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE

# Should show minimal or zero charges for Lambda, S3, DynamoDB
```

---

## Backup Your Code

Before cleanup, consider backing up:

1. **Scripts** - Copy `scripts/` folder locally
2. **Notebooks** - Copy `notebooks/` folder locally
3. **Results** - Download from S3 (see Step 1 above)
4. **DynamoDB data** - Export artifact catalog (see Step 2 above)
5. **Configuration** - Back up `.env` file (remove secrets first!)

```bash
# Create comprehensive backup
mkdir -p backup-$(date +%Y%m%d)
cp -r scripts backup-$(date +%Y%m%d)/
cp -r notebooks backup-$(date +%Y%m%d)/

# Download all S3 data
aws s3 sync s3://$BUCKET_NAME backup-$(date +%Y%m%d)/s3-data/

# Export DynamoDB
aws dynamodb scan --table-name ArtifactCatalog \
  > backup-$(date +%Y%m%d)/artifact_catalog.json

# Create tarball
tar -czf backup-$(date +%Y%m%d).tar.gz backup-$(date +%Y%m%d)/

echo "Backup created: backup-$(date +%Y%m%d).tar.gz"
```

---

## What NOT to Delete

Do NOT delete these (they may be used by other projects):

- Default VPC
- Default security groups
- Your AWS user/account
- Other Lambda functions
- Other S3 buckets
- Other DynamoDB tables
- Other IAM roles

Only delete resources created for this specific project!

---

## Final Checklist

- [ ] Backed up processed results from S3
- [ ] Exported DynamoDB artifact catalog
- [ ] Verified S3 bucket is deleted
- [ ] Verified DynamoDB table is deleted
- [ ] Verified Lambda function is deleted
- [ ] Verified IAM role is deleted
- [ ] Verified CloudWatch logs are deleted
- [ ] Checked AWS Cost Explorer shows no charges
- [ ] Set up billing alert for future projects
- [ ] Saved project code locally

---

## Zero Out Budget

For extra safety, you can set AWS Budget to $0:

```bash
# Create zero budget to catch any unexpected charges
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget '{
    "BudgetName": "Archaeology-Tier2-Zero-Budget",
    "BudgetLimit": {"Amount": "0", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }' \
  --notifications-with-subscribers '[{
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 0.01
    },
    "Subscribers": [{
      "SubscriptionType": "EMAIL",
      "Address": "your-email@example.com"
    }]
  }]'
```

This will alert if any charges appear (good for validation).

---

## Next Steps

After cleanup:

1. **Review what you learned** - See main README.md
2. **Plan next project** - Consider Tier 3 for production
3. **Share results** - Document findings for publication
4. **Keep code** - Version control your analysis scripts

---

## Need Help?

- Check AWS Cost Explorer: https://console.aws.amazon.com/cost-management/
- Review setup_guide.md for resource details
- See troubleshooting section above
- Contact AWS Support: https://console.aws.amazon.com/support/

---

**Important:** You are responsible for all AWS charges. Please verify all resources are deleted and costs have stopped.
