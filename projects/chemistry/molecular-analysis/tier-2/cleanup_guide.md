# Cleanup Guide - Delete AWS Resources

After completing the molecular analysis project, delete all AWS resources to stop incurring charges.

**Important:** Once deleted, resources cannot be recovered. Ensure you've backed up any results you want to keep.

---

## Quick Cleanup (Recommended)

```bash
# Set your bucket name
BUCKET_NAME="molecular-data-xxxx"  # Replace with your bucket name
TABLE_NAME="MolecularProperties"
FUNCTION_NAME="analyze-molecule"
ROLE_NAME="lambda-molecular-analyzer"

# Delete S3 bucket and all contents
aws s3 rm s3://$BUCKET_NAME --recursive
aws s3 rb s3://$BUCKET_NAME

# Delete DynamoDB table
aws dynamodb delete-table --table-name $TABLE_NAME

# Delete Lambda function
aws lambda delete-function --function-name $FUNCTION_NAME

# Delete IAM role (detach policies first)
for policy_arn in $(aws iam list-attached-role-policies --role-name $ROLE_NAME --query 'AttachedPolicies[].PolicyArn' --output text); do
    aws iam detach-role-policy --role-name $ROLE_NAME --policy-arn $policy_arn
done
aws iam delete-role --role-name $ROLE_NAME

echo "All resources deleted!"
```

---

## Step-by-Step Deletion

If you prefer to delete resources individually or want to verify each step:

### Step 1: Back Up Results (Optional)

Save any results before deletion:

```bash
# Download molecular properties from DynamoDB
python scripts/query_results.py \
  --table MolecularProperties \
  --output backup/molecular_properties.csv

# Download molecular structures from S3
aws s3 cp s3://$BUCKET_NAME/molecules/ ./backup/molecules/ --recursive

# Verify backup
ls -lah backup/
```

### Step 2: Delete DynamoDB Table

DynamoDB tables can be deleted immediately.

```bash
# Delete the table
aws dynamodb delete-table --table-name MolecularProperties

# Wait for deletion to complete
aws dynamodb wait table-not-exists --table-name MolecularProperties

# Verify deletion
aws dynamodb list-tables | grep MolecularProperties
# Should return nothing
```

**Console method:**
1. Go to [DynamoDB Console](https://console.aws.amazon.com/dynamodb/)
2. Select `MolecularProperties` table
3. Click "Delete"
4. Confirm deletion

### Step 3: Delete S3 Bucket

S3 buckets must be completely empty before deletion.

```bash
# List bucket contents
aws s3 ls s3://$BUCKET_NAME --recursive

# Delete all objects
aws s3 rm s3://$BUCKET_NAME --recursive

# Delete the bucket
aws s3 rb s3://$BUCKET_NAME

# Verify deletion
aws s3 ls | grep molecular-data
# Should return nothing
```

**Console method:**
1. Go to [S3 Console](https://console.aws.amazon.com/s3/)
2. Select your `molecular-data-xxxx` bucket
3. Click "Empty" to delete all objects
4. Click "Delete" to delete the bucket
5. Confirm deletion

### Step 4: Delete Lambda Function

```bash
# Delete the Lambda function
aws lambda delete-function --function-name analyze-molecule

# Verify deletion
aws lambda get-function --function-name analyze-molecule 2>&1
# Should return: "Function not found"
```

**Console method:**
1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Select `analyze-molecule` function
3. Click "Actions" → "Delete"
4. Confirm deletion

### Step 5: Remove Lambda S3 Event Trigger

If you set up S3→Lambda triggers, remove them first:

```bash
# Remove Lambda permission for S3 invocation
aws lambda remove-permission \
  --function-name analyze-molecule \
  --statement-id s3-invoke-permission 2>&1

# Note: This may fail if function is already deleted (that's OK)

# Remove S3 event notification
aws s3api put-bucket-notification-configuration \
  --bucket $BUCKET_NAME \
  --notification-configuration '{}'

# Note: This may fail if bucket is already deleted (that's OK)
```

### Step 6: Delete IAM Role

IAM roles must have all policies detached before deletion.

```bash
# List attached policies
aws iam list-attached-role-policies \
  --role-name lambda-molecular-analyzer

# Detach managed policies
aws iam detach-role-policy \
  --role-name lambda-molecular-analyzer \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam detach-role-policy \
  --role-name lambda-molecular-analyzer \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam detach-role-policy \
  --role-name lambda-molecular-analyzer \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

# List inline policies
aws iam list-role-policies --role-name lambda-molecular-analyzer

# Delete inline policies (if any)
for policy_name in $(aws iam list-role-policies \
  --role-name lambda-molecular-analyzer \
  --query 'PolicyNames[]' --output text); do
  aws iam delete-role-policy \
    --role-name lambda-molecular-analyzer \
    --policy-name $policy_name
done

# Delete custom policies (if created)
CUSTOM_POLICY_ARN=$(aws iam list-policies \
  --scope Local \
  --query 'Policies[?PolicyName==`MolecularAnalysisPolicy`].Arn' \
  --output text)

if [ ! -z "$CUSTOM_POLICY_ARN" ]; then
  aws iam detach-role-policy \
    --role-name lambda-molecular-analyzer \
    --policy-arn $CUSTOM_POLICY_ARN

  aws iam delete-policy --policy-arn $CUSTOM_POLICY_ARN
fi

# Finally, delete the role
aws iam delete-role --role-name lambda-molecular-analyzer

# Verify deletion
aws iam get-role --role-name lambda-molecular-analyzer 2>&1
# Should return: "NoSuchEntity"
```

**Console method:**
1. Go to [IAM Console](https://console.aws.amazon.com/iam/)
2. Click "Roles"
3. Search for `lambda-molecular-analyzer`
4. Select the role
5. Click "Delete role"
6. Confirm deletion

### Step 7: Delete CloudWatch Logs (Optional)

CloudWatch logs persist after Lambda deletion and incur small storage costs.

```bash
# Delete log group
aws logs delete-log-group \
  --log-group-name /aws/lambda/analyze-molecule

# Verify deletion
aws logs describe-log-groups | grep analyze-molecule
# Should return nothing
```

**Console method:**
1. Go to [CloudWatch Console](https://console.aws.amazon.com/cloudwatch/)
2. Click "Log groups"
3. Search for `/aws/lambda/analyze-molecule`
4. Select the log group
5. Click "Actions" → "Delete log group(s)"
6. Confirm deletion

### Step 8: Check for Remaining Resources

```bash
# Verify all resources are deleted
echo "=== S3 Buckets ==="
aws s3 ls | grep molecular-data || echo "✓ No molecular-data buckets"

echo "=== DynamoDB Tables ==="
aws dynamodb list-tables | grep MolecularProperties || echo "✓ No MolecularProperties table"

echo "=== Lambda Functions ==="
aws lambda list-functions | grep analyze-molecule || echo "✓ No analyze-molecule function"

echo "=== IAM Roles ==="
aws iam list-roles | grep lambda-molecular-analyzer || echo "✓ No lambda-molecular-analyzer role"

echo "=== CloudWatch Logs ==="
aws logs describe-log-groups | grep analyze-molecule || echo "✓ No analyze-molecule logs"

echo ""
echo "All resources cleaned up!"
```

---

## Automated Cleanup Script

You can also use the Python cleanup script:

```bash
# Create cleanup script
cat > cleanup.py << 'EOFPYTHON'
#!/usr/bin/env python3
"""Clean up AWS resources for Molecular Analysis Tier 2 project."""

import boto3
import sys
import time
import argparse
from botocore.exceptions import ClientError

def cleanup_s3(bucket_name):
    """Delete S3 bucket and all contents."""
    s3 = boto3.client('s3')
    print(f"Cleaning up S3 bucket: {bucket_name}")

    try:
        # Delete all objects
        response = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            objects = [{'Key': obj['Key']} for obj in response['Contents']]
            s3.delete_objects(Bucket=bucket_name, Delete={'Objects': objects})
            print(f"  Deleted {len(objects)} objects")

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

def cleanup_dynamodb(table_name):
    """Delete DynamoDB table."""
    dynamodb = boto3.client('dynamodb')
    print(f"Cleaning up DynamoDB table: {table_name}")

    try:
        dynamodb.delete_table(TableName=table_name)
        print(f"✓ Table deleted: {table_name}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"✓ Table already deleted: {table_name}")
            return True
        else:
            print(f"✗ Error deleting table: {e}")
            return False

def cleanup_lambda(function_name):
    """Delete Lambda function."""
    lam = boto3.client('lambda')
    print(f"Cleaning up Lambda function: {function_name}")

    try:
        lam.delete_function(FunctionName=function_name)
        print(f"✓ Lambda function deleted: {function_name}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"✓ Lambda function already deleted: {function_name}")
            return True
        else:
            print(f"✗ Error deleting Lambda: {e}")
            return False

def cleanup_iam_role(role_name):
    """Delete IAM role and policies."""
    iam = boto3.client('iam')
    print(f"Cleaning up IAM role: {role_name}")

    try:
        # Detach managed policies
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
        print(f"✓ IAM role deleted: {role_name}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            print(f"✓ IAM role already deleted: {role_name}")
            return True
        else:
            print(f"✗ Error deleting IAM role: {e}")
            return False

def cleanup_cloudwatch_logs(log_group):
    """Delete CloudWatch logs."""
    logs = boto3.client('logs')
    print(f"Cleaning up CloudWatch logs: {log_group}")

    try:
        logs.delete_log_group(logGroupName=log_group)
        print(f"✓ CloudWatch logs deleted: {log_group}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"✓ CloudWatch logs already deleted: {log_group}")
            return True
        else:
            print(f"✗ Error deleting logs: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Clean up AWS resources')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--table', default='MolecularProperties', help='DynamoDB table name')
    parser.add_argument('--function', default='analyze-molecule', help='Lambda function name')
    parser.add_argument('--role', default='lambda-molecular-analyzer', help='IAM role name')
    parser.add_argument('--yes', action='store_true', help='Skip confirmation')

    args = parser.parse_args()

    print("=" * 60)
    print("Molecular Analysis Tier 2 - AWS Resource Cleanup")
    print("=" * 60)
    print()
    print(f"This will delete:")
    print(f"  - S3 bucket: {args.bucket}")
    print(f"  - DynamoDB table: {args.table}")
    print(f"  - Lambda function: {args.function}")
    print(f"  - IAM role: {args.role}")
    print(f"  - CloudWatch logs: /aws/lambda/{args.function}")
    print()

    if not args.yes:
        response = input("Delete all resources? (yes/no): ")
        if response.lower() != 'yes':
            print("Cleanup cancelled")
            sys.exit(0)

    results = []

    # Clean up in order
    results.append(("S3 Bucket", cleanup_s3(args.bucket)))
    time.sleep(2)

    results.append(("DynamoDB Table", cleanup_dynamodb(args.table)))
    time.sleep(2)

    results.append(("Lambda Function", cleanup_lambda(args.function)))
    time.sleep(2)

    results.append(("CloudWatch Logs", cleanup_cloudwatch_logs(f'/aws/lambda/{args.function}')))
    time.sleep(2)

    results.append(("IAM Role", cleanup_iam_role(args.role)))

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

# Run cleanup (replace with your bucket name)
python cleanup.py --bucket molecular-data-xxxx
```

---

## Verify All Resources Deleted

After cleanup, verify that everything is gone:

```bash
# Check S3
aws s3 ls | grep molecular-data
# Expected: Nothing

# Check DynamoDB
aws dynamodb list-tables | grep MolecularProperties
# Expected: Nothing

# Check Lambda
aws lambda list-functions | grep analyze-molecule
# Expected: Nothing

# Check IAM
aws iam list-roles | grep lambda-molecular-analyzer
# Expected: Nothing

# Check CloudWatch
aws logs describe-log-groups | grep analyze-molecule
# Expected: Nothing
```

---

## Expected Remaining Charges

After cleanup, you should only see:

1. **CloudWatch logs** (if not deleted): ~$0.01 per month
2. **S3 requests** (final cleanup): ~$0.001
3. **Data transfer** (if any): ~$0.01

**If you've completed all steps above, final charges should be < $0.01**

---

## Verify Costs Have Stopped

Monitor that charges stop after cleanup:

```bash
# Check costs for today
aws ce get-cost-and-usage \
  --time-period Start=$(date +%Y-%m-%d),End=$(date -d tomorrow +%Y-%m-%d) \
  --granularity DAILY \
  --metrics "UnblendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE

# Should show zero or minimal charges for S3, Lambda, DynamoDB
```

**Console method:**
1. Go to [AWS Cost Explorer](https://console.aws.amazon.com/cost-management/)
2. View "Daily costs" for last 7 days
3. Filter by service: S3, Lambda, DynamoDB
4. Verify charges are zero or minimal

---

## Troubleshooting Cleanup

### Problem: "Bucket is not empty"

```bash
# Delete all object versions (if versioning was enabled)
aws s3api list-object-versions \
  --bucket $BUCKET_NAME \
  --query 'Versions[].{Key:Key,VersionId:VersionId}' \
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
# Detach all policies (including custom ones)
aws iam list-attached-role-policies --role-name lambda-molecular-analyzer \
  --query 'AttachedPolicies[].PolicyArn' --output text | while read policy_arn; do
  aws iam detach-role-policy \
    --role-name lambda-molecular-analyzer \
    --policy-arn $policy_arn
done

# Then delete the role
aws iam delete-role --role-name lambda-molecular-analyzer
```

### Problem: "Table is being deleted" (DynamoDB)

```bash
# Wait for deletion to complete
aws dynamodb wait table-not-exists --table-name MolecularProperties

# Check table status
aws dynamodb describe-table --table-name MolecularProperties 2>&1
# Should return: "Table not found"
```

---

## Backup Your Analysis

Before cleanup, consider backing up:

1. **Results** - Download molecular properties from DynamoDB
2. **Notebooks** - Save analysis notebooks locally
3. **Scripts** - Keep Python scripts for future use
4. **Configuration** - Back up `.env` file (remove secrets first!)

```bash
# Create backup
mkdir -p backup-$(date +%Y%m%d)

# Backup results
python scripts/query_results.py \
  --table MolecularProperties \
  --output backup-$(date +%Y%m%d)/molecular_properties.csv

# Backup molecular structures
aws s3 cp s3://$BUCKET_NAME/molecules/ \
  backup-$(date +%Y%m%d)/molecules/ \
  --recursive

# Backup notebooks
cp -r notebooks backup-$(date +%Y%m%d)/

# Create archive
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
- Other IAM roles
- Other DynamoDB tables

Only delete resources created specifically for this project!

---

## Final Checklist

- [ ] Backed up molecular properties from DynamoDB
- [ ] Backed up molecular structures from S3
- [ ] Saved analysis notebooks locally
- [ ] Verified S3 bucket is deleted
- [ ] Verified DynamoDB table is deleted
- [ ] Verified Lambda function is deleted
- [ ] Verified IAM role is deleted
- [ ] Verified CloudWatch logs are deleted
- [ ] Checked AWS Cost Explorer shows no charges
- [ ] Set up billing alert (recommended for future projects)

---

## Set Zero Budget Alert

For extra safety after cleanup:

```bash
# Create zero budget alert
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget '{
    "BudgetName": "MolecularAnalysis-ZeroBudget",
    "BudgetLimit": {"Amount": "0.10", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }'
```

This will alert if any charges appear (good for validation).

---

## Next Steps

After cleanup:

1. **Review what you learned** - See main README.md
2. **Plan next project** - Consider Tier 3 for production
3. **Share results** - Document findings from your analysis
4. **Keep code** - Version control your analysis scripts

---

## Need Help?

- Check AWS Cost Explorer: https://console.aws.amazon.com/cost-management/
- Review setup_guide.md for resource details
- See troubleshooting section above
- Contact AWS Support: https://console.aws.amazon.com/support/

---

**Important:** You are responsible for all AWS charges. Please verify all resources are deleted and costs have stopped.

**Last updated:** 2025-11-14 | Research Jumpstart v1.0.0
