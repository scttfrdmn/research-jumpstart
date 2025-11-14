# Cleanup Guide - Delete AWS Resources

After completing the project, delete all AWS resources to stop incurring charges.

**Important:** Once deleted, resources cannot be recovered. Ensure you've backed up any results you want to keep.

---

## Quick Cleanup (Recommended)

```bash
# Delete everything in one script
python cleanup.py

# Or manually with these commands:
BUCKET_NAME="climate-data-xxxx"  # Replace with your bucket name

# Delete S3 bucket and all contents
aws s3 rm s3://$BUCKET_NAME --recursive
aws s3 rb s3://$BUCKET_NAME

# Delete Lambda function
aws lambda delete-function --function-name process-climate-data

# Delete IAM role
aws iam delete-role-policy --role-name lambda-climate-processor \
  --policy-name lambda-policy
aws iam delete-role --role-name lambda-climate-processor

echo "All resources deleted!"
```

---

## Step-by-Step Deletion

If you prefer to delete resources individually or want to verify each step:

### Step 1: Back Up Results (Optional)

Save any results before deletion:

```bash
# Download all results from S3
aws s3 cp s3://$BUCKET_NAME/results/ ./backups/results/ --recursive

# Download processed data
aws s3 cp s3://$BUCKET_NAME/logs/ ./backups/logs/ --recursive

# Verify download
ls -lah backups/
```

### Step 2: Delete S3 Bucket

S3 buckets must be completely empty before deletion.

```bash
# List bucket contents
aws s3 ls s3://$BUCKET_NAME --recursive

# Delete all objects and versions
aws s3 rm s3://$BUCKET_NAME --recursive

# Delete bucket
aws s3 rb s3://$BUCKET_NAME

# Verify deletion
aws s3 ls | grep climate-data
# Should return nothing
```

### Step 3: Delete Lambda Function

```bash
# Delete the Lambda function
aws lambda delete-function --function-name process-climate-data

# Verify deletion
aws lambda get-function --function-name process-climate-data 2>&1
# Should return: "Function not found"
```

### Step 4: Delete Lambda S3 Event Trigger

If you set up S3->Lambda triggers:

```bash
# Remove S3 event notification
aws s3api delete-bucket-notification-configuration \
  --bucket $BUCKET_NAME

# Note: This may fail if bucket is already deleted (that's OK)
```

### Step 5: Delete Lambda Permission

```bash
# Remove Lambda S3 invocation permission
aws lambda remove-permission \
  --function-name process-climate-data \
  --statement-id "AllowS3Invoke" 2>&1
# May fail if function already deleted (that's OK)
```

### Step 6: Delete IAM Role

IAM roles must have all policies detached before deletion.

```bash
# List attached policies
aws iam list-attached-role-policies \
  --role-name lambda-climate-processor

# Detach each policy
aws iam detach-role-policy \
  --role-name lambda-climate-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam detach-role-policy \
  --role-name lambda-climate-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# List inline policies
aws iam list-role-policies --role-name lambda-climate-processor

# Delete inline policies
for policy_name in $(aws iam list-role-policies \
  --role-name lambda-climate-processor \
  --query 'PolicyNames[]' --output text); do
  aws iam delete-role-policy \
    --role-name lambda-climate-processor \
    --policy-name $policy_name
done

# Finally, delete the role
aws iam delete-role --role-name lambda-climate-processor

# Verify deletion
aws iam get-role --role-name lambda-climate-processor 2>&1
# Should return: "NoSuchEntity"
```

### Step 7: Check for Remaining Resources

```bash
# Verify Lambda is gone
aws lambda list-functions | grep process-climate-data

# Verify IAM role is gone
aws iam list-roles | grep lambda-climate-processor

# Verify S3 bucket is gone
aws s3 ls | grep climate-data

# Verify CloudWatch logs are gone (optional)
aws logs describe-log-groups | grep process-climate-data
```

---

## Delete CloudWatch Logs (Optional)

CloudWatch logs persist even after Lambda deletion and incur small storage costs.

```bash
# Delete log group
aws logs delete-log-group \
  --log-group-name /aws/lambda/process-climate-data

# Verify deletion
aws logs describe-log-groups | grep process-climate-data
```

---

## Delete VPC Endpoints (If Created)

If you created VPC endpoints for S3 or Lambda:

```bash
# List VPC endpoints
aws ec2 describe-vpc-endpoints --filters Name=service-name,Values=*s3*

# Delete VPC endpoint
aws ec2 delete-vpc-endpoints \
  --vpc-endpoint-ids vpce-xxxxxxxxx
```

---

## Verify All Resources Deleted

Check that everything is cleaned up:

```bash
# Summary check
echo "=== S3 Buckets ==="
aws s3 ls | grep climate-data || echo "✓ No climate-data buckets"

echo "=== Lambda Functions ==="
aws lambda list-functions | grep process-climate-data || echo "✓ No process-climate-data functions"

echo "=== IAM Roles ==="
aws iam list-roles | grep lambda-climate-processor || echo "✓ No lambda-climate-processor roles"

echo "=== CloudWatch Logs ==="
aws logs describe-log-groups | grep process-climate-data || echo "✓ No process-climate-data logs"

echo ""
echo "All resources cleaned up!"
```

---

## Expected Remaining Charges

After cleanup, you should only see:

1. **CloudWatch logs** (if not deleted): ~$0.03 per month
2. **S3 requests** (any remaining files): ~$0.01
3. **Data transfer** (if S3 has any remaining data): ~$0.02

**If you've completed all steps above, final charges should be < $0.01**

---

## Cleanup Python Script

You can also use the Python cleanup script:

```bash
# Create cleanup.py
cat > cleanup.py << 'EOFPYTHON'
#!/usr/bin/env python3
"""Clean up AWS resources for Climate Tier 2 project."""

import boto3
import sys
import time
from botocore.exceptions import ClientError

def cleanup_s3(bucket_name):
    """Delete S3 bucket and all contents."""
    s3 = boto3.client('s3')

    print(f"Cleaning up S3 bucket: {bucket_name}")

    try:
        # List all objects
        response = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            for obj in response['Contents']:
                print(f"  Deleting: {obj['Key']}")
                s3.delete_object(Bucket=bucket_name, Key=obj['Key'])

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

def cleanup_lambda():
    """Delete Lambda function."""
    lam = boto3.client('lambda')

    print("Cleaning up Lambda function")

    try:
        lam.delete_function(FunctionName='process-climate-data')
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

    print("Cleaning up IAM role")

    try:
        role_name = 'lambda-climate-processor'

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
            logGroupName='/aws/lambda/process-climate-data'
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
    bucket_name = os.environ.get('AWS_S3_BUCKET', 'climate-data-unknown')

    print("=" * 60)
    print("Climate Tier 2 - AWS Resource Cleanup")
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

# Then delete bucket
aws s3 rb s3://$BUCKET_NAME
```

### Problem: "Cannot delete role while it has attached policies"

```bash
# List all attached policies
aws iam list-attached-role-policies \
  --role-name lambda-climate-processor

# Detach each one
aws iam detach-role-policy \
  --role-name lambda-climate-processor \
  --policy-arn arn:aws:iam::aws:policy/AWSLambdaBasicExecutionRole

# Then delete the role
aws iam delete-role --role-name lambda-climate-processor
```

### Problem: "Lambda function still has permissions"

```bash
# Remove S3 invoke permission
aws lambda remove-permission \
  --function-name process-climate-data \
  --statement-id "AllowS3Invoke" || true

# Then delete function
aws lambda delete-function --function-name process-climate-data
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

# Should show minimal or zero charges for Lambda, S3
```

---

## Backup Your Code

Before cleanup, consider backing up:

1. **Scripts** - Copy `scripts/` folder locally
2. **Notebooks** - Copy `notebooks/` folder locally
3. **Results** - Download from S3 (see Step 1 above)
4. **Configuration** - Back up `.env` file (remove secrets first!)

```bash
# Create backup
mkdir -p backup-$(date +%Y%m%d)
cp -r scripts backup-$(date +%Y%m%d)/
cp -r notebooks backup-$(date +%Y%m%d)/
cp -r results backup-$(date +%Y%m%d)/
tar -czf backup-$(date +%Y%m%d).tar.gz backup-$(date +%Y%m%d)/
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

Only delete resources created for this specific project!

---

## Final Checklist

- [ ] Backed up results from S3
- [ ] Verified S3 bucket is deleted
- [ ] Verified Lambda function is deleted
- [ ] Verified IAM role is deleted
- [ ] Verified CloudWatch logs are deleted
- [ ] Checked AWS Cost Explorer shows no charges
- [ ] Set up billing alert (recommended)
- [ ] Saved project code locally

---

## Zero Out Budget

For extra safety, you can set AWS Budget to $0:

```bash
# Create zero budget
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget '{
    "BudgetName": "Climate-Tier2-Zero-Budget",
    "BudgetLimit": {"Amount": "0", "Unit": "USD"},
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
3. **Share results** - Document findings
4. **Keep code** - Version control your analysis scripts

---

## Need Help?

- Check AWS Cost Explorer: https://console.aws.amazon.com/cost-management/
- Review setup_guide.md for resource details
- See troubleshooting section above
- Contact AWS Support: https://console.aws.amazon.com/support/

---

**Important:** You are responsible for all AWS charges. Please verify all resources are deleted and costs have stopped.
