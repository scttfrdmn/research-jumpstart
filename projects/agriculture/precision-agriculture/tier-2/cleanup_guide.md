# Cleanup Guide

Delete all AWS resources after completing the project to avoid unexpected charges.

**Time Estimate:** 5-10 minutes

## IMPORTANT: Cost Prevention

AWS charges continue as long as resources exist:
- S3 storage: ~$0.023 per GB per month
- Lambda: Minimal charge but adds up
- CloudWatch Logs: $0.50 per GB stored

**Total potential cost if left running:** $2-5 per month

## Automated Cleanup

### Option 1: Python Script (Recommended)

Create a cleanup script:

```bash
cat > cleanup.py << 'EOF'
#!/usr/bin/env python3
import boto3
import os
import sys

def cleanup_aws_resources():
    """Delete all resources created by the project"""

    s3 = boto3.client('s3')
    lambda_client = boto3.client('lambda')
    iam = boto3.client('iam')
    logs = boto3.client('logs')

    bucket_name = open('bucket_name.txt').read().strip()

    print("AWS Resource Cleanup")
    print("=" * 50)

    # Step 1: Empty S3 bucket
    print(f"\n1. Deleting S3 bucket contents: {bucket_name}")
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)

        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    print(f"   Deleting: {obj['Key']}")
                    s3.delete_object(Bucket=bucket_name, Key=obj['Key'])
        print("   ✓ S3 bucket emptied")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Step 2: Delete S3 bucket
    print(f"\n2. Deleting S3 bucket: {bucket_name}")
    try:
        s3.delete_bucket(Bucket=bucket_name)
        print("   ✓ S3 bucket deleted")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Step 3: Delete S3 event notification
    print(f"\n3. Removing S3 event notifications")
    try:
        s3.put_bucket_notification_configuration(
            Bucket=bucket_name,
            NotificationConfiguration={}
        )
        print("   ✓ Event notifications removed")
    except Exception as e:
        print(f"   ⚠ Already deleted or error: {e}")

    # Step 4: Delete Lambda function
    print("\n4. Deleting Lambda function: process-ndvi-calculation")
    try:
        lambda_client.delete_function(FunctionName='process-ndvi-calculation')
        print("   ✓ Lambda function deleted")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Step 5: Remove Lambda permission
    print("\n5. Removing Lambda S3 permission")
    try:
        lambda_client.remove_permission(
            FunctionName='process-ndvi-calculation',
            StatementId='AllowExecutionFromS3'
        )
        print("   ✓ Lambda permission removed")
    except Exception as e:
        print(f"   ⚠ Already removed: {e}")

    # Step 6: Delete CloudWatch Logs
    print("\n6. Deleting CloudWatch log group")
    try:
        logs.delete_log_group(logGroupName='/aws/lambda/process-ndvi-calculation')
        print("   ✓ CloudWatch log group deleted")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Step 7: Delete IAM role inline policies
    print("\n7. Deleting IAM inline policies")
    try:
        iam.delete_role_policy(
            RoleName='lambda-ndvi-processor',
            PolicyName='S3Access'
        )
        print("   ✓ S3Access policy deleted")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Step 8: Detach IAM managed policies
    print("\n8. Detaching IAM managed policies")
    try:
        iam.detach_role_policy(
            RoleName='lambda-ndvi-processor',
            PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
        )
        print("   ✓ AWSLambdaBasicExecutionRole detached")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Step 9: Delete IAM role
    print("\n9. Deleting IAM role: lambda-ndvi-processor")
    try:
        iam.delete_role(RoleName='lambda-ndvi-processor')
        print("   ✓ IAM role deleted")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 50)
    print("Cleanup complete!")
    print("\nVerify deletion:")
    print("1. AWS Console -> S3: Verify bucket is gone")
    print("2. AWS Console -> Lambda: Verify function is gone")
    print("3. AWS Console -> IAM: Verify role is gone")
    print("4. AWS Console -> CloudWatch Logs: Verify log group is gone")
    print("\nNo further charges will accrue.")

if __name__ == '__main__':
    try:
        cleanup_aws_resources()
    except Exception as e:
        print(f"Cleanup failed: {e}")
        sys.exit(1)
EOF

# Run cleanup script
python cleanup.py
```

## Manual Cleanup (Step-by-Step)

If the automated script doesn't work, follow these steps manually:

### Step 1: Delete S3 Bucket

```bash
# Load bucket name
BUCKET_NAME=$(cat bucket_name.txt)

# List all objects
aws s3 ls s3://$BUCKET_NAME --recursive

# Empty bucket (delete all objects)
aws s3 rm s3://$BUCKET_NAME --recursive

# Delete bucket
aws s3 rb s3://$BUCKET_NAME
```

### Step 2: Delete Lambda Function

```bash
aws lambda delete-function --function-name process-ndvi-calculation
```

### Step 3: Remove Lambda Permission

```bash
aws lambda remove-permission \
  --function-name process-ndvi-calculation \
  --statement-id AllowExecutionFromS3 2>/dev/null || true
```

### Step 4: Delete CloudWatch Logs

```bash
aws logs delete-log-group --log-group-name /aws/lambda/process-ndvi-calculation
```

### Step 5: Delete IAM Inline Policy

```bash
aws iam delete-role-policy \
  --role-name lambda-ndvi-processor \
  --policy-name S3Access
```

### Step 6: Detach Managed Policies

```bash
aws iam detach-role-policy \
  --role-name lambda-ndvi-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

### Step 7: Delete IAM Role

```bash
aws iam delete-role --role-name lambda-ndvi-processor
```

### Step 8: Clean Up Local Files

```bash
# Remove temporary AWS setup files
rm -f lambda_function.zip
rm -f s3-event-notification.json
rm -f trust-policy.json
rm -f s3-policy.json
rm -f create_table.sql
rm -f lambda_payload.py
rm -f bucket_name.txt
rm -f lambda_role_arn.txt
rm -f aws_resources.txt
rm -rf lambda_package/
```

---

## Verification

After cleanup, verify all resources are deleted:

### AWS CLI Verification

```bash
# Check S3 buckets - should not see satellite-imagery
aws s3 ls | grep satellite-imagery

# Check Lambda functions - should be empty or not exist
aws lambda list-functions --query 'Functions[?FunctionName==`process-ndvi-calculation`]'

# Check IAM roles - should not exist
aws iam get-role --role-name lambda-ndvi-processor 2>&1 | grep "NoSuchEntity"

# Check CloudWatch logs - should not exist
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/process-ndvi-calculation" | grep logGroups
```

### AWS Console Verification

1. **S3:** Go to S3 Dashboard → Buckets → Verify `satellite-imagery-*` is gone
2. **Lambda:** Go to Lambda → Functions → Verify `process-ndvi-calculation` is gone
3. **IAM:** Go to IAM → Roles → Verify `lambda-ndvi-processor` is gone
4. **CloudWatch:** Go to CloudWatch → Logs → Verify `/aws/lambda/process-ndvi-calculation` is gone

**All resources deleted = No more charges!**

---

## Common Issues

### "Access Denied" Errors During Cleanup

**Solution:** Ensure your IAM user has permissions to delete resources. Contact your AWS administrator if needed.

### S3 Bucket Won't Delete

**Check for remaining objects:**
```bash
aws s3 ls s3://satellite-imagery-XXXXX --recursive --summarize
```

**Empty versioned buckets:**
```bash
# Delete all object versions
aws s3api list-object-versions --bucket satellite-imagery-XXXXX \
  --query 'Versions[].{Key:Key, VersionId:VersionId}' \
  --output text | xargs -I {} \
    aws s3api delete-object --bucket satellite-imagery-XXXXX --key {} --version-id {}
```

### Lambda Function Won't Delete

**Check if function is referenced elsewhere:**
```bash
# List all Lambda event source mappings
aws lambda list-event-source-mappings --function-name process-ndvi-calculation
```

### CloudWatch Logs Won't Delete

```bash
# List log groups matching pattern
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/"

# Force delete
aws logs delete-log-group --log-group-name "/aws/lambda/process-ndvi-calculation" --force
```

---

## Cost Summary

**What You'll Save by Cleaning Up:**

| Resource | Monthly Cost | Annual Cost |
|----------|--------------|-------------|
| S3 Bucket (200MB) | $0.01 | $0.12 |
| Lambda Invocations | $0.00 | $0.00 |
| CloudWatch Logs (100MB) | $0.05 | $0.60 |
| **Total** | **$0.06** | **$0.72** |

Small savings, but it's good practice to clean up after projects!

---

## Next Steps After Cleanup

### Option 1: Archive Your Results
Before deleting S3 bucket, download your results locally:

```bash
# Download all results locally
aws s3 sync s3://$(cat bucket_name.txt)/results/ ./results_backup/

# Save results to archive
zip -r results_archive.zip results_backup/
```

### Option 2: Delete and Start Fresh
Once verified deleted, you can create new resources and run the project again with a fresh budget.

### Option 3: Keep Project Running (Not Recommended)
If you want to continue with experiments, you can keep resources running but:
- Set S3 Intelligent-Tiering to reduce storage costs
- Set up AWS Budget Alerts to monitor spending
- Review CloudWatch metrics to optimize

---

## Troubleshooting Cleanup

**Still seeing charges?**

1. Check AWS Billing Dashboard for resource summary
2. Verify resources deleted in AWS Console
3. Look for related resources (e.g., backup S3 buckets, old Lambda layers)
4. Contact AWS Support if resources won't delete

---

**Cleanup complete? Great!** Your AWS account is clean and ready for the next project.

For questions or issues, refer to the [README.md](README.md) troubleshooting section.

---

**AWS Cost Awareness = Happy Budget!**
