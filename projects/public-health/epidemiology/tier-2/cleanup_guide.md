# AWS Cleanup Guide - Disease Surveillance Pipeline

This guide provides step-by-step instructions for deleting all AWS resources created for the epidemiology project to avoid ongoing charges.

**Important:** Follow these steps carefully to ensure all resources are deleted.

**Estimated Time:** 10-15 minutes

---

## Table of Contents

1. [Pre-Cleanup Checklist](#1-pre-cleanup-checklist)
2. [Delete S3 Bucket Contents](#2-delete-s3-bucket-contents)
3. [Delete Lambda Function](#3-delete-lambda-function)
4. [Delete SNS Topic](#4-delete-sns-topic)
5. [Delete DynamoDB Table](#5-delete-dynamodb-table)
6. [Delete IAM Role](#6-delete-iam-role)
7. [Delete Athena Resources (Optional)](#7-delete-athena-resources-optional)
8. [Delete CloudWatch Logs](#8-delete-cloudwatch-logs)
9. [Verify All Resources Deleted](#9-verify-all-resources-deleted)
10. [Final Cost Check](#10-final-cost-check)

---

## 1. Pre-Cleanup Checklist

Before deleting resources, ensure you have:

- [ ] Downloaded any results or data you need to keep
- [ ] Saved analysis notebooks and visualizations locally
- [ ] Exported any important queries or configurations
- [ ] Noted the names of all resources you created

### Get Your Resource Names

```bash
# Set these variables (or retrieve from setup)
source ~/.bashrc
echo "Bucket: ${EPIDEMIOLOGY_BUCKET}"
echo "Topic ARN: ${OUTBREAK_TOPIC_ARN}"
```

If variables are not set, find resources manually:

```bash
# List S3 buckets
aws s3 ls | grep epidemiology

# List Lambda functions
aws lambda list-functions --query 'Functions[?contains(FunctionName, `epidemiology`) || contains(FunctionName, `surveillance`)].FunctionName'

# List DynamoDB tables
aws dynamodb list-tables --query 'TableNames[?contains(@, `Disease`)]'

# List SNS topics
aws sns list-topics --query 'Topics[?contains(TopicArn, `outbreak`)]'
```

---

## 2. Delete S3 Bucket Contents

**Important:** S3 buckets must be empty before deletion.

### 2.1 List Bucket Contents

```bash
aws s3 ls s3://${EPIDEMIOLOGY_BUCKET}/ --recursive
```

### 2.2 Delete All Objects (Including Versions)

```bash
# Delete all objects
aws s3 rm s3://${EPIDEMIOLOGY_BUCKET}/ --recursive

# If versioning was enabled, delete all versions
aws s3api delete-objects \
    --bucket ${EPIDEMIOLOGY_BUCKET} \
    --delete "$(aws s3api list-object-versions \
        --bucket ${EPIDEMIOLOGY_BUCKET} \
        --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' \
        --max-items 1000)"

# Delete all delete markers (if versioning enabled)
aws s3api delete-objects \
    --bucket ${EPIDEMIOLOGY_BUCKET} \
    --delete "$(aws s3api list-object-versions \
        --bucket ${EPIDEMIOLOGY_BUCKET} \
        --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' \
        --max-items 1000)"
```

### 2.3 Delete the Bucket

```bash
aws s3 rb s3://${EPIDEMIOLOGY_BUCKET}
```

### 2.4 Verify Bucket Deleted

```bash
aws s3 ls s3://${EPIDEMIOLOGY_BUCKET}/ 2>&1 | grep "NoSuchBucket" && echo "✓ Bucket deleted successfully"
```

### 2.5 Delete Athena Output Bucket (if created)

```bash
# If you created Athena output bucket
ATHENA_OUTPUT_BUCKET=$(aws s3 ls | grep athena-results | awk '{print $3}')

if [ ! -z "$ATHENA_OUTPUT_BUCKET" ]; then
    echo "Deleting Athena output bucket: ${ATHENA_OUTPUT_BUCKET}"
    aws s3 rm s3://${ATHENA_OUTPUT_BUCKET}/ --recursive
    aws s3 rb s3://${ATHENA_OUTPUT_BUCKET}
    echo "✓ Athena output bucket deleted"
fi
```

---

## 3. Delete Lambda Function

### 3.1 Remove S3 Event Trigger

First, remove the S3 bucket notification (prevents errors):

```bash
aws s3api put-bucket-notification-configuration \
    --bucket ${EPIDEMIOLOGY_BUCKET} \
    --notification-configuration '{}'
```

**Note:** This may fail if bucket is already deleted (that's okay).

### 3.2 Delete Lambda Function

```bash
aws lambda delete-function --function-name analyze-disease-surveillance
```

### 3.3 Verify Deletion

```bash
aws lambda get-function --function-name analyze-disease-surveillance 2>&1 | grep "ResourceNotFoundException" && echo "✓ Lambda function deleted successfully"
```

---

## 4. Delete SNS Topic

### 4.1 List Subscriptions

```bash
aws sns list-subscriptions-by-topic --topic-arn ${OUTBREAK_TOPIC_ARN}
```

### 4.2 Unsubscribe All Subscriptions

```bash
# Get all subscription ARNs
SUBSCRIPTIONS=$(aws sns list-subscriptions-by-topic \
    --topic-arn ${OUTBREAK_TOPIC_ARN} \
    --query 'Subscriptions[].SubscriptionArn' \
    --output text)

# Unsubscribe each
for sub in $SUBSCRIPTIONS; do
    if [ "$sub" != "PendingConfirmation" ]; then
        aws sns unsubscribe --subscription-arn $sub
        echo "✓ Unsubscribed: $sub"
    fi
done
```

### 4.3 Delete Topic

```bash
aws sns delete-topic --topic-arn ${OUTBREAK_TOPIC_ARN}
```

### 4.4 Verify Deletion

```bash
aws sns get-topic-attributes --topic-arn ${OUTBREAK_TOPIC_ARN} 2>&1 | grep "NotFound" && echo "✓ SNS topic deleted successfully"
```

---

## 5. Delete DynamoDB Table

### 5.1 Check Table Size (Optional)

```bash
aws dynamodb describe-table --table-name DiseaseReports \
    --query 'Table.[TableSizeBytes, ItemCount]'
```

### 5.2 Delete Table

```bash
aws dynamodb delete-table --table-name DiseaseReports
```

### 5.3 Wait for Deletion

```bash
aws dynamodb wait table-not-exists --table-name DiseaseReports
echo "✓ DynamoDB table deleted successfully"
```

### 5.4 Verify Deletion

```bash
aws dynamodb list-tables | grep DiseaseReports || echo "✓ Table not found (deleted)"
```

---

## 6. Delete IAM Role

### 6.1 List Attached Policies

```bash
aws iam list-role-policies --role-name lambda-epidemiology-role
```

### 6.2 Delete Inline Policies

```bash
aws iam delete-role-policy \
    --role-name lambda-epidemiology-role \
    --policy-name lambda-epidemiology-policy
```

### 6.3 Detach Managed Policies (if any)

```bash
# List managed policies
MANAGED_POLICIES=$(aws iam list-attached-role-policies \
    --role-name lambda-epidemiology-role \
    --query 'AttachedPolicies[].PolicyArn' \
    --output text)

# Detach each
for policy in $MANAGED_POLICIES; do
    aws iam detach-role-policy \
        --role-name lambda-epidemiology-role \
        --policy-arn $policy
    echo "✓ Detached policy: $policy"
done
```

### 6.4 Delete Role

```bash
aws iam delete-role --role-name lambda-epidemiology-role
```

### 6.5 Verify Deletion

```bash
aws iam get-role --role-name lambda-epidemiology-role 2>&1 | grep "NoSuchEntity" && echo "✓ IAM role deleted successfully"
```

---

## 7. Delete Athena Resources (Optional)

If you set up Athena:

### 7.1 Delete Athena Database

```bash
aws athena start-query-execution \
    --query-string "DROP DATABASE IF EXISTS epidemiology_db CASCADE" \
    --work-group epidemiology-workgroup
```

### 7.2 Delete Workgroup

```bash
aws athena delete-work-group \
    --work-group epidemiology-workgroup \
    --recursive-delete-option
```

### 7.3 Verify Deletion

```bash
aws athena list-work-groups | grep epidemiology-workgroup || echo "✓ Athena workgroup deleted"
```

---

## 8. Delete CloudWatch Logs

### 8.1 List Log Groups

```bash
aws logs describe-log-groups \
    --log-group-name-prefix /aws/lambda/analyze-disease-surveillance
```

### 8.2 Delete Lambda Log Group

```bash
aws logs delete-log-group \
    --log-group-name /aws/lambda/analyze-disease-surveillance
```

### 8.3 Delete Other Related Log Groups (if any)

```bash
# List all log groups that might be related
aws logs describe-log-groups --query 'logGroups[?contains(logGroupName, `epidemiology`) || contains(logGroupName, `surveillance`)].logGroupName'

# Delete each manually if found
# aws logs delete-log-group --log-group-name <log-group-name>
```

### 8.4 Verify Deletion

```bash
aws logs describe-log-groups \
    --log-group-name-prefix /aws/lambda/analyze-disease-surveillance 2>&1 | \
    grep "ResourceNotFoundException" && echo "✓ Log groups deleted successfully"
```

---

## 9. Verify All Resources Deleted

### 9.1 Run Complete Verification

```bash
echo "=== Cleanup Verification ==="
echo ""

# S3 Buckets
echo "Checking S3 buckets..."
aws s3 ls | grep epidemiology && echo "⚠ WARNING: S3 bucket still exists!" || echo "✓ S3 buckets deleted"
echo ""

# Lambda Functions
echo "Checking Lambda functions..."
aws lambda list-functions --query 'Functions[?contains(FunctionName, `surveillance`)].FunctionName' --output text
[ -z "$(aws lambda list-functions --query 'Functions[?contains(FunctionName, `surveillance`)].FunctionName' --output text)" ] && echo "✓ Lambda functions deleted" || echo "⚠ WARNING: Lambda function still exists!"
echo ""

# DynamoDB Tables
echo "Checking DynamoDB tables..."
aws dynamodb list-tables | grep DiseaseReports && echo "⚠ WARNING: DynamoDB table still exists!" || echo "✓ DynamoDB tables deleted"
echo ""

# SNS Topics
echo "Checking SNS topics..."
aws sns list-topics | grep outbreak && echo "⚠ WARNING: SNS topic still exists!" || echo "✓ SNS topics deleted"
echo ""

# IAM Roles
echo "Checking IAM roles..."
aws iam get-role --role-name lambda-epidemiology-role 2>&1 | grep "NoSuchEntity" > /dev/null && echo "✓ IAM roles deleted" || echo "⚠ WARNING: IAM role still exists!"
echo ""

# CloudWatch Logs
echo "Checking CloudWatch logs..."
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/analyze-disease-surveillance 2>&1 | grep "ResourceNotFoundException" > /dev/null && echo "✓ CloudWatch logs deleted" || echo "⚠ WARNING: Log groups still exist!"
echo ""

echo "=== Cleanup Complete ==="
```

### 9.2 Check for Orphaned Resources

```bash
# Search for any resources with 'epidemiology' or 'surveillance' in name
echo "Searching for orphaned resources..."
echo ""

echo "EC2 instances (should be none):"
aws ec2 describe-instances --filters "Name=tag:Project,Values=*epidemiology*" --query 'Reservations[].Instances[].InstanceId'
echo ""

echo "Security groups (default VPC only):"
aws ec2 describe-security-groups --filters "Name=group-name,Values=*epidemiology*" --query 'SecurityGroups[].GroupId'
echo ""

echo "If any resources found above, delete them manually in AWS Console."
```

---

## 10. Final Cost Check

### 10.1 Check Current Month Costs

Wait 24-48 hours after cleanup, then check costs:

```bash
aws ce get-cost-and-usage \
    --time-period Start=$(date -u +%Y-%m-01),End=$(date -u +%Y-%m-%d) \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=SERVICE
```

### 10.2 Expected Costs

**During project (before cleanup):** $6-10 total

**After cleanup:** $0-2 (residual charges for data transfer, storage before cleanup)

**Next month:** $0 (all resources deleted)

### 10.3 Set Up Cost Alert (Recommended)

Ensure you have billing alerts configured:

1. Go to AWS Console → Billing → Budgets
2. Verify budget alert exists for $10
3. If costs exceed expected, investigate in Cost Explorer

### 10.4 Final Verification in Console

Log in to AWS Console and manually check:

1. **S3**: https://console.aws.amazon.com/s3/
   - Search for "epidemiology"
   - Should find no buckets

2. **Lambda**: https://console.aws.amazon.com/lambda/
   - Search for "surveillance"
   - Should find no functions

3. **DynamoDB**: https://console.aws.amazon.com/dynamodb/
   - Search for "DiseaseReports"
   - Should find no tables

4. **SNS**: https://console.aws.amazon.com/sns/
   - Check topics
   - Should not see "outbreak-alerts"

5. **IAM**: https://console.aws.amazon.com/iam/
   - Roles → Search "epidemiology"
   - Should find no roles

6. **CloudWatch**: https://console.aws.amazon.com/cloudwatch/
   - Logs → Log groups
   - Should not see "/aws/lambda/analyze-disease-surveillance"

---

## Troubleshooting Cleanup Issues

### Issue: Cannot delete S3 bucket (not empty)

**Solution:**
```bash
# Force delete all versions
aws s3api list-object-versions --bucket ${EPIDEMIOLOGY_BUCKET} --output json | \
jq '.Versions[], .DeleteMarkers[] | {Key:.Key, VersionId:.VersionId}' | \
jq -s '{Objects: .}' | \
aws s3api delete-objects --bucket ${EPIDEMIOLOGY_BUCKET} --delete file:///dev/stdin

# Then delete bucket
aws s3 rb s3://${EPIDEMIOLOGY_BUCKET}
```

### Issue: IAM role has active sessions

**Solution:**
```bash
# Wait 5 minutes for sessions to expire
sleep 300

# Then try deleting again
aws iam delete-role --role-name lambda-epidemiology-role
```

### Issue: DynamoDB table stuck in "DELETING" state

**Solution:**
```bash
# Wait for deletion to complete
aws dynamodb wait table-not-exists --table-name DiseaseReports

# This can take up to 5 minutes
```

### Issue: Lambda function has active triggers

**Solution:**
```bash
# Remove all event source mappings
MAPPINGS=$(aws lambda list-event-source-mappings \
    --function-name analyze-disease-surveillance \
    --query 'EventSourceMappings[].UUID' --output text)

for uuid in $MAPPINGS; do
    aws lambda delete-event-source-mapping --uuid $uuid
done

# Then delete function
aws lambda delete-function --function-name analyze-disease-surveillance
```

### Issue: CloudWatch log group won't delete

**Solution:**
```bash
# Check for active log streams
aws logs describe-log-streams \
    --log-group-name /aws/lambda/analyze-disease-surveillance

# Delete log group with force
aws logs delete-log-group \
    --log-group-name /aws/lambda/analyze-disease-surveillance
```

---

## Alternative: Automated Cleanup Script

Save this as `cleanup_all.sh`:

```bash
#!/bin/bash
set -e

echo "🧹 Starting automated cleanup..."
echo ""

# Source environment variables
source ~/.bashrc

# Delete S3 bucket
echo "1/7 Deleting S3 bucket..."
aws s3 rm s3://${EPIDEMIOLOGY_BUCKET}/ --recursive
aws s3 rb s3://${EPIDEMIOLOGY_BUCKET}
echo "✓ S3 bucket deleted"

# Delete Lambda function
echo "2/7 Deleting Lambda function..."
aws lambda delete-function --function-name analyze-disease-surveillance
echo "✓ Lambda function deleted"

# Delete SNS topic
echo "3/7 Deleting SNS topic..."
aws sns delete-topic --topic-arn ${OUTBREAK_TOPIC_ARN}
echo "✓ SNS topic deleted"

# Delete DynamoDB table
echo "4/7 Deleting DynamoDB table..."
aws dynamodb delete-table --table-name DiseaseReports
aws dynamodb wait table-not-exists --table-name DiseaseReports
echo "✓ DynamoDB table deleted"

# Delete IAM role
echo "5/7 Deleting IAM role..."
aws iam delete-role-policy --role-name lambda-epidemiology-role --policy-name lambda-epidemiology-policy
aws iam delete-role --role-name lambda-epidemiology-role
echo "✓ IAM role deleted"

# Delete CloudWatch logs
echo "6/7 Deleting CloudWatch logs..."
aws logs delete-log-group --log-group-name /aws/lambda/analyze-disease-surveillance || true
echo "✓ CloudWatch logs deleted"

# Delete Athena resources (if exist)
echo "7/7 Deleting Athena resources (if any)..."
aws athena delete-work-group --work-group epidemiology-workgroup --recursive-delete-option || true
echo "✓ Athena resources deleted"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Please verify in AWS Console and check costs in 24-48 hours."
```

Run with:
```bash
chmod +x cleanup_all.sh
./cleanup_all.sh
```

---

## Summary

You have successfully deleted:

- ✅ S3 bucket and all objects
- ✅ Lambda function
- ✅ SNS topic and subscriptions
- ✅ DynamoDB table
- ✅ IAM role and policies
- ✅ CloudWatch log groups
- ✅ (Optional) Athena workgroup and database

**Expected Cost After Cleanup:** $0/month

**Important:** Check AWS Cost Explorer in 24-48 hours to verify no unexpected charges.

**Note:** Some services may have minimal charges for the partial month before cleanup. These should not exceed $1-2.

---

## Questions?

If you encounter issues during cleanup:

1. Check the troubleshooting section above
2. Verify resources in AWS Console manually
3. Contact AWS Support if charges continue after cleanup
4. Review AWS Cost Explorer for detailed breakdown

**Cleanup completed successfully!** 🎉
