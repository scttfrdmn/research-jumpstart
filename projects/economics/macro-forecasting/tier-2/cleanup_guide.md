# AWS Resource Cleanup Guide

**Time Required:** 10-15 minutes

This guide helps you delete all AWS resources created for the Macroeconomic Forecasting project to avoid ongoing charges.

## Important: When to Clean Up

- **Immediately after completing the project** if you don't need the data
- **Within 7 days** to stay within the estimated $6-11 budget
- **After backing up any important results** you want to keep locally

## Cost Warning

Keeping resources active will incur charges:
- S3 storage: $0.023/GB per month
- DynamoDB storage: $0.25/GB per month
- Lambda invocations: $0.20 per 1M requests
- CloudWatch logs: $0.50/GB stored

**Deleting resources stops all charges immediately.**

---

## Automated Cleanup (Recommended)

### Option 1: Quick Cleanup Script

```bash
# Load your configuration
source aws_config.sh

# Run automated cleanup
python scripts/cleanup.py

# Verify all resources deleted
python scripts/verify_cleanup.py
```

---

## Manual Cleanup (Step-by-Step)

### Step 1: Delete DynamoDB Table (2 minutes)

```bash
# Load configuration
source aws_config.sh

# Delete DynamoDB table
aws dynamodb delete-table \
  --table-name ${TABLE_NAME} \
  --region us-east-1

echo "✓ DynamoDB table '${TABLE_NAME}' deleted"
```

**Verification:**
```bash
# This should return an error (table not found)
aws dynamodb describe-table \
  --table-name ${TABLE_NAME} \
  --region us-east-1
```

### Step 2: Delete S3 Buckets (3 minutes)

```bash
# Delete main data bucket contents
aws s3 rm s3://${BUCKET_NAME} --recursive

# Delete bucket
aws s3 rb s3://${BUCKET_NAME}

echo "✓ S3 bucket '${BUCKET_NAME}' deleted"

# If you created an Athena bucket, delete it too
if [ ! -z "${ATHENA_BUCKET}" ]; then
  aws s3 rm s3://${ATHENA_BUCKET} --recursive
  aws s3 rb s3://${ATHENA_BUCKET}
  echo "✓ Athena bucket '${ATHENA_BUCKET}' deleted"
fi
```

**Verification:**
```bash
# This should return an error (bucket not found)
aws s3 ls s3://${BUCKET_NAME}
```

### Step 3: Delete Lambda Function (2 minutes)

```bash
# Delete Lambda function
aws lambda delete-function \
  --function-name ${FUNCTION_NAME} \
  --region us-east-1

echo "✓ Lambda function '${FUNCTION_NAME}' deleted"
```

**Verification:**
```bash
# This should return an error (function not found)
aws lambda get-function \
  --function-name ${FUNCTION_NAME} \
  --region us-east-1
```

### Step 4: Delete Lambda Layer (1 minute)

```bash
# List layer versions
aws lambda list-layer-versions \
  --layer-name statsmodels \
  --region us-east-1

# Delete each version (usually just version 1)
aws lambda delete-layer-version \
  --layer-name statsmodels \
  --version-number 1 \
  --region us-east-1

echo "✓ Lambda layer 'statsmodels' deleted"
```

### Step 5: Delete IAM Policy and Role (2 minutes)

```bash
# Detach policy from role
aws iam detach-role-policy \
  --role-name ${ROLE_NAME} \
  --policy-arn ${POLICY_ARN}

# Delete policy
aws iam delete-policy \
  --policy-arn ${POLICY_ARN}

echo "✓ IAM policy deleted"

# Delete role
aws iam delete-role \
  --role-name ${ROLE_NAME}

echo "✓ IAM role '${ROLE_NAME}' deleted"
```

**Verification:**
```bash
# This should return an error (role not found)
aws iam get-role \
  --role-name ${ROLE_NAME}
```

### Step 6: Delete CloudWatch Log Groups (2 minutes)

```bash
# Delete Lambda function logs
aws logs delete-log-group \
  --log-group-name /aws/lambda/${FUNCTION_NAME} \
  --region us-east-1

echo "✓ CloudWatch log group deleted"
```

### Step 7: Delete Athena Workgroup (Optional, 1 minute)

```bash
# If you created an Athena workgroup
aws athena delete-work-group \
  --work-group economic-forecasting \
  --recursive-delete-option \
  --region us-east-1

echo "✓ Athena workgroup deleted"
```

---

## Verification Checklist

After cleanup, verify all resources are deleted:

### Check S3 Buckets
```bash
# Should not list your economic-data bucket
aws s3 ls | grep economic-data
```

### Check DynamoDB Tables
```bash
# Should not list EconomicForecasts
aws dynamodb list-tables --region us-east-1 | grep EconomicForecasts
```

### Check Lambda Functions
```bash
# Should not list forecast-economic-indicators
aws lambda list-functions --region us-east-1 | grep forecast-economic-indicators
```

### Check IAM Roles
```bash
# Should not list lambda-economic-forecaster
aws iam list-roles | grep lambda-economic-forecaster
```

### Check CloudWatch Logs
```bash
# Should not list Lambda log group
aws logs describe-log-groups \
  --log-group-name-prefix /aws/lambda/forecast-economic-indicators \
  --region us-east-1
```

---

## Cost Verification

### Final Cost Check (24 hours after cleanup)

```bash
# Check total costs for the project
aws ce get-cost-and-usage \
  --time-period Start=2025-11-01,End=2025-11-30 \
  --granularity MONTHLY \
  --metrics "BlendedCost" \
  --group-by Type=SERVICE \
  --region us-east-1
```

Expected costs (if cleaned up within 7 days):
- S3: $0.02 - $0.50
- Lambda: $0.50 - $1.50
- DynamoDB: $0.65 - $1.00
- CloudWatch: $0.05 - $0.20
- **Total: $1.22 - $3.20**

### Set Up Billing Alerts (Prevent Future Surprises)

```bash
# Navigate to AWS Billing Console
# https://console.aws.amazon.com/billing/home?#/budgets

# Create a budget:
# 1. Click "Create budget"
# 2. Choose "Cost budget"
# 3. Set amount: $10
# 4. Set alert threshold: 80% ($8)
# 5. Enter your email
# 6. Create budget
```

---

## Troubleshooting Cleanup

### Issue: "S3 bucket not empty"

**Solution:** Force delete all objects first
```bash
aws s3 rm s3://${BUCKET_NAME} --recursive --force
aws s3 rb s3://${BUCKET_NAME} --force
```

### Issue: "IAM policy is attached"

**Solution:** Detach from all roles first
```bash
# List attached entities
aws iam list-entities-for-policy \
  --policy-arn ${POLICY_ARN}

# Detach from each role
aws iam detach-role-policy \
  --role-name <role-name> \
  --policy-arn ${POLICY_ARN}

# Then delete policy
aws iam delete-policy --policy-arn ${POLICY_ARN}
```

### Issue: "Lambda function has event source mappings"

**Solution:** Remove S3 trigger first
```bash
# List S3 event notifications
aws s3api get-bucket-notification-configuration \
  --bucket ${BUCKET_NAME}

# Remove notification configuration
aws s3api put-bucket-notification-configuration \
  --bucket ${BUCKET_NAME} \
  --notification-configuration '{}'

# Then delete Lambda function
aws lambda delete-function --function-name ${FUNCTION_NAME}
```

### Issue: "DynamoDB table is being deleted"

**Solution:** Wait for deletion to complete
```bash
# Check status
aws dynamodb describe-table \
  --table-name ${TABLE_NAME} \
  --region us-east-1

# Wait up to 5 minutes for deletion
# Then proceed with other cleanup steps
```

---

## Backup Before Cleanup (Optional)

If you want to keep forecast results:

### Backup DynamoDB Data
```bash
# Export to JSON
python scripts/export_dynamodb.py \
  --table ${TABLE_NAME} \
  --output forecasts_backup.json

# Or use AWS CLI
aws dynamodb scan \
  --table-name ${TABLE_NAME} \
  --region us-east-1 \
  > forecasts_backup.json
```

### Backup S3 Data
```bash
# Download all data locally
aws s3 sync s3://${BUCKET_NAME}/ ./local_backup/

# Or download specific folders
aws s3 sync s3://${BUCKET_NAME}/raw/ ./local_backup/raw/
```

### Backup CloudWatch Logs
```bash
# Download recent logs
aws logs filter-log-events \
  --log-group-name /aws/lambda/${FUNCTION_NAME} \
  --start-time $(date -d '7 days ago' +%s)000 \
  --region us-east-1 \
  > lambda_logs_backup.json
```

---

## Alternative: Pause Resources (Not Recommended)

If you want to pause but not delete (NOT recommended due to costs):

### Disable Lambda Function
```bash
# This doesn't save money, Lambda only charges on invocations
# Better to delete and recreate later
```

### Enable S3 Glacier for Long-term Storage
```bash
# Move data to Glacier (cheaper storage)
cat > glacier_lifecycle.json << 'EOF'
{
  "Rules": [
    {
      "Id": "MoveToGlacier",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 1,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
  --bucket ${BUCKET_NAME} \
  --lifecycle-configuration file://glacier_lifecycle.json

# Note: This reduces storage cost to $0.004/GB per month
# But data retrieval is slow and expensive
```

---

## Re-running the Project Later

If you want to run the project again:

1. **Keep:** aws_config.sh file (save bucket/table names)
2. **Delete:** All AWS resources (follow this guide)
3. **Later:** Re-run setup_guide.md (10-15 minutes)
4. **Result:** Fresh resources, no accumulated costs

---

## Final Checklist

Before closing this guide, confirm:

- [ ] DynamoDB table deleted
- [ ] S3 buckets deleted (main + Athena)
- [ ] Lambda function deleted
- [ ] Lambda layer deleted
- [ ] IAM policy deleted
- [ ] IAM role deleted
- [ ] CloudWatch log groups deleted
- [ ] Athena workgroup deleted (if created)
- [ ] Cost verification completed (24 hours later)
- [ ] Backup saved locally (if needed)

---

## Getting Help

If you have issues deleting resources:

1. **Check AWS Console** for any remaining resources
   - S3: https://console.aws.amazon.com/s3/
   - DynamoDB: https://console.aws.amazon.com/dynamodb/
   - Lambda: https://console.aws.amazon.com/lambda/
   - IAM: https://console.aws.amazon.com/iam/

2. **AWS Support**
   - Free tier: Basic support included
   - Developer tier: $29/month (not needed for this)

3. **Cost Explorer**
   - https://console.aws.amazon.com/cost-management/home#/cost-explorer
   - View detailed cost breakdown by service

---

## Summary

You've successfully cleaned up all AWS resources!

**Time spent:** 10-15 minutes
**Money saved:** ~$2-5 per week (ongoing storage costs)
**Final project cost:** $6-11 total

To run the project again, follow setup_guide.md from scratch.

**Questions?** See project README.md or open a GitHub issue.
