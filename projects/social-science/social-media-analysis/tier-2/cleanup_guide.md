# Cleanup Guide - Delete AWS Resources

After completing the social media sentiment analysis project, delete all AWS resources to stop incurring charges.

**Important:** Once deleted, resources cannot be recovered. Ensure you've backed up any results you want to keep.

---

## Quick Cleanup (5 minutes)

```bash
# Set your bucket name
BUCKET_NAME="social-media-data-xxxx"  # Replace with your actual bucket name

# Delete S3 bucket and all contents
aws s3 rm s3://$BUCKET_NAME --recursive
aws s3 rb s3://$BUCKET_NAME

# Delete DynamoDB table
aws dynamodb delete-table --table-name SocialMediaPosts

# Delete Lambda function
aws lambda delete-function --function-name analyze-sentiment

# Delete IAM role policies and role
aws iam detach-role-policy \
  --role-name lambda-social-analysis \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam detach-role-policy \
  --role-name lambda-social-analysis \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam detach-role-policy \
  --role-name lambda-social-analysis \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

aws iam detach-role-policy \
  --role-name lambda-social-analysis \
  --policy-arn arn:aws:iam::aws:policy/ComprehendReadOnly

aws iam delete-role --role-name lambda-social-analysis

# Delete CloudWatch logs
aws logs delete-log-group --log-group-name /aws/lambda/analyze-sentiment

echo "All resources deleted!"
```

---

## Step-by-Step Deletion

If you prefer to delete resources individually or verify each step:

### Step 1: Back Up Results (Optional)

Save any important data before deletion:

```bash
# Download results from DynamoDB to local file
python scripts/query_results.py \
  --table-name SocialMediaPosts \
  --limit 1000 > results_backup.txt

# Download S3 data
aws s3 sync s3://$BUCKET_NAME/processed/ ./backups/processed/
aws s3 sync s3://$BUCKET_NAME/exports/ ./backups/exports/

# Verify backup
ls -lah backups/
```

### Step 2: Delete S3 Bucket

S3 buckets must be empty before deletion.

```bash
# List bucket contents
aws s3 ls s3://$BUCKET_NAME --recursive

# Delete all objects
aws s3 rm s3://$BUCKET_NAME --recursive

# Verify bucket is empty
aws s3 ls s3://$BUCKET_NAME

# Delete bucket
aws s3 rb s3://$BUCKET_NAME

# Verify deletion
aws s3 ls | grep social-media-data
# Should return nothing
```

### Step 3: Delete DynamoDB Table

```bash
# Delete table
aws dynamodb delete-table --table-name SocialMediaPosts

# Wait for deletion to complete
aws dynamodb wait table-not-exists --table-name SocialMediaPosts

# Verify deletion
aws dynamodb list-tables | grep SocialMediaPosts
# Should return nothing
```

### Step 4: Remove S3 Event Trigger

```bash
# Remove S3 event notification (if bucket still exists)
aws s3api put-bucket-notification-configuration \
  --bucket $BUCKET_NAME \
  --notification-configuration '{}'

# Note: This may fail if bucket is already deleted (that's OK)
```

### Step 5: Delete Lambda Function

```bash
# Remove Lambda permission for S3 trigger
aws lambda remove-permission \
  --function-name analyze-sentiment \
  --statement-id AllowS3Invoke

# Delete Lambda function
aws lambda delete-function --function-name analyze-sentiment

# Verify deletion
aws lambda get-function --function-name analyze-sentiment 2>&1
# Should return: "Function not found"
```

### Step 6: Delete IAM Role

IAM roles must have all policies detached before deletion.

```bash
# List attached policies
aws iam list-attached-role-policies --role-name lambda-social-analysis

# Detach managed policies
aws iam detach-role-policy \
  --role-name lambda-social-analysis \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam detach-role-policy \
  --role-name lambda-social-analysis \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam detach-role-policy \
  --role-name lambda-social-analysis \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

aws iam detach-role-policy \
  --role-name lambda-social-analysis \
  --policy-arn arn:aws:iam::aws:policy/ComprehendReadOnly

# Delete inline policies (if any)
for policy_name in $(aws iam list-role-policies \
  --role-name lambda-social-analysis \
  --query 'PolicyNames[]' --output text); do
  aws iam delete-role-policy \
    --role-name lambda-social-analysis \
    --policy-name $policy_name
done

# Delete the role
aws iam delete-role --role-name lambda-social-analysis

# Verify deletion
aws iam get-role --role-name lambda-social-analysis 2>&1
# Should return: "NoSuchEntity"
```

### Step 7: Delete CloudWatch Logs

CloudWatch logs persist after Lambda deletion and incur small storage costs.

```bash
# Delete log group
aws logs delete-log-group --log-group-name /aws/lambda/analyze-sentiment

# Verify deletion
aws logs describe-log-groups | grep analyze-sentiment
# Should return nothing
```

---

## Verification Checklist

Check that all resources are deleted:

```bash
echo "=== S3 Buckets ==="
aws s3 ls | grep social-media-data || echo "✓ No social-media-data buckets"

echo "=== DynamoDB Tables ==="
aws dynamodb list-tables | grep SocialMediaPosts || echo "✓ No SocialMediaPosts table"

echo "=== Lambda Functions ==="
aws lambda list-functions | grep analyze-sentiment || echo "✓ No analyze-sentiment function"

echo "=== IAM Roles ==="
aws iam list-roles | grep lambda-social-analysis || echo "✓ No lambda-social-analysis role"

echo "=== CloudWatch Logs ==="
aws logs describe-log-groups | grep analyze-sentiment || echo "✓ No analyze-sentiment logs"

echo ""
echo "All resources cleaned up!"
```

---

## Expected Remaining Charges

After cleanup, you should see minimal or zero charges:

| Service | Expected Charge |
|---------|----------------|
| S3 (if completely empty) | $0.00 |
| DynamoDB (after table deleted) | $0.00 |
| Lambda (no invocations) | $0.00 |
| Comprehend (already charged per use) | $0.00 |
| CloudWatch Logs (if deleted) | $0.00 |
| **Total** | **$0.00** |

**If cleanup is complete, no further charges should appear.**

---

## Troubleshooting Cleanup

### Issue: "Bucket is not empty"

```bash
# Force delete all versions and delete markers
aws s3api list-object-versions \
  --bucket $BUCKET_NAME \
  --query 'Versions[].{Key:Key,VersionId:VersionId}' \
  --output text | while read key version; do
  aws s3api delete-object \
    --bucket $BUCKET_NAME \
    --key "$key" \
    --version-id "$version"
done

# Delete all delete markers
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
aws s3 rb s3://$BUCKET_NAME --force
```

### Issue: "Cannot delete role - policies still attached"

```bash
# List ALL attached policies
aws iam list-attached-role-policies --role-name lambda-social-analysis

# Detach each one manually
# (Use the policy ARNs from the list above)
aws iam detach-role-policy \
  --role-name lambda-social-analysis \
  --policy-arn <POLICY_ARN>

# Then delete role
aws iam delete-role --role-name lambda-social-analysis
```

### Issue: "Table is being deleted" (DynamoDB)

```bash
# Wait for deletion to complete
aws dynamodb wait table-not-exists --table-name SocialMediaPosts

# This may take 1-2 minutes
```

---

## Cost Verification

After cleanup, monitor for 24-48 hours to ensure no charges:

```bash
# Check costs for the last 7 days
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '7 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics "UnblendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE

# Should show zero or minimal charges for:
# - Amazon Simple Storage Service (S3)
# - Amazon DynamoDB
# - AWS Lambda
# - Amazon Comprehend
```

---

## AWS Console Verification

Manually verify in AWS Console:

1. **S3**: https://console.aws.amazon.com/s3/
   - Search for "social-media-data"
   - Should find no buckets

2. **DynamoDB**: https://console.aws.amazon.com/dynamodb/
   - Check Tables
   - "SocialMediaPosts" should not exist

3. **Lambda**: https://console.aws.amazon.com/lambda/
   - Check Functions
   - "analyze-sentiment" should not exist

4. **IAM**: https://console.aws.amazon.com/iam/
   - Check Roles
   - "lambda-social-analysis" should not exist

5. **CloudWatch**: https://console.aws.amazon.com/cloudwatch/
   - Logs → Log groups
   - "/aws/lambda/analyze-sentiment" should not exist

6. **Billing**: https://console.aws.amazon.com/billing/
   - Cost Explorer
   - Verify charges are stopping

---

## Keep Local Backups

Before cleanup, save these locally:

```bash
# Create backup directory
mkdir -p backup-$(date +%Y%m%d)

# Copy scripts
cp -r scripts/ backup-$(date +%Y%m%d)/
cp -r notebooks/ backup-$(date +%Y%m%d)/

# Download results if not already done
python scripts/query_results.py > backup-$(date +%Y%m%d)/results.txt

# Create archive
tar -czf backup-$(date +%Y%m%d).tar.gz backup-$(date +%Y%m%d)/

echo "Backup created: backup-$(date +%Y%m%d).tar.gz"
```

---

## Final Checklist

- [ ] Backed up important results from DynamoDB
- [ ] Downloaded sample data from S3
- [ ] Deleted S3 bucket and all contents
- [ ] Deleted DynamoDB table
- [ ] Deleted Lambda function
- [ ] Deleted IAM role and policies
- [ ] Deleted CloudWatch log group
- [ ] Verified all resources deleted via CLI
- [ ] Verified all resources deleted via Console
- [ ] Set up billing alert (recommended)
- [ ] Checked Cost Explorer shows no charges

---

## Set Up Billing Alert

To avoid future unexpected charges:

```bash
# Create SNS topic for billing alerts
aws sns create-topic --name billing-alerts --region us-east-1

# Subscribe your email
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:$(aws sts get-caller-identity --query Account --output text):billing-alerts \
  --protocol email \
  --notification-endpoint your-email@example.com \
  --region us-east-1

# Confirm subscription via email
```

Then create a budget in AWS Console:
1. Go to: https://console.aws.amazon.com/billing/home#/budgets
2. Create budget → Cost budget
3. Amount: $5
4. Alert threshold: 80% ($4)
5. Email: your-email@example.com

---

## Getting Help

If you encounter issues during cleanup:

- AWS Support: https://console.aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- Check AWS documentation for each service
- GitHub Issues: https://github.com/research-jumpstart/research-jumpstart/issues

---

**Cleanup complete!** You should now have zero ongoing AWS charges from this project.

**Last updated:** 2025-11-14 | Research Jumpstart v1.0.0
