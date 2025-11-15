# Cleanup Guide - Delete AWS Resources

This guide helps you delete all AWS resources created for the Corpus Linguistics Tier 2 project to avoid ongoing charges.

**Important:** Follow steps in order to avoid dependency errors.

**Total cleanup time:** 10-15 minutes

---

## Cost After Cleanup

Once all resources are deleted:
- **S3 charges:** $0 (no stored data)
- **DynamoDB charges:** $0 (no table)
- **Lambda charges:** $0 (no function)
- **Total ongoing cost:** $0

---

## Quick Cleanup (Automated)

If you want to delete everything quickly:

```bash
# Set your bucket name
export BUCKET_NAME="linguistic-corpus-YOUR_ID"

# Delete S3 bucket and contents
aws s3 rm s3://$BUCKET_NAME --recursive
aws s3 rb s3://$BUCKET_NAME

# Delete Lambda function
aws lambda delete-function --function-name analyze-linguistic-corpus

# Delete DynamoDB table
aws dynamodb delete-table --table-name LinguisticAnalysis

# Delete IAM role policies and role
aws iam list-role-policies --role-name lambda-linguistic-processor --query 'PolicyNames[]' --output text | \
  xargs -I {} aws iam delete-role-policy --role-name lambda-linguistic-processor --policy-name {}

aws iam list-attached-role-policies --role-name lambda-linguistic-processor --query 'AttachedPolicies[].PolicyArn' --output text | \
  xargs -I {} aws iam detach-role-policy --role-name lambda-linguistic-processor --policy-arn {}

aws iam delete-role --role-name lambda-linguistic-processor

echo "Cleanup complete!"
```

---

## Step-by-Step Cleanup

Follow these steps for detailed control and verification.

### Step 1: Delete S3 Bucket (5 minutes)

**Why first?** S3 triggers must be removed before deleting Lambda.

#### 1.1 Remove S3 Event Notification (if configured)

```bash
# Remove S3 notification configuration
aws s3api put-bucket-notification-configuration \
    --bucket $BUCKET_NAME \
    --notification-configuration '{}'

# Verify removal
aws s3api get-bucket-notification-configuration --bucket $BUCKET_NAME
```

**Expected output:** `{}` (empty configuration)

#### 1.2 Delete All Objects in Bucket

```bash
# List objects to be deleted
aws s3 ls s3://$BUCKET_NAME --recursive

# Delete all objects
aws s3 rm s3://$BUCKET_NAME --recursive

# Verify bucket is empty
aws s3 ls s3://$BUCKET_NAME
```

**Expected output:** No objects listed

#### 1.3 Delete Bucket

```bash
# Delete the bucket itself
aws s3 rb s3://$BUCKET_NAME

# Verify deletion
aws s3 ls | grep linguistic-corpus
```

**Expected output:** No buckets matching name

#### Via AWS Console:

1. Open https://console.aws.amazon.com/s3/
2. Find your bucket (e.g., `linguistic-corpus-123`)
3. Click on bucket name
4. Click **Empty** button
5. Type "permanently delete" to confirm
6. Click **Empty**
7. Go back to S3 buckets list
8. Select bucket checkbox
9. Click **Delete**
10. Type bucket name to confirm
11. Click **Delete bucket**

---

### Step 2: Delete Lambda Function (2 minutes)

#### 2.1 Remove Lambda Permissions

```bash
# List Lambda permissions
aws lambda get-policy --function-name analyze-linguistic-corpus

# Remove S3 invoke permission (if exists)
aws lambda remove-permission \
    --function-name analyze-linguistic-corpus \
    --statement-id s3-invoke-lambda
```

#### 2.2 Delete Lambda Function

```bash
# Delete the function
aws lambda delete-function --function-name analyze-linguistic-corpus

# Verify deletion
aws lambda list-functions | grep analyze-linguistic-corpus
```

**Expected output:** No functions found

#### Via AWS Console:

1. Open https://console.aws.amazon.com/lambda/
2. Find function `analyze-linguistic-corpus`
3. Click on function name
4. Click **Actions** → **Delete function**
5. Type "delete" to confirm
6. Click **Delete**

---

### Step 3: Delete DynamoDB Table (2 minutes)

#### 3.1 (Optional) Export Data Before Deletion

```bash
# Export all items to JSON
aws dynamodb scan --table-name LinguisticAnalysis > corpus_backup.json

echo "Data backed up to corpus_backup.json"
```

#### 3.2 Delete Table

```bash
# Delete the table
aws dynamodb delete-table --table-name LinguisticAnalysis

# Wait for deletion to complete
aws dynamodb wait table-not-exists --table-name LinguisticAnalysis

# Verify deletion
aws dynamodb list-tables | grep LinguisticAnalysis
```

**Expected output:** No tables found

#### Via AWS Console:

1. Open https://console.aws.amazon.com/dynamodb/
2. Click **Tables** in left sidebar
3. Find table `LinguisticAnalysis`
4. Select table checkbox
5. Click **Delete**
6. Check "Delete all CloudWatch alarms for this table"
7. Type "delete" to confirm
8. Click **Delete table**

---

### Step 4: Delete IAM Role (3 minutes)

**Note:** IAM roles cannot be deleted if they have attached policies or inline policies.

#### 4.1 List Attached Policies

```bash
# List attached managed policies
aws iam list-attached-role-policies --role-name lambda-linguistic-processor

# List inline policies
aws iam list-role-policies --role-name lambda-linguistic-processor
```

#### 4.2 Detach Managed Policies

```bash
# Detach AWS managed policy
aws iam detach-role-policy \
    --role-name lambda-linguistic-processor \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Verify detachment
aws iam list-attached-role-policies --role-name lambda-linguistic-processor
```

**Expected output:** Empty list

#### 4.3 Delete Inline Policies

```bash
# Delete inline policy
aws iam delete-role-policy \
    --role-name lambda-linguistic-processor \
    --policy-name linguistic-processor-policy

# Verify deletion
aws iam list-role-policies --role-name lambda-linguistic-processor
```

**Expected output:** Empty list

#### 4.4 Delete IAM Role

```bash
# Delete the role
aws iam delete-role --role-name lambda-linguistic-processor

# Verify deletion
aws iam list-roles | grep lambda-linguistic-processor
```

**Expected output:** No roles found

#### Via AWS Console:

1. Open https://console.aws.amazon.com/iam/
2. Click **Roles** in left sidebar
3. Search for `lambda-linguistic-processor`
4. Click on role name
5. Click **Delete role**
6. Type role name to confirm
7. Click **Delete**

---

### Step 5: Delete CloudWatch Logs (Optional) (1 minute)

Lambda automatically creates CloudWatch log groups that may incur small charges.

```bash
# List Lambda log groups
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/analyze-linguistic-corpus

# Delete log group
aws logs delete-log-group --log-group-name /aws/lambda/analyze-linguistic-corpus

# Verify deletion
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/analyze-linguistic-corpus
```

**Expected output:** No log groups found

#### Via AWS Console:

1. Open https://console.aws.amazon.com/cloudwatch/
2. Click **Logs** → **Log groups** in left sidebar
3. Search for `/aws/lambda/analyze-linguistic-corpus`
4. Select log group checkbox
5. Click **Actions** → **Delete log group(s)**
6. Click **Delete**

---

### Step 6: (Optional) Delete Athena Resources (1 minute)

If you created Athena resources:

```bash
# Delete Athena workgroup
aws athena delete-work-group --work-group corpus-linguistics

# Delete Athena query results bucket (if created separately)
# aws s3 rm s3://aws-athena-query-results-{account-id}-us-east-1 --recursive
# aws s3 rb s3://aws-athena-query-results-{account-id}-us-east-1
```

---

## Verification Checklist

After cleanup, verify all resources are deleted:

```bash
# Check S3 buckets
echo "S3 Buckets:"
aws s3 ls | grep linguistic-corpus
# Expected: No output

# Check Lambda functions
echo "Lambda Functions:"
aws lambda list-functions --query 'Functions[?FunctionName==`analyze-linguistic-corpus`]'
# Expected: []

# Check DynamoDB tables
echo "DynamoDB Tables:"
aws dynamodb list-tables --query 'TableNames[?contains(@, `LinguisticAnalysis`)]'
# Expected: []

# Check IAM roles
echo "IAM Roles:"
aws iam list-roles --query 'Roles[?RoleName==`lambda-linguistic-processor`]'
# Expected: []

# Check CloudWatch log groups
echo "CloudWatch Logs:"
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/analyze-linguistic-corpus
# Expected: No log groups

echo "Verification complete!"
```

---

## Cost Verification

Check your AWS billing to ensure no ongoing charges:

```bash
# Check current month costs
aws ce get-cost-and-usage \
    --time-period Start=2025-11-01,End=2025-11-30 \
    --granularity MONTHLY \
    --metrics "BlendedCost" \
    --group-by Type=SERVICE
```

**Via AWS Console:**

1. Open https://console.aws.amazon.com/billing/
2. Click **Bills** in left sidebar
3. Review current month charges
4. Expand service details to see S3, Lambda, DynamoDB charges
5. Charges should be $0 or minimal (< $1) after cleanup

---

## Common Issues

### "Bucket not empty" error

**Cause:** Objects still exist in bucket

**Solution:**
```bash
# Force delete all objects including versions
aws s3 rm s3://$BUCKET_NAME --recursive
aws s3api delete-bucket --bucket $BUCKET_NAME
```

### "Function not found" error

**Cause:** Lambda already deleted or name mismatch

**Solution:** Verify function name
```bash
aws lambda list-functions --query 'Functions[].FunctionName'
```

### "Role cannot be deleted" error

**Cause:** Policies still attached to role

**Solution:** Force detach all policies
```bash
# Detach all managed policies
aws iam list-attached-role-policies --role-name lambda-linguistic-processor \
    --query 'AttachedPolicies[].PolicyArn' --output text | \
    xargs -I {} aws iam detach-role-policy --role-name lambda-linguistic-processor --policy-arn {}

# Delete all inline policies
aws iam list-role-policies --role-name lambda-linguistic-processor \
    --query 'PolicyNames[]' --output text | \
    xargs -I {} aws iam delete-role-policy --role-name lambda-linguistic-processor --policy-name {}

# Now delete role
aws iam delete-role --role-name lambda-linguistic-processor
```

### "Table is being deleted" error

**Cause:** DynamoDB deletion is in progress

**Solution:** Wait for deletion to complete
```bash
aws dynamodb wait table-not-exists --table-name LinguisticAnalysis
```

---

## Final Costs Summary

After completing this guide:

| Resource | Before Cleanup | After Cleanup |
|----------|----------------|---------------|
| S3 Storage | ~$0.23/month | $0 |
| DynamoDB | ~$0.25/month | $0 |
| Lambda | $0 (free tier) | $0 |
| CloudWatch | $0 (free tier) | $0 |
| **Total** | **~$0.50/month** | **$0** |

---

## Backup Before Cleanup (Optional)

If you want to preserve your corpus and results:

```bash
# Download corpus from S3
mkdir corpus_backup
aws s3 sync s3://$BUCKET_NAME/raw/ corpus_backup/

# Export DynamoDB data
aws dynamodb scan --table-name LinguisticAnalysis > linguistic_analysis_backup.json

# Download CloudWatch logs
aws logs filter-log-events \
    --log-group-name /aws/lambda/analyze-linguistic-corpus \
    --output json > lambda_logs_backup.json

echo "Backup complete!"
echo "Files saved:"
echo "  - corpus_backup/ (corpus files)"
echo "  - linguistic_analysis_backup.json (DynamoDB data)"
echo "  - lambda_logs_backup.json (Lambda logs)"
```

---

## Re-running the Project

If you want to run the project again after cleanup:

1. Follow the setup guide from scratch
2. Use the same bucket naming convention
3. Restore corpus from backup (if saved)
4. Upload corpus and re-process

---

**Cleanup complete!** You should now have $0 in ongoing AWS charges.

**Questions?** See [README.md](README.md) for troubleshooting or open a GitHub issue.

**Last updated:** 2025-11-14
