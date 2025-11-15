# AWS Resource Cleanup Guide - Environmental Monitoring Tier 2

This guide walks you through deleting all AWS resources created for the environmental monitoring project. **Important:** Follow these steps to avoid ongoing charges.

**Estimated time:** 10-15 minutes

---

## Before You Start

### Backup Your Data (Optional)

If you want to keep your sensor data and results:

```bash
# Download all data from S3
aws s3 sync s3://environmental-data-{your-bucket}/ ./backup/

# Export DynamoDB table (optional)
aws dynamodb scan --table-name EnvironmentalReadings > dynamodb_backup.json
```

### Verify Current Costs

Check your current AWS costs before cleanup:

```bash
# View month-to-date costs
aws ce get-cost-and-usage \
  --time-period Start=$(date -v1d +%Y-%m-01),End=$(date +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics "BlendedCost"
```

---

## Step 1: Delete S3 Bucket and Contents

S3 buckets must be empty before deletion.

### Option A: Using AWS Console

1. Go to [S3 Console](https://console.aws.amazon.com/s3/)
2. Find bucket `environmental-data-{your-name}-{date}`
3. Select the bucket (checkbox)
4. Click "Empty" button
5. Type `permanently delete` to confirm
6. Click "Empty"
7. Wait for confirmation
8. Select bucket again
9. Click "Delete" button
10. Type bucket name to confirm
11. Click "Delete bucket"

### Option B: Using AWS CLI

```bash
# Set your bucket name
BUCKET_NAME="environmental-data-{your-bucket}"

# Delete all objects in bucket
aws s3 rm s3://$BUCKET_NAME --recursive

# Delete the bucket
aws s3 rb s3://$BUCKET_NAME

# Verify deletion
aws s3 ls | grep environmental-data
# Should return nothing
```

**Estimated savings:** $0.16 per week for 1GB of data

---

## Step 2: Delete Lambda Function

### Option A: Using AWS Console

1. Go to [Lambda Console](https://console.aws.amazon.com/lambda/)
2. Find function `process-sensor-data`
3. Select the function
4. Click "Actions" → "Delete"
5. Type `delete` to confirm
6. Click "Delete"

### Option B: Using AWS CLI

```bash
# Delete Lambda function
aws lambda delete-function --function-name process-sensor-data

# Verify deletion
aws lambda list-functions | grep process-sensor-data
# Should return nothing
```

**Estimated savings:** $0.40 per 200 invocations

---

## Step 3: Delete DynamoDB Table

**Warning:** This permanently deletes all stored sensor readings.

### Option A: Using AWS Console

1. Go to [DynamoDB Console](https://console.aws.amazon.com/dynamodb/)
2. Click "Tables" in left menu
3. Find table `EnvironmentalReadings`
4. Select the table
5. Click "Delete" button
6. Uncheck "Create backup before deleting"
7. Type `delete` to confirm
8. Click "Delete"

### Option B: Using AWS CLI

```bash
# Delete DynamoDB table
aws dynamodb delete-table --table-name EnvironmentalReadings

# Wait for deletion to complete
aws dynamodb wait table-not-exists --table-name EnvironmentalReadings

# Verify deletion
aws dynamodb list-tables | grep EnvironmentalReadings
# Should return nothing
```

**Estimated savings:** $1.38 per month for 500MB of data

---

## Step 4: Delete SNS Topic and Subscriptions

### Option A: Using AWS Console

1. Go to [SNS Console](https://console.aws.amazon.com/sns/)
2. Click "Topics" in left menu
3. Find topic `environmental-alerts`
4. Select the topic
5. Click "Delete" button
6. Type `delete me` to confirm
7. Click "Delete"

### Option B: Using AWS CLI

```bash
# Get topic ARN
TOPIC_ARN=$(aws sns list-topics --query 'Topics[?contains(TopicArn, `environmental-alerts`)].TopicArn' --output text)

# Delete all subscriptions first
aws sns list-subscriptions-by-topic --topic-arn "$TOPIC_ARN" \
  --query 'Subscriptions[].SubscriptionArn' --output text | \
  xargs -I {} aws sns unsubscribe --subscription-arn {}

# Delete topic
aws sns delete-topic --topic-arn "$TOPIC_ARN"

# Verify deletion
aws sns list-topics | grep environmental-alerts
# Should return nothing
```

**Estimated savings:** $0.50 per 1000 email notifications

---

## Step 5: Delete IAM Role and Policies

**Important:** Only delete this role if you're not using it for other Lambda functions.

### Option A: Using AWS Console

1. Go to [IAM Console](https://console.aws.amazon.com/iam/)
2. Click "Roles" in left menu
3. Search for `lambda-environmental-processor`
4. Click on the role name
5. Click "Delete" button
6. Type role name to confirm
7. Click "Delete"

### Option B: Using AWS CLI

```bash
# Detach all policies from role
aws iam list-attached-role-policies --role-name lambda-environmental-processor \
  --query 'AttachedPolicies[].PolicyArn' --output text | \
  xargs -I {} aws iam detach-role-policy \
    --role-name lambda-environmental-processor \
    --policy-arn {}

# Delete the role
aws iam delete-role --role-name lambda-environmental-processor

# Verify deletion
aws iam get-role --role-name lambda-environmental-processor 2>&1
# Should return error: NoSuchEntity
```

---

## Step 6: Delete CloudWatch Log Groups

Lambda creates log groups automatically. Delete them to stop incurring storage charges.

### Option A: Using AWS Console

1. Go to [CloudWatch Console](https://console.aws.amazon.com/cloudwatch/)
2. Click "Logs" → "Log groups" in left menu
3. Find log group `/aws/lambda/process-sensor-data`
4. Select the log group
5. Click "Actions" → "Delete log group(s)"
6. Click "Delete"

### Option B: Using AWS CLI

```bash
# Delete Lambda log group
aws logs delete-log-group --log-group-name /aws/lambda/process-sensor-data

# Verify deletion
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/process-sensor-data
# Should return empty
```

**Estimated savings:** $0.03 per GB per month

---

## Step 7: (Optional) Delete Athena Queries and Results

If you created Athena resources:

### Using AWS Console

1. Go to [Athena Console](https://console.aws.amazon.com/athena/)
2. Select database containing `environmental_readings` table
3. Drop table: `DROP TABLE IF EXISTS environmental_readings;`
4. Go to S3 and delete `athena-results/` folder in your bucket

### Using AWS CLI

```bash
# Drop Athena table
aws athena start-query-execution \
  --query-string "DROP TABLE IF EXISTS environmental_readings;" \
  --result-configuration "OutputLocation=s3://$BUCKET_NAME/athena-results/"
```

---

## Step 8: Delete Billing Alerts (Optional)

If you created budget alerts:

### Using AWS Console

1. Go to [Billing Console](https://console.aws.amazon.com/billing/)
2. Click "Budgets" in left menu
3. Find budget `environmental-monitoring-budget`
4. Click on budget name
5. Click "Delete" button
6. Confirm deletion

### Using AWS CLI

```bash
# Delete budget
aws budgets delete-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget-name environmental-monitoring-budget
```

---

## Verification Checklist

Run these commands to verify all resources are deleted:

```bash
# Check S3 buckets
aws s3 ls | grep environmental-data
# Expected: Nothing

# Check Lambda functions
aws lambda list-functions --query 'Functions[?contains(FunctionName, `sensor-data`)].FunctionName'
# Expected: []

# Check DynamoDB tables
aws dynamodb list-tables --query 'TableNames[?contains(@, `Environmental`)].@'
# Expected: []

# Check SNS topics
aws sns list-topics --query 'Topics[?contains(TopicArn, `environmental`)].TopicArn'
# Expected: []

# Check IAM roles
aws iam get-role --role-name lambda-environmental-processor 2>&1
# Expected: Error (NoSuchEntity)

# Check CloudWatch log groups
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/process-sensor
# Expected: Empty or error
```

---

## Final Cost Verification

Wait 24 hours after cleanup, then check your AWS bill:

```bash
# Check yesterday's costs
aws ce get-cost-and-usage \
  --time-period Start=$(date -v-1d +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics "BlendedCost"

# Check by service
aws ce get-cost-and-usage \
  --time-period Start=$(date -v-1d +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics "UnblendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE
```

You should see no charges for:
- S3 (environmental-data bucket)
- Lambda (process-sensor-data)
- DynamoDB (EnvironmentalReadings)
- SNS (environmental-alerts)

---

## Troubleshooting

### S3 Bucket Won't Delete

**Error:** "The bucket you tried to delete is not empty"

```bash
# Force delete all versions and delete markers
aws s3api list-object-versions \
  --bucket $BUCKET_NAME \
  --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' \
  --output json | \
  jq '.Objects[]' -c | \
  xargs -I {} aws s3api delete-object --bucket $BUCKET_NAME --cli-input-json '{}'

# Then delete bucket
aws s3 rb s3://$BUCKET_NAME --force
```

### Lambda Function Won't Delete

**Error:** "Function has event source mappings"

```bash
# List event source mappings
aws lambda list-event-source-mappings --function-name process-sensor-data

# Delete each mapping
aws lambda delete-event-source-mapping --uuid <UUID>

# Then delete function
aws lambda delete-function --function-name process-sensor-data
```

### IAM Role Won't Delete

**Error:** "Cannot delete entity, must detach all policies first"

```bash
# List and detach all policies
aws iam list-attached-role-policies --role-name lambda-environmental-processor \
  --query 'AttachedPolicies[].PolicyArn' --output text | \
  xargs -I {} aws iam detach-role-policy \
    --role-name lambda-environmental-processor \
    --policy-arn {}

# Delete role
aws iam delete-role --role-name lambda-environmental-processor
```

### Still Seeing Charges

1. Check AWS Cost Explorer for specific service charges
2. Verify all resources deleted in all regions
3. Check for hidden resources:
   - CloudWatch Logs (storage charges)
   - S3 Glacier (if you had lifecycle policies)
   - Data transfer charges (may appear days later)

---

## Estimated Total Savings

After cleanup, you'll save approximately:

| Resource | Monthly Savings |
|----------|----------------|
| S3 Storage (1GB) | $0.69 |
| Lambda Invocations | $1.20 |
| DynamoDB (500MB) | $1.38 |
| SNS Notifications | $0.50 |
| CloudWatch Logs | $0.03 |
| **Total** | **~$3.80/month** |

For the learning project (1 week), total cost was approximately $7-12.

---

## Keep Learning

You've successfully cleaned up all resources. Next steps:

1. **Review what you learned** - AWS services, environmental monitoring, serverless architecture
2. **Try Tier 3** - Production-grade CloudFormation deployment
3. **Real-world projects** - Connect actual IoT sensors
4. **Other domains** - Try climate science, genomics, or astronomy projects

---

**Cleanup complete!** All resources deleted, no ongoing charges.

For questions or issues, see [GitHub Issues](https://github.com/research-jumpstart/research-jumpstart/issues).

**Last updated:** 2025-11-14 | Research Jumpstart v1.0.0
