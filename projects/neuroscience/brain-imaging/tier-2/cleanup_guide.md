# AWS Resource Cleanup Guide

After completing the fMRI analysis project, you should delete all AWS resources to avoid ongoing charges. Follow this guide to safely clean up.

**Important:** Once deleted, data cannot be recovered. Ensure you've downloaded any results you want to keep.

## Automated Cleanup (Recommended)

We provide a Python script to automate cleanup. This is safer and faster than manual deletion.

### Quick Cleanup

```bash
python scripts/cleanup.py \
    --input-bucket fmri-input-{your-username} \
    --output-bucket fmri-output-{your-username} \
    --lambda-function fmri-preprocessor \
    --iam-role lambda-fmri-processor \
    --confirm
```

### What the script does:
1. Empties S3 input bucket
2. Deletes S3 input bucket
3. Empties S3 output bucket
4. Deletes S3 output bucket
5. Deletes Lambda function
6. Deletes IAM role
7. Verifies all resources are deleted

### Troubleshooting automated cleanup:

If the script fails, see manual cleanup steps below.

## Manual Cleanup (Step-by-Step)

If automated cleanup doesn't work, follow these steps in order. Do NOT skip any steps.

### Step 1: Download Final Results (If Needed)

Before deleting anything, download processed data you want to keep:

```bash
# Download processed fMRI data
python scripts/query_results.py \
    --output-bucket fmri-output-{your-username} \
    --local-path results_backup/
```

Or use AWS CLI:
```bash
aws s3 sync s3://fmri-output-{your-username}/ ./results_backup/
```

### Step 2: Empty S3 Buckets

S3 buckets must be empty before deletion.

#### Using AWS Console:

1. Open AWS Management Console
2. Navigate to **S3** service
3. For each bucket (`fmri-input-{your-username}` and `fmri-output-{your-username}`):
   - Click the bucket name
   - Select all files (checkbox at top)
   - Click **Delete**
   - Type "permanently delete" to confirm
   - Wait for deletion to complete

#### Using AWS CLI:

```bash
# Empty input bucket
aws s3 rm s3://fmri-input-{your-username}/ --recursive

# Empty output bucket
aws s3 rm s3://fmri-output-{your-username}/ --recursive
```

### Step 3: Delete S3 Buckets

#### Using AWS Console:

1. Navigate to **S3** service
2. For each empty bucket:
   - Click the bucket name checkbox
   - Click **Delete**
   - Type the exact bucket name
   - Click **Delete bucket**

#### Using AWS CLI:

```bash
# Delete input bucket
aws s3 rb s3://fmri-input-{your-username}/

# Delete output bucket
aws s3 rb s3://fmri-output-{your-username}/
```

### Step 4: Delete Lambda Function

#### Using AWS Console:

1. Navigate to **Lambda** service
2. Find `fmri-preprocessor` function
3. Click the function name
4. Click **Delete** button at top
5. Type "delete" to confirm

#### Using AWS CLI:

```bash
aws lambda delete-function --function-name fmri-preprocessor
```

### Step 5: Delete IAM Role

#### Using AWS Console:

1. Navigate to **IAM** service
2. Click **Roles** in left menu
3. Find `lambda-fmri-processor` role
4. Click the role name
5. Click **Delete** button (scroll to bottom)
6. Type the role name to confirm

#### Using AWS CLI:

```bash
# First, detach any policies
aws iam detach-role-policy \
    --role-name lambda-fmri-processor \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Then delete the role
aws iam delete-role --role-name lambda-fmri-processor
```

### Step 6: Verify Cleanup

Confirm all resources are deleted:

```bash
# List remaining S3 buckets
aws s3 ls

# List Lambda functions
aws lambda list-functions

# List IAM roles
aws iam list-roles --query 'Roles[?contains(RoleName, `fmri`) || contains(RoleName, `lambda-fmri`)].RoleName'
```

None of these commands should show your fMRI resources.

## Detailed Deletion Instructions by Service

### S3 Bucket Deletion

**Important:** Empty bucket first, then delete

1. Navigate to S3 Console
2. Select bucket
3. Click **Empty** button to empty all contents
4. Click **Delete** button
5. Type bucket name exactly
6. Click **Delete bucket**

**Note:** Deletion takes a few minutes. You may need to refresh to see changes.

### Lambda Function Deletion

**Important:** Cannot be undone once deleted

1. Navigate to Lambda Console
2. Select function: `fmri-preprocessor`
3. Click **Delete** at top
4. Click **Delete** in confirmation dialog

**Note:** Function is immediately deleted. Logs remain in CloudWatch for 30 days (optional to delete).

### IAM Role Deletion

**Important:** Remove inline policies first if created

1. Navigate to IAM Console
2. Click **Roles**
3. Select role: `lambda-fmri-processor`
4. If inline policies exist, delete them first:
   - Find inline policy section
   - Click **X** to delete each policy
5. Click **Delete role** button
6. Click **Delete** in confirmation

## Cleanup Checklist

Use this checklist to ensure complete cleanup:

- [ ] Downloaded and backed up any important results
- [ ] Emptied S3 input bucket (`fmri-input-{your-username}`)
- [ ] Deleted S3 input bucket
- [ ] Emptied S3 output bucket (`fmri-output-{your-username}`)
- [ ] Deleted S3 output bucket
- [ ] Deleted Lambda function (`fmri-preprocessor`)
- [ ] Deleted IAM role (`lambda-fmri-processor`)
- [ ] Ran verification commands (no resources found)
- [ ] Checked AWS Cost Explorer for pending charges

## Cost After Cleanup

After completing cleanup:

| Item | Cost |
|------|------|
| S3 storage | $0 (buckets deleted) |
| Lambda | $0 (function deleted) |
| CloudWatch logs | ~$0.05 (retained for 30 days) |
| **Total remaining** | ~$0.05 |

Optional: Delete CloudWatch logs manually to eliminate remaining charges.

### Delete CloudWatch Logs (Optional)

```bash
# List log groups
aws logs describe-log-groups --query 'logGroups[?contains(logGroupName, `fmri`)].logGroupName'

# Delete log group (adjust name as needed)
aws logs delete-log-group --log-group-name /aws/lambda/fmri-preprocessor
```

## Troubleshooting Cleanup

### Error: "Cannot delete bucket with objects"

**Solution:**
- Ensure bucket is completely empty
- Check for hidden files or versions
- Use `aws s3 rm s3://bucket-name/ --recursive` to force empty
- Try again

### Error: "Role has in-line policies"

**Solution:**
- Delete inline policies first
- List policies: `aws iam list-role-policies --role-name lambda-fmri-processor`
- Delete each: `aws iam delete-role-policy --role-name lambda-fmri-processor --policy-name {policy-name}`

### Error: "NoSuchBucket" when trying to delete

**Solution:**
- Bucket may already be deleted
- Check bucket name spelling
- Verify you're in correct AWS region

### Lambda deletion fails

**Solution:**
- Check function is not currently executing
- Ensure you have permissions to delete
- Try deleting via AWS Console manually
- Check if function has reserved concurrent executions (can cause issues)

## AWS CloudWatch Logs Cleanup

Lambda execution logs are automatically stored in CloudWatch. These may incur small charges.

### View Lambda Logs

```bash
# View recent logs
aws logs tail /aws/lambda/fmri-preprocessor --follow
```

### Delete Log Group

```bash
aws logs delete-log-group --log-group-name /aws/lambda/fmri-preprocessor
```

## Prevention: Avoid Forgotten Resources

To prevent accidental charges from forgotten resources:

1. **Set AWS Billing Alerts:**
   - Go to AWS Console → Billing
   - Click "Billing Preferences"
   - Enable "Receive Billing Alerts"
   - Set threshold (e.g., $5)

2. **Use AWS Budget:**
   - Go to AWS Console → Budgets
   - Create budget for your project
   - Set alerts for exceeding budget

3. **Tag Resources:**
   - Add tag to all resources: `Project: fMRI-Tier2`
   - Makes it easy to find all related resources
   - Helps with cost allocation

4. **Regular Audit:**
   - Weekly review of active AWS resources
   - Monthly review of AWS billing

## Re-Running the Project

To run the project again:

1. Repeat steps in `setup_guide.md`
2. Create new S3 buckets
3. Redeploy Lambda function
4. Upload fresh data
5. Run analysis notebook

Note: You can reuse the same IAM role if you're recreating with same bucket names.

## Support

If you encounter issues during cleanup:

1. Check the "Troubleshooting Cleanup" section above
2. Review AWS documentation for specific service
3. Verify your AWS account permissions
4. Contact AWS Support if needed (AWS Support Plan required)

---

**Cleanup is important!** Uncleaned resources continue to incur charges even if unused. Run this guide immediately after completing your analysis.
