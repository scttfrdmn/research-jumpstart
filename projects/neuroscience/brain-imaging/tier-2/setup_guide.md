# AWS Setup Guide for fMRI Data Processing

This guide walks you through setting up AWS services for the fMRI data processing project. Follow each step carefully to ensure proper configuration.

**Total Setup Time:** 45-60 minutes

## Prerequisites

Before starting, ensure you have:
- [ ] AWS account with billing enabled
- [ ] AWS Management Console access
- [ ] Python 3.8+ installed locally
- [ ] AWS CLI installed (optional but recommended)
- [ ] AWS credentials configured locally

### Configure AWS Credentials Locally

If you haven't already configured your AWS credentials:

```bash
# Configure AWS CLI with your credentials
aws configure

# When prompted, enter:
# - AWS Access Key ID: [your key]
# - AWS Secret Access Key: [your secret]
# - Default region: us-east-1 (or your preferred region)
# - Default output format: json
```

Alternatively, set environment variables:
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
```

Or create `~/.aws/credentials`:
```
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```

## Step 1: Create S3 Buckets (5 minutes)

You'll create two buckets: one for raw input data and one for processed outputs.

### 1.1 Create Input Bucket

1. Open AWS Management Console: https://console.aws.amazon.com
2. Navigate to **S3** service
3. Click **Create bucket**
4. Configure bucket:
   - **Bucket name:** `fmri-input-{your-username}` (must be globally unique)
   - **Region:** `us-east-1` (or your preferred region)
   - **Block Public Access:** Keep all 4 checkboxes enabled (security best practice)
   - Click **Create bucket**

### 1.2 Create Output Bucket

Repeat the same process for output:
- **Bucket name:** `fmri-output-{your-username}`
- Same region as input bucket
- Same security settings

### 1.3 Verify Buckets

In S3 Console, confirm both buckets appear in your bucket list with status "Bucket exists".

**What to note:**
- Input bucket name: `fmri-input-{your-username}`
- Output bucket name: `fmri-output-{your-username}`
- Region: us-east-1

## Step 2: Create IAM Role for Lambda (5 minutes)

Lambda needs permissions to read from S3 input and write to S3 output. We'll create a role with minimal permissions (least-privilege principle).

### 2.1 Navigate to IAM

1. Open AWS Management Console
2. Navigate to **IAM** service
3. Click **Roles** in the left menu

### 2.2 Create New Role

1. Click **Create role**
2. Choose trusted entity:
   - **Trusted entity type:** `AWS service`
   - **Service:** `Lambda`
   - Click **Next**

### 2.3 Attach Permissions

1. Search for `AmazonS3FullAccess` policy (for simplicity, we'll refine later)
2. Select the checkbox
3. Click **Next**

### 2.4 Name and Create Role

1. **Role name:** `lambda-fmri-processor`
2. **Description:** "Role for Lambda fMRI processing functions"
3. Click **Create role**

### 2.5 Create Custom Inline Policy (Recommended)

For better security, add a custom policy with only needed permissions:

1. Find the created role `lambda-fmri-processor`
2. Click on the role name
3. Click **Add inline policy** (or under "Policies" tab)
4. Click **JSON** tab and replace with:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::fmri-input-{your-username}",
                "arn:aws:s3:::fmri-input-{your-username}/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject"
            ],
            "Resource": [
                "arn:aws:s3:::fmri-output-{your-username}",
                "arn:aws:s3:::fmri-output-{your-username}/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}
```

5. Replace `{your-username}` with your actual username or bucket identifier
6. Click **Review policy**
7. **Name:** `fmri-s3-access`
8. Click **Create policy**

**What to note:**
- IAM Role ARN: (found in role details, looks like `arn:aws:iam::123456789012:role/lambda-fmri-processor`)

## Step 3: Deploy Lambda Function (10 minutes)

### 3.1 Navigate to Lambda

1. Open AWS Management Console
2. Navigate to **Lambda** service
3. Click **Create function**

### 3.2 Create Function

Configure the function:
- **Function name:** `fmri-preprocessor`
- **Runtime:** `Python 3.11`
- **Architecture:** `x86_64`
- **Permissions:**
  - Execution role: `Use an existing role`
  - Existing role: Select `lambda-fmri-processor`
- Click **Create function**

### 3.3 Add Lambda Code

1. In the Lambda console, scroll to the **Code** section
2. Replace the default code with content from `scripts/lambda_function.py`
3. Click **Deploy**

### 3.4 Configure Function Settings

1. Click **Configuration** tab
2. Adjust these settings:
   - **General configuration:**
     - Timeout: `5 minutes` (300 seconds)
     - Memory: `512 MB` (good balance for fMRI processing)
   - **Environment variables:**
     - Add variable: `OUTPUT_BUCKET` = `fmri-output-{your-username}`
3. Click **Save**

### 3.5 Test Lambda Function

1. Click **Test** tab
2. Create test event:
   - **Event name:** `test-fmri`
   - **Template:** `API Gateway AWS Proxy`
   - Modify the JSON to:
   ```json
   {
     "body": "{\"input_bucket\": \"fmri-input-{your-username}\", \"input_key\": \"sample_fmri.nii.gz\"}",
     "queryStringParameters": null,
     "requestContext": {
       "requestId": "test-request-id"
     }
   }
   ```
3. Click **Test**
4. Review execution result (should show success or helpful error messages)

**What to note:**
- Lambda function name: `fmri-preprocessor`
- Timeout: 5 minutes
- Memory: 512 MB

## Step 4: Upload Sample fMRI Data (10-20 minutes)

### 4.1 Prepare Python Environment

```bash
# Navigate to project directory
cd /path/to/neuroscience/brain-imaging/tier-2

# Install required packages
pip install -r requirements.txt
```

### 4.2 Create Sample Data (or use provided)

If you need to create sample fMRI data:

```bash
# Create sample data directory
mkdir -p sample_data

# Use the provided script or create your own test data
python scripts/create_sample_fmri.py --output sample_data/sample_fmri.nii.gz
```

### 4.3 Upload to S3

```bash
# Upload using the provided script
python scripts/upload_to_s3.py \
    --bucket fmri-input-{your-username} \
    --local-path sample_data/sample_fmri.nii.gz \
    --s3-key sample_fmri.nii.gz
```

Or use AWS CLI:
```bash
aws s3 cp sample_data/sample_fmri.nii.gz \
    s3://fmri-input-{your-username}/sample_fmri.nii.gz
```

### 4.4 Verify Upload

In AWS Console S3 service:
1. Open `fmri-input-{your-username}` bucket
2. Confirm `sample_fmri.nii.gz` appears in the file list

**What to note:**
- Input file path in S3: `s3://fmri-input-{your-username}/sample_fmri.nii.gz`

## Step 5: Test End-to-End Pipeline (10 minutes)

### 5.1 Trigger Lambda Manually

```bash
# Test Lambda invocation
python scripts/test_lambda.py \
    --function-name fmri-preprocessor \
    --input-bucket fmri-input-{your-username} \
    --input-key sample_fmri.nii.gz
```

### 5.2 Monitor Execution

1. Open Lambda function in AWS Console
2. Click **Monitor** tab
3. View CloudWatch logs to see processing steps
4. Check for errors or warnings

### 5.3 Verify Output

1. Navigate to S3 Console
2. Open `fmri-output-{your-username}` bucket
3. Confirm processed files appear:
   - `sample_fmri_preprocessed.nii.gz` (motion corrected)
   - `sample_fmri_smoothed.nii.gz` (spatially smoothed)

## Step 6: Install Local Analysis Tools (5 minutes)

### 6.1 Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 6.2 Test boto3 Connection

```bash
# Test AWS connectivity
python -c "
import boto3
s3 = boto3.client('s3')
response = s3.list_buckets()
print('Connected! Buckets:', [b['Name'] for b in response['Buckets']])
"
```

## Configuration Summary

Record these values for your scripts and notebooks:

```
INPUT_BUCKET = fmri-input-{your-username}
OUTPUT_BUCKET = fmri-output-{your-username}
LAMBDA_FUNCTION_NAME = fmri-preprocessor
AWS_REGION = us-east-1
IAM_ROLE_ARN = arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-fmri-processor
```

## Troubleshooting Setup Issues

### Issue: "Access Denied" when creating S3 bucket

**Solution:**
- Verify your IAM user has S3 permissions
- Check AWS account isn't restricted by organization policies
- Ensure you're signed into the correct AWS account

### Issue: Lambda role not appearing in dropdown

**Solution:**
- Refresh the page
- Ensure role was created in the same region
- Check role name matches exactly: `lambda-fmri-processor`

### Issue: "Bucket name already exists" error

**Solution:**
- S3 bucket names must be globally unique
- Use more unique suffix: `fmri-input-{your-username}-{random}`
- Update all references in scripts

### Issue: boto3 can't find AWS credentials

**Solution:**
- Run `aws configure` to set up credentials
- Verify `~/.aws/credentials` file exists
- Set environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- Try `aws sts get-caller-identity` to verify credentials

### Issue: Lambda timeout during execution

**Solution:**
- Increase timeout to 5 minutes in Lambda configuration
- Check input file size (should be <100MB for quick processing)
- Review CloudWatch logs for bottlenecks
- Consider increasing memory allocation to 1024MB

## Next Steps

1. ✅ Verify all AWS resources are created
2. ✅ Confirm sample data uploaded to S3
3. ✅ Test Lambda function execution
4. ✅ Record configuration values above
5. Open `notebooks/fmri_analysis.ipynb` in Jupyter
6. Follow notebook cells for data analysis

## Cost Monitoring

After setup:
1. Open AWS Console
2. Navigate to **Cost Explorer**
3. Set up budget alerts for unexpected charges
4. Monitor S3 storage and Lambda invocations

---

**Setup Complete!** You're ready to run the analysis notebook. See cleanup_guide.md when finished to delete resources and avoid ongoing charges.
