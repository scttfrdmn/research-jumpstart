# fMRI Tier 2 Scripts

This directory contains Python scripts for uploading, processing, and analyzing fMRI data on AWS.

## Scripts Overview

### 1. `create_sample_fmri.py`
Generate synthetic fMRI data for testing without needing real neuroimaging data.

**Usage:**
```bash
# Create default sample (64×64×32 voxels, 150 timepoints)
python create_sample_fmri.py --output ../sample_data/sample_fmri.nii.gz

# Create with custom dimensions
python create_sample_fmri.py --shape 128 128 64 300 --output fmri_large.nii.gz

# Create with task activation pattern
python create_sample_fmri.py --task --output sample_fmri_task.nii.gz
```

**Output:**
- NIfTI-format file (.nii.gz) ready for upload to S3
- Includes realistic BOLD signal, noise, and spatial correlations
- Optional simulated task-evoked activation

### 2. `upload_to_s3.py`
Upload fMRI data to S3 bucket with progress tracking.

**Usage:**
```bash
# Upload single file
python upload_to_s3.py \
    --bucket fmri-input-myname \
    --local-path sample_fmri.nii.gz

# Upload entire directory
python upload_to_s3.py \
    --bucket fmri-input-myname \
    --local-path ../sample_data/

# List S3 bucket contents
python upload_to_s3.py --bucket fmri-input-myname --list

# Verify uploads
python upload_to_s3.py \
    --bucket fmri-input-myname \
    --local-path sample_fmri.nii.gz \
    --verify
```

**Features:**
- Progress bar for large file uploads
- Support for single file or directory uploads
- Automatic NIfTI file detection
- File size verification
- Error handling and logging

### 3. `lambda_function.py`
Lambda function code for fMRI preprocessing (motion correction + spatial smoothing).

**What it does:**
1. Downloads raw fMRI from input S3 bucket
2. Applies motion correction (center-of-mass alignment)
3. Applies spatial smoothing (Gaussian filter, FWHM=6mm)
4. Uploads processed results to output S3 bucket
5. Returns execution summary

**Deployment:**
1. Copy entire file content
2. Paste into Lambda Console → Code editor
3. Click Deploy
4. Configure Lambda settings:
   - Timeout: 5 minutes (300 seconds)
   - Memory: 512 MB
   - Environment variable: `OUTPUT_BUCKET=fmri-output-myname`

**Input Event Format:**
```json
{
    "input_bucket": "fmri-input-myname",
    "input_key": "sample_fmri.nii.gz"
}
```

**Output:**
- Motion-corrected file: `{filename}_motion_corrected.nii.gz`
- Smoothed file: `{filename}_smoothed.nii.gz`
- CloudWatch logs with processing steps

### 4. `query_results.py`
Download processed fMRI results from S3 and load for analysis.

**Usage:**
```bash
# Download all results
python query_results.py \
    --output-bucket fmri-output-myname \
    --local-path ../results/

# List results without downloading
python query_results.py \
    --output-bucket fmri-output-myname \
    --list

# Filter downloads (smoothed files only)
python query_results.py \
    --output-bucket fmri-output-myname \
    --local-path ../results/ \
    --filter smoothed

# Load and inspect fMRI file
python query_results.py --load ../results/fmri_smoothed.nii.gz

# Verify downloads
python query_results.py \
    --output-bucket fmri-output-myname \
    --local-path ../results/ \
    --verify
```

**Features:**
- Progress tracking for downloads
- File size verification
- Metadata inspection
- Support for loading NIfTI files directly

### 5. `cleanup.py`
Safely delete all AWS resources (S3 buckets, Lambda, IAM role).

**IMPORTANT:** This is DESTRUCTIVE and cannot be undone!

**Usage:**
```bash
# Interactive cleanup (prompts for confirmation)
python cleanup.py \
    --input-bucket fmri-input-myname \
    --output-bucket fmri-output-myname \
    --lambda-function fmri-preprocessor \
    --iam-role lambda-fmri-processor

# Automatic cleanup (no confirmation)
python cleanup.py \
    --input-bucket fmri-input-myname \
    --output-bucket fmri-output-myname \
    --lambda-function fmri-preprocessor \
    --iam-role lambda-fmri-processor \
    --confirm
```

**What it deletes:**
1. Input S3 bucket and all objects
2. Output S3 bucket and all objects
3. Lambda function
4. IAM role and policies

**Before running:**
- [ ] Download all results from S3
- [ ] Verify results are backed up locally
- [ ] Review what will be deleted (script shows list)
- [ ] Consider the cost if you stop now vs. full cleanup

## Typical Workflow

```bash
# 1. Create sample data
python create_sample_fmri.py --output ../sample_data/sample_fmri.nii.gz

# 2. Upload to S3
python upload_to_s3.py \
    --bucket fmri-input-myname \
    --local-path ../sample_data/sample_fmri.nii.gz

# 3. [Manual step] Deploy Lambda and run it via AWS Console

# 4. Download results
python query_results.py \
    --output-bucket fmri-output-myname \
    --local-path ../results/

# 5. [Open Jupyter] Run fmri_analysis.ipynb

# 6. Cleanup resources
python cleanup.py \
    --input-bucket fmri-input-myname \
    --output-bucket fmri-output-myname \
    --lambda-function fmri-preprocessor \
    --iam-role lambda-fmri-processor \
    --confirm
```

## Requirements

All scripts require:
- Python 3.8+
- Dependencies: `pip install -r ../requirements.txt`
- AWS credentials configured locally

### AWS Credentials

Scripts automatically use credentials from:
1. Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
2. AWS CLI config: `~/.aws/credentials`
3. IAM role (if running on EC2)

Configure once:
```bash
aws configure
# Enter: Access Key ID, Secret Access Key, region, output format
```

## Troubleshooting

### "Access Denied" errors
- Verify AWS credentials: `aws sts get-caller-identity`
- Check IAM user/role has S3 and Lambda permissions
- Verify bucket names are spelled correctly

### Lambda timeout
- Increase timeout in Lambda console (Configuration → General)
- Reduce input file size
- Check CloudWatch logs for bottlenecks

### Upload failures
- Check internet connection
- Verify bucket exists and is accessible
- Check file permissions (should be readable)

### Download errors
- Verify output bucket name is correct
- Check AWS credentials have S3 read permissions
- Use `--list` to verify files exist in bucket

## Cost Monitoring

**Typical costs per project:**
- S3 storage (10GB): $0.23
- Lambda (50 invocations): $0.50
- Data transfer: $0.90
- **Total: $10-14**

**Cost optimization:**
- Delete resources immediately after analysis (see cleanup.py)
- Use smaller input files for testing
- Set Lambda timeout to minimum needed
- Monitor with AWS Cost Explorer

## Development

### Adding new functionality
1. Follow existing patterns for AWS clients
2. Include comprehensive logging
3. Add error handling
4. Support both interactive and automated modes
5. Update this README

### Testing
```bash
# Create test data
python create_sample_fmri.py --output test_fmri.nii.gz

# Test upload (without real AWS)
python upload_to_s3.py --list  # Check help

# Test Lambda locally
python lambda_function.py  # Will try to run as standalone

# Test cleanup (will show resources, ask for confirmation)
python cleanup.py --help
```

## Advanced Usage

### Batch processing multiple files
```bash
for file in ../sample_data/*.nii.gz; do
    python upload_to_s3.py \
        --bucket fmri-input-myname \
        --local-path "$file"
done
```

### Monitor Lambda execution
```bash
# Watch logs in real time
aws logs tail /aws/lambda/fmri-preprocessor --follow
```

### Download specific files
```bash
# Download only motion-corrected files
python query_results.py \
    --output-bucket fmri-output-myname \
    --local-path ../results/ \
    --filter motion_corrected
```

## Support

For issues:
1. Check error messages in script output
2. Review CloudWatch logs (Lambda failures)
3. Verify AWS resource creation in AWS Console
4. See setup_guide.md for detailed setup instructions

---

**Last Updated:** November 2024
**Version:** 1.0
