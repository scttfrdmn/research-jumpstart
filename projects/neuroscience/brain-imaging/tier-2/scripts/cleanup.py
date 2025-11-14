#!/usr/bin/env python3
"""
Cleanup script for fMRI AWS resources.

This script safely deletes all AWS resources created during the fMRI project:
- S3 buckets (input and output)
- Lambda function
- IAM role

IMPORTANT: This is DESTRUCTIVE and cannot be undone. Ensure you've downloaded
all results before running this script.

Usage:
    python cleanup.py \
        --input-bucket fmri-input-myname \
        --output-bucket fmri-output-myname \
        --lambda-function fmri-preprocessor \
        --iam-role lambda-fmri-processor \
        --confirm
"""

import boto3
import argparse
import logging
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS clients
s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')
iam = boto3.client('iam')


def empty_s3_bucket(bucket_name: str) -> bool:
    """
    Empty all objects from an S3 bucket.

    Args:
        bucket_name: S3 bucket name

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Emptying S3 bucket: {bucket_name}")

        # List all objects
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)

        delete_count = 0
        for page in pages:
            if 'Contents' not in page:
                continue

            # Delete objects in batch
            for obj in page['Contents']:
                s3.delete_object(Bucket=bucket_name, Key=obj['Key'])
                delete_count += 1

        logger.info(f"Deleted {delete_count} objects from {bucket_name}")
        return True

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            logger.warning(f"Bucket does not exist: {bucket_name}")
            return True
        logger.error(f"Error emptying bucket: {e}")
        return False
    except Exception as e:
        logger.error(f"Error emptying bucket: {e}")
        return False


def delete_s3_bucket(bucket_name: str) -> bool:
    """
    Delete an S3 bucket.

    Args:
        bucket_name: S3 bucket name

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Deleting S3 bucket: {bucket_name}")
        s3.delete_bucket(Bucket=bucket_name)
        logger.info(f"Successfully deleted bucket: {bucket_name}")
        return True

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            logger.warning(f"Bucket already deleted: {bucket_name}")
            return True
        elif e.response['Error']['Code'] == 'BucketNotEmpty':
            logger.error(f"Bucket not empty: {bucket_name}")
            return False
        logger.error(f"Error deleting bucket: {e}")
        return False
    except Exception as e:
        logger.error(f"Error deleting bucket: {e}")
        return False


def delete_lambda_function(function_name: str) -> bool:
    """
    Delete a Lambda function.

    Args:
        function_name: Lambda function name

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Deleting Lambda function: {function_name}")
        lambda_client.delete_function(FunctionName=function_name)
        logger.info(f"Successfully deleted Lambda function: {function_name}")
        return True

    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            logger.warning(f"Lambda function not found: {function_name}")
            return True
        logger.error(f"Error deleting Lambda function: {e}")
        return False
    except Exception as e:
        logger.error(f"Error deleting Lambda function: {e}")
        return False


def delete_iam_role(role_name: str) -> bool:
    """
    Delete an IAM role and its attached policies.

    Args:
        role_name: IAM role name

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Deleting IAM role: {role_name}")

        # First, detach all attached policies
        try:
            attached_policies = iam.list_attached_role_policies(RoleName=role_name)
            for policy in attached_policies.get('AttachedPolicies', []):
                policy_arn = policy['PolicyArn']
                logger.info(f"Detaching policy: {policy_arn}")
                iam.detach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except ClientError as e:
            logger.warning(f"Error detaching policies: {e}")

        # Delete inline policies
        try:
            inline_policies = iam.list_role_policies(RoleName=role_name)
            for policy_name in inline_policies.get('PolicyNames', []):
                logger.info(f"Deleting inline policy: {policy_name}")
                iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
        except ClientError as e:
            logger.warning(f"Error deleting inline policies: {e}")

        # Delete the role
        iam.delete_role(RoleName=role_name)
        logger.info(f"Successfully deleted IAM role: {role_name}")
        return True

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            logger.warning(f"IAM role not found: {role_name}")
            return True
        logger.error(f"Error deleting IAM role: {e}")
        return False
    except Exception as e:
        logger.error(f"Error deleting IAM role: {e}")
        return False


def list_resources(input_bucket: str, output_bucket: str, function_name: str, role_name: str) -> dict:
    """
    List resources that will be deleted.

    Args:
        input_bucket: Input S3 bucket name
        output_bucket: Output S3 bucket name
        function_name: Lambda function name
        role_name: IAM role name

    Returns:
        Dictionary with resource counts
    """
    resources = {
        'input_bucket': False,
        'output_bucket': False,
        'lambda_function': False,
        'iam_role': False,
        's3_objects': 0
    }

    # Check S3 buckets
    try:
        s3.head_bucket(Bucket=input_bucket)
        resources['input_bucket'] = True
        response = s3.list_objects_v2(Bucket=input_bucket)
        resources['s3_objects'] += len(response.get('Contents', []))
    except ClientError:
        pass

    try:
        s3.head_bucket(Bucket=output_bucket)
        resources['output_bucket'] = True
        response = s3.list_objects_v2(Bucket=output_bucket)
        resources['s3_objects'] += len(response.get('Contents', []))
    except ClientError:
        pass

    # Check Lambda function
    try:
        lambda_client.get_function(FunctionName=function_name)
        resources['lambda_function'] = True
    except ClientError:
        pass

    # Check IAM role
    try:
        iam.get_role(RoleName=role_name)
        resources['iam_role'] = True
    except ClientError:
        pass

    return resources


def cleanup_all(input_bucket: str, output_bucket: str, function_name: str, role_name: str, confirm: bool = False) -> bool:
    """
    Clean up all AWS resources.

    Args:
        input_bucket: Input S3 bucket name
        output_bucket: Output S3 bucket name
        function_name: Lambda function name
        role_name: IAM role name
        confirm: If True, skip confirmation prompt

    Returns:
        True if all cleanup successful, False otherwise
    """
    logger.info("="*60)
    logger.info("AWS Resource Cleanup")
    logger.info("="*60)

    # List resources
    resources = list_resources(input_bucket, output_bucket, function_name, role_name)

    logger.info("\nResources to delete:")
    logger.info(f"  Input S3 bucket: {input_bucket} {'✓' if resources['input_bucket'] else '✗'}")
    logger.info(f"  Output S3 bucket: {output_bucket} {'✓' if resources['output_bucket'] else '✗'}")
    logger.info(f"  Lambda function: {function_name} {'✓' if resources['lambda_function'] else '✗'}")
    logger.info(f"  IAM role: {role_name} {'✓' if resources['iam_role'] else '✗'}")
    logger.info(f"  S3 objects: {resources['s3_objects']}")

    # Confirm before deletion
    if not confirm:
        response = input("\nThis action is PERMANENT and cannot be undone. Continue? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Cleanup cancelled.")
            return False

    logger.info("\nStarting cleanup...")
    all_success = True

    # Empty and delete S3 buckets
    if resources['input_bucket']:
        if not empty_s3_bucket(input_bucket):
            all_success = False
        if not delete_s3_bucket(input_bucket):
            all_success = False

    if resources['output_bucket']:
        if not empty_s3_bucket(output_bucket):
            all_success = False
        if not delete_s3_bucket(output_bucket):
            all_success = False

    # Delete Lambda function
    if resources['lambda_function']:
        if not delete_lambda_function(function_name):
            all_success = False

    # Delete IAM role
    if resources['iam_role']:
        if not delete_iam_role(role_name):
            all_success = False

    # Summary
    logger.info("\n" + "="*60)
    if all_success:
        logger.info("✓ Cleanup completed successfully!")
    else:
        logger.error("✗ Some cleanup steps failed. See errors above.")
    logger.info("="*60)

    return all_success


def main():
    parser = argparse.ArgumentParser(
        description="Clean up AWS resources created for fMRI project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive cleanup (prompts for confirmation)
  python cleanup.py \
      --input-bucket fmri-input-myname \
      --output-bucket fmri-output-myname \
      --lambda-function fmri-preprocessor \
      --iam-role lambda-fmri-processor

  # Automatic cleanup without confirmation
  python cleanup.py \
      --input-bucket fmri-input-myname \
      --output-bucket fmri-output-myname \
      --lambda-function fmri-preprocessor \
      --iam-role lambda-fmri-processor \
      --confirm

WARNING: This script is destructive and cannot be undone!
        """
    )

    parser.add_argument(
        '--input-bucket',
        required=True,
        help='Input S3 bucket name'
    )
    parser.add_argument(
        '--output-bucket',
        required=True,
        help='Output S3 bucket name'
    )
    parser.add_argument(
        '--lambda-function',
        required=True,
        help='Lambda function name'
    )
    parser.add_argument(
        '--iam-role',
        required=True,
        help='IAM role name'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Skip confirmation prompt (DANGEROUS)'
    )

    args = parser.parse_args()

    # Run cleanup
    cleanup_all(
        args.input_bucket,
        args.output_bucket,
        args.lambda_function,
        args.iam_role,
        confirm=args.confirm
    )


if __name__ == '__main__':
    main()
