#!/usr/bin/env python3
"""
Invoke Lambda function for source detection on all S3 images.

Iterates through FITS images in S3 and invokes Lambda for each one.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_environment_variables():
    """Get required environment variables."""
    bucket_raw = os.environ.get("BUCKET_RAW")
    bucket_catalog = os.environ.get("BUCKET_CATALOG")
    lambda_function = os.environ.get("LAMBDA_FUNCTION", "astronomy-source-detection")

    if not bucket_raw or not bucket_catalog:
        print("Error: Environment variables not set")
        print("Set them with:")
        print("  export BUCKET_RAW=your-raw-bucket")
        print("  export BUCKET_CATALOG=your-catalog-bucket")
        sys.exit(1)

    return bucket_raw, bucket_catalog, lambda_function


def list_fits_files(s3, bucket_raw):
    """List all FITS files in S3 bucket."""
    fits_files = []

    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_raw, Prefix="images/")

        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith(".fits") or key.endswith(".fits.bz2"):
                    fits_files.append(key)

    except Exception as e:
        print(f"Error listing files: {e}")
        return []

    return sorted(fits_files)


def invoke_lambda_for_image(lambda_client, function_name, s3_key, bucket_raw, bucket_catalog):
    """Invoke Lambda function for a single image."""
    payload = {"bucket": bucket_raw, "bucket_catalog": bucket_catalog, "key": s3_key}

    try:
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload),
        )

        # Parse response
        response_payload = json.loads(response["Payload"].read())

        if response["StatusCode"] == 200:
            body = json.loads(response_payload.get("body", "{}"))
            return {
                "status": "success",
                "key": s3_key,
                "num_sources": body.get("num_sources", 0),
                "output_key": body.get("output_key", ""),
            }
        else:
            return {
                "status": "error",
                "key": s3_key,
                "error": response_payload.get("error", "Unknown error"),
            }

    except Exception as e:
        return {"status": "error", "key": s3_key, "error": str(e)}


def invoke_lambda_async(lambda_client, function_name, s3_key, bucket_raw, bucket_catalog):
    """Invoke Lambda function asynchronously (fire and forget)."""
    payload = {"bucket": bucket_raw, "bucket_catalog": bucket_catalog, "key": s3_key}

    try:
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType="Event",  # Async invocation
            Payload=json.dumps(payload),
        )

        return {
            "status": "submitted",
            "key": s3_key,
            "request_id": response.get("LogResult", "N/A"),
        }

    except Exception as e:
        return {"status": "error", "key": s3_key, "error": str(e)}


def main():
    """Main function."""
    print("=" * 70)
    print("Lambda Source Detection Invocation")
    print("=" * 70)

    # Get environment variables
    bucket_raw, bucket_catalog, lambda_function = get_environment_variables()
    print("\nConfiguration:")
    print(f"  Raw bucket: s3://{bucket_raw}")
    print(f"  Catalog bucket: s3://{bucket_catalog}")
    print(f"  Lambda function: {lambda_function}\n")

    # Initialize AWS clients
    try:
        s3 = boto3.client("s3")
        lambda_client = boto3.client("lambda")
    except NoCredentialsError:
        print("Error: AWS credentials not configured")
        print("Run: aws configure")
        return 1

    # Verify Lambda function exists
    try:
        lambda_client.get_function(FunctionName=lambda_function)
        print(f"✓ Lambda function exists: {lambda_function}\n")
    except ClientError:
        print(f"✗ Lambda function not found: {lambda_function}")
        print("Deploy it with:")
        print("  aws lambda create-function \\")
        print(f"    --function-name {lambda_function} \\")
        print("    --runtime python3.11 ...")
        return 1

    # List FITS files
    print("Listing FITS files...")
    fits_files = list_fits_files(s3, bucket_raw)

    if not fits_files:
        print(f"✗ No FITS files found in s3://{bucket_raw}/images/")
        print("Upload FITS files with: python scripts/upload_to_s3.py")
        return 1

    print(f"✓ Found {len(fits_files)} FITS files\n")

    # Ask for invocation mode
    print("Invocation modes:")
    print("  1. Synchronous (wait for results) - slower but get results")
    print("  2. Asynchronous (fire and forget) - faster, check logs later")
    print("  3. Sequential (one at a time) - easiest to debug")

    # Invoke Lambda for each file
    print("\n" + "=" * 70)
    print("Invoking Lambda Functions")
    print("=" * 70 + "\n")

    results = []
    errors = []
    total_sources = 0

    # Use synchronous invocation with threading for speed
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}

        for s3_key in fits_files:
            future = executor.submit(
                invoke_lambda_for_image,
                lambda_client,
                lambda_function,
                s3_key,
                bucket_raw,
                bucket_catalog,
            )
            futures[future] = s3_key

        # Track progress
        with tqdm(total=len(futures), desc="Processing images", unit="image") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if result["status"] == "success":
                    total_sources += result.get("num_sources", 0)
                    tqdm.write(f"  ✓ {result['key']}: {result.get('num_sources', 0)} sources")
                else:
                    errors.append(result)
                    tqdm.write(f"  ✗ {result['key']}: {result.get('error', 'Unknown error')}")

                pbar.update(1)

    # Summary
    print("\n" + "=" * 70)
    print("Invocation Summary")
    print("=" * 70)

    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")

    print(f"\n✓ Successful: {successful}/{len(results)}")
    if successful > 0:
        print(f"  Total sources detected: {total_sources}")

    if failed > 0:
        print(f"\n✗ Failed: {failed}/{len(results)}")
        for error in errors:
            print(f"  • {error['key']}: {error.get('error', 'Unknown')}")

    # Next steps
    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print("\n1. Check Lambda logs:")
    print(f"   aws logs tail /aws/lambda/{lambda_function} --follow")
    print("\n2. Query results with Athena:")
    print("   python scripts/query_with_athena.py")
    print("\n3. Visualize in notebook:")
    print("   jupyter notebook notebooks/sky_analysis.ipynb")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    # Check for environment file
    env_file = Path.home() / ".astronomy_env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("export "):
                    key_value = line.replace("export ", "").strip()
                    if "=" in key_value:
                        key, value = key_value.split("=", 1)
                        os.environ[key] = value

    sys.exit(main())
