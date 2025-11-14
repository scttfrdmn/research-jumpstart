#!/usr/bin/env python3
"""
AWS Lambda function for processing CMIP6 climate data.

This function:
1. Triggered by S3 upload of netCDF files
2. Extracts regional statistics from climate data
3. Stores processed results back to S3

Deploy to AWS Lambda:
- Runtime: Python 3.11
- Handler: lambda_function.lambda_handler
- Timeout: 300 seconds (5 minutes)
- Memory: 512 MB
- Environment: BUCKET_NAME, PROCESS_MODE
"""

import json
import boto3
import logging
import os
import traceback
from datetime import datetime
from io import BytesIO

# Initialize clients
s3_client = boto3.client('s3')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Optional: Only imported if netCDF4 is available
try:
    import xarray as xr
    import numpy as np
    NETCDF_AVAILABLE = True
except ImportError:
    logger.warning("xarray/netCDF4 not available - basic processing only")
    NETCDF_AVAILABLE = False


def lambda_handler(event, context):
    """
    AWS Lambda handler for processing climate data.

    Args:
        event (dict): S3 event trigger
        context (LambdaContext): Lambda context

    Returns:
        dict: Response with status and message
    """
    try:
        logger.info("Climate data processing Lambda started")
        logger.info(f"Event: {json.dumps(event)}")

        # Get bucket and file from event
        bucket_name = os.environ.get('BUCKET_NAME', 'climate-data')
        process_mode = os.environ.get('PROCESS_MODE', 'calculate_statistics')

        # Parse S3 event
        if 'Records' in event:
            record = event['Records'][0]
            s3_bucket = record['s3']['bucket']['name']
            s3_key = record['s3']['object']['key']
        else:
            # Direct invocation
            s3_bucket = event.get('bucket', bucket_name)
            s3_key = event.get('key', 'raw/test.nc')

        logger.info(f"Processing file: s3://{s3_bucket}/{s3_key}")

        # Download file from S3
        try:
            obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            file_data = obj['Body'].read()
            logger.info(f"Downloaded {len(file_data)/1e6:.1f}MB from S3")
        except Exception as e:
            error_msg = f"Failed to download from S3: {str(e)}"
            logger.error(error_msg)
            return error_response(error_msg, s3_key)

        # Process file
        try:
            if NETCDF_AVAILABLE and s3_key.endswith(('.nc', '.nc4')):
                result = process_netcdf_file(
                    file_data,
                    s3_key,
                    process_mode
                )
            else:
                result = process_generic_file(file_data, s3_key)
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return error_response(error_msg, s3_key)

        # Upload results to S3
        try:
            results_key = generate_results_key(s3_key)
            upload_results(s3_bucket, results_key, result)
            logger.info(f"Results uploaded to s3://{s3_bucket}/{results_key}")
        except Exception as e:
            error_msg = f"Failed to upload results: {str(e)}"
            logger.error(error_msg)
            return error_response(error_msg, s3_key)

        # Return success
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Processing completed successfully',
                'input_file': s3_key,
                'output_file': results_key,
                'timestamp': datetime.utcnow().isoformat()
            })
        }

        logger.info(f"Success: {response['body']}")
        return response

    except Exception as e:
        error_msg = f"Unhandled error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return error_response(error_msg)


def process_netcdf_file(file_data, filename, mode='calculate_statistics'):
    """
    Process netCDF file and extract statistics.

    Args:
        file_data (bytes): File content
        filename (str): Original filename
        mode (str): Processing mode

    Returns:
        dict: Processed results
    """
    logger.info(f"Processing netCDF file in mode: {mode}")

    try:
        # Load netCDF data
        with BytesIO(file_data) as f:
            ds = xr.open_dataset(f)
            logger.info(f"Dataset variables: {list(ds.data_vars)}")
            logger.info(f"Dataset dimensions: {dict(ds.dims)}")

        # Extract statistics
        statistics = {}

        # Temperature statistics
        if 'tas' in ds:
            tas = ds['tas']
            statistics['temperature'] = {
                'mean': float(tas.mean().values),
                'std': float(tas.std().values),
                'min': float(tas.min().values),
                'max': float(tas.max().values),
                'units': str(tas.attrs.get('units', 'K'))
            }
            logger.info(f"Temperature stats: {statistics['temperature']}")

        # Precipitation statistics
        if 'pr' in ds:
            pr = ds['pr']
            statistics['precipitation'] = {
                'mean': float(pr.mean().values),
                'std': float(pr.std().values),
                'total': float(pr.sum().values),
                'min': float(pr.min().values),
                'max': float(pr.max().values),
                'units': str(pr.attrs.get('units', 'kg m-2 s-1'))
            }
            logger.info(f"Precipitation stats: {statistics['precipitation']}")

        # Dataset info
        statistics['dataset_info'] = {
            'variables': list(ds.data_vars),
            'dimensions': dict(ds.dims),
            'global_attributes': dict(ds.attrs)
        }

        result = {
            'file': filename,
            'timestamp': datetime.utcnow().isoformat(),
            'processing_mode': mode,
            'statistics': statistics,
            'status': 'success'
        }

        ds.close()
        return result

    except Exception as e:
        logger.error(f"Error processing netCDF: {e}")
        raise


def process_generic_file(file_data, filename):
    """
    Process generic file (fallback when netCDF not available).

    Args:
        file_data (bytes): File content
        filename (str): Original filename

    Returns:
        dict: Basic processing results
    """
    logger.info(f"Processing generic file: {filename}")

    result = {
        'file': filename,
        'timestamp': datetime.utcnow().isoformat(),
        'size_bytes': len(file_data),
        'size_mb': len(file_data) / 1e6,
        'status': 'success',
        'note': 'Generic processing - netCDF4 not available in Lambda'
    }

    return result


def generate_results_key(input_key):
    """
    Generate S3 key for results file.

    Args:
        input_key (str): Input S3 key

    Returns:
        str: Results S3 key
    """
    base = os.path.splitext(os.path.basename(input_key))[0]
    return f"results/{base}_processed.json"


def upload_results(bucket, key, results):
    """
    Upload results to S3.

    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        results (dict): Results to upload
    """
    # Convert to JSON
    body = json.dumps(results, indent=2, default=str)

    # Upload to S3
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body.encode('utf-8'),
        ContentType='application/json',
        Metadata={
            'processed-at': datetime.utcnow().isoformat(),
            'source': 'lambda-climate-processor'
        }
    )

    logger.info(f"Uploaded results: {len(body)} bytes")


def error_response(error_msg, filename=None):
    """
    Generate error response.

    Args:
        error_msg (str): Error message
        filename (str): Optional filename being processed

    Returns:
        dict: Error response
    """
    response = {
        'statusCode': 500,
        'body': json.dumps({
            'message': 'Processing failed',
            'error': error_msg,
            'timestamp': datetime.utcnow().isoformat()
        })
    }

    if filename:
        response['body'] = json.dumps({
            'file': filename,
            **json.loads(response['body'])
        })

    return response


# Local testing
if __name__ == '__main__':
    # Set environment variables
    os.environ['BUCKET_NAME'] = 'climate-data-test'
    os.environ['PROCESS_MODE'] = 'calculate_statistics'

    # Create test event
    test_event = {
        'Records': [
            {
                's3': {
                    'bucket': {'name': 'climate-data-test'},
                    'object': {'key': 'raw/test_data.nc'}
                }
            }
        ]
    }

    class MockContext:
        pass

    # Test handler (will fail without actual S3 bucket)
    print("Testing lambda_handler...")
    print("Note: This will fail without a valid S3 bucket")

    try:
        result = lambda_handler(test_event, MockContext())
        print(f"Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
