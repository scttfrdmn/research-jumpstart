"""
AWS Lambda function for preprocessing medical images.

This Lambda function:
1. Triggered by S3 upload events (raw-images/ prefix)
2. Downloads image from S3
3. Preprocesses image (resize, normalize)
4. Uploads processed image to S3
5. Stores metadata in DynamoDB
6. Returns processing results

Handler: lambda_handler
Memory: 256 MB
Timeout: 300 seconds
Environment Variables:
  - PROCESSED_BUCKET: S3 bucket for processed images
  - DYNAMODB_TABLE: DynamoDB table for metadata
  - IMAGE_SIZE: Target image size (default: 224)

Supported formats: PNG, JPG, JPEG, DICOM
"""

import json
import os
import io
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import boto3
import numpy as np
from PIL import Image
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Environment variables
PROCESSED_BUCKET = os.environ.get('PROCESSED_BUCKET', os.environ.get('S3_BUCKET_NAME'))
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'medical-predictions')
TARGET_IMAGE_SIZE = int(os.environ.get('IMAGE_SIZE', 224))

# Constants
SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB


def get_image_extension(key: str) -> str:
    """Extract file extension from S3 key."""
    return os.path.splitext(key.lower())[1]


def download_image_from_s3(bucket: str, key: str) -> Tuple[Optional[bytes], Dict[str, Any]]:
    """
    Download image from S3.

    Args:
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        Tuple of (image_bytes, metadata_dict)
    """
    try:
        logger.info(f"Downloading {key} from {bucket}")

        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_bytes = response['Body'].read()

        # Check file size
        file_size = len(image_bytes)
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File too large: {file_size} bytes > {MAX_FILE_SIZE}")

        metadata = {
            'source_bucket': bucket,
            'source_key': key,
            'source_size': file_size,
            'content_type': response.get('ContentType', 'unknown'),
            'last_modified': response.get('LastModified', datetime.utcnow()).isoformat()
        }

        logger.info(f"Successfully downloaded {key} ({file_size} bytes)")
        return image_bytes, metadata

    except s3_client.exceptions.NoSuchKey:
        logger.error(f"Image not found: {key}")
        return None, {'error': 'Image not found'}
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        return None, {'error': str(e)}


def preprocess_image(image_bytes: bytes, target_size: int = 224) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Preprocess medical image.

    Steps:
    1. Load image
    2. Convert to grayscale (for medical images)
    3. Resize to target size
    4. Normalize to [0, 1] range
    5. Return numpy array and metadata

    Args:
        image_bytes: Raw image bytes
        target_size: Target image size (square)

    Returns:
        Tuple of (processed_image_array, preprocessing_metadata)
    """
    metadata = {}
    start_time = time.time()

    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        logger.info(f"Image format: {image.format}, Mode: {image.mode}, Size: {image.size}")

        # Store original properties
        metadata['original_format'] = str(image.format)
        metadata['original_mode'] = image.mode
        metadata['original_size'] = image.size

        # Convert to grayscale (standard for medical images)
        if image.mode != 'L':
            image = image.convert('L')

        # Resize to target size
        image_resized = image.resize((target_size, target_size), Image.Resampling.LANCZOS)

        # Convert to numpy array
        image_array = np.array(image_resized, dtype=np.float32)

        # Normalize to [0, 1] range
        image_normalized = image_array / 255.0

        # Store processing metadata
        metadata['processed_size'] = (target_size, target_size)
        metadata['dtype'] = str(image_normalized.dtype)
        metadata['min_value'] = float(image_normalized.min())
        metadata['max_value'] = float(image_normalized.max())
        metadata['mean_value'] = float(image_normalized.mean())
        metadata['std_value'] = float(image_normalized.std())
        metadata['processing_time_ms'] = (time.time() - start_time) * 1000

        logger.info(f"Image preprocessing completed: {target_size}x{target_size}, "
                   f"normalized [{metadata['min_value']:.3f}, {metadata['max_value']:.3f}]")

        return image_normalized, metadata

    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        metadata['error'] = str(e)
        return None, metadata


def save_processed_image(image_array: np.ndarray, bucket: str, output_prefix: str,
                        source_key: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Save processed image to S3.

    Args:
        image_array: Processed image as numpy array
        bucket: S3 bucket name
        output_prefix: S3 prefix for output
        source_key: Original S3 key (for naming)

    Returns:
        Tuple of (success, metadata)
    """
    metadata = {}
    start_time = time.time()

    try:
        # Generate output key
        base_name = os.path.basename(source_key)
        name_without_ext = os.path.splitext(base_name)[0]
        output_key = f"{output_prefix}{name_without_ext}_processed.png"

        # Convert numpy array back to image for saving as PNG
        image_normalized = (image_array * 255).astype(np.uint8)
        image = Image.fromarray(image_normalized, mode='L')

        # Save to bytes buffer
        image_buffer = io.BytesIO()
        image.save(image_buffer, format='PNG')
        image_buffer.seek(0)
        image_bytes = image_buffer.getvalue()

        # Upload to S3
        logger.info(f"Uploading processed image to {output_key}")
        s3_client.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=image_bytes,
            ContentType='image/png',
            Metadata={
                'processed_at': datetime.utcnow().isoformat(),
                'source_key': source_key,
                'image_size': f"{image_array.shape[0]}x{image_array.shape[1]}"
            }
        )

        metadata['output_bucket'] = bucket
        metadata['output_key'] = output_key
        metadata['output_size'] = len(image_bytes)
        metadata['save_time_ms'] = (time.time() - start_time) * 1000

        logger.info(f"Successfully saved processed image: {output_key} ({len(image_bytes)} bytes)")
        return True, metadata

    except Exception as e:
        logger.error(f"Error saving processed image: {str(e)}")
        metadata['error'] = str(e)
        return False, metadata


def store_metadata_dynamodb(image_id: str, table_name: str, metadata: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Store prediction metadata in DynamoDB.

    Args:
        image_id: Unique image identifier
        table_name: DynamoDB table name
        metadata: Metadata dictionary

    Returns:
        Tuple of (success, message)
    """
    try:
        table = dynamodb.Table(table_name)

        # Prepare item for DynamoDB
        item = {
            'image_id': image_id,
            'timestamp': int(datetime.utcnow().timestamp() * 1000),  # milliseconds
            'metadata': json.dumps(metadata, default=str)  # Store metadata as JSON string
        }

        logger.info(f"Storing metadata for {image_id} in DynamoDB table {table_name}")
        response = table.put_item(Item=item)

        logger.info(f"Successfully stored metadata: {image_id}")
        return True, f"Metadata stored: {image_id}"

    except Exception as e:
        logger.error(f"Error storing metadata in DynamoDB: {str(e)}")
        return False, str(e)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for image preprocessing.

    Args:
        event: S3 event containing bucket and key information
        context: Lambda context

    Returns:
        Response dictionary with status and results
    """
    logger.info(f"Lambda invoked with event: {json.dumps(event)}")

    response = {
        'statusCode': 200,
        'body': {},
        'errors': []
    }

    try:
        # Extract S3 bucket and key from event
        # Handle both direct S3 events and test events
        if 'Records' in event:
            s3_record = event['Records'][0]
            bucket = s3_record['s3']['bucket']['name']
            key = s3_record['s3']['object']['key']
        else:
            bucket = event.get('s3', {}).get('bucket', {}).get('name')
            key = event.get('s3', {}).get('object', {}).get('key')

        if not bucket or not key:
            raise ValueError("Invalid event: bucket and key not found")

        logger.info(f"Processing image: s3://{bucket}/{key}")

        # Skip if not in raw-images folder
        if 'raw-images' not in key:
            logger.info(f"Skipping {key} - not in raw-images prefix")
            response['statusCode'] = 200
            response['body'] = {'message': 'Skipped - not raw image'}
            return response

        # Generate unique image ID
        image_id = str(uuid.uuid4())

        # Download image from S3
        image_bytes, download_metadata = download_image_from_s3(bucket, key)
        if image_bytes is None:
            response['statusCode'] = 400
            response['errors'].append(f"Failed to download image: {download_metadata.get('error')}")
            return response

        # Preprocess image
        processed_image, preprocess_metadata = preprocess_image(image_bytes, TARGET_IMAGE_SIZE)
        if processed_image is None:
            response['statusCode'] = 400
            response['errors'].append(f"Failed to preprocess image: {preprocess_metadata.get('error')}")
            return response

        # Save processed image
        processed_bucket = PROCESSED_BUCKET or bucket
        save_success, save_metadata = save_processed_image(
            processed_image,
            processed_bucket,
            'processed-images/',
            key
        )
        if not save_success:
            response['statusCode'] = 400
            response['errors'].append(f"Failed to save processed image: {save_metadata.get('error')}")
            return response

        # Combine all metadata
        combined_metadata = {
            **download_metadata,
            **preprocess_metadata,
            **save_metadata
        }

        # Store metadata in DynamoDB
        store_success, store_message = store_metadata_dynamodb(
            image_id,
            DYNAMODB_TABLE,
            combined_metadata
        )
        if not store_success:
            logger.warning(f"Failed to store metadata: {store_message}")
            response['errors'].append(f"Warning: {store_message}")

        # Build success response
        response['statusCode'] = 200
        response['body'] = {
            'image_id': image_id,
            'source_key': key,
            'source_size': download_metadata.get('source_size'),
            'processed_key': save_metadata.get('output_key'),
            'processed_size': save_metadata.get('output_size'),
            'processing_time_ms': preprocess_metadata.get('processing_time_ms'),
            'image_stats': {
                'min': preprocess_metadata.get('min_value'),
                'max': preprocess_metadata.get('max_value'),
                'mean': preprocess_metadata.get('mean_value'),
                'std': preprocess_metadata.get('std_value')
            },
            'metadata_stored': store_success
        }

        logger.info(f"Successfully processed image: {image_id}")
        logger.info(f"Response: {json.dumps(response, default=str)}")

        return response

    except Exception as e:
        logger.error(f"Unexpected error in lambda_handler: {str(e)}", exc_info=True)
        response['statusCode'] = 500
        response['errors'].append(f"Internal error: {str(e)}")
        return response


if __name__ == '__main__':
    # For local testing
    test_event = {
        'Records': [
            {
                's3': {
                    'bucket': {'name': 'medical-images-test'},
                    'object': {'key': 'raw-images/sample.png'}
                }
            }
        ]
    }

    class MockContext:
        pass

    result = lambda_handler(test_event, MockContext())
    print(json.dumps(result, indent=2, default=str))
