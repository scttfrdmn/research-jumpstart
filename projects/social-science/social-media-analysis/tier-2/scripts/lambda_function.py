"""
AWS Lambda function for social media sentiment analysis.

This Lambda function:
1. Triggered by S3 upload events (raw/ prefix)
2. Downloads social media posts from S3
3. Analyzes sentiment using AWS Comprehend
4. Extracts hashtags and mentions
5. Stores results in DynamoDB
6. Returns processing statistics

Handler: lambda_handler
Memory: 256 MB
Timeout: 30 seconds
Environment Variables:
  - BUCKET_NAME: S3 bucket for processed data
  - DYNAMODB_TABLE: DynamoDB table name (default: SocialMediaPosts)
  - AWS_REGION: AWS region (default: us-east-1)

Supported formats: JSON arrays of posts
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import logging

import boto3
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3_client = boto3.client('s3')
comprehend_client = boto3.client('comprehend')
dynamodb = boto3.resource('dynamodb')

# Environment variables
BUCKET_NAME = os.environ.get('BUCKET_NAME', os.environ.get('S3_BUCKET_NAME'))
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'SocialMediaPosts')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

# Constants
MAX_TEXT_LENGTH = 5000  # AWS Comprehend limit
HASHTAG_PATTERN = re.compile(r'#\w+')
MENTION_PATTERN = re.compile(r'@\w+')


def extract_hashtags(text: str) -> List[str]:
    """
    Extract hashtags from text.

    Args:
        text: Post text

    Returns:
        List of hashtags (without # symbol)
    """
    hashtags = HASHTAG_PATTERN.findall(text)
    return [tag[1:] for tag in hashtags]  # Remove # symbol


def extract_mentions(text: str) -> List[str]:
    """
    Extract @mentions from text.

    Args:
        text: Post text

    Returns:
        List of mentions (without @ symbol)
    """
    mentions = MENTION_PATTERN.findall(text)
    return [mention[1:] for mention in mentions]  # Remove @ symbol


def analyze_sentiment(text: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Analyze sentiment using AWS Comprehend.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (sentiment_result, error_message)
    """
    try:
        # Truncate text if too long
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(f"Text truncated from {len(text)} to {MAX_TEXT_LENGTH} chars")
            text = text[:MAX_TEXT_LENGTH]

        # Call Comprehend DetectSentiment API
        response = comprehend_client.detect_sentiment(
            Text=text,
            LanguageCode='en'
        )

        result = {
            'sentiment': response['Sentiment'],
            'positive_score': response['SentimentScore']['Positive'],
            'negative_score': response['SentimentScore']['Negative'],
            'neutral_score': response['SentimentScore']['Neutral'],
            'mixed_score': response['SentimentScore']['Mixed']
        }

        logger.info(f"Sentiment analysis: {result['sentiment']}")
        return result, None

    except ClientError as e:
        error_msg = f"Comprehend API error: {e}"
        logger.error(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Unexpected sentiment analysis error: {e}"
        logger.error(error_msg)
        return None, error_msg


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extract entities using AWS Comprehend (optional, costs extra).

    Args:
        text: Text to analyze

    Returns:
        List of entity dictionaries
    """
    try:
        # Truncate text if too long
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]

        response = comprehend_client.detect_entities(
            Text=text,
            LanguageCode='en'
        )

        entities = []
        for entity in response.get('Entities', [])[:10]:  # Limit to top 10
            entities.append({
                'text': entity['Text'],
                'type': entity['Type'],
                'score': entity['Score']
            })

        return entities

    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return []


def store_in_dynamodb(post_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Store post analysis in DynamoDB.

    Args:
        post_data: Post data with sentiment analysis

    Returns:
        Tuple of (success, message)
    """
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)

        # Prepare item for DynamoDB
        item = {
            'post_id': post_data['post_id'],
            'timestamp': int(post_data.get('timestamp', time.time())),
            'text': post_data['text'],
            'sentiment': post_data['sentiment'],
            'positive_score': post_data['positive_score'],
            'negative_score': post_data['negative_score'],
            'neutral_score': post_data['neutral_score'],
            'mixed_score': post_data['mixed_score'],
            'hashtags': post_data.get('hashtags', []),
            'mentions': post_data.get('mentions', []),
            'processed_at': datetime.utcnow().isoformat()
        }

        # Add optional fields
        if 'user_id' in post_data:
            item['user_id'] = post_data['user_id']
        if 'username' in post_data:
            item['username'] = post_data['username']
        if 'entities' in post_data:
            item['entities'] = post_data['entities']

        # Write to DynamoDB
        table.put_item(Item=item)

        logger.info(f"Stored post {post_data['post_id']} in DynamoDB")
        return True, f"Stored: {post_data['post_id']}"

    except ClientError as e:
        error_msg = f"DynamoDB error: {e}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected DynamoDB error: {e}"
        logger.error(error_msg)
        return False, error_msg


def process_post(post: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process single social media post.

    Args:
        post: Post dictionary with at least 'post_id' and 'text'

    Returns:
        Processing result dictionary
    """
    start_time = time.time()
    result = {
        'post_id': post.get('post_id', 'unknown'),
        'status': 'pending',
        'errors': []
    }

    try:
        # Validate required fields
        if 'text' not in post or not post['text']:
            result['status'] = 'failed'
            result['errors'].append('Missing or empty text field')
            return result

        text = post['text']

        # Extract hashtags and mentions
        hashtags = extract_hashtags(text)
        mentions = extract_mentions(text)

        # Analyze sentiment
        sentiment_result, sentiment_error = analyze_sentiment(text)
        if sentiment_error:
            result['status'] = 'failed'
            result['errors'].append(sentiment_error)
            return result

        # Build complete post data
        post_data = {
            'post_id': post.get('post_id', f"post_{int(time.time())}"),
            'timestamp': post.get('timestamp', int(time.time())),
            'text': text,
            'user_id': post.get('user_id', ''),
            'username': post.get('username', ''),
            'sentiment': sentiment_result['sentiment'],
            'positive_score': sentiment_result['positive_score'],
            'negative_score': sentiment_result['negative_score'],
            'neutral_score': sentiment_result['neutral_score'],
            'mixed_score': sentiment_result['mixed_score'],
            'hashtags': hashtags,
            'mentions': mentions
        }

        # Optional: Extract entities (costs extra, comment out if not needed)
        # entities = extract_entities(text)
        # post_data['entities'] = entities

        # Store in DynamoDB
        store_success, store_message = store_in_dynamodb(post_data)
        if not store_success:
            result['status'] = 'failed'
            result['errors'].append(store_message)
            return result

        # Success
        result['status'] = 'success'
        result['sentiment'] = sentiment_result['sentiment']
        result['processing_time_ms'] = (time.time() - start_time) * 1000

        return result

    except Exception as e:
        logger.error(f"Error processing post {result['post_id']}: {e}")
        result['status'] = 'failed'
        result['errors'].append(str(e))
        return result


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for sentiment analysis.

    Args:
        event: S3 event containing bucket and key information
        context: Lambda context

    Returns:
        Response dictionary with status and results
    """
    logger.info(f"Lambda invoked with event: {json.dumps(event)}")

    response = {
        'statusCode': 200,
        'body': {
            'processed': 0,
            'successful': 0,
            'failed': 0
        },
        'errors': []
    }

    try:
        # Extract S3 bucket and key from event
        if 'Records' in event:
            s3_record = event['Records'][0]
            bucket = s3_record['s3']['bucket']['name']
            key = s3_record['s3']['object']['key']
        else:
            bucket = event.get('bucket')
            key = event.get('key')

        if not bucket or not key:
            raise ValueError("Invalid event: bucket and key not found")

        logger.info(f"Processing file: s3://{bucket}/{key}")

        # Download JSON from S3
        try:
            s3_response = s3_client.get_object(Bucket=bucket, Key=key)
            file_content = s3_response['Body'].read().decode('utf-8')
            posts = json.loads(file_content)

            # Handle both single post and array of posts
            if isinstance(posts, dict):
                posts = [posts]

            logger.info(f"Loaded {len(posts)} posts from S3")

        except ClientError as e:
            error_msg = f"S3 download error: {e}"
            logger.error(error_msg)
            response['statusCode'] = 400
            response['errors'].append(error_msg)
            return response

        # Process each post
        for post in posts:
            result = process_post(post)
            response['body']['processed'] += 1

            if result['status'] == 'success':
                response['body']['successful'] += 1
            else:
                response['body']['failed'] += 1
                response['errors'].extend(result.get('errors', []))

        # Log summary
        logger.info(f"Processing complete: {response['body']}")

        return response

    except Exception as e:
        logger.error(f"Unexpected error in lambda_handler: {e}", exc_info=True)
        response['statusCode'] = 500
        response['errors'].append(f"Internal error: {str(e)}")
        return response


if __name__ == '__main__':
    # For local testing
    test_event = {
        'bucket': 'social-media-data-test',
        'key': 'raw/sample_posts.json'
    }

    # Or test with sample posts directly
    test_posts = [
        {
            'post_id': 'test001',
            'text': 'I love this new product! It works great and exceeded my expectations.',
            'timestamp': 1705246800,
            'user_id': 'user123',
            'username': 'happy_customer'
        },
        {
            'post_id': 'test002',
            'text': 'Terrible service. Very disappointed with this experience. #frustrated',
            'timestamp': 1705246801,
            'user_id': 'user456',
            'username': 'angry_customer'
        }
    ]

    # Test sentiment analysis locally
    for post in test_posts:
        print(f"\nProcessing: {post['post_id']}")
        result = process_post(post)
        print(f"Result: {json.dumps(result, indent=2)}")
