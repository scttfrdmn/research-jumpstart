"""
AWS Lambda function for student learning analytics.

This function processes student activity data uploaded to S3 and:
- Calculates grade averages and trends
- Computes completion rates
- Calculates engagement scores
- Identifies at-risk students
- Analyzes learning curves
- Computes mastery learning metrics
- Stores results in DynamoDB

Trigger: S3 PUT events on raw-data/ prefix
Runtime: Python 3.11
Memory: 512 MB
Timeout: 60 seconds
"""

import json
import os
import io
from datetime import datetime
from typing import Dict, List, Tuple
import logging

import boto3
import pandas as pd
import numpy as np
from scipy import stats

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Environment variables
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'StudentAnalytics')
S3_BUCKET = os.environ.get('S3_BUCKET', '')

# DynamoDB table
table = dynamodb.Table(DYNAMODB_TABLE)


def lambda_handler(event, context):
    """
    Lambda handler triggered by S3 PUT events.

    Args:
        event: S3 event containing bucket and object key
        context: Lambda context object

    Returns:
        Response with processing results
    """
    try:
        # Parse S3 event
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        logger.info(f"Processing file: s3://{bucket}/{key}")

        # Download and read CSV from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content))

        logger.info(f"Loaded {len(df)} records from CSV")

        # Process student analytics
        results = process_student_analytics(df)

        # Store results in DynamoDB
        stored_count = store_results_dynamodb(results)

        # Save aggregated results to S3
        save_results_to_s3(results, bucket, key)

        logger.info(f"Successfully processed {len(results)} students")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Analytics processing complete',
                'students_processed': len(results),
                'records_analyzed': len(df),
                'stored_to_dynamodb': stored_count,
                'source_file': key
            })
        }

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }


def process_student_analytics(df: pd.DataFrame) -> List[Dict]:
    """
    Process student data and calculate analytics metrics.

    Args:
        df: DataFrame with student activity records

    Returns:
        List of dictionaries with student metrics
    """
    results = []

    # Group by student and course
    for (student_id, course_id), group in df.groupby(['student_id', 'course_id']):
        metrics = calculate_student_metrics(student_id, course_id, group)
        results.append(metrics)

    return results


def calculate_student_metrics(student_id: str, course_id: str,
                              activity_df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive metrics for a single student in a course.

    Args:
        student_id: Anonymized student identifier
        course_id: Course identifier
        activity_df: DataFrame of student's activities

    Returns:
        Dictionary with calculated metrics
    """
    # Grade metrics
    scores = activity_df[activity_df['submitted'] == True]['score']

    if len(scores) > 0:
        avg_grade = float(scores.mean())
        median_grade = float(scores.median())
        std_grade = float(scores.std()) if len(scores) > 1 else 0.0

        # Calculate grade trend (linear regression slope)
        if len(scores) > 2:
            x = np.arange(len(scores))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
            grade_trend = float(slope)
        else:
            grade_trend = 0.0
    else:
        avg_grade = 0.0
        median_grade = 0.0
        std_grade = 0.0
        grade_trend = 0.0

    # Completion metrics
    total_assessments = len(activity_df)
    completed_assessments = int(activity_df['submitted'].sum())
    completion_rate = (completed_assessments / total_assessments * 100) if total_assessments > 0 else 0.0

    # Engagement metrics
    total_time = float(activity_df['time_on_task_minutes'].sum())
    avg_time_per_task = float(activity_df['time_on_task_minutes'].mean())
    total_resources = int(activity_df['resource_views'].sum())

    # Normalize engagement score (0-100)
    # Based on time on task and resource views
    time_score = min(100, (total_time / (total_assessments * 30)) * 100)  # 30 min per task expected
    resource_score = min(100, (total_resources / (total_assessments * 5)) * 100)  # 5 views per task expected
    engagement_score = (time_score * 0.6 + resource_score * 0.4)

    # At-risk identification
    risk_level, risk_factors = identify_risk_level(
        avg_grade, grade_trend, completion_rate, engagement_score
    )

    # Mastery learning metrics
    mastery_metrics = calculate_mastery_metrics(activity_df)

    # Construct result
    return {
        'student_id': student_id,
        'course_id': course_id,
        'avg_grade': round(avg_grade, 2),
        'median_grade': round(median_grade, 2),
        'std_grade': round(std_grade, 2),
        'grade_trend': round(grade_trend, 2),
        'total_assessments': total_assessments,
        'completed_assessments': completed_assessments,
        'completion_rate': round(completion_rate, 2),
        'total_time_minutes': round(total_time, 2),
        'avg_time_per_task': round(avg_time_per_task, 2),
        'total_resources_viewed': total_resources,
        'engagement_score': round(engagement_score, 2),
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'learning_gain': mastery_metrics['learning_gain'],
        'mastery_level': mastery_metrics['mastery_level'],
        'consistency_score': mastery_metrics['consistency_score'],
        'last_updated': datetime.utcnow().isoformat(),
        'record_count': len(activity_df)
    }


def identify_risk_level(avg_grade: float, grade_trend: float,
                       completion_rate: float, engagement_score: float) -> Tuple[str, List[str]]:
    """
    Identify student risk level based on multiple factors.

    Args:
        avg_grade: Average grade percentage
        grade_trend: Grade trend (slope)
        completion_rate: Completion rate percentage
        engagement_score: Engagement score (0-100)

    Returns:
        Tuple of (risk_level, list of risk factors)
    """
    risk_factors = []
    risk_score = 0

    # Low grade
    if avg_grade < 60:
        risk_factors.append('failing_grade')
        risk_score += 3
    elif avg_grade < 70:
        risk_factors.append('low_grade')
        risk_score += 2

    # Declining performance
    if grade_trend < -2:
        risk_factors.append('declining_performance')
        risk_score += 2
    elif grade_trend < 0:
        risk_factors.append('slight_decline')
        risk_score += 1

    # Low completion
    if completion_rate < 60:
        risk_factors.append('very_low_completion')
        risk_score += 3
    elif completion_rate < 80:
        risk_factors.append('low_completion')
        risk_score += 1

    # Low engagement
    if engagement_score < 40:
        risk_factors.append('very_low_engagement')
        risk_score += 2
    elif engagement_score < 60:
        risk_factors.append('low_engagement')
        risk_score += 1

    # Determine risk level
    if risk_score >= 5:
        risk_level = 'high'
    elif risk_score >= 3:
        risk_level = 'medium'
    elif risk_score >= 1:
        risk_level = 'low'
    else:
        risk_level = 'none'

    return risk_level, risk_factors


def calculate_mastery_metrics(activity_df: pd.DataFrame) -> Dict:
    """
    Calculate mastery learning metrics.

    Args:
        activity_df: DataFrame of student activities

    Returns:
        Dictionary with mastery metrics
    """
    submitted_df = activity_df[activity_df['submitted'] == True].sort_values('assessment_number')
    scores = submitted_df['score'].values

    if len(scores) >= 2:
        # Learning gain: improvement from first to last
        learning_gain = float(scores[-1] - scores[0])

        # Mastery level: percentage of assessments above 80%
        mastery_count = (scores >= 80).sum()
        mastery_level = (mastery_count / len(scores) * 100)

        # Consistency: inverse of coefficient of variation
        if scores.mean() > 0:
            cv = scores.std() / scores.mean()
            consistency_score = max(0, 100 - (cv * 100))
        else:
            consistency_score = 0.0

    else:
        learning_gain = 0.0
        mastery_level = 0.0
        consistency_score = 0.0

    return {
        'learning_gain': round(learning_gain, 2),
        'mastery_level': round(mastery_level, 2),
        'consistency_score': round(consistency_score, 2)
    }


def store_results_dynamodb(results: List[Dict]) -> int:
    """
    Store student metrics in DynamoDB.

    Args:
        results: List of student metric dictionaries

    Returns:
        Number of records successfully stored
    """
    stored_count = 0

    for metrics in results:
        try:
            # Convert lists to strings for DynamoDB
            item = metrics.copy()
            item['risk_factors'] = json.dumps(item['risk_factors'])

            # Put item in DynamoDB
            table.put_item(Item=item)
            stored_count += 1

        except Exception as e:
            logger.error(f"Error storing item for student {metrics['student_id']}: {str(e)}")

    logger.info(f"Stored {stored_count}/{len(results)} records to DynamoDB")
    return stored_count


def save_results_to_s3(results: List[Dict], bucket: str, source_key: str):
    """
    Save aggregated results to S3 for Athena queries.

    Args:
        results: List of student metrics
        bucket: S3 bucket name
        source_key: Original source file key
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Convert risk_factors list to string
        df['risk_factors'] = df['risk_factors'].apply(lambda x: ','.join(x) if x else '')

        # Generate output key
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        output_key = f"processed-data/analytics_{timestamp}.csv"

        # Upload to S3
        csv_buffer = df.to_csv(index=False)
        s3_client.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=csv_buffer.encode('utf-8'),
            ContentType='text/csv',
            Metadata={
                'source_file': source_key,
                'processed_at': datetime.utcnow().isoformat(),
                'record_count': str(len(df))
            }
        )

        logger.info(f"Saved results to s3://{bucket}/{output_key}")

    except Exception as e:
        logger.error(f"Error saving results to S3: {str(e)}")


# For local testing
if __name__ == '__main__':
    # Test event
    test_event = {
        'Records': [{
            's3': {
                'bucket': {'name': 'learning-analytics-test'},
                'object': {'key': 'raw-data/test.csv'}
            }
        }]
    }

    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))
