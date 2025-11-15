"""
AWS Lambda function for epidemiological analysis and outbreak detection.

This Lambda function:
1. Triggered by S3 upload events (case-data/ prefix)
2. Downloads case report CSV from S3
3. Calculates epidemiological metrics (incidence, prevalence, CFR)
4. Detects outbreak signals using statistical methods
5. Estimates R0 (basic reproductive number)
6. Generates epidemic curve data
7. Sends SNS alerts if outbreak detected
8. Stores results in DynamoDB

Handler: lambda_handler
Memory: 256-512 MB
Timeout: 60-300 seconds

Environment Variables:
  - S3_BUCKET_NAME: S3 bucket for data
  - DYNAMODB_TABLE: DynamoDB table name
  - SNS_TOPIC_ARN: SNS topic ARN for alerts
  - OUTBREAK_THRESHOLD: Standard deviations for outbreak detection (default: 2.0)
  - POPULATION_SIZE: Population for rate calculations (default: 100000)

Author: Research Jumpstart
"""

import json
import os
import io
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

import boto3
import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sns_client = boto3.client('sns')

# Environment variables
S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
DYNAMODB_TABLE_NAME = os.environ.get('DYNAMODB_TABLE', 'DiseaseReports')
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN')
OUTBREAK_THRESHOLD = float(os.environ.get('OUTBREAK_THRESHOLD', '2.0'))
POPULATION_SIZE = int(os.environ.get('POPULATION_SIZE', '100000'))

# Constants
MIN_CASES_FOR_R0 = 10  # Minimum cases needed for R0 estimation
SERIAL_INTERVAL = 3.0  # Days (typical for influenza, adjust per disease)


def lambda_handler(event, context):
    """
    Main Lambda handler function.

    Args:
        event: S3 event notification
        context: Lambda context

    Returns:
        Response dict with status code and body
    """
    logger.info("=== Starting epidemiological analysis ===")
    logger.info(f"Event: {json.dumps(event)}")

    try:
        # Extract S3 bucket and key from event
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        logger.info(f"Processing: s3://{bucket}/{key}")

        # Download and parse CSV data
        case_data = download_and_parse_csv(bucket, key)

        if case_data.empty:
            logger.warning("No case data found in CSV")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No data to process'})
            }

        logger.info(f"Loaded {len(case_data)} case records")

        # Calculate epidemiological metrics
        metrics = calculate_epi_metrics(case_data)
        logger.info(f"Calculated metrics: {metrics}")

        # Detect outbreak signals
        outbreak_signals = detect_outbreak(case_data)
        logger.info(f"Outbreak detection: {outbreak_signals}")

        # Estimate R0 if sufficient data
        r0_estimate = estimate_r0(case_data)
        if r0_estimate:
            logger.info(f"R0 estimate: {r0_estimate}")

        # Generate epidemic curve data
        epi_curve = generate_epidemic_curve(case_data)

        # Store results in DynamoDB
        store_results(case_data, metrics, outbreak_signals, r0_estimate, epi_curve)

        # Send SNS alert if outbreak detected
        if outbreak_signals['outbreak_detected']:
            send_outbreak_alert(outbreak_signals, metrics, r0_estimate)

        # Return success
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Analysis complete',
                'cases_processed': len(case_data),
                'outbreak_detected': outbreak_signals['outbreak_detected'],
                'metrics': metrics,
                'r0_estimate': r0_estimate
            })
        }

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def download_and_parse_csv(bucket: str, key: str) -> pd.DataFrame:
    """
    Download CSV from S3 and parse into DataFrame.

    Args:
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        DataFrame with case data
    """
    try:
        # Download file
        response = s3_client.get_object(Bucket=bucket, Key=key)
        csv_content = response['Body'].read().decode('utf-8')

        # Parse CSV
        df = pd.read_csv(io.StringIO(csv_content))

        # Convert date columns
        if 'report_date' in df.columns:
            df['report_date'] = pd.to_datetime(df['report_date'])

        if 'symptom_onset_date' in df.columns:
            df['symptom_onset_date'] = pd.to_datetime(df['symptom_onset_date'], errors='coerce')

        # Sort by date
        df = df.sort_values('report_date')

        return df

    except Exception as e:
        logger.error(f"Error downloading/parsing CSV: {str(e)}")
        raise


def calculate_epi_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate key epidemiological metrics.

    Args:
        df: DataFrame with case data

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    total_cases = len(df)
    metrics['total_cases'] = total_cases

    # Incidence rate (per 100,000 population)
    time_span_days = (df['report_date'].max() - df['report_date'].min()).days + 1
    metrics['incidence_rate'] = (total_cases / POPULATION_SIZE) * 100000
    metrics['time_span_days'] = time_span_days

    # Prevalence (active cases per 100 population)
    # Assume active if reported in last 14 days
    recent_cutoff = df['report_date'].max() - timedelta(days=14)
    active_cases = len(df[df['report_date'] >= recent_cutoff])
    metrics['prevalence'] = (active_cases / POPULATION_SIZE) * 100
    metrics['active_cases'] = active_cases

    # Case fatality rate (CFR)
    if 'outcome' in df.columns:
        fatal_cases = len(df[df['outcome'] == 'fatal'])
        metrics['case_fatality_rate'] = (fatal_cases / total_cases) * 100 if total_cases > 0 else 0
        metrics['fatal_cases'] = fatal_cases

        # Hospitalization rate
        hospitalized = len(df[df['outcome'].isin(['hospitalized', 'icu'])])
        metrics['hospitalization_rate'] = (hospitalized / total_cases) * 100 if total_cases > 0 else 0
        metrics['hospitalized_cases'] = hospitalized

    # Attack rate by region
    if 'region' in df.columns:
        attack_rates = {}
        for region in df['region'].unique():
            region_cases = len(df[df['region'] == region])
            attack_rates[str(region)] = (region_cases / POPULATION_SIZE) * 100

        metrics['attack_rate_by_region'] = attack_rates

    # Demographics
    if 'age_group' in df.columns:
        age_distribution = df['age_group'].value_counts().to_dict()
        metrics['age_distribution'] = {str(k): int(v) for k, v in age_distribution.items()}

    if 'sex' in df.columns:
        sex_distribution = df['sex'].value_counts().to_dict()
        metrics['sex_distribution'] = {str(k): int(v) for k, v in sex_distribution.items()}

    # Daily case rate
    daily_cases = df.groupby(df['report_date'].dt.date).size()
    metrics['avg_daily_cases'] = float(daily_cases.mean())
    metrics['max_daily_cases'] = int(daily_cases.max())
    metrics['min_daily_cases'] = int(daily_cases.min())

    return metrics


def detect_outbreak(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect outbreak signals using statistical methods.

    Args:
        df: DataFrame with case data

    Returns:
        Dictionary with outbreak detection results
    """
    outbreak_signals = {
        'outbreak_detected': False,
        'confidence': 'none',
        'reasons': [],
        'detection_method': []
    }

    # Need minimum data for detection
    if len(df) < 7:
        outbreak_signals['reasons'].append('Insufficient data for outbreak detection')
        return outbreak_signals

    # Method 1: Moving average threshold
    daily_cases = df.groupby(df['report_date'].dt.date).size()

    if len(daily_cases) >= 7:
        # Calculate 7-day moving average
        ma_7 = daily_cases.rolling(window=7, min_periods=1).mean()

        # Calculate baseline (first 7 days or all historical)
        baseline = daily_cases.iloc[:7].mean()
        baseline_std = daily_cases.iloc[:7].std()

        if baseline_std > 0:
            # Current average vs baseline
            current_avg = ma_7.iloc[-3:].mean()  # Last 3 days average

            # Z-score calculation
            z_score = (current_avg - baseline) / baseline_std

            if z_score > OUTBREAK_THRESHOLD:
                outbreak_signals['outbreak_detected'] = True
                outbreak_signals['detection_method'].append('moving_average')
                outbreak_signals['reasons'].append(
                    f'Cases exceed baseline by {z_score:.1f} standard deviations'
                )
                outbreak_signals['z_score'] = float(z_score)

    # Method 2: Rapid growth detection
    if len(daily_cases) >= 3:
        recent_trend = daily_cases.iloc[-3:].values
        if len(recent_trend) >= 2:
            growth_rate = (recent_trend[-1] - recent_trend[0]) / (recent_trend[0] + 1)

            if growth_rate > 0.5:  # 50% increase
                outbreak_signals['outbreak_detected'] = True
                outbreak_signals['detection_method'].append('rapid_growth')
                outbreak_signals['reasons'].append(
                    f'Rapid growth detected: {growth_rate*100:.1f}% increase'
                )
                outbreak_signals['growth_rate'] = float(growth_rate)

    # Method 3: Geographic clustering
    if 'region' in df.columns:
        region_counts = df['region'].value_counts()

        # Check if one region has disproportionate cases
        total_regions = len(region_counts)
        if total_regions > 1:
            max_region_cases = region_counts.iloc[0]
            expected_per_region = len(df) / total_regions

            if max_region_cases > expected_per_region * 2:
                outbreak_signals['outbreak_detected'] = True
                outbreak_signals['detection_method'].append('geographic_clustering')
                outbreak_signals['reasons'].append(
                    f'Geographic clustering in {region_counts.index[0]}'
                )
                outbreak_signals['hotspot_region'] = str(region_counts.index[0])

    # Set confidence level
    if outbreak_signals['outbreak_detected']:
        detection_count = len(outbreak_signals['detection_method'])
        if detection_count >= 2:
            outbreak_signals['confidence'] = 'high'
        elif detection_count == 1:
            outbreak_signals['confidence'] = 'medium'
    else:
        outbreak_signals['reasons'] = ['No outbreak signals detected']

    return outbreak_signals


def estimate_r0(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Estimate basic reproductive number (R0).

    Uses exponential growth method:
    R0 = 1 + r * T
    where r is growth rate, T is serial interval

    Args:
        df: DataFrame with case data

    Returns:
        Dictionary with R0 estimate or None
    """
    if len(df) < MIN_CASES_FOR_R0:
        logger.info(f"Insufficient cases for R0 estimation (need {MIN_CASES_FOR_R0})")
        return None

    try:
        # Use symptom onset date if available, otherwise report date
        date_col = 'symptom_onset_date' if 'symptom_onset_date' in df.columns else 'report_date'

        # Daily case counts
        daily_cases = df.groupby(df[date_col].dt.date).size()

        if len(daily_cases) < 5:
            return None

        # Exponential growth fit on early phase (first 70% of data)
        early_phase_size = int(len(daily_cases) * 0.7)
        early_cases = daily_cases.iloc[:early_phase_size]

        # Remove zeros for log transform
        early_cases_nonzero = early_cases[early_cases > 0]

        if len(early_cases_nonzero) < 3:
            return None

        # Log-linear regression
        x = np.arange(len(early_cases_nonzero))
        y = np.log(early_cases_nonzero.values)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Growth rate per day
        growth_rate = slope

        # R0 estimation
        r0 = 1 + (growth_rate * SERIAL_INTERVAL)

        # Confidence interval (approximate)
        r0_lower = 1 + ((growth_rate - 1.96 * std_err) * SERIAL_INTERVAL)
        r0_upper = 1 + ((growth_rate + 1.96 * std_err) * SERIAL_INTERVAL)

        return {
            'r0': float(max(0, r0)),  # R0 cannot be negative
            'r0_lower': float(max(0, r0_lower)),
            'r0_upper': float(max(0, r0_upper)),
            'growth_rate': float(growth_rate),
            'r_squared': float(r_value ** 2),
            'method': 'exponential_growth'
        }

    except Exception as e:
        logger.warning(f"Error estimating R0: {str(e)}")
        return None


def generate_epidemic_curve(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate epidemic curve data (cases by date).

    Args:
        df: DataFrame with case data

    Returns:
        List of dicts with date and case count
    """
    date_col = 'symptom_onset_date' if 'symptom_onset_date' in df.columns else 'report_date'

    daily_cases = df.groupby(df[date_col].dt.date).size()

    epi_curve = [
        {
            'date': str(date),
            'cases': int(count)
        }
        for date, count in daily_cases.items()
    ]

    return epi_curve


def store_results(
    df: pd.DataFrame,
    metrics: Dict,
    outbreak_signals: Dict,
    r0_estimate: Optional[Dict],
    epi_curve: List[Dict]
):
    """
    Store analysis results in DynamoDB.

    Args:
        df: Original case data
        metrics: Epidemiological metrics
        outbreak_signals: Outbreak detection results
        r0_estimate: R0 estimation results
        epi_curve: Epidemic curve data
    """
    try:
        table = dynamodb.Table(DYNAMODB_TABLE_NAME)

        # Store summary record
        summary_item = {
            'case_id': f"SUMMARY_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'report_date': datetime.utcnow().isoformat(),
            'record_type': 'summary',
            'total_cases': metrics['total_cases'],
            'incidence_rate': metrics['incidence_rate'],
            'outbreak_detected': outbreak_signals['outbreak_detected'],
            'outbreak_confidence': outbreak_signals['confidence'],
            'metrics': json.dumps(metrics),
            'outbreak_signals': json.dumps(outbreak_signals),
            'epi_curve': json.dumps(epi_curve)
        }

        if r0_estimate:
            summary_item['r0_estimate'] = r0_estimate['r0']
            summary_item['r0_details'] = json.dumps(r0_estimate)

        table.put_item(Item=summary_item)
        logger.info("✓ Stored summary in DynamoDB")

        # Store individual case records (sample for large datasets)
        max_individual_records = 100
        for idx, row in df.head(max_individual_records).iterrows():
            case_item = {
                'case_id': str(row['case_id']),
                'report_date': row['report_date'].isoformat(),
                'record_type': 'case',
                'disease': str(row.get('disease', 'unknown')),
                'region': str(row.get('region', 'unknown')),
                'age_group': str(row.get('age_group', 'unknown')),
                'sex': str(row.get('sex', 'unknown')),
                'outcome': str(row.get('outcome', 'unknown'))
            }

            table.put_item(Item=case_item)

        logger.info(f"✓ Stored {min(len(df), max_individual_records)} case records in DynamoDB")

    except Exception as e:
        logger.error(f"Error storing results in DynamoDB: {str(e)}")
        raise


def send_outbreak_alert(
    outbreak_signals: Dict,
    metrics: Dict,
    r0_estimate: Optional[Dict]
):
    """
    Send SNS alert for detected outbreak.

    Args:
        outbreak_signals: Outbreak detection results
        metrics: Epidemiological metrics
        r0_estimate: R0 estimation results
    """
    if not SNS_TOPIC_ARN:
        logger.warning("SNS_TOPIC_ARN not configured, skipping alert")
        return

    try:
        # Construct alert message
        subject = f"⚠️ Outbreak Alert - {metrics['total_cases']} Cases Detected"

        message_lines = [
            "DISEASE SURVEILLANCE ALERT",
            "=" * 50,
            "",
            f"Outbreak Confidence: {outbreak_signals['confidence'].upper()}",
            f"Total Cases: {metrics['total_cases']}",
            f"Active Cases: {metrics.get('active_cases', 'N/A')}",
            f"Incidence Rate: {metrics['incidence_rate']:.2f} per 100,000",
            "",
            "Detection Methods:",
        ]

        for method in outbreak_signals['detection_method']:
            message_lines.append(f"  • {method}")

        message_lines.append("")
        message_lines.append("Reasons:")
        for reason in outbreak_signals['reasons']:
            message_lines.append(f"  • {reason}")

        if r0_estimate:
            message_lines.extend([
                "",
                "Reproductive Number (R0):",
                f"  R0 estimate: {r0_estimate['r0']:.2f}",
                f"  95% CI: [{r0_estimate['r0_lower']:.2f}, {r0_estimate['r0_upper']:.2f}]",
                f"  Growth rate: {r0_estimate['growth_rate']:.3f} per day"
            ])

        if 'hotspot_region' in outbreak_signals:
            message_lines.extend([
                "",
                f"Hotspot Region: {outbreak_signals['hotspot_region']}"
            ])

        message_lines.extend([
            "",
            "=" * 50,
            f"Generated: {datetime.utcnow().isoformat()} UTC",
            "",
            "RECOMMENDED ACTIONS:",
            "1. Verify case data and outbreak signal",
            "2. Coordinate with regional health authorities",
            "3. Increase surveillance in affected areas",
            "4. Review intervention protocols"
        ])

        message = "\n".join(message_lines)

        # Send SNS notification
        response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=message
        )

        logger.info(f"✓ Sent outbreak alert via SNS: {response['MessageId']}")

    except Exception as e:
        logger.error(f"Error sending SNS alert: {str(e)}")
        # Don't raise - this is non-critical
