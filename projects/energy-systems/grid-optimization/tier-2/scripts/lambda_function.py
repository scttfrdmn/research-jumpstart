#!/usr/bin/env python3
"""
AWS Lambda Function: Smart Grid Optimization and Analysis

This function is triggered when grid data is uploaded to S3.
It performs load forecasting, renewable integration analysis, and grid stability assessment.

To deploy:
    1. Package with: zip -r lambda_function.zip lambda_function.py
    2. Upload to AWS Lambda console or use AWS CLI:
       aws lambda create-function --function-name optimize-energy-grid \
           --runtime python3.11 --role <ROLE_ARN> \
           --handler lambda_function.lambda_handler --zip-file fileb://lambda_function.zip

Environment variables:
    BUCKET_NAME: S3 bucket name for grid data
    DYNAMODB_TABLE: DynamoDB table name (default: GridAnalysis)
    SNS_TOPIC_ARN: SNS topic ARN for alerts
    AWS_REGION: AWS region (default: us-east-1)

Expected S3 event format:
    {
        "Records": [{
            "s3": {
                "bucket": {"name": "energy-grid-12345"},
                "object": {"key": "raw/grid_data_substation_001_20250114.csv"}
            }
        }]
    }

Output:
    - DynamoDB: GridAnalysis table with metrics
    - S3: results/{filename}_analysis.json
    - SNS: Alerts for grid anomalies
"""

import json
import boto3
import csv
import io
import os
import sys
from datetime import datetime
from decimal import Decimal

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sns_client = boto3.client('sns')

# Configuration from environment variables
BUCKET_NAME = os.environ.get('BUCKET_NAME', '')
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'GridAnalysis')
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', '')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

print(f"Lambda initialized - Region: {AWS_REGION}, Table: {DYNAMODB_TABLE}")


def lambda_handler(event, context):
    """
    Main Lambda handler function.

    Args:
        event (dict): S3 event triggering the Lambda
        context (object): Lambda context object

    Returns:
        dict: Response with status code and message
    """
    try:
        print(f"Event: {json.dumps(event)}")

        # Parse S3 event
        if 'Records' not in event or len(event['Records']) == 0:
            return error_response(400, "No S3 records in event")

        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']

        print(f"Processing: s3://{bucket}/{key}")

        # Download and parse CSV data from S3
        grid_data = download_and_parse_csv(bucket, key)

        if not grid_data:
            return error_response(500, "Failed to parse grid data")

        print(f"Parsed {len(grid_data)} data points")

        # Perform grid analysis
        analysis_results = analyze_grid_data(grid_data)

        if not analysis_results:
            return error_response(500, "Failed to analyze grid data")

        # Store results in DynamoDB
        store_results_dynamodb(analysis_results)

        # Save detailed results to S3
        results_key = key.replace('raw/', 'results/').replace('.csv', '_analysis.json')
        save_results_to_s3(bucket, results_key, analysis_results)

        # Check for anomalies and send alerts
        check_and_alert_anomalies(analysis_results)

        print(f"✓ Analysis complete. Results saved to DynamoDB and S3")

        return success_response(
            message="Grid optimization analysis successful",
            records_processed=len(grid_data),
            location=analysis_results.get('location', 'unknown')
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return error_response(500, f"Internal error: {str(e)}")


def download_and_parse_csv(bucket, key):
    """
    Download CSV from S3 and parse grid data.

    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key

    Returns:
        list: Parsed grid data records
    """
    try:
        # Download file from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        csv_content = response['Body'].read().decode('utf-8')

        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        data = []

        for row in csv_reader:
            # Convert to proper types
            record = {
                'timestamp': row['timestamp'],
                'location': row['location'],
                'load_mw': float(row['load_mw']),
                'generation_mw': float(row['generation_mw']),
                'voltage_kv': float(row['voltage_kv']),
                'frequency_hz': float(row['frequency_hz']),
                'solar_mw': float(row['solar_mw']),
                'wind_mw': float(row['wind_mw']),
                'power_factor': float(row.get('power_factor', 0.95))
            }
            data.append(record)

        return data

    except Exception as e:
        print(f"Error downloading/parsing CSV: {e}")
        return None


def analyze_grid_data(grid_data):
    """
    Perform comprehensive grid analysis.

    Analyzes:
    - Load patterns and forecasting
    - Renewable energy integration
    - Grid stability (voltage, frequency)
    - Peak demand identification
    - Energy efficiency metrics

    Args:
        grid_data (list): List of grid data records

    Returns:
        dict: Analysis results
    """
    try:
        if not grid_data:
            return None

        # Extract location (assumes all records from same location)
        location = grid_data[0]['location']

        # Calculate load metrics
        loads = [r['load_mw'] for r in grid_data]
        load_avg = sum(loads) / len(loads)
        load_min = min(loads)
        load_max = max(loads)
        load_std = calculate_std(loads)

        # Calculate generation metrics
        generations = [r['generation_mw'] for r in grid_data]
        gen_avg = sum(generations) / len(generations)

        # Calculate renewable metrics
        solar_total = sum(r['solar_mw'] for r in grid_data)
        wind_total = sum(r['wind_mw'] for r in grid_data)
        renewable_total = solar_total + wind_total
        total_generation = sum(generations)
        renewable_penetration = renewable_total / total_generation if total_generation > 0 else 0

        # Calculate voltage metrics
        voltages = [r['voltage_kv'] for r in grid_data]
        voltage_avg = sum(voltages) / len(voltages)
        voltage_min = min(voltages)
        voltage_max = max(voltages)
        voltage_std = calculate_std(voltages)

        # Calculate frequency metrics
        frequencies = [r['frequency_hz'] for r in grid_data]
        frequency_avg = sum(frequencies) / len(frequencies)
        frequency_min = min(frequencies)
        frequency_max = max(frequencies)
        frequency_std = calculate_std(frequencies)

        # Calculate power factor metrics
        power_factors = [r['power_factor'] for r in grid_data]
        power_factor_avg = sum(power_factors) / len(power_factors)

        # Grid stability score (0-1, higher is better)
        # Based on voltage and frequency stability
        voltage_stability = 1.0 - min(abs(voltage_avg - 13.8) / 13.8, 1.0)
        frequency_stability = 1.0 - min(abs(frequency_avg - 60.0) / 60.0, 1.0)
        stability_score = (voltage_stability + frequency_stability) / 2

        # Energy efficiency score (0-1, higher is better)
        # Based on generation efficiency and power factor
        gen_efficiency = min(load_avg / gen_avg, 1.0) if gen_avg > 0 else 0
        efficiency_score = (gen_efficiency + power_factor_avg) / 2

        # Peak demand analysis
        peak_demand_mw = load_max
        peak_demand_time = grid_data[loads.index(load_max)]['timestamp']

        # Alert status determination
        alert_status = "normal"
        alerts = []

        if voltage_min < 13.5 or voltage_max > 14.1:
            alert_status = "warning"
            alerts.append("Voltage out of range")

        if abs(frequency_avg - 60.0) > 0.05:
            alert_status = "warning"
            alerts.append("Frequency deviation detected")

        if load_max / gen_avg > 0.95:
            alert_status = "warning"
            alerts.append("High load/generation ratio")

        if stability_score < 0.85:
            alert_status = "critical"
            alerts.append("Low grid stability")

        # Compile results
        results = {
            'location': location,
            'timestamp': datetime.now().isoformat(),
            'time_range': {
                'start': grid_data[0]['timestamp'],
                'end': grid_data[-1]['timestamp']
            },
            'data_points': len(grid_data),

            # Load metrics
            'load_metrics': {
                'avg_mw': round(load_avg, 2),
                'min_mw': round(load_min, 2),
                'max_mw': round(load_max, 2),
                'std_mw': round(load_std, 2),
                'peak_time': peak_demand_time
            },

            # Generation metrics
            'generation_metrics': {
                'avg_mw': round(gen_avg, 2),
                'total_mwh': round(sum(generations) / 4, 2)  # Assuming 15-min intervals
            },

            # Renewable metrics
            'renewable_metrics': {
                'solar_total_mw': round(solar_total, 2),
                'wind_total_mw': round(wind_total, 2),
                'renewable_penetration': round(renewable_penetration, 4),
                'renewable_percentage': round(renewable_penetration * 100, 2)
            },

            # Voltage metrics
            'voltage_metrics': {
                'avg_kv': round(voltage_avg, 3),
                'min_kv': round(voltage_min, 3),
                'max_kv': round(voltage_max, 3),
                'std_kv': round(voltage_std, 3)
            },

            # Frequency metrics
            'frequency_metrics': {
                'avg_hz': round(frequency_avg, 4),
                'min_hz': round(frequency_min, 4),
                'max_hz': round(frequency_max, 4),
                'std_hz': round(frequency_std, 4)
            },

            # Power quality
            'power_quality': {
                'power_factor_avg': round(power_factor_avg, 3),
                'stability_score': round(stability_score, 3),
                'efficiency_score': round(efficiency_score, 3)
            },

            # Alerts
            'alert_status': alert_status,
            'alerts': alerts,

            # Processing metadata
            'processor_version': '1.0',
            'processing_timestamp': datetime.now().isoformat()
        }

        print(f"Analysis complete for {location}:")
        print(f"  Load: {results['load_metrics']['avg_mw']}MW avg, {results['load_metrics']['max_mw']}MW peak")
        print(f"  Renewable: {results['renewable_metrics']['renewable_percentage']}%")
        print(f"  Stability: {results['power_quality']['stability_score']}")
        print(f"  Status: {alert_status}")

        return results

    except Exception as e:
        print(f"Error analyzing grid data: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_std(values):
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def store_results_dynamodb(results):
    """
    Store analysis results in DynamoDB.

    Args:
        results (dict): Analysis results
    """
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)

        # Convert float to Decimal for DynamoDB
        item = json.loads(json.dumps(results), parse_float=Decimal)

        # Use location and timestamp as keys
        item['location'] = results['location']
        item['timestamp'] = results['timestamp']

        # Store in DynamoDB
        table.put_item(Item=item)

        print(f"✓ Stored results in DynamoDB table: {DYNAMODB_TABLE}")

    except Exception as e:
        print(f"Error storing in DynamoDB: {e}")
        # Don't fail the entire function if DynamoDB write fails
        pass


def save_results_to_s3(bucket, key, results):
    """
    Save detailed results to S3 as JSON.

    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        results (dict): Analysis results
    """
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(results, indent=2),
            ContentType='application/json'
        )
        print(f"✓ Saved results to: s3://{bucket}/{key}")

    except Exception as e:
        print(f"Error saving results to S3: {e}")
        raise


def check_and_alert_anomalies(results):
    """
    Check for grid anomalies and send SNS alerts.

    Args:
        results (dict): Analysis results
    """
    try:
        if not SNS_TOPIC_ARN:
            print("No SNS topic configured, skipping alerts")
            return

        alert_status = results.get('alert_status', 'normal')

        if alert_status in ['warning', 'critical']:
            # Compose alert message
            location = results['location']
            alerts = results.get('alerts', [])
            stability = results['power_quality']['stability_score']
            voltage_avg = results['voltage_metrics']['avg_kv']
            frequency_avg = results['frequency_metrics']['avg_hz']

            message = f"""
GRID ALERT: {alert_status.upper()}

Location: {location}
Timestamp: {results['timestamp']}

Issues Detected:
{chr(10).join('  - ' + alert for alert in alerts)}

Current Metrics:
  - Voltage: {voltage_avg:.2f} kV (nominal: 13.8 kV)
  - Frequency: {frequency_avg:.4f} Hz (nominal: 60.0 Hz)
  - Stability Score: {stability:.3f}

Action Required: Review grid conditions and take corrective measures.
"""

            subject = f"Grid Alert [{alert_status.upper()}]: {location}"

            # Send SNS notification
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=subject,
                Message=message
            )

            print(f"✓ Sent {alert_status} alert to SNS")

    except Exception as e:
        print(f"Error sending SNS alert: {e}")
        # Don't fail the function if alert fails
        pass


def success_response(message, records_processed=0, location=None):
    """Generate successful Lambda response."""
    body = {
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'records_processed': records_processed
    }

    if location:
        body['location'] = location

    return {
        'statusCode': 200,
        'body': json.dumps(body)
    }


def error_response(status_code, error_message):
    """Generate error Lambda response."""
    return {
        'statusCode': status_code,
        'body': json.dumps({
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        })
    }


# Test function for local development
def test_lambda_locally():
    """Test Lambda function with sample data."""
    print("Testing Lambda function locally...")
    print("=" * 60)

    # Create sample grid data
    sample_data = []
    for i in range(96):  # 24 hours of 15-min intervals
        hour = i // 4
        base_load = 100 + 30 * (hour / 24)

        sample_data.append({
            'timestamp': f'2025-01-14T{hour:02d}:{(i%4)*15:02d}:00',
            'location': 'substation_001',
            'load_mw': base_load + (i % 10),
            'generation_mw': base_load * 1.05,
            'voltage_kv': 13.8 + (i % 5) * 0.01,
            'frequency_hz': 60.0 + (i % 3) * 0.001,
            'solar_mw': 20 if 6 <= hour <= 18 else 0,
            'wind_mw': 15 + (i % 8),
            'power_factor': 0.95
        })

    # Run analysis
    results = analyze_grid_data(sample_data)

    print("\nAnalysis Results:")
    print(json.dumps(results, indent=2))
    print("=" * 60)


if __name__ == '__main__':
    # Local testing
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_lambda_locally()
    else:
        print("Use --test flag for local testing")
        print("Example: python lambda_function.py --test")
