"""
AWS Lambda function to process oceanographic data.
Analyzes marine parameters and detects anomalies.
"""

import os
import json
import csv
from datetime import datetime
from io import StringIO
import boto3
from decimal import Decimal


# AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sns_client = boto3.client('sns')

# Environment variables
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'OceanObservations')
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', '')
BUCKET_NAME = os.environ.get('BUCKET_NAME', '')

# Thresholds for anomaly detection
TEMPERATURE_ANOMALY_THRESHOLD = float(os.environ.get('TEMPERATURE_ANOMALY_THRESHOLD', '3.0'))
PH_WARNING_THRESHOLD = float(os.environ.get('PH_WARNING_THRESHOLD', '7.8'))
PH_CRITICAL_THRESHOLD = float(os.environ.get('PH_CRITICAL_THRESHOLD', '7.6'))
DO_WARNING_THRESHOLD = float(os.environ.get('DO_WARNING_THRESHOLD', '4.0'))
DO_CRITICAL_THRESHOLD = float(os.environ.get('DO_CRITICAL_THRESHOLD', '2.0'))
CHLOROPHYLL_BLOOM_THRESHOLD = float(os.environ.get('CHLOROPHYLL_BLOOM_THRESHOLD', '20.0'))


def lambda_handler(event, context):
    """
    Lambda handler for ocean data processing.

    Event: S3 PUT event when new ocean data is uploaded
    """
    start_time = datetime.utcnow()
    print(f"Lambda function started at {start_time.isoformat()}")

    try:
        # Parse S3 event
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        print(f"Processing file: s3://{bucket}/{key}")

        # Download file from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')

        # Parse CSV data
        observations = parse_csv_data(content)
        print(f"Parsed {len(observations)} observations")

        # Process each observation
        processed_count = 0
        alert_count = 0

        for obs in observations:
            # Calculate marine metrics
            processed_obs = process_observation(obs)

            # Store in DynamoDB
            store_in_dynamodb(processed_obs)
            processed_count += 1

            # Check for anomalies and send alerts
            if processed_obs.get('alert_sent', False):
                alert_count += 1

        # Calculate execution time
        end_time = datetime.utcnow()
        execution_ms = (end_time - start_time).total_seconds() * 1000

        result = {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Ocean data processed successfully',
                'file': key,
                'observations_processed': processed_count,
                'alerts_sent': alert_count,
                'execution_time_ms': round(execution_ms, 2)
            })
        }

        print(f"Processing complete: {processed_count} observations, {alert_count} alerts")
        return result

    except Exception as e:
        print(f"Error processing ocean data: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def parse_csv_data(content):
    """Parse CSV content into list of observations."""
    observations = []
    csv_file = StringIO(content)
    reader = csv.DictReader(csv_file)

    for row in reader:
        obs = {
            'timestamp': row.get('timestamp', ''),
            'location_name': row.get('location_name', ''),
            'latitude': float(row.get('latitude', 0)),
            'longitude': float(row.get('longitude', 0)),
            'depth': float(row.get('depth', 0)),
            'temperature': float(row.get('temperature', 0)),
            'salinity': float(row.get('salinity', 0)),
            'ph': float(row.get('ph', 0)),
            'dissolved_oxygen': float(row.get('dissolved_oxygen', 0)),
            'chlorophyll': float(row.get('chlorophyll', 0)),
        }
        observations.append(obs)

    return observations


def process_observation(obs):
    """
    Process ocean observation and calculate marine metrics.

    Calculates:
    - Temperature anomaly
    - Stratification index
    - Upwelling index
    - Ocean acidification metrics
    - Primary production estimates
    - Anomaly detection
    """
    processed = obs.copy()

    # Generate unique observation ID
    timestamp = obs['timestamp'].replace(':', '-').replace('.', '-')
    location = obs['location_name'].replace(' ', '-')
    depth = int(obs['depth'])
    obs_id = f"{location}-{timestamp}-{depth}m"
    processed['observation_id'] = obs_id

    # Calculate temperature anomaly (deviation from climatology)
    # Using simple baseline: 15°C at surface, 4°C at 500m depth
    expected_temp = 15.0 * (1 - obs['depth'] / 1000) + 4.0 * (obs['depth'] / 1000)
    temp_anomaly = obs['temperature'] - expected_temp
    processed['temperature_anomaly'] = round(temp_anomaly, 2)

    # Stratification index (proxy using temperature gradient)
    # Higher values indicate stronger stratification
    if obs['depth'] < 50:
        stratification = obs['temperature'] - (obs['temperature'] - 5.0)
        processed['stratification_index'] = round(max(0, stratification), 2)
    else:
        processed['stratification_index'] = 0.0

    # Upwelling index (simplified: cold water at surface indicates upwelling)
    if obs['depth'] < 20 and obs['temperature'] < 12.0:
        upwelling = (12.0 - obs['temperature']) * 10
        processed['upwelling_index'] = round(upwelling, 2)
    else:
        processed['upwelling_index'] = 0.0

    # Aragonite saturation state (simplified calculation)
    # Ωarag = [CO3^2-][Ca^2+] / Ksp
    # Simplified proxy using pH (higher pH = higher saturation)
    omega_arag = (obs['ph'] - 7.0) * 2.0
    processed['aragonite_saturation'] = round(omega_arag, 2)

    # Primary production estimate (from chlorophyll-a)
    # Using simple conversion: PP (mg C/m²/day) ≈ 50 * Chl-a
    primary_production = obs['chlorophyll'] * 50.0
    processed['primary_production'] = round(primary_production, 1)

    # Detect anomalies
    anomaly_status, anomaly_type, alert_message = detect_anomalies(processed)
    processed['anomaly_status'] = anomaly_status
    processed['anomaly_type'] = anomaly_type

    # Send alert if critical anomaly detected
    alert_sent = False
    if anomaly_status in ['warning', 'critical'] and SNS_TOPIC_ARN:
        send_alert(processed, alert_message)
        alert_sent = True

    processed['alert_sent'] = alert_sent

    # Data quality assessment
    processed['data_quality'] = assess_data_quality(obs)

    return processed


def detect_anomalies(obs):
    """
    Detect marine anomalies.

    Returns:
    - anomaly_status: 'normal', 'warning', or 'critical'
    - anomaly_type: type of anomaly detected
    - alert_message: description for SNS alert
    """
    anomalies = []
    status = 'normal'
    alert_message = ''

    # Marine heatwave detection
    if obs['temperature_anomaly'] > TEMPERATURE_ANOMALY_THRESHOLD:
        anomalies.append('marine_heatwave')
        status = 'critical'
        alert_message += f"MARINE HEATWAVE: {obs['location_name']} shows +{obs['temperature_anomaly']:.1f}°C anomaly. "

    # Ocean acidification
    if obs['ph'] < PH_CRITICAL_THRESHOLD:
        anomalies.append('severe_acidification')
        status = 'critical'
        alert_message += f"SEVERE ACIDIFICATION: pH {obs['ph']:.2f} at {obs['location_name']}. "
    elif obs['ph'] < PH_WARNING_THRESHOLD:
        anomalies.append('acidification_warning')
        if status == 'normal':
            status = 'warning'
        alert_message += f"Acidification warning: pH {obs['ph']:.2f} at {obs['location_name']}. "

    # Hypoxia detection
    if obs['dissolved_oxygen'] < DO_CRITICAL_THRESHOLD:
        anomalies.append('severe_hypoxia')
        status = 'critical'
        alert_message += f"SEVERE HYPOXIA: DO {obs['dissolved_oxygen']:.1f} mg/L at {obs['depth']:.0f}m depth. "
    elif obs['dissolved_oxygen'] < DO_WARNING_THRESHOLD:
        anomalies.append('hypoxia_warning')
        if status == 'normal':
            status = 'warning'
        alert_message += f"Hypoxia warning: DO {obs['dissolved_oxygen']:.1f} mg/L. "

    # Harmful algal bloom
    if obs['chlorophyll'] > CHLOROPHYLL_BLOOM_THRESHOLD:
        anomalies.append('harmful_algal_bloom')
        if status == 'normal':
            status = 'warning'
        alert_message += f"Possible bloom: Chlorophyll {obs['chlorophyll']:.1f} mg/m³. "

    # Biological desert (very low chlorophyll)
    if obs['depth'] < 50 and obs['chlorophyll'] < 0.1:
        anomalies.append('biological_desert')
        if status == 'normal':
            status = 'warning'
        alert_message += f"Very low productivity: Chlorophyll {obs['chlorophyll']:.2f} mg/m³. "

    anomaly_type = ','.join(anomalies) if anomalies else 'none'

    return status, anomaly_type, alert_message.strip()


def send_alert(obs, message):
    """Send SNS alert for marine anomaly."""
    try:
        subject = f"Ocean Anomaly Alert: {obs['location_name']}"

        body = f"""
Ocean Anomaly Detected
======================

Location: {obs['location_name']}
Coordinates: {obs['latitude']:.2f}°, {obs['longitude']:.2f}°
Depth: {obs['depth']:.0f} m
Time: {obs['timestamp']}

Anomaly Status: {obs['anomaly_status'].upper()}
Anomaly Type: {obs['anomaly_type']}

Observations:
- Temperature: {obs['temperature']:.2f}°C (anomaly: {obs['temperature_anomaly']:.2f}°C)
- Salinity: {obs['salinity']:.2f} PSU
- pH: {obs['ph']:.3f}
- Dissolved Oxygen: {obs['dissolved_oxygen']:.2f} mg/L
- Chlorophyll-a: {obs['chlorophyll']:.2f} mg/m³

Details:
{message}

Ocean Metrics:
- Stratification Index: {obs['stratification_index']:.2f} kg/m³
- Upwelling Index: {obs['upwelling_index']:.2f}
- Aragonite Saturation: {obs['aragonite_saturation']:.2f}
- Primary Production: {obs['primary_production']:.1f} mg C/m²/day

Observation ID: {obs['observation_id']}
"""

        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=body
        )

        print(f"Alert sent for {obs['observation_id']}: {obs['anomaly_type']}")

    except Exception as e:
        print(f"Error sending SNS alert: {str(e)}")


def assess_data_quality(obs):
    """Assess quality of ocean data."""
    # Check for valid ranges
    issues = 0

    # Temperature: -2°C to 40°C
    if obs['temperature'] < -2 or obs['temperature'] > 40:
        issues += 1

    # Salinity: 0-42 PSU
    if obs['salinity'] < 0 or obs['salinity'] > 42:
        issues += 1

    # pH: 7.0-8.5
    if obs['ph'] < 7.0 or obs['ph'] > 8.5:
        issues += 1

    # Dissolved oxygen: 0-15 mg/L
    if obs['dissolved_oxygen'] < 0 or obs['dissolved_oxygen'] > 15:
        issues += 1

    # Chlorophyll: 0-100 mg/m³
    if obs['chlorophyll'] < 0 or obs['chlorophyll'] > 100:
        issues += 1

    if issues == 0:
        return 'excellent'
    elif issues == 1:
        return 'good'
    elif issues == 2:
        return 'fair'
    else:
        return 'poor'


def store_in_dynamodb(obs):
    """Store processed observation in DynamoDB."""
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)

        # Convert float values to Decimal for DynamoDB
        item = {
            'observation_id': obs['observation_id'],
            'timestamp': obs['timestamp'],
            'location_name': obs['location_name'],
            'latitude': Decimal(str(obs['latitude'])),
            'longitude': Decimal(str(obs['longitude'])),
            'depth': Decimal(str(obs['depth'])),
            'temperature': Decimal(str(obs['temperature'])),
            'salinity': Decimal(str(obs['salinity'])),
            'ph': Decimal(str(obs['ph'])),
            'dissolved_oxygen': Decimal(str(obs['dissolved_oxygen'])),
            'chlorophyll': Decimal(str(obs['chlorophyll'])),
            'temperature_anomaly': Decimal(str(obs['temperature_anomaly'])),
            'stratification_index': Decimal(str(obs['stratification_index'])),
            'upwelling_index': Decimal(str(obs['upwelling_index'])),
            'aragonite_saturation': Decimal(str(obs['aragonite_saturation'])),
            'primary_production': Decimal(str(obs['primary_production'])),
            'anomaly_status': obs['anomaly_status'],
            'anomaly_type': obs['anomaly_type'],
            'alert_sent': obs['alert_sent'],
            'data_quality': obs['data_quality'],
        }

        table.put_item(Item=item)

    except Exception as e:
        print(f"Error storing in DynamoDB: {str(e)}")
        raise
