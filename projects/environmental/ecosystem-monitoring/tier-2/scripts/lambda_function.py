#!/usr/bin/env python3
"""
AWS Lambda function for processing environmental sensor data.

This function:
1. Triggered by S3 upload of sensor data (CSV/JSON)
2. Calculates environmental indices (AQI, WQI)
3. Detects threshold violations and anomalies
4. Sends SNS alerts for critical pollution levels
5. Stores processed results in DynamoDB

Deploy to AWS Lambda:
- Runtime: Python 3.11
- Handler: lambda_function.lambda_handler
- Timeout: 30 seconds
- Memory: 256 MB
- Environment variables:
  - BUCKET_NAME: S3 bucket name
  - DYNAMODB_TABLE: DynamoDB table name
  - SNS_TOPIC_ARN: SNS topic ARN for alerts
  - ALERT_THRESHOLD_PM25: PM2.5 threshold (default: 35.4)
  - ALERT_THRESHOLD_AQI: AQI threshold (default: 101)
  - ALERT_THRESHOLD_PH_MIN: pH minimum (default: 6.5)
  - ALERT_THRESHOLD_PH_MAX: pH maximum (default: 8.5)
"""

import json
import logging
import os
import traceback
from datetime import datetime
from decimal import Decimal
from io import StringIO

import boto3

# Initialize clients
s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
sns_client = boto3.client("sns")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
BUCKET_NAME = os.environ.get("BUCKET_NAME", "environmental-data")
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "EnvironmentalReadings")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")

# Alert thresholds
ALERT_THRESHOLD_PM25 = float(os.environ.get("ALERT_THRESHOLD_PM25", "35.4"))
ALERT_THRESHOLD_AQI = int(os.environ.get("ALERT_THRESHOLD_AQI", "101"))
ALERT_THRESHOLD_PH_MIN = float(os.environ.get("ALERT_THRESHOLD_PH_MIN", "6.5"))
ALERT_THRESHOLD_PH_MAX = float(os.environ.get("ALERT_THRESHOLD_PH_MAX", "8.5"))
ALERT_THRESHOLD_DO_MIN = float(os.environ.get("ALERT_THRESHOLD_DO_MIN", "5.0"))


def lambda_handler(event, context):
    """
    AWS Lambda handler for processing environmental sensor data.

    Args:
        event (dict): S3 event trigger
        context (LambdaContext): Lambda context

    Returns:
        dict: Response with status and message
    """
    try:
        logger.info("Environmental data processing Lambda started")
        start_time = datetime.utcnow()

        # Parse S3 event
        if "Records" in event:
            record = event["Records"][0]
            s3_bucket = record["s3"]["bucket"]["name"]
            s3_key = record["s3"]["object"]["key"]
        else:
            # Direct invocation
            s3_bucket = event.get("bucket", BUCKET_NAME)
            s3_key = event.get("key", "raw/test_data.csv")

        logger.info(f"Processing file: s3://{s3_bucket}/{s3_key}")

        # Download and parse data
        try:
            obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            file_content = obj["Body"].read().decode("utf-8")
            logger.info(f"Downloaded {len(file_content)} bytes")

            # Parse CSV or JSON
            if s3_key.endswith(".csv"):
                data = parse_csv(file_content)
            elif s3_key.endswith(".json"):
                data = parse_json(file_content)
            else:
                return error_response(f"Unsupported file format: {s3_key}")

            logger.info(f"Parsed {len(data)} sensor readings")

        except Exception as e:
            return error_response(f"Failed to download/parse file: {e!s}")

        # Process each sensor reading
        results = []
        alerts = []

        for reading in data:
            try:
                processed = process_sensor_reading(reading)
                results.append(processed)

                # Check for alerts
                if processed["alert_status"] != "none":
                    alerts.append(processed)

            except Exception as e:
                logger.error(f"Error processing reading: {e}")
                continue

        # Store results in DynamoDB
        try:
            store_results_dynamodb(results)
            logger.info(f"Stored {len(results)} results in DynamoDB")
        except Exception as e:
            logger.error(f"Failed to store in DynamoDB: {e}")

        # Send alerts via SNS
        if alerts and SNS_TOPIC_ARN:
            try:
                send_alerts(alerts)
                logger.info(f"Sent {len(alerts)} alerts via SNS")
            except Exception as e:
                logger.error(f"Failed to send alerts: {e}")

        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

        # Return success
        response = {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Processing completed successfully",
                    "input_file": s3_key,
                    "readings_processed": len(results),
                    "alerts_triggered": len(alerts),
                    "processing_time_ms": processing_time_ms,
                    "timestamp": end_time.isoformat(),
                }
            ),
        }

        logger.info(f"Success: {response['body']}")
        return response

    except Exception as e:
        error_msg = f"Unhandled error: {e!s}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return error_response(error_msg)


def parse_csv(content):
    """Parse CSV content into list of dictionaries."""
    import csv

    reader = csv.DictReader(StringIO(content))
    return list(reader)


def parse_json(content):
    """Parse JSON content into list of dictionaries."""
    data = json.loads(content)
    if isinstance(data, list):
        return data
    return [data]


def process_sensor_reading(reading):
    """
    Process a single sensor reading.

    Args:
        reading (dict): Raw sensor reading

    Returns:
        dict: Processed reading with calculated metrics
    """
    sensor_type = reading.get("sensor_type", "unknown")

    if sensor_type == "air":
        return process_air_quality(reading)
    elif sensor_type == "water":
        return process_water_quality(reading)
    elif sensor_type == "weather":
        return process_weather_data(reading)
    else:
        return process_generic_sensor(reading)


def process_air_quality(reading):
    """
    Process air quality sensor reading and calculate AQI.

    Args:
        reading (dict): Air quality sensor data

    Returns:
        dict: Processed reading with AQI and alert status
    """
    # Extract parameters
    pm25 = float(reading.get("pm25", 0))
    pm10 = float(reading.get("pm10", 0))
    co2 = float(reading.get("co2", 0))
    no2 = float(reading.get("no2", 0))
    o3 = float(reading.get("o3", 0))
    co = float(reading.get("co", 0))

    # Calculate AQI for each pollutant
    aqi_pm25 = calculate_aqi_pm25(pm25)
    aqi_pm10 = calculate_aqi_pm10(pm10)
    aqi_o3 = calculate_aqi_o3(o3)
    aqi_no2 = calculate_aqi_no2(no2)
    aqi_co = calculate_aqi_co(co)

    # Overall AQI is the maximum
    aqi = max(aqi_pm25, aqi_pm10, aqi_o3, aqi_no2, aqi_co)

    # Determine AQI category
    aqi_category = get_aqi_category(aqi)

    # Check alert conditions
    alert_status = "none"
    alert_message = ""

    if pm25 > ALERT_THRESHOLD_PM25:
        alert_status = "warning" if pm25 < 55.4 else "critical"
        alert_message = f"PM2.5 level high: {pm25:.1f} μg/m³ (threshold: {ALERT_THRESHOLD_PM25})"
    elif aqi >= ALERT_THRESHOLD_AQI:
        if aqi >= 151:
            alert_status = "critical"
            alert_message = f"AQI unhealthy: {aqi} ({aqi_category})"
        else:
            alert_status = "warning"
            alert_message = f"AQI elevated: {aqi} ({aqi_category})"

    # Build result
    result = {
        "location_id": reading.get("location_id", "unknown"),
        "timestamp": reading.get("timestamp", datetime.utcnow().isoformat() + "Z"),
        "reading_id": f"{reading.get('location_id', 'unknown')}_{reading.get('timestamp', '')}",
        "sensor_type": "air",
        "parameters": {
            "pm25": pm25,
            "pm10": pm10,
            "co2": co2,
            "no2": no2,
            "o3": o3,
            "co": co,
            "temperature": float(reading.get("temperature", 0)),
            "humidity": float(reading.get("humidity", 0)),
        },
        "calculated_metrics": {
            "aqi": aqi,
            "aqi_category": aqi_category,
            "aqi_pm25": aqi_pm25,
            "aqi_pm10": aqi_pm10,
            "aqi_o3": aqi_o3,
            "dominant_pollutant": get_dominant_pollutant(
                aqi_pm25, aqi_pm10, aqi_o3, aqi_no2, aqi_co
            ),
        },
        "alert_status": alert_status,
        "alert_message": alert_message,
        "coordinates": {
            "latitude": float(reading.get("latitude", 0)),
            "longitude": float(reading.get("longitude", 0)),
        },
        "data_quality_score": 95,  # Simplified
    }

    return result


def calculate_aqi_pm25(pm25):
    """
    Calculate AQI for PM2.5 using EPA breakpoints.

    Args:
        pm25 (float): PM2.5 concentration in μg/m³

    Returns:
        int: AQI value
    """
    # EPA AQI breakpoints for PM2.5 (24-hour)
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]

    return calculate_aqi_from_breakpoints(pm25, breakpoints)


def calculate_aqi_pm10(pm10):
    """Calculate AQI for PM10."""
    breakpoints = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ]
    return calculate_aqi_from_breakpoints(pm10, breakpoints)


def calculate_aqi_o3(o3):
    """Calculate AQI for O3 (8-hour average)."""
    breakpoints = [
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300),
    ]
    return calculate_aqi_from_breakpoints(o3, breakpoints)


def calculate_aqi_no2(no2):
    """Calculate AQI for NO2."""
    breakpoints = [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
    ]
    return calculate_aqi_from_breakpoints(no2, breakpoints)


def calculate_aqi_co(co):
    """Calculate AQI for CO (ppm)."""
    breakpoints = [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
    ]
    return calculate_aqi_from_breakpoints(co, breakpoints)


def calculate_aqi_from_breakpoints(concentration, breakpoints):
    """
    Calculate AQI from concentration using EPA breakpoints.

    Args:
        concentration (float): Pollutant concentration
        breakpoints (list): List of (C_low, C_high, I_low, I_high) tuples

    Returns:
        int: AQI value
    """
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            # Linear interpolation
            aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(aqi)

    # If concentration exceeds all breakpoints, return maximum
    return 500


def get_aqi_category(aqi):
    """Get AQI category from AQI value."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def get_dominant_pollutant(aqi_pm25, aqi_pm10, aqi_o3, aqi_no2, aqi_co):
    """Determine which pollutant is driving the AQI."""
    pollutants = {"PM2.5": aqi_pm25, "PM10": aqi_pm10, "O3": aqi_o3, "NO2": aqi_no2, "CO": aqi_co}
    return max(pollutants, key=pollutants.get)


def process_water_quality(reading):
    """
    Process water quality sensor reading and calculate WQI.

    Args:
        reading (dict): Water quality sensor data

    Returns:
        dict: Processed reading with WQI and alert status
    """
    # Extract parameters
    ph = float(reading.get("ph", 7.0))
    do = float(reading.get("dissolved_oxygen", 8.0))  # mg/L
    turbidity = float(reading.get("turbidity", 5.0))  # NTU
    conductivity = float(reading.get("conductivity", 300.0))  # μS/cm
    temperature = float(reading.get("temperature", 20.0))
    tds = float(reading.get("tds", conductivity * 0.64))

    # Calculate Water Quality Index (simplified)
    wqi = calculate_wqi(ph, do, turbidity, tds)

    # Determine WQI category
    wqi_category = get_wqi_category(wqi)

    # Check alert conditions
    alert_status = "none"
    alert_message = ""

    if ph < ALERT_THRESHOLD_PH_MIN or ph > ALERT_THRESHOLD_PH_MAX:
        alert_status = "warning" if 6.0 <= ph <= 9.0 else "critical"
        alert_message = f"pH out of range: {ph:.2f} (acceptable: {ALERT_THRESHOLD_PH_MIN}-{ALERT_THRESHOLD_PH_MAX})"
    elif do < ALERT_THRESHOLD_DO_MIN:
        alert_status = "critical" if do < 4.0 else "warning"
        alert_message = f"Low dissolved oxygen: {do:.2f} mg/L (threshold: {ALERT_THRESHOLD_DO_MIN})"
    elif wqi > 75:
        alert_status = "warning" if wqi < 100 else "critical"
        alert_message = f"Poor water quality: WQI {wqi} ({wqi_category})"

    # Build result
    result = {
        "location_id": reading.get("location_id", "unknown"),
        "timestamp": reading.get("timestamp", datetime.utcnow().isoformat() + "Z"),
        "reading_id": f"{reading.get('location_id', 'unknown')}_{reading.get('timestamp', '')}",
        "sensor_type": "water",
        "parameters": {
            "ph": ph,
            "dissolved_oxygen": do,
            "turbidity": turbidity,
            "conductivity": conductivity,
            "temperature": temperature,
            "tds": tds,
        },
        "calculated_metrics": {"wqi": wqi, "wqi_category": wqi_category},
        "alert_status": alert_status,
        "alert_message": alert_message,
        "coordinates": {
            "latitude": float(reading.get("latitude", 0)),
            "longitude": float(reading.get("longitude", 0)),
        },
        "data_quality_score": 95,
    }

    return result


def calculate_wqi(ph, do, turbidity, tds):
    """
    Calculate Water Quality Index (simplified version).

    Args:
        ph (float): pH value
        do (float): Dissolved oxygen in mg/L
        turbidity (float): Turbidity in NTU
        tds (float): Total dissolved solids in mg/L

    Returns:
        int: WQI value (0-100+)
    """
    # Sub-index calculations (simplified)
    # pH: ideal is 7.0, range 6.5-8.5
    ph_si = abs(ph - 7.0) * 20

    # DO: ideal is > 8 mg/L
    do_si = max(0, (8.0 - do) * 15)

    # Turbidity: ideal is < 5 NTU
    turbidity_si = min(50, turbidity * 2)

    # TDS: ideal is < 500 mg/L
    tds_si = min(50, tds / 10)

    # Weighted average (weights sum to 1.0)
    wqi = ph_si * 0.25 + do_si * 0.35 + turbidity_si * 0.20 + tds_si * 0.20

    return round(wqi)


def get_wqi_category(wqi):
    """Get WQI category from WQI value."""
    if wqi <= 25:
        return "Excellent"
    elif wqi <= 50:
        return "Good"
    elif wqi <= 75:
        return "Fair"
    elif wqi <= 100:
        return "Poor"
    else:
        return "Very Poor"


def process_weather_data(reading):
    """Process weather sensor reading."""
    result = {
        "location_id": reading.get("location_id", "unknown"),
        "timestamp": reading.get("timestamp", datetime.utcnow().isoformat() + "Z"),
        "reading_id": f"{reading.get('location_id', 'unknown')}_{reading.get('timestamp', '')}",
        "sensor_type": "weather",
        "parameters": {
            "temperature": float(reading.get("temperature", 0)),
            "humidity": float(reading.get("humidity", 0)),
            "pressure": float(reading.get("pressure", 1013)),
            "wind_speed": float(reading.get("wind_speed", 0)),
            "precipitation": float(reading.get("precipitation", 0)),
        },
        "calculated_metrics": {},
        "alert_status": "none",
        "alert_message": "",
        "coordinates": {
            "latitude": float(reading.get("latitude", 0)),
            "longitude": float(reading.get("longitude", 0)),
        },
        "data_quality_score": 95,
    }
    return result


def process_generic_sensor(reading):
    """Process generic sensor reading."""
    result = {
        "location_id": reading.get("location_id", "unknown"),
        "timestamp": reading.get("timestamp", datetime.utcnow().isoformat() + "Z"),
        "reading_id": f"{reading.get('location_id', 'unknown')}_{reading.get('timestamp', '')}",
        "sensor_type": reading.get("sensor_type", "unknown"),
        "parameters": reading,
        "calculated_metrics": {},
        "alert_status": "none",
        "alert_message": "",
        "coordinates": {
            "latitude": float(reading.get("latitude", 0)),
            "longitude": float(reading.get("longitude", 0)),
        },
        "data_quality_score": 90,
    }
    return result


def store_results_dynamodb(results):
    """
    Store processed results in DynamoDB.

    Args:
        results (list): List of processed readings
    """
    table = dynamodb.Table(DYNAMODB_TABLE)

    for result in results:
        # Convert floats to Decimal for DynamoDB
        item = convert_to_dynamodb_item(result)

        try:
            table.put_item(Item=item)
        except Exception as e:
            logger.error(f"Failed to store item in DynamoDB: {e}")
            logger.error(f"Item: {item}")


def convert_to_dynamodb_item(data):
    """Convert data to DynamoDB compatible format (float to Decimal)."""
    if isinstance(data, dict):
        return {k: convert_to_dynamodb_item(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_dynamodb_item(v) for v in data]
    elif isinstance(data, float):
        return Decimal(str(data))
    else:
        return data


def send_alerts(alerts):
    """
    Send SNS alerts for critical readings.

    Args:
        alerts (list): List of readings with alert status
    """
    if not SNS_TOPIC_ARN:
        logger.warning("SNS_TOPIC_ARN not configured, skipping alerts")
        return

    # Group alerts by severity
    critical_alerts = [a for a in alerts if a["alert_status"] == "critical"]
    warning_alerts = [a for a in alerts if a["alert_status"] == "warning"]

    # Send critical alerts
    if critical_alerts:
        subject = f"CRITICAL: Environmental Alert - {len(critical_alerts)} critical readings"
        message = format_alert_message(critical_alerts, "CRITICAL")
        sns_client.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject, Message=message)
        logger.info(f"Sent critical alert for {len(critical_alerts)} readings")

    # Send warning alerts (if no critical)
    elif warning_alerts:
        subject = f"WARNING: Environmental Alert - {len(warning_alerts)} warnings"
        message = format_alert_message(warning_alerts, "WARNING")
        sns_client.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject, Message=message)
        logger.info(f"Sent warning alert for {len(warning_alerts)} readings")


def format_alert_message(alerts, severity):
    """Format alert message for SNS."""
    message = f"Environmental Monitoring Alert - {severity}\n"
    message += f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
    message += f"Total alerts: {len(alerts)}\n\n"

    for i, alert in enumerate(alerts[:10], 1):  # Limit to 10 alerts
        message += f"{i}. {alert['alert_message']}\n"
        message += f"   Location: {alert['location_id']}\n"
        message += f"   Time: {alert['timestamp']}\n"
        message += f"   Type: {alert['sensor_type']}\n\n"

    if len(alerts) > 10:
        message += f"... and {len(alerts) - 10} more alerts\n\n"

    message += "Check DynamoDB for full details.\n"
    return message


def error_response(error_msg, filename=None):
    """Generate error response."""
    response = {
        "statusCode": 500,
        "body": json.dumps(
            {
                "message": "Processing failed",
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ),
    }

    if filename:
        response["body"] = json.dumps({"file": filename, **json.loads(response["body"])})

    return response


# Local testing
if __name__ == "__main__":
    # Set environment variables for testing
    os.environ["BUCKET_NAME"] = "environmental-data-test"
    os.environ["DYNAMODB_TABLE"] = "EnvironmentalReadings"
    os.environ["SNS_TOPIC_ARN"] = "arn:aws:sns:us-east-1:123456789012:environmental-alerts"

    # Create test event
    test_event = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "environmental-data-test"},
                    "object": {"key": "raw/test_data.csv"},
                }
            }
        ]
    }

    class MockContext:
        pass

    # Test handler
    print("Testing lambda_handler...")
    print("Note: This will fail without actual S3 bucket and DynamoDB table")

    try:
        result = lambda_handler(test_event, MockContext())
        print(f"Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
