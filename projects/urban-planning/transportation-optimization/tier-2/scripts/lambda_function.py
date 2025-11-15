#!/usr/bin/env python3
"""
AWS Lambda function for traffic flow analysis.

This function processes traffic data uploaded to S3, calculates transportation
metrics, and stores results in DynamoDB.

Metrics calculated:
- Average speed by segment
- Volume/Capacity (V/C) ratio
- Level of Service (LOS) rating
- Congestion detection and hotspots
- Peak hour identification
- Travel time reliability
- Speed performance index

Environment Variables Required:
- DYNAMODB_TABLE: Name of DynamoDB table for results
- S3_BUCKET: S3 bucket name (optional, from event)
- RESULTS_PREFIX: S3 prefix for result files (default: results/)

Trigger: S3 ObjectCreated events for CSV files in raw/ prefix
"""

import csv
import io
import json
import os
import urllib.parse
from datetime import datetime

import boto3

# Initialize AWS clients
s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

# Environment variables
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "TrafficAnalysis")
RESULTS_PREFIX = os.environ.get("RESULTS_PREFIX", "results/")

# Transportation constants
SPEED_LIMIT_DEFAULT = 55  # mph
CAPACITY_DEFAULT = 2000  # vehicles per hour per lane
LANES_DEFAULT = 2

# Level of Service (LOS) thresholds based on Highway Capacity Manual
LOS_THRESHOLDS = {
    "A": {"max_vc": 0.35, "min_speed": 55},
    "B": {"max_vc": 0.54, "min_speed": 50},
    "C": {"max_vc": 0.77, "min_speed": 45},
    "D": {"max_vc": 0.90, "min_speed": 40},
    "E": {"max_vc": 1.00, "min_speed": 30},
    "F": {"max_vc": 999, "min_speed": 0},  # Over capacity
}


def lambda_handler(event, context):
    """
    Lambda handler function triggered by S3 upload events.

    Args:
        event: S3 event notification
        context: Lambda context

    Returns:
        Response dictionary with status and results
    """
    print(f"Event received: {json.dumps(event)}")

    try:
        # Parse S3 event
        for record in event["Records"]:
            bucket = record["s3"]["bucket"]["name"]
            key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])

            print(f"Processing file: s3://{bucket}/{key}")

            # Process traffic data
            results = process_traffic_file(bucket, key)

            # Store results in DynamoDB
            store_results_dynamodb(results)

            # Save summary to S3
            save_results_summary(bucket, key, results)

            print(f"Successfully processed {len(results)} records")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Traffic data processed successfully",
                    "records_processed": len(results),
                }
            ),
        }

    except Exception as e:
        print(f"Error processing traffic data: {e!s}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


def process_traffic_file(bucket: str, key: str) -> list[dict]:
    """
    Download and process traffic data from S3.

    Args:
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        List of processed traffic records with metrics
    """
    print(f"Downloading file from S3: {bucket}/{key}")

    # Download file from S3
    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")

    # Parse CSV
    csv_reader = csv.DictReader(io.StringIO(content))
    records = []

    for row in csv_reader:
        try:
            # Extract and validate data
            record = parse_traffic_record(row)

            # Calculate metrics
            metrics = calculate_traffic_metrics(record)

            # Combine record and metrics
            result = {**record, **metrics}
            records.append(result)

        except Exception as e:
            print(f"Error processing row: {e!s}")
            continue

    print(f"Processed {len(records)} traffic records")
    return records


def parse_traffic_record(row: dict) -> dict:
    """
    Parse and validate a traffic data record.

    Args:
        row: CSV row as dictionary

    Returns:
        Parsed traffic record

    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = [
        "timestamp",
        "segment_id",
        "latitude",
        "longitude",
        "vehicle_count",
        "avg_speed",
    ]

    # Check required fields
    for field in required_fields:
        if field not in row:
            raise ValueError(f"Missing required field: {field}")

    # Parse and validate
    record = {
        "timestamp": row["timestamp"],
        "segment_id": row["segment_id"],
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "vehicle_count": int(row["vehicle_count"]),
        "avg_speed": float(row["avg_speed"]),
        "occupancy": float(row.get("occupancy", 0.0)),
        "congestion_level": int(row.get("congestion_level", 0)),
        "speed_limit": float(row.get("speed_limit", SPEED_LIMIT_DEFAULT)),
        "capacity": int(row.get("capacity", CAPACITY_DEFAULT * LANES_DEFAULT)),
        "lanes": int(row.get("lanes", LANES_DEFAULT)),
    }

    # Validate ranges
    if not (-90 <= record["latitude"] <= 90):
        raise ValueError(f"Invalid latitude: {record['latitude']}")
    if not (-180 <= record["longitude"] <= 180):
        raise ValueError(f"Invalid longitude: {record['longitude']}")
    if record["avg_speed"] < 0:
        raise ValueError(f"Invalid speed: {record['avg_speed']}")
    if record["vehicle_count"] < 0:
        raise ValueError(f"Invalid vehicle count: {record['vehicle_count']}")

    return record


def calculate_traffic_metrics(record: dict) -> dict:
    """
    Calculate transportation engineering metrics.

    Args:
        record: Traffic data record

    Returns:
        Dictionary of calculated metrics
    """
    # Volume/Capacity ratio
    vc_ratio = record["vehicle_count"] / record["capacity"]

    # Level of Service
    los = calculate_los(vc_ratio, record["avg_speed"])

    # Speed Performance Index (actual speed / speed limit)
    speed_performance = record["avg_speed"] / record["speed_limit"]

    # Travel Time Index (free-flow travel time / actual travel time)
    # Assuming 1 mile segment
    segment_length = 1.0  # miles
    free_flow_time = segment_length / record["speed_limit"]
    actual_time = segment_length / record["avg_speed"] if record["avg_speed"] > 0 else 999
    travel_time_index = actual_time / free_flow_time

    # Congestion detection
    is_congested = vc_ratio > 0.8 or record["avg_speed"] < 0.7 * record["speed_limit"]

    # Peak hour detection (from timestamp)
    is_peak_hour = check_peak_hour(record["timestamp"])

    # Reliability metric (coefficient of variation approximation)
    # Higher = less reliable
    reliability_score = calculate_reliability(vc_ratio, record["avg_speed"], record["speed_limit"])

    metrics = {
        "vc_ratio": round(vc_ratio, 3),
        "los": los,
        "speed_performance_index": round(speed_performance, 3),
        "travel_time_index": round(travel_time_index, 3),
        "is_congested": is_congested,
        "is_peak_hour": is_peak_hour,
        "reliability_score": round(reliability_score, 3),
        "timestamp_unix": convert_to_unix_timestamp(record["timestamp"]),
    }

    return metrics


def calculate_los(vc_ratio: float, avg_speed: float) -> str:
    """
    Calculate Level of Service based on V/C ratio and speed.

    Args:
        vc_ratio: Volume/Capacity ratio
        avg_speed: Average speed in mph

    Returns:
        LOS letter grade (A-F)
    """
    for los_grade, thresholds in LOS_THRESHOLDS.items():
        if vc_ratio <= thresholds["max_vc"] and avg_speed >= thresholds["min_speed"]:
            return los_grade

    return "F"  # Default to worst LOS


def check_peak_hour(timestamp_str: str) -> bool:
    """
    Check if timestamp falls within peak hours.

    Peak hours: 7-9 AM and 5-7 PM

    Args:
        timestamp_str: ISO format timestamp string

    Returns:
        True if peak hour, False otherwise
    """
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        hour = dt.hour

        # Morning peak: 7-9 AM
        # Evening peak: 5-7 PM (17-19)
        return (7 <= hour < 9) or (17 <= hour < 19)

    except Exception:
        return False


def calculate_reliability(vc_ratio: float, avg_speed: float, speed_limit: float) -> float:
    """
    Calculate travel time reliability score.

    Lower score = more reliable
    Higher score = less reliable (more variable)

    Args:
        vc_ratio: Volume/Capacity ratio
        avg_speed: Average speed
        speed_limit: Posted speed limit

    Returns:
        Reliability score (0-1, higher is worse)
    """
    # Reliability decreases as V/C increases
    vc_factor = min(vc_ratio, 1.5) / 1.5

    # Reliability decreases as speed drops below speed limit
    speed_ratio = avg_speed / speed_limit
    speed_factor = 1.0 - min(speed_ratio, 1.0)

    # Combined reliability score
    reliability = 0.6 * vc_factor + 0.4 * speed_factor

    return min(reliability, 1.0)


def convert_to_unix_timestamp(timestamp_str: str) -> int:
    """
    Convert ISO timestamp to Unix timestamp.

    Args:
        timestamp_str: ISO format timestamp

    Returns:
        Unix timestamp (seconds since epoch)
    """
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return int(datetime.utcnow().timestamp())


def store_results_dynamodb(results: list[dict]):
    """
    Store traffic analysis results in DynamoDB.

    Args:
        results: List of processed traffic records with metrics
    """
    table = dynamodb.Table(DYNAMODB_TABLE)

    print(f"Storing {len(results)} records in DynamoDB table: {DYNAMODB_TABLE}")

    batch_size = 25  # DynamoDB batch write limit
    for i in range(0, len(results), batch_size):
        batch = results[i : i + batch_size]

        with table.batch_writer() as writer:
            for record in batch:
                try:
                    # Prepare item for DynamoDB
                    item = {
                        "segment_id": record["segment_id"],
                        "timestamp": record["timestamp_unix"],
                        "timestamp_iso": record["timestamp"],
                        "latitude": record["latitude"],
                        "longitude": record["longitude"],
                        "vehicle_count": record["vehicle_count"],
                        "avg_speed": record["avg_speed"],
                        "occupancy": record.get("occupancy", 0.0),
                        "vc_ratio": record["vc_ratio"],
                        "los": record["los"],
                        "speed_performance_index": record["speed_performance_index"],
                        "travel_time_index": record["travel_time_index"],
                        "is_congested": record["is_congested"],
                        "is_peak_hour": record["is_peak_hour"],
                        "reliability_score": record["reliability_score"],
                        "speed_limit": record.get("speed_limit", SPEED_LIMIT_DEFAULT),
                        "capacity": record.get("capacity", CAPACITY_DEFAULT * LANES_DEFAULT),
                    }

                    writer.put_item(Item=item)

                except Exception as e:
                    print(f"Error writing to DynamoDB: {e!s}")
                    continue

    print(f"Successfully stored {len(results)} records in DynamoDB")


def save_results_summary(bucket: str, input_key: str, results: list[dict]):
    """
    Save analysis summary to S3 as JSON.

    Args:
        bucket: S3 bucket name
        input_key: Original input file key
        results: Processed traffic records
    """
    # Generate summary statistics
    summary = generate_summary_statistics(results)

    # Add metadata
    summary["metadata"] = {
        "input_file": input_key,
        "processing_time": datetime.utcnow().isoformat(),
        "records_processed": len(results),
    }

    # Create output key
    filename = os.path.basename(input_key).replace(".csv", "_analysis.json")
    output_key = f"{RESULTS_PREFIX}{filename}"

    # Upload to S3
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=json.dumps(summary, indent=2),
            ContentType="application/json",
        )
        print(f"Summary saved to s3://{bucket}/{output_key}")
    except Exception as e:
        print(f"Error saving summary to S3: {e!s}")


def generate_summary_statistics(results: list[dict]) -> dict:
    """
    Generate summary statistics from traffic analysis results.

    Args:
        results: List of processed traffic records

    Returns:
        Dictionary of summary statistics
    """
    if not results:
        return {}

    # Calculate aggregates
    total_records = len(results)
    avg_speed = sum(r["avg_speed"] for r in results) / total_records
    avg_vc_ratio = sum(r["vc_ratio"] for r in results) / total_records
    avg_tti = sum(r["travel_time_index"] for r in results) / total_records

    # Count by LOS
    los_counts = {}
    for los_grade in ["A", "B", "C", "D", "E", "F"]:
        los_counts[los_grade] = sum(1 for r in results if r["los"] == los_grade)

    # Congestion statistics
    congested_count = sum(1 for r in results if r["is_congested"])
    congestion_rate = congested_count / total_records

    # Peak hour statistics
    peak_hour_count = sum(1 for r in results if r["is_peak_hour"])
    peak_hour_rate = peak_hour_count / total_records

    # Identify hotspots (segments with worst LOS)
    hotspots = sorted(
        [
            {"segment_id": r["segment_id"], "los": r["los"], "vc_ratio": r["vc_ratio"]}
            for r in results
            if r["los"] in ["E", "F"]
        ],
        key=lambda x: x["vc_ratio"],
        reverse=True,
    )[:10]  # Top 10 hotspots

    summary = {
        "overall_metrics": {
            "total_records": total_records,
            "avg_speed_mph": round(avg_speed, 2),
            "avg_vc_ratio": round(avg_vc_ratio, 3),
            "avg_travel_time_index": round(avg_tti, 3),
            "congestion_rate": round(congestion_rate, 3),
            "peak_hour_rate": round(peak_hour_rate, 3),
        },
        "level_of_service_distribution": los_counts,
        "congestion_statistics": {
            "congested_segments": congested_count,
            "total_segments": total_records,
            "congestion_rate": round(congestion_rate * 100, 1),
        },
        "top_hotspots": hotspots,
    }

    return summary


# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        "Records": [
            {"s3": {"bucket": {"name": "test-bucket"}, "object": {"key": "raw/test_traffic.csv"}}}
        ]
    }

    print("Testing Lambda function locally...")
    result = lambda_handler(test_event, None)
    print(f"Result: {json.dumps(result, indent=2)}")
