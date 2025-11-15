#!/usr/bin/env python3
"""
Query environmental sensor data from DynamoDB.

This script retrieves processed sensor readings from DynamoDB and displays
results in a formatted table. Supports filtering by location, time range,
sensor type, and alert status.
"""

import argparse
import boto3
from datetime import datetime, timedelta
from decimal import Decimal
import json
import sys
from boto3.dynamodb.conditions import Key, Attr
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnvironmentalDataQuery:
    """Query environmental data from DynamoDB."""

    def __init__(self, table_name='EnvironmentalReadings', region='us-east-1', profile=None):
        """
        Initialize DynamoDB client.

        Args:
            table_name (str): DynamoDB table name
            region (str): AWS region
            profile (str): AWS profile name
        """
        if profile:
            session = boto3.Session(profile_name=profile)
        else:
            session = boto3.Session()

        self.dynamodb = session.resource('dynamodb', region_name=region)
        self.table = self.dynamodb.Table(table_name)
        self.table_name = table_name

        logger.info(f"Connected to DynamoDB table: {table_name}")

    def query_by_location(self, location_id, days=7, limit=100):
        """
        Query readings for a specific location.

        Args:
            location_id (str): Location identifier
            days (int): Number of days to query
            limit (int): Maximum number of results

        Returns:
            list: Query results
        """
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        start_timestamp = start_time.isoformat() + 'Z'

        logger.info(f"Querying location: {location_id}")
        logger.info(f"Time range: {start_timestamp} to present")

        try:
            response = self.table.query(
                KeyConditionExpression=Key('location_id').eq(location_id) &
                                      Key('timestamp').gte(start_timestamp),
                Limit=limit,
                ScanIndexForward=False  # Most recent first
            )

            items = response.get('Items', [])
            logger.info(f"Found {len(items)} readings")

            return items

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    def query_by_alert_status(self, alert_status='critical', limit=50):
        """
        Query readings with specific alert status.

        Args:
            alert_status (str): Alert status ('warning', 'critical')
            limit (int): Maximum number of results

        Returns:
            list: Query results
        """
        logger.info(f"Scanning for alerts with status: {alert_status}")

        try:
            response = self.table.scan(
                FilterExpression=Attr('alert_status').eq(alert_status),
                Limit=limit
            )

            items = response.get('Items', [])
            logger.info(f"Found {len(items)} {alert_status} alerts")

            return items

        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return []

    def query_by_sensor_type(self, sensor_type='air', days=7, limit=100):
        """
        Query readings by sensor type.

        Args:
            sensor_type (str): Sensor type ('air', 'water', 'weather')
            days (int): Number of days to query
            limit (int): Maximum number of results

        Returns:
            list: Query results
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        start_timestamp = start_time.isoformat() + 'Z'

        logger.info(f"Scanning for {sensor_type} sensor readings")

        try:
            response = self.table.scan(
                FilterExpression=Attr('sensor_type').eq(sensor_type) &
                                Attr('timestamp').gte(start_timestamp),
                Limit=limit
            )

            items = response.get('Items', [])
            logger.info(f"Found {len(items)} {sensor_type} readings")

            return items

        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return []

    def get_all_locations(self):
        """Get list of all unique locations."""
        logger.info("Scanning for unique locations")

        try:
            response = self.table.scan(
                ProjectionExpression='location_id',
                Limit=1000
            )

            items = response.get('Items', [])
            locations = list(set([item['location_id'] for item in items]))
            locations.sort()

            logger.info(f"Found {len(locations)} unique locations")
            return locations

        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return []

    def get_statistics(self, location_id, days=7):
        """
        Calculate statistics for a location.

        Args:
            location_id (str): Location identifier
            days (int): Number of days to analyze

        Returns:
            dict: Statistics
        """
        items = self.query_by_location(location_id, days=days, limit=10000)

        if not items:
            return {}

        stats = {
            'location_id': location_id,
            'total_readings': len(items),
            'time_range_days': days,
            'alerts': {
                'critical': 0,
                'warning': 0,
                'none': 0
            },
            'sensor_types': {}
        }

        for item in items:
            # Count alerts
            alert_status = item.get('alert_status', 'none')
            stats['alerts'][alert_status] = stats['alerts'].get(alert_status, 0) + 1

            # Count sensor types
            sensor_type = item.get('sensor_type', 'unknown')
            stats['sensor_types'][sensor_type] = stats['sensor_types'].get(sensor_type, 0) + 1

            # Sensor-specific statistics
            if sensor_type == 'air':
                if 'air_quality' not in stats:
                    stats['air_quality'] = {
                        'aqi_values': [],
                        'pm25_values': []
                    }
                metrics = item.get('calculated_metrics', {})
                params = item.get('parameters', {})
                if 'aqi' in metrics:
                    stats['air_quality']['aqi_values'].append(float(metrics['aqi']))
                if 'pm25' in params:
                    stats['air_quality']['pm25_values'].append(float(params['pm25']))

            elif sensor_type == 'water':
                if 'water_quality' not in stats:
                    stats['water_quality'] = {
                        'wqi_values': [],
                        'ph_values': []
                    }
                metrics = item.get('calculated_metrics', {})
                params = item.get('parameters', {})
                if 'wqi' in metrics:
                    stats['water_quality']['wqi_values'].append(float(metrics['wqi']))
                if 'ph' in params:
                    stats['water_quality']['ph_values'].append(float(params['ph']))

        # Calculate averages
        if 'air_quality' in stats:
            aqi_vals = stats['air_quality']['aqi_values']
            pm25_vals = stats['air_quality']['pm25_values']
            stats['air_quality']['avg_aqi'] = sum(aqi_vals) / len(aqi_vals) if aqi_vals else 0
            stats['air_quality']['max_aqi'] = max(aqi_vals) if aqi_vals else 0
            stats['air_quality']['avg_pm25'] = sum(pm25_vals) / len(pm25_vals) if pm25_vals else 0
            stats['air_quality']['max_pm25'] = max(pm25_vals) if pm25_vals else 0
            del stats['air_quality']['aqi_values']
            del stats['air_quality']['pm25_values']

        if 'water_quality' in stats:
            wqi_vals = stats['water_quality']['wqi_values']
            ph_vals = stats['water_quality']['ph_values']
            stats['water_quality']['avg_wqi'] = sum(wqi_vals) / len(wqi_vals) if wqi_vals else 0
            stats['water_quality']['max_wqi'] = max(wqi_vals) if wqi_vals else 0
            stats['water_quality']['avg_ph'] = sum(ph_vals) / len(ph_vals) if ph_vals else 0
            del stats['water_quality']['wqi_values']
            del stats['water_quality']['ph_values']

        return stats


def convert_decimal(obj):
    """Convert DynamoDB Decimal to float for JSON serialization."""
    if isinstance(obj, list):
        return [convert_decimal(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj


def format_table(items, max_rows=20):
    """Format query results as ASCII table."""
    if not items:
        print("No results found.")
        return

    print(f"\nResults: {len(items)} readings")
    print("=" * 120)

    # Table header
    print(f"{'Timestamp':<20} {'Location':<15} {'Type':<8} {'Alert':<10} {'Message':<50}")
    print("-" * 120)

    # Table rows
    for i, item in enumerate(items[:max_rows]):
        timestamp = item.get('timestamp', 'N/A')[:19]
        location = item.get('location_id', 'N/A')[:15]
        sensor_type = item.get('sensor_type', 'N/A')[:8]
        alert = item.get('alert_status', 'none')[:10]
        message = item.get('alert_message', '')[:50]

        print(f"{timestamp:<20} {location:<15} {sensor_type:<8} {alert:<10} {message:<50}")

    if len(items) > max_rows:
        print(f"\n... and {len(items) - max_rows} more results (use --limit to see more)")

    print("=" * 120)


def format_statistics(stats):
    """Format statistics as readable output."""
    print("\n" + "=" * 80)
    print(f"Statistics for Location: {stats['location_id']}")
    print("=" * 80)

    print(f"\nTime Range: {stats['time_range_days']} days")
    print(f"Total Readings: {stats['total_readings']}")

    print("\nAlerts:")
    print(f"  Critical: {stats['alerts']['critical']}")
    print(f"  Warning:  {stats['alerts']['warning']}")
    print(f"  None:     {stats['alerts']['none']}")

    print("\nSensor Types:")
    for sensor_type, count in stats['sensor_types'].items():
        print(f"  {sensor_type}: {count}")

    if 'air_quality' in stats:
        aq = stats['air_quality']
        print("\nAir Quality:")
        print(f"  Average AQI: {aq['avg_aqi']:.1f}")
        print(f"  Maximum AQI: {aq['max_aqi']:.1f}")
        print(f"  Average PM2.5: {aq['avg_pm25']:.2f} μg/m³")
        print(f"  Maximum PM2.5: {aq['max_pm25']:.2f} μg/m³")

    if 'water_quality' in stats:
        wq = stats['water_quality']
        print("\nWater Quality:")
        print(f"  Average WQI: {wq['avg_wqi']:.1f}")
        print(f"  Maximum WQI: {wq['max_wqi']:.1f}")
        print(f"  Average pH: {wq['avg_ph']:.2f}")

    print("=" * 80)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Query environmental sensor data from DynamoDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query specific location
  python query_results.py --location station-01 --days 7

  # Query critical alerts
  python query_results.py --alert-status critical

  # Query by sensor type
  python query_results.py --sensor-type water --days 14

  # Get statistics
  python query_results.py --location station-01 --statistics

  # List all locations
  python query_results.py --list-locations

  # Export to JSON
  python query_results.py --location station-01 --format json > results.json
        """
    )
    parser.add_argument(
        '--table',
        default='EnvironmentalReadings',
        help='DynamoDB table name (default: EnvironmentalReadings)'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--profile',
        help='AWS profile name'
    )
    parser.add_argument(
        '--location',
        help='Query specific location ID'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to query (default: 7)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum number of results (default: 100)'
    )
    parser.add_argument(
        '--alert-status',
        choices=['warning', 'critical'],
        help='Filter by alert status'
    )
    parser.add_argument(
        '--sensor-type',
        choices=['air', 'water', 'weather'],
        help='Filter by sensor type'
    )
    parser.add_argument(
        '--statistics',
        action='store_true',
        help='Show statistics instead of raw results'
    )
    parser.add_argument(
        '--list-locations',
        action='store_true',
        help='List all unique locations'
    )
    parser.add_argument(
        '--format',
        choices=['table', 'json'],
        default='table',
        help='Output format (default: table)'
    )

    args = parser.parse_args()

    try:
        query = EnvironmentalDataQuery(
            table_name=args.table,
            region=args.region,
            profile=args.profile
        )

        # List locations
        if args.list_locations:
            locations = query.get_all_locations()
            print("\nAvailable Locations:")
            for loc in locations:
                print(f"  {loc}")
            sys.exit(0)

        # Statistics mode
        if args.statistics:
            if not args.location:
                logger.error("--location required for statistics")
                sys.exit(1)

            stats = query.get_statistics(args.location, days=args.days)
            if stats:
                if args.format == 'json':
                    print(json.dumps(convert_decimal(stats), indent=2))
                else:
                    format_statistics(stats)
            else:
                print(f"No data found for location: {args.location}")
            sys.exit(0)

        # Query mode
        items = []

        if args.location:
            items = query.query_by_location(args.location, days=args.days, limit=args.limit)
        elif args.alert_status:
            items = query.query_by_alert_status(args.alert_status, limit=args.limit)
        elif args.sensor_type:
            items = query.query_by_sensor_type(args.sensor_type, days=args.days, limit=args.limit)
        else:
            logger.error("Must specify --location, --alert-status, or --sensor-type")
            parser.print_help()
            sys.exit(1)

        # Output results
        if args.format == 'json':
            print(json.dumps(convert_decimal(items), indent=2))
        else:
            format_table(items, max_rows=args.limit)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
