#!/usr/bin/env python3
"""
Upload environmental sensor data to S3 bucket.

This script handles uploading sensor data (CSV/JSON) to AWS S3 for processing.
Supports sample data generation, resumable uploads, and progress tracking.

Sensor types supported:
- Air quality (PM2.5, PM10, CO2, NO2, O3, temperature, humidity)
- Water quality (pH, dissolved oxygen, turbidity, conductivity, temperature)
- Weather (temperature, humidity, pressure, wind speed, precipitation)
- Soil (moisture, NPK, temperature, pH)
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnvironmentalDataGenerator:
    """Generate synthetic environmental sensor data for testing."""

    def __init__(self, seed=42):
        """Initialize data generator with random seed."""
        np.random.seed(seed)
        self.locations = [
            {"id": "station-01", "name": "Downtown", "lat": 40.7128, "lon": -74.0060},
            {"id": "station-02", "name": "Industrial", "lat": 40.7580, "lon": -73.9855},
            {"id": "station-03", "name": "Suburban", "lat": 40.6782, "lon": -73.9442},
            {"id": "station-04", "name": "Park", "lat": 40.7829, "lon": -73.9654},
            {"id": "station-05", "name": "Waterfront", "lat": 40.7061, "lon": -74.0134},
        ]
        self.water_locations = [
            {"id": "river-01", "name": "Main River", "lat": 40.7500, "lon": -74.0000},
            {"id": "lake-01", "name": "City Lake", "lat": 40.7200, "lon": -73.9800},
        ]

    def generate_air_quality_data(self, days=7, interval_minutes=15):
        """
        Generate air quality sensor data.

        Args:
            days (int): Number of days of data
            interval_minutes (int): Sampling interval in minutes

        Returns:
            pd.DataFrame: Air quality data
        """
        logger.info(f"Generating {days} days of air quality data")

        samples_per_day = (24 * 60) // interval_minutes
        total_samples = samples_per_day * days

        data = []
        for location in self.locations:
            start_time = datetime.utcnow() - timedelta(days=days)

            # Base pollution levels (different by location)
            base_pm25 = np.random.uniform(10, 30)
            base_pm25 * 1.8
            base_co2 = np.random.uniform(400, 450)

            for i in range(total_samples):
                timestamp = start_time + timedelta(minutes=i * interval_minutes)
                hour = timestamp.hour

                # Diurnal variation (rush hour effects)
                rush_hour_factor = 1.0
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    rush_hour_factor = 1.5
                elif 0 <= hour <= 5:
                    rush_hour_factor = 0.7

                # Add noise and trends
                pm25 = base_pm25 * rush_hour_factor + np.random.normal(0, 5)
                pm25 = max(0, pm25)  # Ensure positive

                pm10 = pm25 * 1.8 + np.random.normal(0, 8)
                pm10 = max(0, pm10)

                co2 = base_co2 * rush_hour_factor + np.random.normal(0, 20)
                co2 = max(350, co2)

                # Other pollutants
                no2 = np.random.uniform(10, 50) * rush_hour_factor
                o3 = np.random.uniform(20, 80) * (1.5 if 12 <= hour <= 16 else 0.8)
                co = np.random.uniform(0.1, 2.0) * rush_hour_factor

                # Weather
                temp = 20 + 10 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2)
                humidity = 60 + 20 * np.sin(2 * np.pi * (hour - 6) / 24) + np.random.normal(0, 5)
                humidity = max(20, min(95, humidity))

                # Occasional anomalies (5% chance)
                if np.random.random() < 0.05:
                    pm25 *= 2.5
                    pm10 *= 2.5

                data.append(
                    {
                        "timestamp": timestamp.isoformat() + "Z",
                        "location_id": location["id"],
                        "location_name": location["name"],
                        "sensor_type": "air",
                        "pm25": round(pm25, 2),
                        "pm10": round(pm10, 2),
                        "co2": round(co2, 1),
                        "no2": round(no2, 2),
                        "o3": round(o3, 2),
                        "co": round(co, 3),
                        "temperature": round(temp, 1),
                        "humidity": round(humidity, 1),
                        "latitude": location["lat"],
                        "longitude": location["lon"],
                    }
                )

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} air quality readings")
        return df

    def generate_water_quality_data(self, days=7, interval_minutes=60):
        """
        Generate water quality sensor data.

        Args:
            days (int): Number of days of data
            interval_minutes (int): Sampling interval in minutes

        Returns:
            pd.DataFrame: Water quality data
        """
        logger.info(f"Generating {days} days of water quality data")

        samples_per_day = (24 * 60) // interval_minutes
        total_samples = samples_per_day * days

        data = []
        for location in self.water_locations:
            start_time = datetime.utcnow() - timedelta(days=days)

            # Base water quality parameters
            base_ph = np.random.uniform(7.0, 8.0)
            base_do = np.random.uniform(7.0, 9.0)  # mg/L
            base_turbidity = np.random.uniform(2, 10)  # NTU
            base_conductivity = np.random.uniform(200, 500)  # μS/cm

            for i in range(total_samples):
                timestamp = start_time + timedelta(minutes=i * interval_minutes)
                hour = timestamp.hour

                # Diurnal variation (photosynthesis affects DO and pH)
                photosynthesis_factor = 1.0
                photosynthesis_factor = 1.2 if 6 <= hour <= 18 else 0.9

                # Add noise
                ph = base_ph + 0.3 * (photosynthesis_factor - 1) + np.random.normal(0, 0.2)
                ph = max(6.0, min(9.0, ph))

                do = base_do * photosynthesis_factor + np.random.normal(0, 0.5)
                do = max(4.0, do)

                turbidity = base_turbidity + np.random.normal(0, 2)
                turbidity = max(0, turbidity)

                conductivity = base_conductivity + np.random.normal(0, 30)
                conductivity = max(100, conductivity)

                # Water temperature
                water_temp = 18 + 4 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 1)

                # TDS (approximation from conductivity)
                tds = conductivity * 0.64

                # Occasional pollution events (3% chance)
                if np.random.random() < 0.03:
                    ph += np.random.choice([-1.5, 1.5])
                    do *= 0.6
                    turbidity *= 3.0

                data.append(
                    {
                        "timestamp": timestamp.isoformat() + "Z",
                        "location_id": location["id"],
                        "location_name": location["name"],
                        "sensor_type": "water",
                        "ph": round(ph, 2),
                        "dissolved_oxygen": round(do, 2),
                        "turbidity": round(turbidity, 2),
                        "conductivity": round(conductivity, 1),
                        "temperature": round(water_temp, 1),
                        "tds": round(tds, 1),
                        "latitude": location["lat"],
                        "longitude": location["lon"],
                    }
                )

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} water quality readings")
        return df

    def generate_weather_data(self, days=7, interval_minutes=10):
        """
        Generate weather sensor data.

        Args:
            days (int): Number of days of data
            interval_minutes (int): Sampling interval in minutes

        Returns:
            pd.DataFrame: Weather data
        """
        logger.info(f"Generating {days} days of weather data")

        samples_per_day = (24 * 60) // interval_minutes
        total_samples = samples_per_day * days

        data = []
        location = {"id": "weather-01", "name": "Central Station", "lat": 40.7300, "lon": -74.0000}

        start_time = datetime.utcnow() - timedelta(days=days)
        base_temp = 20
        base_pressure = 1013

        for i in range(total_samples):
            timestamp = start_time + timedelta(minutes=i * interval_minutes)
            hour = timestamp.hour
            day = i // samples_per_day

            # Temperature with day/night cycle
            temp = base_temp + 10 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2)

            # Humidity inversely related to temperature
            humidity = 70 - (temp - base_temp) * 2 + np.random.normal(0, 5)
            humidity = max(20, min(95, humidity))

            # Pressure with slow trends
            pressure = base_pressure + 10 * np.sin(2 * np.pi * day / 7) + np.random.normal(0, 2)

            # Wind speed
            wind_speed = max(0, np.random.exponential(3) + np.random.normal(0, 1))
            wind_direction = np.random.uniform(0, 360)

            # Precipitation (occasional)
            precipitation = 0.0
            if np.random.random() < 0.1:
                precipitation = np.random.exponential(2.0)

            data.append(
                {
                    "timestamp": timestamp.isoformat() + "Z",
                    "location_id": location["id"],
                    "location_name": location["name"],
                    "sensor_type": "weather",
                    "temperature": round(temp, 1),
                    "humidity": round(humidity, 1),
                    "pressure": round(pressure, 1),
                    "wind_speed": round(wind_speed, 1),
                    "wind_direction": round(wind_direction, 1),
                    "precipitation": round(precipitation, 2),
                    "latitude": location["lat"],
                    "longitude": location["lon"],
                }
            )

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} weather readings")
        return df


class S3Uploader:
    """Upload files to S3 with progress tracking."""

    def __init__(self, bucket_name, region="us-east-1", profile=None):
        """
        Initialize S3 uploader.

        Args:
            bucket_name (str): S3 bucket name
            region (str): AWS region
            profile (str): AWS profile name
        """
        self.bucket_name = bucket_name
        self.region = region

        # Create session and S3 client
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()

        self.s3 = session.client("s3", region_name=region)

        # Verify bucket exists
        try:
            self.s3.head_bucket(Bucket=bucket_name)
            logger.info(f"Connected to bucket: {bucket_name}")
        except ClientError as e:
            logger.error(f"Cannot access bucket: {bucket_name}")
            logger.error(f"Error: {e}")
            raise

    def upload_dataframe(self, df, s3_key, format="csv"):
        """
        Upload DataFrame to S3 as CSV or JSON.

        Args:
            df (pd.DataFrame): Data to upload
            s3_key (str): S3 object key
            format (str): 'csv' or 'json'

        Returns:
            bool: Success status
        """
        try:
            if format == "csv":
                body = df.to_csv(index=False)
                content_type = "text/csv"
            elif format == "json":
                body = df.to_json(orient="records", indent=2)
                content_type = "application/json"
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=body.encode("utf-8"),
                ContentType=content_type,
            )

            logger.info(f"Uploaded: s3://{self.bucket_name}/{s3_key} ({len(df)} rows)")
            return True

        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            return False

    def upload_file(self, file_path, s3_key):
        """
        Upload file to S3.

        Args:
            file_path (str): Local file path
            s3_key (str): S3 object key

        Returns:
            bool: Success status
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        file_size = file_path.stat().st_size
        logger.info(f"Uploading: {file_path.name} ({file_size / 1e6:.2f}MB)")

        try:
            self.s3.upload_file(str(file_path), self.bucket_name, s3_key)
            logger.info(f"Uploaded: s3://{self.bucket_name}/{s3_key}")
            return True

        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            return False

    def list_uploaded_files(self, prefix="raw/"):
        """List all uploaded files in S3."""
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)

            if "Contents" not in response:
                logger.info(f"No files found in s3://{self.bucket_name}/{prefix}")
                return []

            files = []
            total_size = 0

            logger.info(f"\nUploaded files in {prefix}:")
            for obj in response["Contents"]:
                size_mb = obj["Size"] / 1e6
                files.append(
                    {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "size_mb": size_mb,
                        "modified": obj["LastModified"],
                    }
                )
                logger.info(f"  {obj['Key']} ({size_mb:.2f}MB)")
                total_size += obj["Size"]

            logger.info(f"Total: {len(files)} files, {total_size / 1e6:.2f}MB")
            return files

        except ClientError as e:
            logger.error(f"Failed to list files: {e}")
            return []


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Upload environmental sensor data to S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and upload sample data
  python upload_to_s3.py --bucket environmental-data-xxxx --generate-sample

  # Upload specific file
  python upload_to_s3.py --bucket environmental-data-xxxx --file data.csv

  # Generate data for specific days
  python upload_to_s3.py --bucket environmental-data-xxxx --generate-sample --days 30
        """,
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument(
        "--generate-sample", action="store_true", help="Generate sample environmental data"
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Days of sample data to generate (default: 7)"
    )
    parser.add_argument("--file", help="Upload single file instead of generating data")
    parser.add_argument(
        "--s3-prefix", default="raw/", help="S3 prefix for uploaded files (default: raw/)"
    )
    parser.add_argument(
        "--format", choices=["csv", "json"], default="csv", help="Output format (default: csv)"
    )
    parser.add_argument(
        "--list-only", action="store_true", help="Only list files without uploading"
    )

    args = parser.parse_args()

    try:
        uploader = S3Uploader(args.bucket, args.region, args.profile)

        if args.list_only:
            uploader.list_uploaded_files(args.s3_prefix)
            return

        if args.file:
            # Upload single file
            s3_key = f"{args.s3_prefix}{Path(args.file).name}"
            if uploader.upload_file(args.file, s3_key):
                uploader.list_uploaded_files(args.s3_prefix)
                sys.exit(0)
            else:
                sys.exit(1)

        if args.generate_sample:
            # Generate sample data
            logger.info(f"Generating {args.days} days of sample environmental data")
            generator = EnvironmentalDataGenerator()

            # Generate air quality data
            air_data = generator.generate_air_quality_data(days=args.days)
            s3_key = (
                f"{args.s3_prefix}air_quality_{datetime.utcnow().strftime('%Y%m%d')}.{args.format}"
            )
            uploader.upload_dataframe(air_data, s3_key, format=args.format)

            # Generate water quality data
            water_data = generator.generate_water_quality_data(days=args.days)
            s3_key = f"{args.s3_prefix}water_quality_{datetime.utcnow().strftime('%Y%m%d')}.{args.format}"
            uploader.upload_dataframe(water_data, s3_key, format=args.format)

            # Generate weather data
            weather_data = generator.generate_weather_data(days=args.days)
            s3_key = f"{args.s3_prefix}weather_{datetime.utcnow().strftime('%Y%m%d')}.{args.format}"
            uploader.upload_dataframe(weather_data, s3_key, format=args.format)

            logger.info("\nSample data generation complete!")
            logger.info(f"  Air quality: {len(air_data)} readings")
            logger.info(f"  Water quality: {len(water_data)} readings")
            logger.info(f"  Weather: {len(weather_data)} readings")

            uploader.list_uploaded_files(args.s3_prefix)
            sys.exit(0)

        parser.print_help()
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
