"""
Lambda function for archaeological artifact classification and analysis.

This function:
1. Reads artifact CSV from S3
2. Performs artifact classification:
   - Typological classification
   - Morphometric analysis
   - Period assignment
3. Calculates spatial distribution
4. Performs chronological analysis
5. Stores results in DynamoDB
6. Writes summary to S3

Triggers: S3 upload, direct invocation
Output: DynamoDB records + JSON summary in S3
"""

import json
import os
from datetime import datetime
from io import StringIO
from typing import Any

import boto3
import numpy as np
import pandas as pd

# Initialize AWS clients
s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Main Lambda handler for artifact classification.

    Expected event payload:
    {
        "bucket": "archaeology-data-xxxx",
        "key": "raw/SITE_A_artifacts.csv"
    }

    Or S3 trigger event format:
    {
        "Records": [{
            "s3": {
                "bucket": {"name": "archaeology-data-xxxx"},
                "object": {"key": "raw/SITE_A_artifacts.csv"}
            }
        }]
    }

    Returns:
        {
            "statusCode": 200,
            "body": {
                "artifacts_processed": number,
                "site_id": string,
                "summary_file": S3 path,
                "message": success message
            }
        }
    """

    try:
        # Parse input (handle both direct and S3 trigger formats)
        if "Records" in event:
            # S3 trigger format
            bucket = event["Records"][0]["s3"]["bucket"]["name"]
            key = event["Records"][0]["s3"]["object"]["key"]
        else:
            # Direct invocation format
            bucket = event.get("bucket")
            key = event.get("key")

        if not bucket or not key:
            raise ValueError("Missing required parameters: bucket, key")

        # Skip non-CSV files
        if not key.endswith(".csv"):
            print(f"Skipping non-CSV file: {key}")
            return {"statusCode": 200, "body": {"message": "Skipped non-CSV file"}}

        print(f"Processing: s3://{bucket}/{key}")

        # Get environment variables
        table_name = os.environ.get("TABLE_NAME", "ArtifactCatalog")
        output_bucket = bucket  # Use same bucket for output

        # Download and parse CSV
        print("Downloading artifact data...")
        df = download_csv_from_s3(bucket, key)
        print(f"Loaded {len(df)} artifacts")

        # Extract site ID from filename or data
        site_id = extract_site_id(key, df)
        print(f"Site ID: {site_id}")

        # Classify artifacts
        print("Classifying artifacts...")
        df_classified = classify_artifacts(df)

        # Calculate morphometric indices
        print("Calculating morphometric indices...")
        df_classified = calculate_morphometrics(df_classified)

        # Perform spatial analysis
        print("Analyzing spatial distribution...")
        spatial_summary = analyze_spatial_distribution(df_classified)

        # Perform chronological analysis
        print("Analyzing chronology...")
        chrono_summary = analyze_chronology(df_classified)

        # Calculate typological statistics
        print("Calculating typology statistics...")
        typo_summary = analyze_typology(df_classified)

        # Write artifacts to DynamoDB
        print(f"Writing {len(df_classified)} records to DynamoDB...")
        write_to_dynamodb(df_classified, table_name)

        # Create comprehensive summary
        summary = {
            "site_id": site_id,
            "processing_date": datetime.now().isoformat(),
            "source_file": f"s3://{bucket}/{key}",
            "artifacts_processed": len(df_classified),
            "spatial_analysis": spatial_summary,
            "chronological_analysis": chrono_summary,
            "typological_analysis": typo_summary,
            "artifact_counts": {
                "by_type": df_classified["artifact_type"].value_counts().to_dict(),
                "by_period": df_classified["period"].value_counts().to_dict(),
                "by_material": df_classified["material"].value_counts().to_dict(),
            },
        }

        # Upload summary to S3
        summary_key = f"processed/{site_id}_summary.json"
        print(f"Uploading summary to s3://{output_bucket}/{summary_key}")
        upload_json_to_s3(summary, output_bucket, summary_key)

        # Upload classified data to S3
        processed_key = f"processed/{site_id}_classified.csv"
        print(f"Uploading classified data to s3://{output_bucket}/{processed_key}")
        upload_dataframe_to_s3(df_classified, output_bucket, processed_key)

        print("✓ Processing complete!")

        return {
            "statusCode": 200,
            "body": {
                "artifacts_processed": len(df_classified),
                "site_id": site_id,
                "summary_file": f"s3://{output_bucket}/{summary_key}",
                "processed_file": f"s3://{output_bucket}/{processed_key}",
                "dynamodb_table": table_name,
                "message": f"Successfully processed {len(df_classified)} artifacts",
            },
        }

    except Exception as e:
        print(f"Error: {e!s}")
        import traceback

        traceback.print_exc()
        return {
            "statusCode": 500,
            "body": {"error": str(e), "message": "Artifact processing failed"},
        }


def download_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """Download CSV file from S3 and parse as DataFrame."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        csv_content = response["Body"].read().decode("utf-8")
        df = pd.read_csv(StringIO(csv_content))
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to download {key} from {bucket}: {e!s}") from e


def extract_site_id(key: str, df: pd.DataFrame) -> str:
    """Extract site ID from filename or DataFrame."""
    # Try to extract from filename (e.g., "SITE_A_artifacts.csv")
    filename = key.split("/")[-1]
    parts = filename.replace(".csv", "").split("_")

    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]  # e.g., "SITE_A"

    # Try to extract from DataFrame
    if "site_id" in df.columns:
        return df["site_id"].iloc[0]

    # Default
    return "UNKNOWN_SITE"


def classify_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify artifacts based on typological characteristics.

    Assigns:
    - Artifact subtype (e.g., pottery -> bowl, jar, etc.)
    - Functional category
    - Classification confidence
    """
    df = df.copy()

    # Initialize classification columns
    df["artifact_subtype"] = ""
    df["functional_category"] = ""
    df["classification_confidence"] = 0.0

    for idx, row in df.iterrows():
        artifact_type = row["artifact_type"]
        length = row["length"]
        width = row["width"]
        thickness = row["thickness"]

        # Classify based on type and morphology
        if artifact_type == "pottery":
            # Classify pottery by shape ratios
            l_w_ratio = length / width if width > 0 else 1
            if l_w_ratio > 1.5:
                subtype = "jar"
                function = "storage"
            elif l_w_ratio > 0.8:
                subtype = "bowl"
                function = "serving"
            else:
                subtype = "plate"
                function = "serving"
            confidence = 0.85

        elif artifact_type == "lithic":
            # Classify lithics by size and material
            if length > 80:
                subtype = "blade"
                function = "cutting"
            elif length > 40:
                subtype = "flake"
                function = "tool"
            else:
                subtype = "debitage"
                function = "waste"
            confidence = 0.75

        elif artifact_type == "bone":
            # Classify bones by size
            if length > 100:
                subtype = "long_bone"
                function = "food_remains"
            elif length > 50:
                subtype = "medium_bone"
                function = "food_remains"
            else:
                subtype = "small_bone"
                function = "food_remains"
            confidence = 0.70

        elif artifact_type == "coin":
            # Classify coins by size
            if length > 25:
                subtype = "large_denomination"
                function = "currency"
            elif length > 18:
                subtype = "medium_denomination"
                function = "currency"
            else:
                subtype = "small_denomination"
                function = "currency"
            confidence = 0.90

        elif artifact_type == "architecture":
            # Classify architectural elements
            if length > 300:
                subtype = "structural_element"
                function = "construction"
            elif thickness > 40:
                subtype = "foundation"
                function = "construction"
            else:
                subtype = "decorative_element"
                function = "decoration"
            confidence = 0.80

        else:
            subtype = "unclassified"
            function = "unknown"
            confidence = 0.50

        df.at[idx, "artifact_subtype"] = subtype
        df.at[idx, "functional_category"] = function
        df.at[idx, "classification_confidence"] = confidence

    return df


def calculate_morphometrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate morphometric indices for artifacts.

    Common indices:
    - Length/Width ratio
    - Thickness index
    - Volume estimate
    - Shape index
    """
    df = df.copy()

    # Length/Width ratio
    df["l_w_ratio"] = df["length"] / df["width"]

    # Thickness index (relative thickness)
    df["thickness_index"] = df["thickness"] / df["length"]

    # Volume estimate (simplified as rectangular prism)
    df["volume_est"] = df["length"] * df["width"] * df["thickness"]

    # Shape index (sphericity measure)
    df["shape_index"] = np.minimum(df["length"], df["width"]) / np.maximum(
        df["length"], df["width"]
    )

    # Weight/volume ratio (density proxy)
    df["density_proxy"] = df["weight"] / (df["volume_est"] + 1)  # +1 to avoid division by zero

    return df


def analyze_spatial_distribution(df: pd.DataFrame) -> dict:
    """Analyze spatial distribution of artifacts."""

    spatial_summary = {
        "center_of_mass": {
            "latitude": float(df["gps_lat"].mean()),
            "longitude": float(df["gps_lon"].mean()),
        },
        "bounding_box": {
            "north": float(df["gps_lat"].max()),
            "south": float(df["gps_lat"].min()),
            "east": float(df["gps_lon"].max()),
            "west": float(df["gps_lon"].min()),
        },
        "spread": {"lat_std": float(df["gps_lat"].std()), "lon_std": float(df["gps_lon"].std())},
        "clusters_by_type": {},
    }

    # Calculate centroids by artifact type
    for artifact_type in df["artifact_type"].unique():
        type_df = df[df["artifact_type"] == artifact_type]
        spatial_summary["clusters_by_type"][artifact_type] = {
            "count": len(type_df),
            "centroid_lat": float(type_df["gps_lat"].mean()),
            "centroid_lon": float(type_df["gps_lon"].mean()),
        }

    return spatial_summary


def analyze_chronology(df: pd.DataFrame) -> dict:
    """Analyze chronological distribution of artifacts."""

    chrono_summary = {
        "date_range": {
            "earliest": int(df["dating_value"].min()),
            "latest": int(df["dating_value"].max()),
            "span_years": int(df["dating_value"].max() - df["dating_value"].min()),
        },
        "periods": {},
        "stratigraphic_sequence": {},
    }

    # Analyze by period
    for period in df["period"].unique():
        period_df = df[df["period"] == period]
        chrono_summary["periods"][period] = {
            "count": len(period_df),
            "date_mean": float(period_df["dating_value"].mean()),
            "date_std": float(period_df["dating_value"].std()),
            "artifact_types": period_df["artifact_type"].value_counts().to_dict(),
        }

    # Analyze stratigraphic sequence
    for unit in sorted(df["stratigraphic_unit"].unique()):
        unit_df = df[df["stratigraphic_unit"] == unit]
        chrono_summary["stratigraphic_sequence"][unit] = {
            "count": len(unit_df),
            "periods": unit_df["period"].value_counts().to_dict(),
            "date_range": {
                "min": int(unit_df["dating_value"].min()),
                "max": int(unit_df["dating_value"].max()),
            },
        }

    return chrono_summary


def analyze_typology(df: pd.DataFrame) -> dict:
    """Analyze typological patterns."""

    typo_summary = {"diversity_indices": {}, "type_statistics": {}, "material_patterns": {}}

    # Calculate diversity (Simpson's diversity index)
    total = len(df)
    type_counts = df["artifact_type"].value_counts()
    simpson = 1 - sum((n / total) ** 2 for n in type_counts)
    typo_summary["diversity_indices"]["simpson"] = float(simpson)

    # Type statistics
    for artifact_type in df["artifact_type"].unique():
        type_df = df[df["artifact_type"] == artifact_type]
        typo_summary["type_statistics"][artifact_type] = {
            "count": len(type_df),
            "percentage": float(len(type_df) / total * 100),
            "mean_length": float(type_df["length"].mean()),
            "mean_width": float(type_df["width"].mean()),
            "mean_weight": float(type_df["weight"].mean()),
            "materials": type_df["material"].value_counts().to_dict(),
        }

    # Material patterns
    for material in df["material"].unique():
        mat_df = df[df["material"] == material]
        typo_summary["material_patterns"][material] = {
            "count": len(mat_df),
            "artifact_types": mat_df["artifact_type"].value_counts().to_dict(),
            "periods": mat_df["period"].value_counts().to_dict(),
        }

    return typo_summary


def write_to_dynamodb(df: pd.DataFrame, table_name: str) -> None:
    """Write artifact records to DynamoDB."""

    table = dynamodb.Table(table_name)

    # Write in batches to avoid throttling
    batch_size = 25  # DynamoDB batch write limit
    total_batches = (len(df) + batch_size - 1) // batch_size

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]

        with table.batch_writer() as writer:
            for _, row in batch.iterrows():
                item = {
                    "artifact_id": str(row["artifact_id"]),
                    "site_id": str(row["site_id"]),
                    "artifact_type": str(row["artifact_type"]),
                    "artifact_subtype": str(row.get("artifact_subtype", "")),
                    "material": str(row["material"]),
                    "functional_category": str(row.get("functional_category", "")),
                    "period": str(row["period"]),
                    "length": float(row["length"]),
                    "width": float(row["width"]),
                    "thickness": float(row["thickness"]),
                    "weight": float(row["weight"]),
                    "gps_lat": float(row["gps_lat"]),
                    "gps_lon": float(row["gps_lon"]),
                    "stratigraphic_unit": str(row["stratigraphic_unit"]),
                    "dating_method": str(row["dating_method"]),
                    "dating_value": int(row["dating_value"]),
                    "excavation_date": str(row["excavation_date"]),
                    "classification_confidence": float(row.get("classification_confidence", 0.5)),
                    "l_w_ratio": float(row.get("l_w_ratio", 0)),
                    "thickness_index": float(row.get("thickness_index", 0)),
                    "shape_index": float(row.get("shape_index", 0)),
                    "notes": str(row.get("notes", "")),
                    "processed_date": datetime.now().isoformat(),
                }

                writer.put_item(Item=item)

        print(f"  Wrote batch {i // batch_size + 1}/{total_batches}")


def upload_json_to_s3(data: dict, bucket: str, key: str) -> None:
    """Upload JSON data to S3."""
    try:
        json_str = json.dumps(data, indent=2)
        s3_client.put_object(
            Bucket=bucket, Key=key, Body=json_str.encode("utf-8"), ContentType="application/json"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to upload JSON to s3://{bucket}/{key}: {e!s}") from e


def upload_dataframe_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    """Upload DataFrame as CSV to S3."""
    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=csv_buffer.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to upload CSV to s3://{bucket}/{key}: {e!s}") from e


# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {"bucket": "archaeology-data-test", "key": "raw/SITE_A_artifacts.csv"}

    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))
