#!/usr/bin/env python3
"""
Query and analyze quantum circuit simulation results from DynamoDB.

This script queries DynamoDB for circuit results, filters by algorithm type,
displays results in formatted tables, and optionally downloads detailed results from S3.

Usage:
    python query_results.py [options]

Environment Variables:
    AWS_DYNAMODB_TABLE: DynamoDB table name (default: QuantumResults)
    AWS_S3_BUCKET: S3 bucket name for detailed results
    AWS_REGION: AWS region (default: us-east-1)
"""

import argparse
import json
import os
from decimal import Decimal
from typing import Any, Optional

import boto3
import pandas as pd
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError


def get_aws_clients():
    """Create and return AWS clients."""
    region = os.environ.get("AWS_REGION", "us-east-1")
    dynamodb = boto3.resource("dynamodb", region_name=region)
    s3_client = boto3.client("s3", region_name=region)

    table_name = os.environ.get("AWS_DYNAMODB_TABLE", "QuantumResults")
    table = dynamodb.Table(table_name)

    return table, s3_client


def decimal_to_float(obj):
    """Convert DynamoDB Decimal types to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_float(item) for item in obj]
    return obj


def scan_all_results(table) -> list[dict[str, Any]]:
    """
    Scan all results from DynamoDB table.

    Args:
        table: DynamoDB table resource

    Returns:
        List of result items
    """
    print("📊 Scanning DynamoDB table...")

    try:
        response = table.scan()
        items = response.get("Items", [])

        # Handle pagination
        while "LastEvaluatedKey" in response:
            response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            items.extend(response.get("Items", []))

        print(f"✅ Found {len(items)} results")
        return items

    except ClientError as e:
        print(f"❌ Error scanning table: {e}")
        return []


def query_by_algorithm(table, algorithm_type: str) -> list[dict[str, Any]]:
    """
    Query results by algorithm type using a scan with filter.

    Args:
        table: DynamoDB table resource
        algorithm_type: Algorithm type to filter (bell, grover, ghz, etc.)

    Returns:
        List of matching result items
    """
    print(f"🔍 Querying circuits of type: {algorithm_type}")

    try:
        response = table.scan(FilterExpression=Attr("AlgorithmType").eq(algorithm_type))
        items = response.get("Items", [])

        # Handle pagination
        while "LastEvaluatedKey" in response:
            response = table.scan(
                FilterExpression=Attr("AlgorithmType").eq(algorithm_type),
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            items.extend(response.get("Items", []))

        print(f"✅ Found {len(items)} {algorithm_type} circuits")
        return items

    except ClientError as e:
        print(f"❌ Error querying table: {e}")
        return []


def query_by_qubit_count(table, min_qubits: int, max_qubits: int) -> list[dict[str, Any]]:
    """
    Query results by qubit count range.

    Args:
        table: DynamoDB table resource
        min_qubits: Minimum number of qubits
        max_qubits: Maximum number of qubits

    Returns:
        List of matching result items
    """
    print(f"🔍 Querying circuits with {min_qubits}-{max_qubits} qubits")

    try:
        response = table.scan(FilterExpression=Attr("NumQubits").between(min_qubits, max_qubits))
        items = response.get("Items", [])

        # Handle pagination
        while "LastEvaluatedKey" in response:
            response = table.scan(
                FilterExpression=Attr("NumQubits").between(min_qubits, max_qubits),
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            items.extend(response.get("Items", []))

        print(f"✅ Found {len(items)} circuits")
        return items

    except ClientError as e:
        print(f"❌ Error querying table: {e}")
        return []


def query_high_fidelity(table, min_fidelity: float = 0.95) -> list[dict[str, Any]]:
    """
    Query results with high fidelity.

    Args:
        table: DynamoDB table resource
        min_fidelity: Minimum fidelity threshold

    Returns:
        List of matching result items
    """
    print(f"🔍 Querying circuits with fidelity ≥ {min_fidelity}")

    try:
        response = table.scan(FilterExpression=Attr("Fidelity").gte(Decimal(str(min_fidelity))))
        items = response.get("Items", [])

        # Handle pagination
        while "LastEvaluatedKey" in response:
            response = table.scan(
                FilterExpression=Attr("Fidelity").gte(Decimal(str(min_fidelity))),
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            items.extend(response.get("Items", []))

        print(f"✅ Found {len(items)} high-fidelity circuits")
        return items

    except ClientError as e:
        print(f"❌ Error querying table: {e}")
        return []


def format_results_table(items: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Format results as a pandas DataFrame for display.

    Args:
        items: List of DynamoDB items

    Returns:
        Pandas DataFrame
    """
    if not items:
        return pd.DataFrame()

    # Convert to list of dicts with selected fields
    data = []
    for item in items:
        item = decimal_to_float(item)
        data.append(
            {
                "Circuit ID": item.get("CircuitID", "N/A"),
                "Algorithm": item.get("AlgorithmType", "N/A"),
                "Qubits": item.get("NumQubits", 0),
                "Gates": item.get("NumGates", 0),
                "Fidelity": f"{item.get('Fidelity', 0):.3f}",
                "Entanglement": f"{item.get('Entanglement', 0):.3f}",
                "Exec Time (ms)": item.get("ExecutionTimeMs", 0),
                "Timestamp": item.get("Timestamp", "N/A")[:19],  # Truncate timestamp
            }
        )

    df = pd.DataFrame(data)
    return df


def display_measurement_probabilities(item: dict[str, Any]):
    """
    Display measurement probabilities for a circuit.

    Args:
        item: DynamoDB item
    """
    circuit_id = item.get("CircuitID", "Unknown")
    probs = item.get("MeasurementProbabilities", {})

    if not probs:
        print(f"  No measurement data for {circuit_id}")
        return

    print(f"\n  Circuit: {circuit_id}")
    print(f"  Algorithm: {item.get('AlgorithmType', 'N/A')}")
    print(f"  Qubits: {item.get('NumQubits', 0)}")
    print("  Measurement Probabilities:")

    # Sort by probability (descending)
    sorted_probs = sorted(probs.items(), key=lambda x: float(x[1]), reverse=True)

    for state, prob in sorted_probs[:10]:  # Show top 10
        prob_float = float(prob)
        bar_length = int(prob_float * 50)
        bar = "█" * bar_length
        print(f"    |{state}⟩: {prob_float:.4f} {bar}")


def download_detailed_results(
    s3_client, bucket_name: str, s3_key: str, output_dir: str = "results"
) -> Optional[dict]:
    """
    Download detailed results from S3.

    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        s3_key: S3 object key
        output_dir: Local output directory

    Returns:
        Parsed JSON results or None if error
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Download file
        local_filename = os.path.join(output_dir, os.path.basename(s3_key))
        s3_client.download_file(bucket_name, s3_key, local_filename)

        # Read and parse JSON
        with open(local_filename) as f:
            data = json.load(f)

        print(f"  ✅ Downloaded: {local_filename}")
        return data

    except ClientError as e:
        print(f"  ❌ Error downloading {s3_key}: {e}")
        return None


def calculate_statistics(items: list[dict[str, Any]]):
    """
    Calculate and display statistics across all results.

    Args:
        items: List of DynamoDB items
    """
    if not items:
        print("No results to analyze")
        return

    items = [decimal_to_float(item) for item in items]

    # Group by algorithm type
    algorithm_counts = {}
    algorithm_qubits = {}
    algorithm_fidelity = {}
    algorithm_time = {}

    for item in items:
        algo = item.get("AlgorithmType", "unknown")
        algorithm_counts[algo] = algorithm_counts.get(algo, 0) + 1

        if algo not in algorithm_qubits:
            algorithm_qubits[algo] = []
            algorithm_fidelity[algo] = []
            algorithm_time[algo] = []

        algorithm_qubits[algo].append(item.get("NumQubits", 0))
        algorithm_fidelity[algo].append(item.get("Fidelity", 0))
        algorithm_time[algo].append(item.get("ExecutionTimeMs", 0))

    print("\n" + "=" * 70)
    print("Circuit Statistics by Algorithm Type")
    print("=" * 70)

    for algo in sorted(algorithm_counts.keys()):
        count = algorithm_counts[algo]
        avg_qubits = sum(algorithm_qubits[algo]) / len(algorithm_qubits[algo])
        avg_fidelity = sum(algorithm_fidelity[algo]) / len(algorithm_fidelity[algo])
        avg_time = sum(algorithm_time[algo]) / len(algorithm_time[algo])

        print(f"\n{algo.upper()}:")
        print(f"  Count: {count}")
        print(f"  Avg Qubits: {avg_qubits:.1f}")
        print(f"  Avg Fidelity: {avg_fidelity:.3f}")
        print(f"  Avg Execution Time: {avg_time:.1f} ms")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Query quantum circuit simulation results from DynamoDB"
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        help="Filter by algorithm type (bell, grover, ghz, vqe, etc.)",
    )
    parser.add_argument("--min-qubits", type=int, help="Minimum number of qubits")
    parser.add_argument("--max-qubits", type=int, help="Maximum number of qubits")
    parser.add_argument(
        "--min-fidelity", type=float, default=0.0, help="Minimum fidelity threshold (default: 0.0)"
    )
    parser.add_argument(
        "--show-measurements", "-m", action="store_true", help="Show measurement probabilities"
    )
    parser.add_argument(
        "--download-results", "-d", action="store_true", help="Download detailed results from S3"
    )
    parser.add_argument(
        "--stats", "-s", action="store_true", help="Show statistics across all results"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results.csv",
        help="Output CSV filename (default: results.csv)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Quantum Circuit Results Query")
    print("=" * 70)
    print()

    # Get AWS clients
    table, s3_client = get_aws_clients()

    # Query based on filters
    if args.algorithm:
        items = query_by_algorithm(table, args.algorithm)
    elif args.min_qubits or args.max_qubits:
        min_q = args.min_qubits or 1
        max_q = args.max_qubits or 20
        items = query_by_qubit_count(table, min_q, max_q)
    elif args.min_fidelity > 0:
        items = query_high_fidelity(table, args.min_fidelity)
    else:
        items = scan_all_results(table)

    if not items:
        print("❌ No results found")
        return 1

    # Format and display results table
    print("\n" + "=" * 70)
    print("Query Results")
    print("=" * 70)
    df = format_results_table(items)
    print(df.to_string(index=False))
    print()

    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"✅ Results saved to: {args.output}")
    print()

    # Show measurement probabilities if requested
    if args.show_measurements:
        print("=" * 70)
        print("Measurement Probabilities")
        print("=" * 70)
        for item in items[:5]:  # Show first 5
            display_measurement_probabilities(item)
        print()

    # Download detailed results if requested
    if args.download_results:
        bucket_name = os.environ.get("AWS_S3_BUCKET")
        if not bucket_name:
            print("⚠️  AWS_S3_BUCKET not set, skipping download")
        else:
            print("=" * 70)
            print("Downloading Detailed Results from S3")
            print("=" * 70)
            for item in items:
                s3_key = item.get("S3ResultsKey")
                if s3_key:
                    download_detailed_results(s3_client, bucket_name, s3_key)
            print()

    # Show statistics if requested
    if args.stats:
        calculate_statistics(items)
        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total circuits: {len(items)}")

    if items:
        items_float = [decimal_to_float(item) for item in items]
        avg_qubits = sum(item.get("NumQubits", 0) for item in items_float) / len(items_float)
        avg_fidelity = sum(item.get("Fidelity", 0) for item in items_float) / len(items_float)
        avg_time = sum(item.get("ExecutionTimeMs", 0) for item in items_float) / len(items_float)

        print(f"Average qubits: {avg_qubits:.1f}")
        print(f"Average fidelity: {avg_fidelity:.3f}")
        print(f"Average execution time: {avg_time:.1f} ms")

    print()
    print("🎯 Next Steps:")
    print("1. Open Jupyter notebook for visualization:")
    print("   jupyter notebook notebooks/quantum_analysis.ipynb")
    print()
    print("2. Query specific algorithm:")
    print(f"   python {__file__} --algorithm grover")
    print()
    print("3. Download detailed results:")
    print(f"   python {__file__} --download-results")
    print()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
