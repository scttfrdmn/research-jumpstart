"""
Unit tests for AWS Lambda functions using moto mocking.

Tests Lambda handlers with mocked S3 events and AWS services.
"""

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws


# Helper function to discover Lambda functions
def discover_lambda_functions():
    """Discover all lambda_function.py files in the project."""
    projects_dir = Path(__file__).parent.parent / "projects"
    lambda_files = list(projects_dir.rglob("**/tier-2/scripts/lambda_function.py"))
    return lambda_files


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def s3_event():
    """Sample S3 event for Lambda testing."""
    return {
        "Records": [
            {
                "eventVersion": "2.1",
                "eventSource": "aws:s3",
                "awsRegion": "us-east-1",
                "eventTime": "2024-01-01T12:00:00.000Z",
                "eventName": "ObjectCreated:Put",
                "s3": {
                    "s3SchemaVersion": "1.0",
                    "configurationId": "test-config",
                    "bucket": {
                        "name": "test-bucket",
                        "arn": "arn:aws:s3:::test-bucket",
                    },
                    "object": {
                        "key": "raw/test_data_20240101.tif",
                        "size": 1024,
                        "eTag": "abc123",
                    },
                },
            }
        ]
    }


@pytest.fixture
def lambda_context():
    """Mock Lambda context object."""

    class LambdaContext:
        def __init__(self):
            self.function_name = "test-function"
            self.function_version = "$LATEST"
            self.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test-function"
            self.memory_limit_in_mb = 128
            self.aws_request_id = "test-request-id-123"
            self.log_group_name = "/aws/lambda/test-function"
            self.log_stream_name = "2024/01/01/[$LATEST]test123"

        def get_remaining_time_in_millis(self):
            return 300000

    return LambdaContext()


# ============================================================================
# Lambda Function Tests
# ============================================================================


@mock_aws
def test_agriculture_lambda_processes_s3_event(s3_event, lambda_context, aws_credentials):
    """Test agriculture Lambda function processes S3 event correctly."""
    # Setup mocked S3
    s3_client = boto3.client("s3", region_name="us-east-1")
    s3_client.create_bucket(Bucket="test-bucket")

    # Upload test file
    s3_client.put_object(
        Bucket="test-bucket",
        Key="raw/field_001_20240615.tif",
        Body=b"fake satellite image data",
    )

    # Import Lambda function INSIDE mock context
    import sys
    import importlib

    lambda_path = (
        Path(__file__).parent.parent
        / "projects/agriculture/precision-agriculture/tier-2/scripts"
    )
    sys.path.insert(0, str(lambda_path))

    try:
        # Import fresh copy of module
        if "lambda_function" in sys.modules:
            del sys.modules["lambda_function"]
        import lambda_function

        # Update event with correct key
        s3_event["Records"][0]["s3"]["object"]["key"] = "raw/field_001_20240615.tif"

        # Invoke Lambda
        response = lambda_function.lambda_handler(s3_event, lambda_context)

        # Assertions
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "message" in body
        assert "metrics" in body
        assert body["metrics"]["field_id"] == "field_001"
        assert "avg_ndvi" in body["metrics"]

        # Verify metrics file was created in S3
        objects = s3_client.list_objects_v2(Bucket="test-bucket", Prefix="results/")
        assert "Contents" in objects
        assert any("metrics.json" in obj["Key"] for obj in objects["Contents"])

    finally:
        sys.path.remove(str(lambda_path))
        if "lambda_function" in sys.modules:
            del sys.modules["lambda_function"]


@mock_aws
def test_lambda_handles_invalid_event(lambda_context, aws_credentials):
    """Test Lambda handles invalid S3 event gracefully."""
    s3_client = boto3.client("s3", region_name="us-east-1")
    s3_client.create_bucket(Bucket="test-bucket")

    import sys

    lambda_path = (
        Path(__file__).parent.parent
        / "projects/agriculture/precision-agriculture/tier-2/scripts"
    )
    sys.path.insert(0, str(lambda_path))

    try:
        if "lambda_function" in sys.modules:
            del sys.modules["lambda_function"]
        import lambda_function

        # Invalid event (no Records)
        invalid_event = {"invalid": "event"}

        response = lambda_function.lambda_handler(invalid_event, lambda_context)

        # Should return error response
        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "error" in body

    finally:
        sys.path.remove(str(lambda_path))
        if "lambda_function" in sys.modules:
            del sys.modules["lambda_function"]


@mock_aws
def test_lambda_calculates_metrics_correctly(aws_credentials):
    """Test NDVI metrics calculation logic."""
    import sys

    lambda_path = (
        Path(__file__).parent.parent
        / "projects/agriculture/precision-agriculture/tier-2/scripts"
    )
    sys.path.insert(0, str(lambda_path))

    try:
        if "lambda_function" in sys.modules:
            del sys.modules["lambda_function"]
        import lambda_function

        metrics = lambda_function.calculate_ndvi_metrics("field_test", "20240101")

        # Validate metrics structure
        assert metrics is not None
        assert "field_id" in metrics
        assert "avg_ndvi" in metrics
        assert "min_ndvi" in metrics
        assert "max_ndvi" in metrics
        assert "health_status" in metrics

        # Validate value ranges
        assert 0.0 <= metrics["avg_ndvi"] <= 1.0
        assert 0.0 <= metrics["min_ndvi"] <= 1.0
        assert 0.0 <= metrics["max_ndvi"] <= 1.0
        assert metrics["min_ndvi"] <= metrics["avg_ndvi"] <= metrics["max_ndvi"]
        assert metrics["health_status"] in ["Healthy", "Moderate", "Stressed"]

    finally:
        sys.path.remove(str(lambda_path))
        if "lambda_function" in sys.modules:
            del sys.modules["lambda_function"]


@mock_aws
def test_lambda_saves_metrics_to_s3(aws_credentials):
    """Test metrics are saved correctly to S3."""
    s3_client = boto3.client("s3", region_name="us-east-1")
    s3_client.create_bucket(Bucket="test-bucket")

    import sys

    lambda_path = (
        Path(__file__).parent.parent
        / "projects/agriculture/precision-agriculture/tier-2/scripts"
    )
    sys.path.insert(0, str(lambda_path))

    try:
        if "lambda_function" in sys.modules:
            del sys.modules["lambda_function"]
        import lambda_function

        test_metrics = {
            "field_id": "test_field",
            "avg_ndvi": 0.75,
            "health_status": "Healthy",
        }

        # Save metrics
        lambda_function.save_metrics_to_s3(
            "test-bucket", "results/test_metrics.json", test_metrics
        )

        # Verify saved
        response = s3_client.get_object(Bucket="test-bucket", Key="results/test_metrics.json")
        saved_data = json.loads(response["Body"].read())

        assert saved_data == test_metrics

    finally:
        sys.path.remove(str(lambda_path))
        if "lambda_function" in sys.modules:
            del sys.modules["lambda_function"]


# ============================================================================
# Summary Test
# ============================================================================


@pytest.mark.unit
def test_lambda_functions_discoverable():
    """Test that we can discover all Lambda functions."""
    lambda_files = discover_lambda_functions()

    assert len(lambda_files) > 0, "Should find Lambda functions"
    assert all(f.name == "lambda_function.py" for f in lambda_files)
    assert all("tier-2" in str(f) for f in lambda_files)

    print(f"\n✓ Found {len(lambda_files)} Lambda functions:")
    for f in sorted(lambda_files):
        domain = f.parts[-5]
        project = f.parts[-4]
        print(f"  - {domain}/{project}")
