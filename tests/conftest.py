"""
Pytest configuration and shared fixtures for Research Jumpstart tests.

This file is automatically loaded by pytest and provides:
- Common fixtures used across multiple test files
- Test discovery helpers
- AWS mocking setup
- Test data generators
"""

import os
import sys
from pathlib import Path
from typing import List

import pytest

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# File Discovery Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def projects_dir(project_root: Path) -> Path:
    """Return the projects directory."""
    return project_root / "projects"


@pytest.fixture(scope="session")
def python_files(projects_dir: Path) -> List[Path]:
    """Discover all Python files in the projects directory."""
    files = list(projects_dir.rglob("*.py"))
    # Exclude common non-importable files
    excluded = ["__pycache__", "setup.py", ".aws-sam"]
    return [f for f in files if not any(ex in str(f) for ex in excluded)]


@pytest.fixture(scope="session")
def notebook_files(projects_dir: Path) -> List[Path]:
    """Discover all Jupyter notebook files."""
    files = list(projects_dir.rglob("*.ipynb"))
    # Exclude checkpoint directories
    return [f for f in files if ".ipynb_checkpoints" not in str(f)]


@pytest.fixture(scope="session")
def lambda_functions(projects_dir: Path) -> List[Path]:
    """Discover all Lambda function files."""
    return list(projects_dir.rglob("lambda_function.py"))


@pytest.fixture(scope="session")
def cloudformation_templates(projects_dir: Path) -> List[Path]:
    """Discover all CloudFormation template files."""
    return list(projects_dir.glob("**/cloudformation/*.yaml"))


# ============================================================================
# AWS Mocking Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def aws_credentials():
    """Mock AWS credentials for testing."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["AWS_REGION"] = "us-east-1"
    yield
    # Cleanup
    for key in [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SECURITY_TOKEN",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
    ]:
        os.environ.pop(key, None)


@pytest.fixture(scope="function")
def mock_s3_bucket(aws_credentials):
    """Create a mock S3 bucket for testing."""
    from moto import mock_aws
    import boto3

    with mock_aws():
        s3_client = boto3.client("s3", region_name="us-east-1")
        bucket_name = "test-research-bucket"
        s3_client.create_bucket(Bucket=bucket_name)
        yield s3_client, bucket_name


@pytest.fixture(scope="function")
def mock_dynamodb_table(aws_credentials):
    """Create a mock DynamoDB table for testing."""
    from moto import mock_aws
    import boto3

    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="test-table",
            KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        yield table


@pytest.fixture(scope="function")
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
                        "key": "raw/test_data.txt",
                        "size": 1024,
                        "eTag": "abc123",
                    },
                },
            }
        ]
    }


@pytest.fixture(scope="function")
def lambda_context():
    """Mock Lambda context object."""

    class LambdaContext:
        def __init__(self):
            self.function_name = "test-function"
            self.function_version = "$LATEST"
            self.invoked_function_arn = (
                "arn:aws:lambda:us-east-1:123456789012:function:test-function"
            )
            self.memory_limit_in_mb = 128
            self.aws_request_id = "test-request-id-123"
            self.log_group_name = "/aws/lambda/test-function"
            self.log_stream_name = "2024/01/01/[$LATEST]test123"

        def get_remaining_time_in_millis(self):
            return 300000

    return LambdaContext()


# ============================================================================
# Test Markers for Filtering
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "aws: marks tests that require AWS services"
    )
    config.addinivalue_line(
        "markers", "notebook: marks notebook execution tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks fast unit tests"
    )


# ============================================================================
# Test Collection Hooks
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark slow tests based on name
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Auto-mark AWS tests
        if "aws" in item.nodeid.lower() or "lambda" in item.nodeid.lower():
            item.add_marker(pytest.mark.aws)

        # Auto-mark notebook tests
        if "notebook" in item.nodeid.lower():
            item.add_marker(pytest.mark.notebook)
