"""Climate Data Analysis Tier 2 - AWS Lambda and S3 integration scripts."""

__version__ = "1.0.0"
__author__ = "Research Jumpstart Community"

# Import key functions for easy access
try:
    from .upload_to_s3 import upload_climate_data, upload_file
except ImportError:
    pass

try:
    from .lambda_function import lambda_handler, process_netcdf_file
except ImportError:
    pass

try:
    from .query_results import download_results, query_s3_results
except ImportError:
    pass

__all__ = [
    'upload_climate_data',
    'upload_file',
    'lambda_handler',
    'process_netcdf_file',
    'download_results',
    'query_s3_results',
]
