"""Climate Data Analysis Tier 2 - AWS Lambda and S3 integration scripts."""
import contextlib

__version__ = "1.0.0"
__author__ = "Research Jumpstart Community"

# Import key functions for easy access
with contextlib.suppress(ImportError):
    from .upload_to_s3 import upload_climate_data, upload_file

with contextlib.suppress(ImportError):
    from .lambda_function import lambda_handler, process_netcdf_file

with contextlib.suppress(ImportError):
    from .query_results import download_results, query_s3_results

__all__ = [
    "download_results",
    "lambda_handler",
    "process_netcdf_file",
    "query_s3_results",
    "upload_climate_data",
    "upload_file",
]
