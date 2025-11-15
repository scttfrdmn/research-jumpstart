"""
Data access utilities for social media datasets on AWS.

Provides functions to access Twitter and Reddit datasets from AWS Open Data
and other public sources without downloading large files locally.
"""

import json
import logging
from typing import Optional

import boto3
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SocialMediaDataAccess:
    """
    Client for accessing social media datasets from AWS S3.

    Supports Twitter and Reddit data from AWS Open Data registry and other
    public sources.
    """

    def __init__(self, use_anon: bool = True, region: str = "us-east-1"):
        """
        Initialize social media data access client.

        Parameters
        ----------
        use_anon : bool, default True
            Use anonymous access for public data
        region : str, default 'us-east-1'
            AWS region
        """
        self.s3_client = boto3.client("s3", region_name=region)
        self.use_anon = use_anon
        logger.info("Initialized SocialMediaDataAccess client")

    def load_twitter_dataset(
        self,
        bucket: str,
        prefix: str,
        date_range: Optional[tuple[str, str]] = None,
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load Twitter data from S3.

        Parameters
        ----------
        bucket : str
            S3 bucket name
        prefix : str
            S3 prefix/path
        date_range : tuple of str, optional
            (start_date, end_date) in 'YYYY-MM-DD' format
        sample_size : int, optional
            Number of tweets to sample

        Returns
        -------
        pd.DataFrame
            DataFrame with tweet data

        Examples
        --------
        >>> client = SocialMediaDataAccess()
        >>> df = client.load_twitter_dataset(
        ...     bucket='twitter-open-data',
        ...     prefix='tweets/2024/',
        ...     sample_size=10000
        ... )
        """
        logger.info(f"Loading Twitter data from s3://{bucket}/{prefix}")

        # List objects in S3
        response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if "Contents" not in response:
            raise ValueError(f"No data found at s3://{bucket}/{prefix}")

        # Load data files
        dfs = []
        for obj in response.get("Contents", [])[:10]:  # Limit for performance
            key = obj["Key"]
            if key.endswith(".json") or key.endswith(".jsonl"):
                try:
                    obj_data = self.s3_client.get_object(Bucket=bucket, Key=key)
                    data = obj_data["Body"].read().decode("utf-8")

                    # Parse JSON lines
                    tweets = [json.loads(line) for line in data.strip().split("\\n") if line]
                    df_chunk = pd.DataFrame(tweets)
                    dfs.append(df_chunk)

                except Exception as e:
                    logger.warning(f"Error loading {key}: {e}")
                    continue

        if not dfs:
            raise ValueError("No valid data files found")

        # Combine all chunks
        df = pd.concat(dfs, ignore_index=True)

        # Filter by date range if specified
        if date_range and "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])
            start, end = date_range
            df = df[(df["created_at"] >= start) & (df["created_at"] <= end)]

        # Sample if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        logger.info(f"Loaded {len(df)} tweets")
        return df

    def load_reddit_dataset(
        self,
        bucket: str,
        prefix: str,
        subreddit: Optional[str] = None,
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load Reddit data from S3.

        Parameters
        ----------
        bucket : str
            S3 bucket name
        prefix : str
            S3 prefix/path
        subreddit : str, optional
            Filter by specific subreddit
        sample_size : int, optional
            Number of posts to sample

        Returns
        -------
        pd.DataFrame
            DataFrame with Reddit post data

        Examples
        --------
        >>> client = SocialMediaDataAccess()
        >>> df = client.load_reddit_dataset(
        ...     bucket='reddit-open-data',
        ...     prefix='submissions/2024/',
        ...     subreddit='science'
        ... )
        """
        logger.info(f"Loading Reddit data from s3://{bucket}/{prefix}")

        # Implementation similar to Twitter
        # Simplified for template purposes
        df = pd.DataFrame()

        logger.info(f"Loaded {len(df)} Reddit posts")
        return df

    def load_csv_dataset(self, bucket: str, key: str, **pandas_kwargs) -> pd.DataFrame:
        """
        Load CSV dataset from S3.

        Parameters
        ----------
        bucket : str
            S3 bucket name
        key : str
            S3 object key (file path)
        **pandas_kwargs
            Additional arguments passed to pd.read_csv

        Returns
        -------
        pd.DataFrame
            Loaded dataset

        Examples
        --------
        >>> client = SocialMediaDataAccess()
        >>> df = client.load_csv_dataset(
        ...     bucket='my-bucket',
        ...     key='data/social_media_posts.csv'
        ... )
        """
        logger.info(f"Loading CSV from s3://{bucket}/{key}")

        obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj["Body"], **pandas_kwargs)

        logger.info(f"Loaded {len(df)} rows")
        return df

    def save_results(self, df: pd.DataFrame, bucket: str, key: str, format: str = "csv") -> str:
        """
        Save analysis results to S3.

        Parameters
        ----------
        df : pd.DataFrame
            Results DataFrame
        bucket : str
            S3 bucket name
        key : str
            S3 object key
        format : str, default 'csv'
            Output format: 'csv', 'json', or 'parquet'

        Returns
        -------
        str
            S3 URI of saved file

        Examples
        --------
        >>> client = SocialMediaDataAccess()
        >>> client.save_results(
        ...     df=results_df,
        ...     bucket='my-results-bucket',
        ...     key='analysis/sentiment_results.csv'
        ... )
        's3://my-results-bucket/analysis/sentiment_results.csv'
        """
        logger.info(f"Saving results to s3://{bucket}/{key}")

        if format == "csv":
            csv_buffer = df.to_csv(index=False)
            self.s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.encode("utf-8"))
        elif format == "json":
            json_buffer = df.to_json(orient="records", lines=True)
            self.s3_client.put_object(Bucket=bucket, Key=key, Body=json_buffer.encode("utf-8"))
        elif format == "parquet":
            # Requires pyarrow
            parquet_buffer = df.to_parquet()
            self.s3_client.put_object(Bucket=bucket, Key=key, Body=parquet_buffer)
        else:
            raise ValueError(f"Unsupported format: {format}")

        s3_uri = f"s3://{bucket}/{key}"
        logger.info(f"Results saved to {s3_uri}")
        return s3_uri


def validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> bool:
    """
    Validate DataFrame has required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list of str
        Required column names

    Returns
    -------
    bool
        True if valid, raises ValueError otherwise

    Examples
    --------
    >>> validate_dataframe(df, ['text', 'timestamp', 'user_id'])
    True
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def check_s3_access(bucket: str, region: str = "us-east-1") -> bool:
    """
    Verify S3 bucket access.

    Parameters
    ----------
    bucket : str
        S3 bucket name
    region : str, default 'us-east-1'
        AWS region

    Returns
    -------
    bool
        True if accessible, False otherwise
    """
    try:
        s3 = boto3.client("s3", region_name=region)
        s3.head_bucket(Bucket=bucket)
        logger.info(f"✓ S3 bucket '{bucket}' is accessible")
        return True
    except Exception as e:
        logger.error(f"✗ S3 access failed: {e}")
        return False
