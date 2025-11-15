"""
Data ingestion module for economic data sources.

Provides classes for loading data from FRED, World Bank, and OECD APIs.
"""

import logging
import os
from datetime import datetime
from typing import Optional, Union

import awswrangler as wr
import boto3
import pandas as pd
import wbgapi as wb
from fredapi import Fred

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FREDDataLoader:
    """Load economic data from Federal Reserve Economic Data (FRED)."""

    def __init__(self, api_key: Optional[str] = None, bucket_name: Optional[str] = None):
        """
        Initialize FRED data loader.

        Args:
            api_key: FRED API key (defaults to FRED_API_KEY env var)
            bucket_name: S3 bucket for caching data
        """
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED API key required. Set FRED_API_KEY env var or pass api_key.")

        self.fred = Fred(api_key=self.api_key)
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3") if bucket_name else None

    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cache: bool = True,
    ) -> pd.Series:
        """
        Get a FRED time series.

        Args:
            series_id: FRED series ID (e.g., 'GDP', 'CPIAUCSL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cache: Whether to cache to S3

        Returns:
            Time series data
        """
        logger.info(f"Fetching FRED series: {series_id}")

        # Check S3 cache first
        if cache and self.bucket_name:
            try:
                s3_path = f"s3://{self.bucket_name}/fred/{series_id}.parquet"
                cached_df = wr.s3.read_parquet(path=s3_path)
                logger.info(f"Loaded {series_id} from cache")
                return cached_df[series_id]
            except Exception as e:
                logger.debug(f"Cache miss for {series_id}: {e}")

        # Fetch from FRED API
        series = self.fred.get_series(
            series_id, observation_start=start_date, observation_end=end_date
        )

        # Cache to S3
        if cache and self.bucket_name:
            try:
                df = pd.DataFrame({series_id: series})
                s3_path = f"s3://{self.bucket_name}/fred/{series_id}.parquet"
                wr.s3.to_parquet(df=df, path=s3_path)
                logger.info(f"Cached {series_id} to S3")
            except Exception as e:
                logger.warning(f"Failed to cache {series_id}: {e}")

        return series

    def get_multiple_series(
        self,
        series_ids: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get multiple FRED series as a DataFrame.

        Args:
            series_ids: List of FRED series IDs
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with series as columns
        """
        logger.info(f"Fetching {len(series_ids)} FRED series")

        series_dict = {}
        for series_id in series_ids:
            try:
                series_dict[series_id] = self.get_series(series_id, start_date, end_date)
            except Exception as e:
                logger.warning(f"Failed to fetch {series_id}: {e}")

        df = pd.DataFrame(series_dict)
        return df

    def get_gdp_components(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get GDP and major components.

        Args:
            start_date: Start date (defaults to 1960-01-01)

        Returns:
            DataFrame with GDP components
        """
        start_date = start_date or "1960-01-01"

        series_ids = {
            "GDP": "GDP",
            "PCEC": "PCEC",  # Personal Consumption
            "GPDI": "GPDI",  # Gross Private Domestic Investment
            "GCE": "GCE",  # Government Consumption
            "NETEXP": "NETEXP",  # Net Exports
        }

        return self.get_multiple_series(list(series_ids.values()), start_date)

    def get_inflation_measures(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get various inflation measures.

        Args:
            start_date: Start date (defaults to 1960-01-01)

        Returns:
            DataFrame with inflation measures
        """
        start_date = start_date or "1960-01-01"

        series_ids = {
            "CPI": "CPIAUCSL",  # Consumer Price Index
            "PCE": "PCEPI",  # Personal Consumption Expenditures Price Index
            "CORE_PCE": "PCEPILFE",  # Core PCE (excl. food & energy)
            "PPI": "PPIACO",  # Producer Price Index
        }

        return self.get_multiple_series(list(series_ids.values()), start_date)

    def get_labor_market(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get labor market indicators.

        Args:
            start_date: Start date (defaults to 1960-01-01)

        Returns:
            DataFrame with labor market data
        """
        start_date = start_date or "1960-01-01"

        series_ids = {
            "UNRATE": "UNRATE",  # Unemployment Rate
            "PAYEMS": "PAYEMS",  # Nonfarm Payrolls
            "CIVPART": "CIVPART",  # Labor Force Participation
            "AWHAE": "AWHAE",  # Average Weekly Hours
            "CES0500000003": "CES0500000003",  # Average Hourly Earnings
        }

        return self.get_multiple_series(list(series_ids.values()), start_date)

    def get_financial_indicators(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get financial and monetary indicators.

        Args:
            start_date: Start date (defaults to 1960-01-01)

        Returns:
            DataFrame with financial indicators
        """
        start_date = start_date or "1960-01-01"

        series_ids = {
            "FEDFUNDS": "FEDFUNDS",  # Federal Funds Rate
            "DGS10": "DGS10",  # 10-Year Treasury
            "DGS2": "DGS2",  # 2-Year Treasury
            "DGS3MO": "DGS3MO",  # 3-Month Treasury
            "BAMLH0A0HYM2": "BAMLH0A0HYM2",  # High Yield Spread
            "VIXCLS": "VIXCLS",  # VIX
        }

        return self.get_multiple_series(list(series_ids.values()), start_date)


class WorldBankDataLoader:
    """Load economic data from World Bank APIs."""

    def __init__(self, bucket_name: Optional[str] = None):
        """
        Initialize World Bank data loader.

        Args:
            bucket_name: S3 bucket for caching data
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3") if bucket_name else None

    def get_indicator(
        self,
        indicator: str,
        countries: Union[str, list[str]] = "all",
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get World Bank indicator data.

        Args:
            indicator: World Bank indicator code (e.g., 'NY.GDP.MKTP.CD')
            countries: Country code(s) or 'all'
            start_year: Start year
            end_year: End year

        Returns:
            DataFrame with indicator data
        """
        logger.info(f"Fetching World Bank indicator: {indicator}")

        # Fetch from World Bank API
        df = wb.data.DataFrame(
            indicator,
            countries,
            time=range(start_year or 1960, end_year or datetime.now().year + 1),
            skipBlanks=True,
            labels=True,
        )

        # Cache to S3
        if self.bucket_name:
            try:
                s3_path = f"s3://{self.bucket_name}/worldbank/{indicator}.parquet"
                wr.s3.to_parquet(df=df, path=s3_path)
                logger.info(f"Cached {indicator} to S3")
            except Exception as e:
                logger.warning(f"Failed to cache {indicator}: {e}")

        return df

    def get_gdp_data(
        self, countries: Union[str, list[str]] = "all", start_year: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get GDP data for countries.

        Args:
            countries: Country code(s) or 'all'
            start_year: Start year

        Returns:
            DataFrame with GDP data
        """
        return self.get_indicator(
            "NY.GDP.MKTP.CD",  # GDP (current US$)
            countries,
            start_year,
        )

    def get_gdp_growth(
        self, countries: Union[str, list[str]] = "all", start_year: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get GDP growth rate data.

        Args:
            countries: Country code(s) or 'all'
            start_year: Start year

        Returns:
            DataFrame with GDP growth rates
        """
        return self.get_indicator(
            "NY.GDP.MKTP.KD.ZG",  # GDP growth (annual %)
            countries,
            start_year,
        )

    def get_g7_indicators(self, start_year: Optional[int] = None) -> dict[str, pd.DataFrame]:
        """
        Get major indicators for G7 countries.

        Args:
            start_year: Start year

        Returns:
            Dictionary of DataFrames for each indicator
        """
        g7_countries = ["USA", "CAN", "GBR", "FRA", "DEU", "ITA", "JPN"]

        indicators = {
            "gdp": "NY.GDP.MKTP.CD",
            "gdp_growth": "NY.GDP.MKTP.KD.ZG",
            "inflation": "FP.CPI.TOTL.ZG",
            "unemployment": "SL.UEM.TOTL.ZS",
            "trade_balance": "NE.RSB.GNFS.CD",
        }

        results = {}
        for name, indicator in indicators.items():
            logger.info(f"Fetching {name} for G7 countries")
            results[name] = self.get_indicator(indicator, g7_countries, start_year)

        return results


class EconomicDataPipeline:
    """
    Combined pipeline for ingesting and processing economic data.
    """

    def __init__(self, fred_api_key: Optional[str] = None, bucket_name: Optional[str] = None):
        """
        Initialize data pipeline.

        Args:
            fred_api_key: FRED API key
            bucket_name: S3 bucket for data storage
        """
        self.fred_loader = FREDDataLoader(fred_api_key, bucket_name)
        self.wb_loader = WorldBankDataLoader(bucket_name)
        self.bucket_name = bucket_name

    def build_us_macro_dataset(self, start_date: str = "2000-01-01") -> pd.DataFrame:
        """
        Build comprehensive US macroeconomic dataset.

        Args:
            start_date: Start date

        Returns:
            DataFrame with aligned macro indicators
        """
        logger.info("Building US macro dataset")

        # Get all components
        gdp = self.fred_loader.get_gdp_components(start_date)
        inflation = self.fred_loader.get_inflation_measures(start_date)
        labor = self.fred_loader.get_labor_market(start_date)
        financial = self.fred_loader.get_financial_indicators(start_date)

        # Combine all data
        df = pd.concat([gdp, inflation, labor, financial], axis=1)

        # Forward fill missing values
        df = df.fillna(method="ffill")

        logger.info(f"Built dataset with {len(df)} observations and {len(df.columns)} variables")

        return df

    def export_to_s3(
        self, df: pd.DataFrame, s3_key: str, partition_cols: Optional[list[str]] = None
    ) -> str:
        """
        Export DataFrame to S3 in Parquet format.

        Args:
            df: DataFrame to export
            s3_key: S3 key (path within bucket)
            partition_cols: Columns to partition by

        Returns:
            S3 path
        """
        if not self.bucket_name:
            raise ValueError("bucket_name not set")

        s3_path = f"s3://{self.bucket_name}/{s3_key}"
        wr.s3.to_parquet(df=df, path=s3_path, partition_cols=partition_cols, dataset=True)

        logger.info(f"Exported data to {s3_path}")
        return s3_path


def main():
    """
    Main function for data ingestion.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Ingest economic data")
    parser.add_argument("--bucket", type=str, help="S3 bucket name")
    parser.add_argument(
        "--start-date", type=str, default="2000-01-01", help="Start date (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    # Build and export dataset
    pipeline = EconomicDataPipeline(bucket_name=args.bucket)
    df = pipeline.build_us_macro_dataset(start_date=args.start_date)

    if args.bucket:
        pipeline.export_to_s3(df, "processed/us_macro.parquet")
        print(f"Data exported to s3://{args.bucket}/processed/us_macro.parquet")
    else:
        print(df.head())


if __name__ == "__main__":
    main()
