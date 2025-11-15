"""
Lambda function for economic time series forecasting.

This function:
1. Reads CSV time series from S3 (triggered by S3 upload)
2. Fits ARIMA and Exponential Smoothing models
3. Generates forecasts with confidence intervals
4. Stores predictions in DynamoDB
5. Logs metadata to CloudWatch

Triggers: S3 upload event
Output: Forecasts in DynamoDB table
"""

import json
import os
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import tempfile
import traceback

# Import forecasting libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
except ImportError as e:
    print(f"Warning: statsmodels not available: {e}")
    print("Ensure Lambda Layer with statsmodels is attached")


# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for economic forecasting.

    Expected event payload (S3 trigger):
    {
        "Records": [{
            "s3": {
                "bucket": {"name": "bucket-name"},
                "object": {"key": "raw/gdp/usa_gdp_quarterly.csv"}
            }
        }]
    }

    Or direct invocation:
    {
        "bucket": "bucket-name",
        "key": "raw/gdp/usa_gdp_quarterly.csv",
        "indicator": "GDP",
        "country": "USA"
    }

    Returns:
        {
            "statusCode": 200,
            "body": {
                "indicator": "GDP",
                "country": "USA",
                "forecasts_stored": 8,
                "models": ["ARIMA", "ExponentialSmoothing"]
            }
        }
    """

    start_time = datetime.now()

    try:
        # Parse event (S3 trigger or direct invocation)
        if 'Records' in event:
            # S3 event trigger
            record = event['Records'][0]
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']

            # Extract indicator and country from key
            # Example: raw/gdp/usa_gdp_quarterly.csv → GDP, USA
            parts = key.split('/')
            indicator = parts[1].upper() if len(parts) > 1 else "UNKNOWN"
            filename = parts[-1] if len(parts) > 0 else ""
            country = filename.split('_')[0].upper() if '_' in filename else "UNKNOWN"

        else:
            # Direct invocation
            bucket = event.get('bucket')
            key = event.get('key')
            indicator = event.get('indicator', 'UNKNOWN')
            country = event.get('country', 'UNKNOWN')

        if not bucket or not key:
            raise ValueError("Missing required parameters: bucket, key")

        print(f"Processing: s3://{bucket}/{key}")
        print(f"Indicator: {indicator}, Country: {country}")

        # Download CSV from S3
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name

        print(f"Downloading from S3...")
        s3_client.download_file(bucket, key, tmp_path)

        # Load time series data
        df = pd.read_csv(tmp_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        print(f"Loaded {len(df)} data points")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")

        # Validate data
        if len(df) < 10:
            raise ValueError(f"Insufficient data points: {len(df)} (need at least 10)")

        # Prepare time series
        ts = df.set_index('date')['value']

        # Determine frequency (monthly or quarterly)
        freq = infer_frequency(ts)
        print(f"Detected frequency: {freq}")

        # Run forecasting models
        results = []

        # ARIMA model
        try:
            print("Fitting ARIMA model...")
            arima_forecasts = forecast_arima(ts, periods=8)
            results.append({
                'model_type': 'ARIMA',
                'forecasts': arima_forecasts
            })
            print(f"✓ ARIMA: {len(arima_forecasts)} forecasts generated")
        except Exception as e:
            print(f"✗ ARIMA failed: {e}")

        # Exponential Smoothing model
        try:
            print("Fitting Exponential Smoothing model...")
            es_forecasts = forecast_exponential_smoothing(ts, periods=8)
            results.append({
                'model_type': 'ExponentialSmoothing',
                'forecasts': es_forecasts
            })
            print(f"✓ Exponential Smoothing: {len(es_forecasts)} forecasts generated")
        except Exception as e:
            print(f"✗ Exponential Smoothing failed: {e}")

        if not results:
            raise Exception("All forecasting models failed")

        # Store forecasts in DynamoDB
        table_name = os.environ.get('TABLE_NAME', 'EconomicForecasts')
        table = dynamodb.Table(table_name)

        total_stored = 0
        for result in results:
            model_type = result['model_type']
            forecasts = result['forecasts']

            for forecast in forecasts:
                item = {
                    'indicator_country': f"{indicator}_{country}",
                    'forecast_date': int(forecast['date'].timestamp()),
                    'indicator': indicator,
                    'country': country,
                    'forecast_value': float(forecast['value']),
                    'confidence_80_lower': float(forecast.get('ci_80_lower', forecast['value'])),
                    'confidence_80_upper': float(forecast.get('ci_80_upper', forecast['value'])),
                    'confidence_95_lower': float(forecast.get('ci_95_lower', forecast['value'])),
                    'confidence_95_upper': float(forecast.get('ci_95_upper', forecast['value'])),
                    'model_type': model_type,
                    'model_params': json.dumps(forecast.get('model_params', {})),
                    'processing_time_ms': int((datetime.now() - start_time).total_seconds() * 1000),
                    'timestamp': datetime.now().isoformat(),
                    'data_points': len(df),
                    'forecast_horizon': 8,
                    's3_source': f"s3://{bucket}/{key}"
                }

                table.put_item(Item=item)
                total_stored += 1

        print(f"✓ Stored {total_stored} forecasts in DynamoDB")

        # Clean up temp file
        os.remove(tmp_path)

        # Return success response
        return {
            'statusCode': 200,
            'body': json.dumps({
                'indicator': indicator,
                'country': country,
                'data_points': len(df),
                'forecasts_stored': total_stored,
                'models': [r['model_type'] for r in results],
                'processing_time_ms': int((datetime.now() - start_time).total_seconds() * 1000)
            })
        }

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)

        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'traceback': traceback.format_exc()
            })
        }


def infer_frequency(ts: pd.Series) -> str:
    """Infer time series frequency (monthly or quarterly)."""
    if len(ts) < 2:
        return 'M'

    # Calculate average time difference
    diffs = ts.index.to_series().diff().dropna()
    avg_days = diffs.mean().days

    if avg_days < 45:
        return 'M'  # Monthly
    else:
        return 'Q'  # Quarterly


def forecast_arima(ts: pd.Series, periods: int = 8) -> List[Dict]:
    """
    Forecast using ARIMA model.

    Args:
        ts: Time series data
        periods: Number of periods to forecast

    Returns:
        List of forecast dictionaries with confidence intervals
    """
    # Auto-select ARIMA parameters (simple approach)
    # For production, use auto_arima from pmdarima
    order = (1, 1, 1)  # Default ARIMA(1,1,1)

    # Fit ARIMA model
    model = ARIMA(ts, order=order)
    fitted_model = model.fit()

    # Generate forecasts
    forecast_result = fitted_model.forecast(steps=periods, alpha=0.05)

    # Get confidence intervals
    forecast_df = fitted_model.get_forecast(steps=periods)
    conf_int_95 = forecast_df.conf_int(alpha=0.05)
    conf_int_80 = forecast_df.conf_int(alpha=0.20)

    # Determine forecast dates
    freq = infer_frequency(ts)
    last_date = ts.index[-1]

    forecasts = []
    for i in range(periods):
        if freq == 'Q':
            forecast_date = last_date + pd.DateOffset(months=3 * (i + 1))
        else:
            forecast_date = last_date + pd.DateOffset(months=(i + 1))

        forecasts.append({
            'date': forecast_date,
            'value': forecast_result.iloc[i],
            'ci_95_lower': conf_int_95.iloc[i, 0],
            'ci_95_upper': conf_int_95.iloc[i, 1],
            'ci_80_lower': conf_int_80.iloc[i, 0],
            'ci_80_upper': conf_int_80.iloc[i, 1],
            'model_params': {
                'order': order,
                'aic': float(fitted_model.aic),
                'bic': float(fitted_model.bic)
            }
        })

    return forecasts


def forecast_exponential_smoothing(ts: pd.Series, periods: int = 8) -> List[Dict]:
    """
    Forecast using Exponential Smoothing (Holt-Winters).

    Args:
        ts: Time series data
        periods: Number of periods to forecast

    Returns:
        List of forecast dictionaries with confidence intervals
    """
    # Determine if data has trend and seasonality
    freq = infer_frequency(ts)
    seasonal_periods = 12 if freq == 'M' else 4

    # Simple Exponential Smoothing (no trend, no seasonality)
    # For production, add trend and seasonality detection
    try:
        model = ExponentialSmoothing(
            ts,
            trend='add',
            seasonal=None,
            seasonal_periods=None
        )
        fitted_model = model.fit()
    except:
        # Fallback: simple exponential smoothing
        model = ExponentialSmoothing(ts, trend=None, seasonal=None)
        fitted_model = model.fit()

    # Generate forecasts
    forecast_result = fitted_model.forecast(steps=periods)

    # Approximate confidence intervals (ES doesn't provide built-in CI)
    # Use standard error approximation
    residuals = ts - fitted_model.fittedvalues
    std_error = residuals.std()

    # Determine forecast dates
    last_date = ts.index[-1]

    forecasts = []
    for i in range(periods):
        if freq == 'Q':
            forecast_date = last_date + pd.DateOffset(months=3 * (i + 1))
        else:
            forecast_date = last_date + pd.DateOffset(months=(i + 1))

        # Approximate confidence intervals
        forecast_value = forecast_result.iloc[i]
        margin_95 = 1.96 * std_error * np.sqrt(i + 1)
        margin_80 = 1.28 * std_error * np.sqrt(i + 1)

        forecasts.append({
            'date': forecast_date,
            'value': forecast_value,
            'ci_95_lower': forecast_value - margin_95,
            'ci_95_upper': forecast_value + margin_95,
            'ci_80_lower': forecast_value - margin_80,
            'ci_80_upper': forecast_value + margin_80,
            'model_params': {
                'smoothing_level': float(fitted_model.params.get('smoothing_level', 0.0)),
                'smoothing_trend': float(fitted_model.params.get('smoothing_trend', 0.0)),
                'aic': float(fitted_model.aic) if hasattr(fitted_model, 'aic') else None
            }
        })

    return forecasts


# For local testing
if __name__ == "__main__":
    # Test with sample event
    test_event = {
        "bucket": "economic-data-test",
        "key": "raw/gdp/usa_gdp_quarterly.csv",
        "indicator": "GDP",
        "country": "USA"
    }

    class Context:
        function_name = "forecast-economic-indicators"
        memory_limit_in_mb = 512
        invoked_function_arn = "arn:aws:lambda:us-east-1:123456789:function:test"

    result = lambda_handler(test_event, Context())
    print(json.dumps(result, indent=2))
