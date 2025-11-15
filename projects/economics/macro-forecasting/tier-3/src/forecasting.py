"""
Forecasting module for macroeconomic time series.

Implements ARIMA, VAR, LSTM, Prophet, and ensemble methods.
"""

import logging
from typing import Optional

# AWS
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Machine learning
from sklearn.preprocessing import StandardScaler

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from tensorflow import keras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GDPForecaster:
    """Forecast GDP using multiple time series methods."""

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize GDP forecaster.

        Args:
            data: Historical GDP data (datetime index)
        """
        self.data = data
        self.models = {}
        self.forecasts = {}
        self.scaler = StandardScaler()

    def fit_arima(
        self, series: pd.Series, order: Optional[tuple[int, int, int]] = None, auto: bool = True
    ) -> ARIMA:
        """
        Fit ARIMA model.

        Args:
            series: Time series data
            order: ARIMA order (p, d, q)
            auto: Use auto_arima to find best order

        Returns:
            Fitted ARIMA model
        """
        logger.info("Fitting ARIMA model")

        if auto:
            # Use auto_arima to find best parameters
            model = auto_arima(
                series,
                start_p=0,
                start_q=0,
                max_p=5,
                max_q=5,
                seasonal=False,
                trace=True,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )
            logger.info(f"Best ARIMA order: {model.order}")
        else:
            order = order or (1, 1, 1)
            model = ARIMA(series, order=order)
            model = model.fit()

        self.models["arima"] = model
        return model

    def fit_sarimax(
        self,
        series: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 4),
    ) -> SARIMAX:
        """
        Fit SARIMAX model with exogenous variables.

        Args:
            series: Time series data
            exog: Exogenous variables
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)

        Returns:
            Fitted SARIMAX model
        """
        logger.info("Fitting SARIMAX model")

        model = SARIMAX(
            series,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted_model = model.fit(disp=False)

        self.models["sarimax"] = fitted_model
        return fitted_model

    def fit_var(self, data: pd.DataFrame, maxlags: int = 15) -> VAR:
        """
        Fit Vector Autoregression model.

        Args:
            data: Multivariate time series data
            maxlags: Maximum number of lags to consider

        Returns:
            Fitted VAR model
        """
        logger.info("Fitting VAR model")

        model = VAR(data)
        fitted_model = model.fit(maxlags=maxlags, ic="aic")

        logger.info(f"VAR lag order: {fitted_model.k_ar}")
        self.models["var"] = fitted_model
        return fitted_model

    def fit_lstm(
        self, series: pd.Series, lookback: int = 12, epochs: int = 50, batch_size: int = 32
    ) -> keras.Model:
        """
        Fit LSTM neural network.

        Args:
            series: Time series data
            lookback: Number of time steps to look back
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            Fitted LSTM model
        """
        logger.info("Fitting LSTM model")

        # Prepare data
        X, y = self._prepare_lstm_data(series, lookback)

        # Build model
        model = keras.Sequential(
            [
                keras.layers.LSTM(
                    50, activation="relu", return_sequences=True, input_shape=(lookback, 1)
                ),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(50, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(25, activation="relu"),
                keras.layers.Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Train
        history = model.fit(
            X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
        )

        logger.info(f"LSTM training loss: {history.history['loss'][-1]:.4f}")
        self.models["lstm"] = model
        return model

    def _prepare_lstm_data(self, series: pd.Series, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM."""
        # Scale data
        values = series.values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values)

        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled)):
            X.append(scaled[i - lookback : i, 0])
            y.append(scaled[i, 0])

        X = np.array(X)
        y = np.array(y)

        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        return X, y

    def fit_prophet(
        self, series: pd.Series, yearly_seasonality: bool = True, weekly_seasonality: bool = False
    ) -> Prophet:
        """
        Fit Facebook Prophet model.

        Args:
            series: Time series data
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality

        Returns:
            Fitted Prophet model
        """
        logger.info("Fitting Prophet model")

        # Prepare data for Prophet
        df = pd.DataFrame({"ds": series.index, "y": series.values})

        # Create and fit model
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            changepoint_prior_scale=0.05,
        )
        model.fit(df)

        self.models["prophet"] = model
        return model

    def forecast(
        self, model_name: str, steps: int, exog: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Generate forecast from a fitted model.

        Args:
            model_name: Name of the model ('arima', 'sarimax', 'var', 'lstm', 'prophet')
            steps: Number of steps to forecast
            exog: Exogenous variables for SARIMAX

        Returns:
            Forecast series
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not fitted")

        model = self.models[model_name]

        if model_name in ["arima", "sarimax"]:
            forecast = model.forecast(steps=steps, exog=exog)
        elif model_name == "var":
            forecast = model.forecast(model.endog[-model.k_ar :], steps=steps)
            forecast = pd.DataFrame(forecast, columns=model.names)
        elif model_name == "lstm":
            forecast = self._forecast_lstm(model, steps)
        elif model_name == "prophet":
            future = model.make_future_dataframe(periods=steps, freq="Q")
            forecast = model.predict(future)
            forecast = pd.Series(
                forecast["yhat"].values[-steps:], index=future["ds"].values[-steps:]
            )

        self.forecasts[model_name] = forecast
        logger.info(f"Generated {steps}-step forecast with {model_name}")

        return forecast

    def _forecast_lstm(self, model: keras.Model, steps: int) -> pd.Series:
        """Generate forecast from LSTM model."""
        # Use last available data as starting point
        last_sequence = self.scaler.transform(self.data.iloc[-12:].values.reshape(-1, 1))

        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(steps):
            # Reshape for prediction
            X = current_sequence.reshape((1, 12, 1))

            # Predict next step
            pred = model.predict(X, verbose=0)
            predictions.append(pred[0, 0])

            # Update sequence (roll forward)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred

        # Inverse transform
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        return pd.Series(predictions.flatten())

    def ensemble_forecast(
        self, steps: int, weights: Optional[dict[str, float]] = None
    ) -> pd.Series:
        """
        Create ensemble forecast from multiple models.

        Args:
            steps: Number of steps to forecast
            weights: Model weights (defaults to equal weights)

        Returns:
            Ensemble forecast
        """
        logger.info("Creating ensemble forecast")

        if not self.forecasts:
            raise ValueError("No forecasts available. Run forecast() first.")

        # Default to equal weights
        if weights is None:
            weights = {name: 1.0 / len(self.forecasts) for name in self.forecasts}

        # Weighted average
        ensemble = None
        for name, forecast in self.forecasts.items():
            weight = weights.get(name, 0)
            if ensemble is None:
                ensemble = forecast * weight
            else:
                ensemble += forecast * weight

        return ensemble

    def evaluate(self, actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
        """
        Evaluate forecast accuracy.

        Args:
            actual: Actual values
            forecast: Forecasted values

        Returns:
            Dictionary of error metrics
        """
        # Align indices
        common_idx = actual.index.intersection(forecast.index)
        actual_aligned = actual.loc[common_idx]
        forecast_aligned = forecast.loc[common_idx]

        metrics = {
            "rmse": np.sqrt(mean_squared_error(actual_aligned, forecast_aligned)),
            "mae": mean_absolute_error(actual_aligned, forecast_aligned),
            "mape": np.mean(np.abs((actual_aligned - forecast_aligned) / actual_aligned)) * 100,
        }

        logger.info(
            f"Forecast metrics: RMSE={metrics['rmse']:.2f}, "
            f"MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%"
        )

        return metrics


class RecessionPredictor:
    """Predict recession probability using probit model."""

    def __init__(self):
        """Initialize recession predictor."""
        self.model = None

    def prepare_features(
        self,
        yield_curve: pd.Series,
        unemployment: pd.Series,
        stock_returns: pd.Series,
        credit_spread: pd.Series,
    ) -> pd.DataFrame:
        """
        Prepare features for recession prediction.

        Args:
            yield_curve: 10Y - 3M yield spread
            unemployment: Unemployment rate changes
            stock_returns: Stock market returns
            credit_spread: Corporate credit spread

        Returns:
            Feature DataFrame
        """
        features = pd.DataFrame(
            {
                "yield_curve": yield_curve,
                "unemployment_change": unemployment.diff(),
                "stock_returns": stock_returns,
                "credit_spread": credit_spread,
            }
        )

        return features.dropna()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit probit model for recession prediction.

        Args:
            X: Features
            y: Recession indicator (0/1)
        """
        from statsmodels.discrete.discrete_model import Probit

        self.model = Probit(y, X).fit()
        logger.info(f"Recession model fitted. Pseudo R-squared: {self.model.prsquared:.4f}")

    def predict_probability(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict recession probability.

        Args:
            X: Features

        Returns:
            Recession probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        probabilities = self.model.predict(X)
        return probabilities


def main():
    """
    Main function for GDP forecasting.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Forecast GDP")
    parser.add_argument("--series-id", type=str, default="GDP", help="FRED series ID")
    parser.add_argument("--steps", type=int, default=8, help="Forecast horizon (quarters)")
    parser.add_argument(
        "--models", type=str, nargs="+", default=["arima", "prophet"], help="Models to use"
    )
    args = parser.parse_args()

    # Load data (would need FRED API key)
    from .data_ingestion import FREDDataLoader

    loader = FREDDataLoader()
    series = loader.get_series(args.series_id, start_date="2000-01-01")

    # Initialize forecaster
    forecaster = GDPForecaster(series)

    # Fit models
    for model_name in args.models:
        if model_name == "arima":
            forecaster.fit_arima(series)
        elif model_name == "prophet":
            forecaster.fit_prophet(series)
        # Add other models as needed

    # Generate forecasts
    for model_name in args.models:
        forecast = forecaster.forecast(model_name, args.steps)
        print(f"\n{model_name.upper()} Forecast:")
        print(forecast)

    # Ensemble forecast
    if len(args.models) > 1:
        ensemble = forecaster.ensemble_forecast(args.steps)
        print("\nEnsemble Forecast:")
        print(ensemble)


if __name__ == "__main__":
    main()
