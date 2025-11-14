"""
LSTM-based forecasting models for disease surveillance.

Provides functions to train single models and ensembles,
generate forecasts, and create probabilistic predictions.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")


def prepare_sequences(
    data: np.ndarray,
    lookback_weeks: int,
    forecast_horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM training.

    Args:
        data: Time series data array
        lookback_weeks: Number of past weeks to use as input
        forecast_horizon: Number of weeks to forecast ahead

    Returns:
        X (input sequences), y (target sequences)
    """
    X, y = [], []

    for i in range(len(data) - lookback_weeks - forecast_horizon + 1):
        X.append(data[i:i + lookback_weeks])
        y.append(data[i + lookback_weeks:i + lookback_weeks + forecast_horizon])

    return np.array(X), np.array(y)


def create_lstm_model(
    input_shape: Tuple[int, int],
    forecast_horizon: int,
    lstm_units: List[int] = [128, 64],
    dense_units: int = 32,
    dropout_rate: float = 0.2
) -> keras.Model:
    """
    Create LSTM model architecture.

    Args:
        input_shape: (lookback_weeks, n_features)
        forecast_horizon: Number of time steps to forecast
        lstm_units: List of LSTM layer sizes
        dense_units: Dense layer size
        dropout_rate: Dropout rate

    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for LSTM models")

    model = Sequential()

    # First LSTM layer
    model.add(LSTM(
        lstm_units[0],
        return_sequences=len(lstm_units) > 1,
        input_shape=input_shape
    ))
    model.add(Dropout(dropout_rate))

    # Additional LSTM layers
    for i in range(1, len(lstm_units)):
        return_seq = i < len(lstm_units) - 1
        model.add(LSTM(lstm_units[i], return_sequences=return_seq))
        model.add(Dropout(dropout_rate))

    # Dense layers
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(forecast_horizon))

    # Compile
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    forecast_horizon: int = 4,
    epochs: int = 50,
    batch_size: int = 32,
    checkpoint_path: Optional[Path] = None,
    verbose: int = 1
) -> keras.Model:
    """
    Train LSTM forecasting model.

    Args:
        X_train: Training input sequences
        y_train: Training target sequences
        X_val: Validation input sequences (optional)
        y_val: Validation target sequences (optional)
        forecast_horizon: Forecast horizon
        epochs: Number of training epochs
        batch_size: Training batch size
        checkpoint_path: Path to save model checkpoints
        verbose: Training verbosity

    Returns:
        Trained model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for LSTM training")

    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape, forecast_horizon)

    # Callbacks
    callbacks = []

    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                save_best_only=True,
                monitor='val_loss' if X_val is not None else 'loss',
                verbose=verbose
            )
        )

    callbacks.append(
        EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=verbose
        )
    )

    # Train
    validation_data = (X_val, y_val) if X_val is not None else None

    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )

    return model


def train_lstm_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_models: int = 5,
    forecast_horizon: int = 4,
    epochs: int = 50,
    checkpoint_dir: Optional[Path] = None,
    verbose: int = 1
) -> List[keras.Model]:
    """
    Train ensemble of LSTM models with different initializations.

    Args:
        X_train: Training input sequences
        y_train: Training target sequences
        X_val: Validation sequences
        y_val: Validation targets
        n_models: Number of models in ensemble
        forecast_horizon: Forecast horizon
        epochs: Training epochs
        checkpoint_dir: Directory to save checkpoints
        verbose: Training verbosity

    Returns:
        List of trained models
    """
    models = []

    for i in range(n_models):
        print(f"\nTraining model {i+1}/{n_models}...")

        checkpoint_path = None
        if checkpoint_dir:
            checkpoint_path = checkpoint_dir / f"model_{i+1}.h5"

        model = train_lstm_model(
            X_train, y_train,
            X_val, y_val,
            forecast_horizon=forecast_horizon,
            epochs=epochs,
            checkpoint_path=checkpoint_path,
            verbose=verbose
        )

        models.append(model)

    return models


def generate_forecast(
    model: keras.Model,
    last_sequence: np.ndarray
) -> np.ndarray:
    """
    Generate forecast using trained model.

    Args:
        model: Trained LSTM model
        last_sequence: Last sequence of input data (lookback_weeks, n_features)

    Returns:
        Forecast array (forecast_horizon,)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for forecasting")

    # Reshape for model input
    input_seq = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])

    # Generate prediction
    forecast = model.predict(input_seq, verbose=0)

    return forecast[0]


def generate_probabilistic_forecast(
    models: List[keras.Model],
    last_sequence: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Generate probabilistic forecast from ensemble.

    Args:
        models: List of trained models
        last_sequence: Last input sequence

    Returns:
        Dictionary with mean, std, and quantiles
    """
    # Generate forecasts from all models
    forecasts = []
    for model in models:
        forecast = generate_forecast(model, last_sequence)
        forecasts.append(forecast)

    forecasts = np.array(forecasts)

    # Calculate statistics
    return {
        'mean': np.mean(forecasts, axis=0),
        'median': np.median(forecasts, axis=0),
        'std': np.std(forecasts, axis=0),
        'q05': np.percentile(forecasts, 5, axis=0),
        'q25': np.percentile(forecasts, 25, axis=0),
        'q75': np.percentile(forecasts, 75, axis=0),
        'q95': np.percentile(forecasts, 95, axis=0),
        'all_forecasts': forecasts
    }


def save_ensemble(models: List[keras.Model], save_dir: Path):
    """Save ensemble models to directory."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required")

    save_dir.mkdir(parents=True, exist_ok=True)

    for i, model in enumerate(models):
        model.save(save_dir / f"model_{i+1}.h5")

    print(f"Saved {len(models)} models to {save_dir}")


def load_ensemble(load_dir: Path) -> List[keras.Model]:
    """Load ensemble models from directory."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required")

    models = []
    model_files = sorted(load_dir.glob("model_*.h5"))

    for model_file in model_files:
        model = keras.models.load_model(model_file)
        models.append(model)

    print(f"Loaded {len(models)} models from {load_dir}")
    return models
