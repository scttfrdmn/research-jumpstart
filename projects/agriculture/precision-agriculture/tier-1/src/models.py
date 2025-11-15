"""
Model architectures for yield prediction.

Includes CNN, LSTM, and ensemble models optimized for multi-sensor
agricultural data.
"""


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models


def build_cnn_model(
    input_shape: tuple[int, int, int], num_outputs: int = 1, architecture: str = "resnet"
) -> keras.Model:
    """
    Build CNN model for spatial pattern recognition.

    Args:
        input_shape: Input image shape (height, width, channels)
        num_outputs: Number of output values (1 for regression)
        architecture: Model architecture ('simple', 'resnet')

    Returns:
        Compiled Keras model
    """
    if architecture == "simple":
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                # Conv block 1
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.2),
                # Conv block 2
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.2),
                # Conv block 3
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.2),
                # Dense layers
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                # Output
                layers.Dense(num_outputs, activation="linear"),
            ]
        )

    elif architecture == "resnet":
        # ResNet-like architecture with skip connections
        inputs = layers.Input(shape=input_shape)

        # Initial conv
        x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # Residual blocks
        for filters in [32, 64, 128]:
            # Main path
            shortcut = x
            x = layers.Conv2D(filters, (3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Conv2D(filters, (3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)

            # Skip connection (with projection if needed)
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, (1, 1), padding="same")(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)

            x = layers.Add()([x, shortcut])
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.2)(x)

        # Global pooling and dense
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_outputs, activation="linear")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model


def build_lstm_model(
    input_shape: tuple[int, int], num_outputs: int = 1, architecture: str = "bidirectional"
) -> keras.Model:
    """
    Build LSTM model for temporal pattern recognition.

    Args:
        input_shape: Input shape (time_steps, features)
        num_outputs: Number of output values
        architecture: Model architecture ('simple', 'bidirectional', 'stacked')

    Returns:
        Compiled Keras model
    """
    if architecture == "simple":
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.LSTM(64, return_sequences=True),
                layers.Dropout(0.3),
                layers.LSTM(32),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(num_outputs, activation="linear"),
            ]
        )

    elif architecture == "bidirectional":
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
                layers.Dropout(0.3),
                layers.Bidirectional(layers.LSTM(32)),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(num_outputs, activation="linear"),
            ]
        )

    elif architecture == "stacked":
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.LSTM(128, return_sequences=True),
                layers.Dropout(0.3),
                layers.LSTM(64, return_sequences=True),
                layers.Dropout(0.3),
                layers.LSTM(32),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(num_outputs, activation="linear"),
            ]
        )

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model


def build_ensemble_model(
    spatial_input_shape: tuple[int, int, int],
    temporal_input_shape: tuple[int, int],
    num_outputs: int = 1,
) -> keras.Model:
    """
    Build ensemble model combining CNN and LSTM.

    Args:
        spatial_input_shape: Spatial input shape (height, width, channels)
        temporal_input_shape: Temporal input shape (time_steps, features)
        num_outputs: Number of output values

    Returns:
        Compiled Keras model
    """
    # Spatial branch (CNN)
    spatial_input = layers.Input(shape=spatial_input_shape, name="spatial_input")
    x_spatial = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(spatial_input)
    x_spatial = layers.BatchNormalization()(x_spatial)
    x_spatial = layers.MaxPooling2D((2, 2))(x_spatial)
    x_spatial = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x_spatial)
    x_spatial = layers.BatchNormalization()(x_spatial)
    x_spatial = layers.MaxPooling2D((2, 2))(x_spatial)
    x_spatial = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x_spatial)
    x_spatial = layers.BatchNormalization()(x_spatial)
    x_spatial = layers.GlobalAveragePooling2D()(x_spatial)
    x_spatial = layers.Dense(128, activation="relu")(x_spatial)
    x_spatial = layers.Dropout(0.3)(x_spatial)

    # Temporal branch (LSTM)
    temporal_input = layers.Input(shape=temporal_input_shape, name="temporal_input")
    x_temporal = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(temporal_input)
    x_temporal = layers.Dropout(0.3)(x_temporal)
    x_temporal = layers.Bidirectional(layers.LSTM(32))(x_temporal)
    x_temporal = layers.Dropout(0.3)(x_temporal)
    x_temporal = layers.Dense(64, activation="relu")(x_temporal)
    x_temporal = layers.Dropout(0.3)(x_temporal)

    # Merge branches
    merged = layers.Concatenate()([x_spatial, x_temporal])
    x = layers.Dense(128, activation="relu")(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_outputs, activation="linear")(x)

    model = keras.Model(
        inputs=[spatial_input, temporal_input], outputs=outputs, name="ensemble_model"
    )

    return model


def build_random_forest(n_estimators: int = 100, max_depth: int = 20):
    """
    Build Random Forest model for baseline comparison.

    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth

    Returns:
        Scikit-learn RandomForestRegressor
    """
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    return model


def build_gradient_boosting(n_estimators: int = 100, learning_rate: float = 0.1):
    """
    Build Gradient Boosting model.

    Args:
        n_estimators: Number of boosting stages
        learning_rate: Learning rate

    Returns:
        Scikit-learn GradientBoostingRegressor
    """
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )

    return model


class EnsembleStacker:
    """
    Stack multiple models using meta-learner.
    """

    def __init__(self, base_models, meta_model=None):
        """
        Initialize ensemble stacker.

        Args:
            base_models: List of (name, model) tuples
            meta_model: Meta-learner model (default: linear regression)
        """
        self.base_models = base_models

        if meta_model is None:
            from sklearn.linear_model import Ridge

            self.meta_model = Ridge(alpha=1.0)
        else:
            self.meta_model = meta_model

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit base models and meta-learner.

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        # Train base models
        base_predictions = []

        for name, model in self.base_models:
            print(f"Training {name}...")
            model.fit(X, y)

            # Get predictions for meta-learner
            pred = model.predict(X_val) if X_val is not None else model.predict(X)

            base_predictions.append(pred)

        # Stack predictions
        X_meta = np.column_stack(base_predictions)

        # Train meta-learner
        print("Training meta-learner...")
        if X_val is not None and y_val is not None:
            self.meta_model.fit(X_meta, y_val)
        else:
            self.meta_model.fit(X_meta, y)

        print("Ensemble training complete!")

    def predict(self, X):
        """
        Make predictions using ensemble.

        Args:
            X: Input features

        Returns:
            Ensemble predictions
        """
        # Get base model predictions
        base_predictions = []
        for _name, model in self.base_models:
            pred = model.predict(X)
            base_predictions.append(pred)

        # Stack predictions
        X_meta = np.column_stack(base_predictions)

        # Meta-learner prediction
        return self.meta_model.predict(X_meta)


def compile_model(
    model: keras.Model, loss: str = "mse", optimizer: str = "adam", learning_rate: float = 0.001
):
    """
    Compile Keras model with standard settings.

    Args:
        model: Keras model
        loss: Loss function ('mse', 'mae')
        optimizer: Optimizer name
        learning_rate: Learning rate
    """
    if optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        opt = optimizer

    model.compile(optimizer=opt, loss=loss, metrics=["mae", "mse"])

    return model
