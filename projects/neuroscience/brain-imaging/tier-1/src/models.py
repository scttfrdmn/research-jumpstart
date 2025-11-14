"""
Deep learning models for brain state classification.
"""

import numpy as np
from pathlib import Path


# Define base directory
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'saved_models'
MODELS_DIR.mkdir(exist_ok=True)


def build_connectivity_classifier(n_rois, n_classes):
    """
    Build a simple MLP for connectivity-based classification.

    Parameters
    ----------
    n_rois : int
        Number of ROIs (connectivity matrix is n_rois x n_rois)
    n_classes : int
        Number of classes to predict

    Returns
    -------
    model : keras Model
        Compiled model
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    print(f"Building connectivity classifier...")
    print(f"  Input: {n_rois}x{n_rois} connectivity matrix")
    print(f"  Output: {n_classes} classes")

    # Input is flattened upper triangle of connectivity matrix
    n_features = n_rois * (n_rois - 1) // 2

    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"✓ Model built with {model.count_params():,} parameters")
    return model


def build_3d_cnn(input_shape, n_classes):
    """
    Build 3D CNN for volumetric fMRI classification.

    Parameters
    ----------
    input_shape : tuple
        Input shape (x, y, z, channels)
    n_classes : int
        Number of classes

    Returns
    -------
    model : keras Model
        Compiled model
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    print(f"Building 3D CNN...")
    print(f"  Input shape: {input_shape}")
    print(f"  Output classes: {n_classes}")

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # First conv block
        layers.Conv3D(32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=2),
        layers.BatchNormalization(),

        # Second conv block
        layers.Conv3D(64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=2),
        layers.BatchNormalization(),

        # Third conv block
        layers.Conv3D(128, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=2),
        layers.BatchNormalization(),

        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"✓ 3D CNN built with {model.count_params():,} parameters")
    return model


def flatten_connectivity_matrix(conn_matrix):
    """
    Flatten connectivity matrix to feature vector (upper triangle).

    Parameters
    ----------
    conn_matrix : ndarray, shape (n_rois, n_rois)
        Connectivity matrix

    Returns
    -------
    features : ndarray, shape (n_features,)
        Flattened features
    """
    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(conn_matrix.shape[0], k=1)
    features = conn_matrix[triu_idx]
    return features


def unflatten_connectivity_features(features, n_rois):
    """
    Reconstruct symmetric connectivity matrix from flattened features.

    Parameters
    ----------
    features : ndarray
        Flattened upper triangle
    n_rois : int
        Number of ROIs

    Returns
    -------
    conn_matrix : ndarray, shape (n_rois, n_rois)
        Symmetric connectivity matrix
    """
    conn_matrix = np.zeros((n_rois, n_rois))

    # Fill upper triangle
    triu_idx = np.triu_indices(n_rois, k=1)
    conn_matrix[triu_idx] = features

    # Make symmetric
    conn_matrix = conn_matrix + conn_matrix.T

    return conn_matrix


def train_ensemble(models, X_train, y_train, X_val=None, y_val=None, epochs=50):
    """
    Train ensemble of models.

    Parameters
    ----------
    models : list
        List of keras models
    X_train : ndarray
        Training data
    y_train : ndarray
        Training labels
    X_val : ndarray or None
        Validation data
    y_val : ndarray or None
        Validation labels
    epochs : int
        Number of epochs

    Returns
    -------
    histories : list
        Training histories
    """
    try:
        from tensorflow import keras
    except ImportError:
        raise ImportError("TensorFlow not installed")

    print(f"Training ensemble of {len(models)} models...")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5
        )
    ]

    histories = []
    for i, model in enumerate(models):
        print(f"\nTraining model {i+1}/{len(models)}...")

        validation_data = (X_val, y_val) if X_val is not None else None

        history = model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        histories.append(history)

    print(f"\n✓ Ensemble training complete!")
    return histories


def ensemble_predict(models, X):
    """
    Make predictions using ensemble (average probabilities).

    Parameters
    ----------
    models : list
        List of trained models
    X : ndarray
        Input data

    Returns
    -------
    predictions : ndarray
        Ensemble predictions
    """
    print(f"Making ensemble predictions with {len(models)} models...")

    # Get predictions from all models
    all_probs = []
    for model in models:
        probs = model.predict(X, verbose=0)
        all_probs.append(probs)

    # Average probabilities
    avg_probs = np.mean(all_probs, axis=0)

    # Get class predictions
    predictions = np.argmax(avg_probs, axis=1)

    print(f"✓ Predictions complete: {len(predictions)} samples")
    return predictions, avg_probs


def save_model(model, name):
    """
    Save trained model.

    Parameters
    ----------
    model : keras Model
        Trained model
    name : str
        Model name
    """
    output_path = MODELS_DIR / f'{name}.h5'
    model.save(output_path)
    print(f"✓ Model saved to {output_path}")


def load_model(name):
    """
    Load saved model.

    Parameters
    ----------
    name : str
        Model name

    Returns
    -------
    model : keras Model
        Loaded model
    """
    try:
        from tensorflow import keras
    except ImportError:
        raise ImportError("TensorFlow not installed")

    input_path = MODELS_DIR / f'{name}.h5'

    if not input_path.exists():
        raise FileNotFoundError(f"Model not found: {input_path}")

    model = keras.models.load_model(input_path)
    print(f"✓ Model loaded from {input_path}")
    return model


if __name__ == '__main__':
    # Test model building
    print("Testing model utilities...")

    # Test connectivity classifier
    print("\n1. Connectivity Classifier:")
    model_conn = build_connectivity_classifier(n_rois=200, n_classes=2)
    print(f"   Parameters: {model_conn.count_params():,}")

    # Test flattening
    print("\n2. Connectivity Matrix Flattening:")
    conn_matrix = np.random.randn(200, 200)
    features = flatten_connectivity_matrix(conn_matrix)
    print(f"   Flattened shape: {features.shape}")

    reconstructed = unflatten_connectivity_features(features, 200)
    print(f"   Reconstructed shape: {reconstructed.shape}")

    # Test 3D CNN (with small input for testing)
    print("\n3. 3D CNN:")
    model_3d = build_3d_cnn(input_shape=(32, 32, 32, 1), n_classes=2)
    print(f"   Parameters: {model_3d.count_params():,}")
