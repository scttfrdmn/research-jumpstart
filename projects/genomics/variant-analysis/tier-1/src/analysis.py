"""
Variant calling and analysis functions for genomic data.

This module provides functions for training variant caller models,
making predictions, combining ensemble models, and evaluating performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras


def build_cnn_variant_caller(
    input_shape: Tuple[int, int, int] = (221, 100, 7),
    architecture: str = "resnet"
) -> keras.Model:
    """
    Build CNN architecture for variant calling.

    Args:
        input_shape: Shape of input pileup tensor (positions, depth, channels)
        architecture: Model architecture ("resnet", "simple", "inception")

    Returns:
        Keras model
    """
    if architecture == "resnet":
        # ResNet-inspired architecture
        inputs = keras.layers.Input(shape=input_shape)

        # Initial convolution
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        # Residual blocks
        for filters in [64, 128, 256]:
            # Residual connection
            shortcut = x

            # Conv block
            x = keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = keras.layers.BatchNormalization()(x)

            # Match dimensions if needed
            if shortcut.shape[-1] != filters:
                shortcut = keras.layers.Conv2D(filters, (1, 1), padding='same')(shortcut)

            # Add residual connection
            x = keras.layers.Add()([x, shortcut])
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.MaxPooling2D((2, 2))(x)

        # Global pooling and dense layers
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)

        # Output: per-position variant probabilities
        outputs = keras.layers.Dense(input_shape[0], activation='sigmoid')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

    elif architecture == "simple":
        # Simpler CNN for faster training
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(input_shape[0], activation='sigmoid')
        ])

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model


def train_variant_caller(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    architecture: str = "resnet",
    epochs: int = 50,
    batch_size: int = 32,
    checkpoint_path: Optional[Path] = None
) -> Tuple[keras.Model, Dict]:
    """
    Train a CNN variant caller model.

    Args:
        X_train: Training pileup tensors
        y_train: Training labels
        X_val: Validation pileup tensors
        y_val: Validation labels
        architecture: Model architecture
        epochs: Number of training epochs
        batch_size: Batch size
        checkpoint_path: Path to save model checkpoints

    Returns:
        Tuple of (trained_model, training_history)
    """
    # Build model
    model = build_cnn_variant_caller(
        input_shape=X_train.shape[1:],
        architecture=architecture
    )

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3
        )
    ]

    if checkpoint_path:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor='val_loss',
                save_best_only=True
            )
        )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history.history


def call_variants(
    model: keras.Model,
    X: np.ndarray,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Call variants using trained model.

    Args:
        model: Trained variant caller model
        X: Pileup tensors
        threshold: Probability threshold for variant calling

    Returns:
        Tuple of (predictions, probabilities)
    """
    # Predict probabilities
    probs = model.predict(X)

    # Threshold to binary predictions
    preds = (probs > threshold).astype(int)

    return preds, probs


def ensemble_predict(
    models: List[keras.Model],
    X: np.ndarray,
    weights: Optional[List[float]] = None,
    strategy: str = "weighted_average"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine predictions from multiple models using ensemble strategy.

    Args:
        models: List of trained models
        X: Input data
        weights: Optional weights for each model (for weighted strategies)
        strategy: Ensemble strategy ("weighted_average", "majority_vote", "max")

    Returns:
        Tuple of (ensemble_predictions, ensemble_probabilities)
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    # Collect predictions from all models
    all_probs = []
    for model in models:
        probs = model.predict(X, verbose=0)
        all_probs.append(probs)

    all_probs = np.array(all_probs)  # Shape: (n_models, n_samples, n_positions)

    if strategy == "weighted_average":
        # Weighted average of probabilities
        weights_array = np.array(weights).reshape(-1, 1, 1)
        ensemble_probs = np.sum(all_probs * weights_array, axis=0)

    elif strategy == "majority_vote":
        # Majority vote on binary predictions
        binary_preds = (all_probs > 0.5).astype(int)
        ensemble_probs = np.mean(binary_preds, axis=0)

    elif strategy == "max":
        # Max probability across models
        ensemble_probs = np.max(all_probs, axis=0)

    else:
        raise ValueError(f"Unknown ensemble strategy: {strategy}")

    # Threshold to binary
    ensemble_preds = (ensemble_probs > 0.5).astype(int)

    return ensemble_preds, ensemble_probs


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate variant calling performance metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Optional prediction probabilities (for AUC)

    Returns:
        Dictionary of metrics
    """
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, average='binary', zero_division=0
    )

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    # Confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    tn, fp, fn, tp = cm.ravel()

    metrics.update({
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
    })

    # AUC if probabilities provided
    if y_prob is not None:
        y_prob_flat = y_prob.flatten()
        try:
            auc = roc_auc_score(y_true_flat, y_prob_flat)
            metrics['auc'] = auc
        except ValueError:
            # Handle case where only one class present
            metrics['auc'] = 0.0

    return metrics


def compare_with_truth_set(
    predicted_variants: pd.DataFrame,
    truth_vcf_path: Path,
    chromosome: str,
    tolerance: int = 0
) -> Dict[str, any]:
    """
    Compare predicted variants with GIAB truth set.

    Args:
        predicted_variants: DataFrame with columns ['chrom', 'pos', 'prob']
        truth_vcf_path: Path to truth VCF file
        chromosome: Chromosome to compare
        tolerance: Position tolerance for matching (base pairs)

    Returns:
        Dictionary with comparison results
    """
    try:
        import cyvcf2
    except ImportError:
        print("Warning: cyvcf2 not installed. Using simplified comparison.")
        return {'error': 'cyvcf2 not available'}

    # Load truth set
    vcf = cyvcf2.VCF(str(truth_vcf_path))

    # Extract truth variants for chromosome
    truth_positions = set()
    for variant in vcf(f"{chromosome}"):
        truth_positions.add(variant.POS)

    vcf.close()

    # Compare predictions
    pred_positions = set(predicted_variants['pos'].values)

    # Find matches (allowing tolerance)
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    matched_truth = set()
    matched_pred = set()

    for pred_pos in pred_positions:
        # Check if any truth position within tolerance
        match_found = False
        for truth_pos in truth_positions:
            if abs(pred_pos - truth_pos) <= tolerance:
                match_found = True
                matched_truth.add(truth_pos)
                matched_pred.add(pred_pos)
                break

        if match_found:
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(truth_positions - matched_truth)

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'truth_total': len(truth_positions),
        'predicted_total': len(pred_positions)
    }


def save_variants_to_vcf(
    variants: pd.DataFrame,
    output_path: Path,
    sample_id: str,
    reference_genome: str = "GRCh37"
):
    """
    Save predicted variants to VCF file.

    Args:
        variants: DataFrame with columns ['chrom', 'pos', 'ref', 'alt', 'qual']
        output_path: Path to output VCF file
        sample_id: Sample ID
        reference_genome: Reference genome name
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##reference={reference_genome}\n")
        f.write("##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">\n")
        f.write("##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">\n")
        f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{}\n".format(sample_id))

        # Write variants
        for _, row in variants.iterrows():
            chrom = row['chrom']
            pos = row['pos']
            ref = row.get('ref', 'N')
            alt = row.get('alt', '.')
            qual = row.get('qual', 0)

            # Simple format
            f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual:.2f}\t.\t.\tGT\t0/1\n")

    print(f"✓ Saved variants to {output_path}")
