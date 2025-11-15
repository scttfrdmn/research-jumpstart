"""
Artifact Analysis Utilities

Functions for artifact image classification, feature extraction,
and typological analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image


def load_artifact_images(
    image_dir: Path, image_size: tuple[int, int] = (224, 224)
) -> list[np.ndarray]:
    """
    Load and preprocess artifact images from directory.

    Args:
        image_dir: Directory containing artifact images
        image_size: Target size for images (height, width)

    Returns:
        List of preprocessed image arrays
    """
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    images = []
    image_paths = sorted(Path(image_dir).glob("*.jpg")) + sorted(Path(image_dir).glob("*.png"))

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            images.append(img_tensor.numpy())
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    return images


def extract_artifact_features(images: np.ndarray, model) -> np.ndarray:
    """
    Extract deep features from artifact images using trained model.

    Args:
        images: Array of preprocessed images
        model: Trained CNN model for feature extraction

    Returns:
        Array of extracted features
    """
    model.eval()
    features = []

    with torch.no_grad():
        for img in images:
            img_tensor = torch.from_numpy(img).unsqueeze(0)
            feature_vector = model(img_tensor)
            features.append(feature_vector.numpy())

    return np.vstack(features)


def classify_artifacts(images: np.ndarray, model, class_names: list[str]) -> pd.DataFrame:
    """
    Classify artifact images into typological categories.

    Args:
        images: Array of preprocessed images
        model: Trained classification model
        class_names: List of artifact class names

    Returns:
        DataFrame with predictions and confidence scores
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for img in images:
            img_tensor = torch.from_numpy(img).unsqueeze(0)
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1).numpy()[0]

            pred_class = np.argmax(probabilities)
            pred_conf = probabilities[pred_class]

            predictions.append(
                {
                    "predicted_class": class_names[pred_class],
                    "confidence": pred_conf,
                    "probabilities": probabilities,
                }
            )

    return pd.DataFrame(predictions)


def compare_artifact_assemblages(
    site_a_features: np.ndarray, site_b_features: np.ndarray
) -> dict[str, float]:
    """
    Compare artifact assemblages between two sites using feature distributions.

    Args:
        site_a_features: Feature vectors from site A artifacts
        site_b_features: Feature vectors from site B artifacts

    Returns:
        Dictionary of similarity metrics
    """
    from scipy.spatial.distance import cosine
    from scipy.stats import ks_2samp

    # Mean feature comparison
    mean_a = np.mean(site_a_features, axis=0)
    mean_b = np.mean(site_b_features, axis=0)
    cosine_similarity = 1 - cosine(mean_a, mean_b)

    # Distribution comparison (Kolmogorov-Smirnov test)
    ks_stats = []
    for i in range(site_a_features.shape[1]):
        ks_stat, _p_value = ks_2samp(site_a_features[:, i], site_b_features[:, i])
        ks_stats.append(ks_stat)

    return {
        "cosine_similarity": float(cosine_similarity),
        "mean_ks_statistic": float(np.mean(ks_stats)),
        "assemblage_difference": float(1 - cosine_similarity),
    }


def identify_diagnostic_artifacts(
    features: np.ndarray, labels: np.ndarray, top_n: int = 10
) -> list[int]:
    """
    Identify most diagnostic artifacts for classification.

    Args:
        features: Feature vectors of all artifacts
        labels: Class labels for artifacts
        top_n: Number of top diagnostic artifacts to return

    Returns:
        Indices of most diagnostic artifacts
    """
    from sklearn.ensemble import RandomForestClassifier

    # Train classifier to identify most important artifacts
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features, labels)

    # Get feature importances (proxy for diagnostic value)
    # In real implementation, would use more sophisticated methods
    importances = clf.feature_importances_

    # Find artifacts with highest average feature importance
    artifact_scores = np.mean(features * importances, axis=1)
    diagnostic_indices = np.argsort(artifact_scores)[-top_n:]

    return diagnostic_indices.tolist()


def calculate_artifact_diversity(features: np.ndarray) -> dict[str, float]:
    """
    Calculate diversity metrics for artifact assemblage.

    Args:
        features: Feature vectors of artifacts

    Returns:
        Dictionary of diversity metrics
    """
    from scipy.spatial.distance import pdist
    from sklearn.decomposition import PCA

    # PCA-based diversity (variance in principal components)
    pca = PCA(n_components=min(10, features.shape[1]))
    pca.fit_transform(features)
    pca_diversity = np.sum(pca.explained_variance_)

    # Pairwise distance-based diversity
    distances = pdist(features, metric="euclidean")
    mean_distance = np.mean(distances)

    return {
        "pca_diversity": float(pca_diversity),
        "mean_pairwise_distance": float(mean_distance),
        "assemblage_richness": int(features.shape[0]),
    }
