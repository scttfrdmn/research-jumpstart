"""
Facial expression analysis utilities for emotion recognition.

This module provides functions to extract features from facial images,
detect emotions, and compute Action Units (AUs).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def extract_facial_features(
    image: np.ndarray,
    method: str = 'hog'
) -> np.ndarray:
    """
    Extract features from facial image.

    Args:
        image: Facial image, shape (height, width, 3) or (height, width)
        method: Feature extraction method ('hog', 'lbp', 'pixel')

    Returns:
        Feature vector
    """
    if method == 'hog':
        # Histogram of Oriented Gradients (simplified)
        if len(image.shape) == 3:
            image_gray = np.mean(image, axis=2)
        else:
            image_gray = image

        # Compute gradients
        gx = np.gradient(image_gray, axis=1)
        gy = np.gradient(image_gray, axis=0)

        # Compute magnitude and orientation
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)

        # Create histogram (simplified)
        features = np.histogram(orientation.flatten(), bins=9, weights=magnitude.flatten())[0]

        return features

    elif method == 'lbp':
        # Local Binary Pattern (simplified)
        if len(image.shape) == 3:
            image_gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            image_gray = image.astype(np.uint8)

        # Simple LBP implementation
        h, w = image_gray.shape
        lbp = np.zeros((h-2, w-2))

        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image_gray[i, j]
                code = 0
                code |= (image_gray[i-1, j-1] >= center) << 7
                code |= (image_gray[i-1, j] >= center) << 6
                code |= (image_gray[i-1, j+1] >= center) << 5
                code |= (image_gray[i, j+1] >= center) << 4
                code |= (image_gray[i+1, j+1] >= center) << 3
                code |= (image_gray[i+1, j] >= center) << 2
                code |= (image_gray[i+1, j-1] >= center) << 1
                code |= (image_gray[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code

        features = np.histogram(lbp.flatten(), bins=256, range=(0, 256))[0]
        return features

    elif method == 'pixel':
        # Raw pixel values (downsampled)
        from scipy.ndimage import zoom
        target_size = (64, 64)

        if len(image.shape) == 3:
            image_gray = np.mean(image, axis=2)
        else:
            image_gray = image

        h, w = image_gray.shape
        zoom_factors = (target_size[0] / h, target_size[1] / w)
        resized = zoom(image_gray, zoom_factors, order=1)

        return resized.flatten()

    else:
        raise ValueError(f"Unknown method: {method}")


def detect_emotions_from_face(
    features: np.ndarray,
    model=None
) -> Dict[str, float]:
    """
    Detect emotions from facial features using a pre-trained model.

    Args:
        features: Facial feature vector
        model: Pre-trained emotion recognition model (optional)

    Returns:
        Dictionary of emotion probabilities
    """
    # Placeholder: In real implementation, would use a trained model
    # For demo, return random probabilities
    emotions = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']

    if model is not None:
        # Use provided model
        probabilities = model.predict(features.reshape(1, -1))[0]
    else:
        # Demo: generate normalized random probabilities
        probabilities = np.random.random(len(emotions))
        probabilities = probabilities / probabilities.sum()

    return {emotion: prob for emotion, prob in zip(emotions, probabilities)}


def compute_action_units(
    landmarks: np.ndarray
) -> Dict[str, float]:
    """
    Compute Facial Action Units (FACS) from facial landmarks.

    Action Units are atomic facial movements defined in the Facial Action Coding System.

    Args:
        landmarks: Facial landmarks, shape (68, 2) for 68-point model

    Returns:
        Dictionary of Action Unit activations
    """
    # Simplified AU computation based on geometric features
    # In practice, would use OpenFace or similar library

    aus = {}

    # AU1 (Inner Brow Raiser): Distance between inner brow points
    if landmarks.shape[0] >= 68:
        brow_inner_dist = np.linalg.norm(landmarks[21] - landmarks[22])
        aus['AU1'] = brow_inner_dist

        # AU2 (Outer Brow Raiser): Distance of outer brow from center
        brow_outer_dist_left = np.linalg.norm(landmarks[17] - landmarks[21])
        brow_outer_dist_right = np.linalg.norm(landmarks[26] - landmarks[22])
        aus['AU2'] = (brow_outer_dist_left + brow_outer_dist_right) / 2

        # AU4 (Brow Lowerer): Vertical distance between brow and eye
        brow_eye_dist = np.mean([
            np.linalg.norm(landmarks[19] - landmarks[37]),
            np.linalg.norm(landmarks[24] - landmarks[44])
        ])
        aus['AU4'] = brow_eye_dist

        # AU6 (Cheek Raiser): Eye opening
        eye_opening_left = np.linalg.norm(landmarks[37] - landmarks[41])
        eye_opening_right = np.linalg.norm(landmarks[43] - landmarks[47])
        aus['AU6'] = (eye_opening_left + eye_opening_right) / 2

        # AU12 (Lip Corner Puller): Mouth width
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
        aus['AU12'] = mouth_width

        # AU15 (Lip Corner Depressor): Similar to AU12 but inverse
        aus['AU15'] = 1.0 / (mouth_width + 1e-6)

        # AU25 (Lips Part): Vertical mouth opening
        mouth_opening = np.mean([
            np.linalg.norm(landmarks[51] - landmarks[57]),
            np.linalg.norm(landmarks[62] - landmarks[66])
        ])
        aus['AU25'] = mouth_opening

    return aus


def detect_facial_landmarks(
    image: np.ndarray,
    detector=None
) -> np.ndarray:
    """
    Detect facial landmarks in an image.

    Args:
        image: Face image, shape (height, width, 3) or (height, width)
        detector: Pre-trained landmark detector (optional)

    Returns:
        Landmarks array, shape (68, 2) for 68-point model
    """
    # Placeholder: In real implementation, would use dlib or MediaPipe
    # For demo, generate synthetic landmarks

    h, w = image.shape[:2]

    # Generate approximate facial landmarks (68-point model)
    # Coordinates normalized to image size
    landmarks = np.array([
        # Jaw (0-16)
        *[(w * 0.2 + i * w * 0.04, h * 0.8) for i in range(17)],
        # Left eyebrow (17-21)
        *[(w * 0.25 + i * w * 0.05, h * 0.35) for i in range(5)],
        # Right eyebrow (22-26)
        *[(w * 0.55 + i * w * 0.05, h * 0.35) for i in range(5)],
        # Nose (27-35)
        *[(w * 0.5, h * 0.4 + i * h * 0.05) for i in range(4)],
        *[(w * 0.4 + i * w * 0.05, h * 0.55) for i in range(5)],
        # Left eye (36-41)
        *[(w * 0.35 + i * w * 0.03, h * 0.4) for i in range(6)],
        # Right eye (42-47)
        *[(w * 0.6 + i * w * 0.03, h * 0.4) for i in range(6)],
        # Mouth outer (48-59)
        *[(w * 0.3 + i * w * 0.04, h * 0.7) for i in range(12)],
        # Mouth inner (60-67)
        *[(w * 0.35 + i * w * 0.04, h * 0.72) for i in range(8)],
    ])

    # Add some random variation
    landmarks += np.random.normal(0, min(h, w) * 0.01, landmarks.shape)

    return landmarks


def extract_temporal_facial_features(
    video_frames: List[np.ndarray],
    fps: float = 24.0
) -> Dict[str, np.ndarray]:
    """
    Extract temporal features from facial video sequence.

    Args:
        video_frames: List of face images
        fps: Frames per second

    Returns:
        Dictionary of temporal feature sequences
    """
    n_frames = len(video_frames)
    temporal_features = {
        'frame_features': [],
        'landmarks': [],
        'action_units': []
    }

    for frame in video_frames:
        # Extract features per frame
        features = extract_facial_features(frame, method='hog')
        temporal_features['frame_features'].append(features)

        # Detect landmarks
        landmarks = detect_facial_landmarks(frame)
        temporal_features['landmarks'].append(landmarks)

        # Compute action units
        aus = compute_action_units(landmarks)
        temporal_features['action_units'].append(list(aus.values()))

    # Convert to arrays
    temporal_features['frame_features'] = np.array(temporal_features['frame_features'])
    temporal_features['landmarks'] = np.array(temporal_features['landmarks'])
    temporal_features['action_units'] = np.array(temporal_features['action_units'])

    return temporal_features


def compute_facial_motion(
    landmarks_sequence: np.ndarray
) -> np.ndarray:
    """
    Compute motion features from landmark sequence.

    Args:
        landmarks_sequence: Landmark sequence, shape (n_frames, n_landmarks, 2)

    Returns:
        Motion features, shape (n_frames-1, n_landmarks, 2)
    """
    # Compute frame-to-frame differences
    motion = np.diff(landmarks_sequence, axis=0)

    return motion
