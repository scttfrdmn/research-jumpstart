"""
EEG signal processing utilities for emotion recognition.

This module provides functions to preprocess EEG data, extract spectral features,
and compute connectivity measures.
"""


import numpy as np
from scipy import signal


def preprocess_eeg(
    eeg_data: np.ndarray,
    sampling_rate: int = 256,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    remove_baseline: bool = True,
) -> np.ndarray:
    """
    Preprocess EEG data with filtering and baseline removal.

    Args:
        eeg_data: EEG signals, shape (n_channels, n_samples)
        sampling_rate: Sampling frequency in Hz
        lowcut: Low cutoff frequency for bandpass filter
        highcut: High cutoff frequency for bandpass filter
        remove_baseline: Whether to remove baseline (mean)

    Returns:
        Preprocessed EEG data, same shape as input
    """
    # Bandpass filter
    nyquist = sampling_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = signal.butter(4, [low, high], btype="band")
    filtered = signal.filtfilt(b, a, eeg_data, axis=1)

    # Remove baseline
    if remove_baseline:
        filtered = filtered - np.mean(filtered, axis=1, keepdims=True)

    return filtered


def extract_spectral_features(
    eeg_data: np.ndarray, sampling_rate: int = 256, window_size: float = 2.0
) -> dict[str, np.ndarray]:
    """
    Extract spectral power features from EEG data.

    Computes average power in standard frequency bands:
    - Delta (0.5-4 Hz)
    - Theta (4-8 Hz)
    - Alpha (8-13 Hz)
    - Beta (13-30 Hz)
    - Gamma (30-45 Hz)

    Args:
        eeg_data: EEG signals, shape (n_channels, n_samples)
        sampling_rate: Sampling frequency in Hz
        window_size: Window size for spectral estimation in seconds

    Returns:
        Dictionary with band powers for each channel
    """
    _n_channels, _n_samples = eeg_data.shape

    # Define frequency bands
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    # Compute power spectral density
    nperseg = int(window_size * sampling_rate)
    freqs, psd = signal.welch(eeg_data, fs=sampling_rate, nperseg=nperseg, axis=1)

    # Extract band powers
    features = {}
    for band_name, (low_freq, high_freq) in bands.items():
        freq_mask = (freqs >= low_freq) & (freqs < high_freq)
        band_power = np.mean(psd[:, freq_mask], axis=1)
        features[band_name] = band_power

    # Add relative powers (normalized by total power)
    total_power = np.sum([features[band] for band in bands], axis=0)
    for band_name in bands:
        features[f"{band_name}_relative"] = features[band_name] / (total_power + 1e-10)

    return features


def compute_eeg_connectivity(
    eeg_data: np.ndarray, method: str = "coherence", sampling_rate: int = 256
) -> np.ndarray:
    """
    Compute functional connectivity between EEG channels.

    Args:
        eeg_data: EEG signals, shape (n_channels, n_samples)
        method: Connectivity method ('coherence', 'correlation', 'plv')
        sampling_rate: Sampling frequency in Hz

    Returns:
        Connectivity matrix, shape (n_channels, n_channels)
    """
    n_channels = eeg_data.shape[0]
    connectivity = np.zeros((n_channels, n_channels))

    if method == "correlation":
        # Simple Pearson correlation
        connectivity = np.corrcoef(eeg_data)

    elif method == "coherence":
        # Magnitude squared coherence
        for i in range(n_channels):
            for j in range(i, n_channels):
                freqs, Cxy = signal.coherence(
                    eeg_data[i], eeg_data[j], fs=sampling_rate, nperseg=256
                )
                # Average coherence across alpha band (8-13 Hz)
                alpha_mask = (freqs >= 8) & (freqs <= 13)
                connectivity[i, j] = np.mean(Cxy[alpha_mask])
                connectivity[j, i] = connectivity[i, j]

    elif method == "plv":
        # Phase Locking Value (simplified)
        # Extract instantaneous phase using Hilbert transform
        analytic_signal = signal.hilbert(eeg_data, axis=1)
        phase = np.angle(analytic_signal)

        for i in range(n_channels):
            for j in range(i, n_channels):
                phase_diff = phase[i] - phase[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                connectivity[i, j] = plv
                connectivity[j, i] = plv

    else:
        raise ValueError(f"Unknown connectivity method: {method}")

    return connectivity


def extract_eeg_features_vector(eeg_data: np.ndarray, sampling_rate: int = 256) -> np.ndarray:
    """
    Extract comprehensive feature vector from EEG data.

    Combines spectral features and connectivity measures into a single vector.

    Args:
        eeg_data: EEG signals, shape (n_channels, n_samples)
        sampling_rate: Sampling frequency in Hz

    Returns:
        Feature vector (1D array)
    """
    features = []

    # Spectral features
    spectral = extract_spectral_features(eeg_data, sampling_rate)
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
        features.append(spectral[band])
        features.append(spectral[f"{band}_relative"])

    # Connectivity features (upper triangle)
    connectivity = compute_eeg_connectivity(
        eeg_data, method="coherence", sampling_rate=sampling_rate
    )
    triu_indices = np.triu_indices_from(connectivity, k=1)
    features.append(connectivity[triu_indices])

    # Statistical features per channel
    features.append(np.mean(eeg_data, axis=1))
    features.append(np.std(eeg_data, axis=1))
    features.append(np.max(eeg_data, axis=1) - np.min(eeg_data, axis=1))  # Range

    # Flatten and concatenate
    feature_vector = np.concatenate([f.flatten() for f in features])

    return feature_vector


def detect_artifacts(eeg_data: np.ndarray, threshold_std: float = 3.0) -> np.ndarray:
    """
    Detect artifacts in EEG data using simple threshold method.

    Args:
        eeg_data: EEG signals, shape (n_channels, n_samples)
        threshold_std: Threshold in standard deviations

    Returns:
        Boolean mask indicating artifact samples, shape (n_samples,)
    """
    # Compute z-scores
    mean = np.mean(eeg_data, axis=1, keepdims=True)
    std = np.std(eeg_data, axis=1, keepdims=True)
    z_scores = np.abs((eeg_data - mean) / (std + 1e-10))

    # Mark samples exceeding threshold in any channel
    artifacts = np.any(z_scores > threshold_std, axis=0)

    return artifacts


def segment_eeg(
    eeg_data: np.ndarray,
    sampling_rate: int = 256,
    segment_duration: float = 2.0,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Segment continuous EEG data into overlapping windows.

    Args:
        eeg_data: EEG signals, shape (n_channels, n_samples)
        sampling_rate: Sampling frequency in Hz
        segment_duration: Duration of each segment in seconds
        overlap: Overlap fraction (0-1)

    Returns:
        Segmented data, shape (n_segments, n_channels, segment_samples)
    """
    _n_channels, n_samples = eeg_data.shape
    segment_samples = int(segment_duration * sampling_rate)
    step_samples = int(segment_samples * (1 - overlap))

    segments = []
    for start in range(0, n_samples - segment_samples + 1, step_samples):
        segment = eeg_data[:, start : start + segment_samples]
        segments.append(segment)

    return np.array(segments)
