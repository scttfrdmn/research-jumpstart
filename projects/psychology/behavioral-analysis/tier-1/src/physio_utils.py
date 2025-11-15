"""
Physiological signal processing utilities for emotion recognition.

This module provides functions to process ECG, GSR, respiration, and temperature signals.
"""


import numpy as np
from scipy import signal


def preprocess_physiological(
    signal_data: np.ndarray, sampling_rate: int = 256, signal_type: str = "ecg"
) -> np.ndarray:
    """
    Preprocess physiological signals with appropriate filtering.

    Args:
        signal_data: Raw physiological signal, shape (n_samples,)
        sampling_rate: Sampling frequency in Hz
        signal_type: Type of signal ('ecg', 'gsr', 'respiration', 'temperature')

    Returns:
        Preprocessed signal
    """
    if signal_type == "ecg":
        # Bandpass filter for ECG (0.5-40 Hz)
        nyquist = sampling_rate / 2
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        b, a = signal.butter(4, [low, high], btype="band")
        filtered = signal.filtfilt(b, a, signal_data)

    elif signal_type == "gsr":
        # Lowpass filter for GSR (5 Hz)
        nyquist = sampling_rate / 2
        cutoff = 5.0 / nyquist
        b, a = signal.butter(4, cutoff, btype="low")
        filtered = signal.filtfilt(b, a, signal_data)

    elif signal_type == "respiration":
        # Bandpass filter for respiration (0.1-0.5 Hz)
        nyquist = sampling_rate / 2
        low = 0.1 / nyquist
        high = 0.5 / nyquist
        b, a = signal.butter(4, [low, high], btype="band")
        filtered = signal.filtfilt(b, a, signal_data)

    elif signal_type == "temperature":
        # Lowpass filter for temperature (0.1 Hz)
        nyquist = sampling_rate / 2
        cutoff = 0.1 / nyquist
        b, a = signal.butter(4, cutoff, btype="low")
        filtered = signal.filtfilt(b, a, signal_data)

    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

    return filtered


def compute_hrv(ecg_signal: np.ndarray, sampling_rate: int = 256) -> dict[str, float]:
    """
    Compute Heart Rate Variability (HRV) features from ECG signal.

    Args:
        ecg_signal: Preprocessed ECG signal, shape (n_samples,)
        sampling_rate: Sampling frequency in Hz

    Returns:
        Dictionary of HRV features
    """
    # Detect R-peaks (simplified peak detection)
    peaks, _ = signal.find_peaks(ecg_signal, distance=sampling_rate * 0.5, prominence=0.5)

    if len(peaks) < 2:
        return {"mean_hr": 0, "std_hr": 0, "rmssd": 0, "sdnn": 0, "pnn50": 0}

    # Compute RR intervals (in milliseconds)
    rr_intervals = np.diff(peaks) / sampling_rate * 1000

    if len(rr_intervals) < 2:
        return {"mean_hr": 0, "std_hr": 0, "rmssd": 0, "sdnn": 0, "pnn50": 0}

    # Time-domain HRV features
    hrv_features = {}

    # Mean heart rate
    hrv_features["mean_hr"] = 60000 / np.mean(rr_intervals)

    # Standard deviation of heart rate
    hrv_features["std_hr"] = np.std(60000 / rr_intervals)

    # RMSSD: Root mean square of successive differences
    successive_diffs = np.diff(rr_intervals)
    hrv_features["rmssd"] = np.sqrt(np.mean(successive_diffs**2))

    # SDNN: Standard deviation of NN intervals
    hrv_features["sdnn"] = np.std(rr_intervals)

    # pNN50: Percentage of successive RR intervals differing by more than 50 ms
    hrv_features["pnn50"] = np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100

    # Frequency-domain features (simplified)
    # Compute power spectral density
    freqs, psd = signal.welch(
        rr_intervals, fs=1000 / np.mean(rr_intervals), nperseg=min(256, len(rr_intervals))
    )

    # VLF (0.003-0.04 Hz)
    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    hrv_features["vlf_power"] = np.trapz(psd[vlf_mask], freqs[vlf_mask])

    # LF (0.04-0.15 Hz)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hrv_features["lf_power"] = np.trapz(psd[lf_mask], freqs[lf_mask])

    # HF (0.15-0.4 Hz)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)
    hrv_features["hf_power"] = np.trapz(psd[hf_mask], freqs[hf_mask])

    # LF/HF ratio
    hrv_features["lf_hf_ratio"] = hrv_features["lf_power"] / (hrv_features["hf_power"] + 1e-10)

    return hrv_features


def extract_scr_features(gsr_signal: np.ndarray, sampling_rate: int = 32) -> dict[str, float]:
    """
    Extract Skin Conductance Response (SCR) features from GSR signal.

    Args:
        gsr_signal: Preprocessed GSR signal, shape (n_samples,)
        sampling_rate: Sampling frequency in Hz

    Returns:
        Dictionary of SCR features
    """
    features = {}

    # Skin Conductance Level (SCL): tonic component (low-pass filtered)
    nyquist = sampling_rate / 2
    cutoff = 0.05 / nyquist
    b, a = signal.butter(4, cutoff, btype="low")
    scl = signal.filtfilt(b, a, gsr_signal)

    # Skin Conductance Response (SCR): phasic component
    scr = gsr_signal - scl

    # Statistical features
    features["scl_mean"] = np.mean(scl)
    features["scl_std"] = np.std(scl)
    features["scr_mean"] = np.mean(np.abs(scr))
    features["scr_std"] = np.std(scr)

    # Peak detection in SCR
    peaks, peak_properties = signal.find_peaks(scr, height=0.01, distance=sampling_rate)

    features["scr_num_peaks"] = len(peaks)
    if len(peaks) > 0:
        features["scr_peak_amplitude"] = np.mean(peak_properties["peak_heights"])
        features["scr_peak_rate"] = len(peaks) / (
            len(gsr_signal) / sampling_rate
        )  # peaks per second
    else:
        features["scr_peak_amplitude"] = 0
        features["scr_peak_rate"] = 0

    # Derivative features
    gsr_derivative = np.diff(gsr_signal)
    features["gsr_derivative_mean"] = np.mean(gsr_derivative)
    features["gsr_derivative_std"] = np.std(gsr_derivative)

    return features


def extract_respiratory_features(
    resp_signal: np.ndarray, sampling_rate: int = 32
) -> dict[str, float]:
    """
    Extract respiratory features from respiration signal.

    Args:
        resp_signal: Preprocessed respiration signal, shape (n_samples,)
        sampling_rate: Sampling frequency in Hz

    Returns:
        Dictionary of respiratory features
    """
    features = {}

    # Detect breathing cycles (peaks = inhalations)
    peaks, _ = signal.find_peaks(resp_signal, distance=sampling_rate * 1.5)

    if len(peaks) > 1:
        # Respiratory rate
        breath_intervals = np.diff(peaks) / sampling_rate
        features["resp_rate"] = 60 / np.mean(breath_intervals)  # breaths per minute
        features["resp_rate_std"] = np.std(60 / breath_intervals)

        # Respiratory amplitude
        troughs, _ = signal.find_peaks(-resp_signal, distance=sampling_rate * 1.5)
        if len(troughs) > 0 and len(peaks) > 0:
            min_len = min(len(peaks), len(troughs))
            amplitudes = resp_signal[peaks[:min_len]] - resp_signal[troughs[:min_len]]
            features["resp_amplitude_mean"] = np.mean(amplitudes)
            features["resp_amplitude_std"] = np.std(amplitudes)
        else:
            features["resp_amplitude_mean"] = 0
            features["resp_amplitude_std"] = 0

    else:
        features["resp_rate"] = 0
        features["resp_rate_std"] = 0
        features["resp_amplitude_mean"] = 0
        features["resp_amplitude_std"] = 0

    # Statistical features
    features["resp_signal_mean"] = np.mean(resp_signal)
    features["resp_signal_std"] = np.std(resp_signal)

    return features


def extract_temperature_features(temp_signal: np.ndarray) -> dict[str, float]:
    """
    Extract features from skin temperature signal.

    Args:
        temp_signal: Preprocessed temperature signal, shape (n_samples,)

    Returns:
        Dictionary of temperature features
    """
    features = {}

    # Statistical features
    features["temp_mean"] = np.mean(temp_signal)
    features["temp_std"] = np.std(temp_signal)
    features["temp_min"] = np.min(temp_signal)
    features["temp_max"] = np.max(temp_signal)
    features["temp_range"] = features["temp_max"] - features["temp_min"]

    # Trend
    time = np.arange(len(temp_signal))
    slope, _ = np.polyfit(time, temp_signal, 1)
    features["temp_slope"] = slope

    return features


def extract_all_physiological_features(
    ecg_signal: np.ndarray,
    gsr_signal: np.ndarray,
    resp_signal: np.ndarray,
    temp_signal: np.ndarray,
    ecg_sr: int = 256,
    gsr_sr: int = 32,
    resp_sr: int = 32,
    temp_sr: int = 32,
) -> dict[str, float]:
    """
    Extract comprehensive feature set from all physiological signals.

    Args:
        ecg_signal: ECG signal
        gsr_signal: GSR signal
        resp_signal: Respiration signal
        temp_signal: Temperature signal
        ecg_sr: ECG sampling rate
        gsr_sr: GSR sampling rate
        resp_sr: Respiration sampling rate
        temp_sr: Temperature sampling rate

    Returns:
        Dictionary of all physiological features
    """
    features = {}

    # Preprocess signals
    ecg_clean = preprocess_physiological(ecg_signal, ecg_sr, "ecg")
    gsr_clean = preprocess_physiological(gsr_signal, gsr_sr, "gsr")
    resp_clean = preprocess_physiological(resp_signal, resp_sr, "respiration")
    temp_clean = preprocess_physiological(temp_signal, temp_sr, "temperature")

    # Extract features from each modality
    hrv_features = compute_hrv(ecg_clean, ecg_sr)
    scr_features = extract_scr_features(gsr_clean, gsr_sr)
    resp_features = extract_respiratory_features(resp_clean, resp_sr)
    temp_features = extract_temperature_features(temp_clean)

    # Combine all features
    features.update(hrv_features)
    features.update(scr_features)
    features.update(resp_features)
    features.update(temp_features)

    return features


def create_physiological_feature_vector(features_dict: dict[str, float]) -> np.ndarray:
    """
    Convert feature dictionary to numpy array.

    Args:
        features_dict: Dictionary of features

    Returns:
        Feature vector as numpy array
    """
    return np.array(list(features_dict.values()))
