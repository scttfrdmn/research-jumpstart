"""
Data loading and caching utilities for dialect corpus management.

This module provides functions to download, cache, and load dialect data
from various sources. Data is stored in persistent SageMaker Studio Lab
storage for instant access in future sessions.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests


def get_data_path(create_if_missing: bool = True) -> Path:
    """
    Get path to persistent data directory.

    Args:
        create_if_missing: Create directory if it doesn't exist

    Returns:
        Path to data directory
    """
    data_dir = Path(__file__).parent.parent / 'data'
    if create_if_missing:
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / 'raw').mkdir(exist_ok=True)
        (data_dir / 'processed').mkdir(exist_ok=True)
    return data_dir


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """
    Download file with progress bar.

    Args:
        url: URL to download from
        destination: Local file path to save to
        chunk_size: Download chunk size in bytes
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def download_corpus(
    language: str,
    force_download: bool = False
) -> Path:
    """
    Download dialect corpus for specified language.

    Args:
        language: Language code ('english', 'spanish', 'mandarin', etc.)
        force_download: Re-download even if cached

    Returns:
        Path to downloaded corpus directory
    """
    data_dir = get_data_path()
    corpus_dir = data_dir / 'raw' / f'{language}_dialects'

    if corpus_dir.exists() and not force_download:
        print(f"✓ {language.capitalize()} corpus already cached")
        return corpus_dir

    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder for actual download logic
    # In real implementation, would download from linguistic archives
    print(f"Downloading {language.capitalize()} dialect corpus...")
    print(f"  → Saving to {corpus_dir}")

    return corpus_dir


def load_dialect_corpus(
    language: str,
    split: Optional[str] = None,
    force_download: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Load dialect corpus for specified language.

    Args:
        language: Language code ('english', 'spanish', 'mandarin', etc.)
        split: Data split ('train', 'val', 'test') or None for all
        force_download: Re-download corpus if True

    Returns:
        Dictionary with 'audio_paths', 'transcriptions', 'dialects'
    """
    corpus_dir = download_corpus(language, force_download)

    # Load cached processed data if available
    data_dir = get_data_path()
    processed_file = data_dir / 'processed' / f'{language}_processed.pkl'

    if processed_file.exists() and not force_download:
        print(f"✓ Loading cached {language} data")
        return pd.read_pickle(processed_file)

    # Placeholder for actual data loading
    # In real implementation, would parse audio files and annotations
    print(f"Processing {language} corpus...")

    data = {
        'audio_paths': [],
        'transcriptions': [],
        'dialects': [],
        'language': language
    }

    # Save processed data for future use
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(data, processed_file)

    return data


def load_multiple_languages(
    languages: List[str],
    force_download: bool = False
) -> Dict[str, Dict]:
    """
    Load dialect corpora for multiple languages.

    Args:
        languages: List of language codes
        force_download: Re-download all corpora

    Returns:
        Dictionary mapping language -> corpus data
    """
    corpora = {}
    for lang in tqdm(languages, desc="Loading languages"):
        corpora[lang] = load_dialect_corpus(lang, force_download=force_download)
    return corpora


def get_dialect_statistics(corpus_data: Dict) -> pd.DataFrame:
    """
    Calculate statistics about dialect distribution in corpus.

    Args:
        corpus_data: Corpus data dictionary

    Returns:
        DataFrame with dialect statistics
    """
    if not corpus_data.get('dialects'):
        return pd.DataFrame()

    df = pd.DataFrame({'dialect': corpus_data['dialects']})
    stats = df['dialect'].value_counts().reset_index()
    stats.columns = ['dialect', 'count']
    stats['percentage'] = 100 * stats['count'] / stats['count'].sum()

    return stats


def create_train_val_test_split(
    corpus_data: Dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[Dict, Dict, Dict]:
    """
    Split corpus into train/validation/test sets.

    Args:
        corpus_data: Full corpus data
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    np.random.seed(random_seed)

    n = len(corpus_data['dialects'])
    indices = np.random.permutation(n)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def split_data(idx):
        return {
            'audio_paths': [corpus_data['audio_paths'][i] for i in idx],
            'transcriptions': [corpus_data['transcriptions'][i] for i in idx],
            'dialects': [corpus_data['dialects'][i] for i in idx],
        }

    return split_data(train_idx), split_data(val_idx), split_data(test_idx)
