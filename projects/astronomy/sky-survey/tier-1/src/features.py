"""
Feature engineering for astronomical object classification.
"""

import numpy as np
import pandas as pd


def calculate_colors(catalog):
    """
    Calculate optical and infrared colors for classification.

    Parameters
    ----------
    catalog : astropy.table.Table or pandas.DataFrame
        Catalog with photometry from multiple surveys

    Returns
    -------
    pandas.DataFrame
        Feature dataframe with colors
    """
    features = {}

    # SDSS optical colors (if available)
    if 'u' in catalog.colnames and 'g' in catalog.colnames:
        features['u_g'] = catalog['u'] - catalog['g']
        features['g_r'] = catalog['g'] - catalog['r']
        features['r_i'] = catalog['r'] - catalog['i']
        features['i_z'] = catalog['i'] - catalog['z']

        # Wide colors
        features['u_r'] = catalog['u'] - catalog['r']
        features['g_i'] = catalog['g'] - catalog['i']

    # 2MASS near-IR colors (if available)
    if 'j_m' in catalog.colnames:
        features['J_H'] = catalog['j_m'] - catalog['h_m']
        features['H_K'] = catalog['h_m'] - catalog['k_m']
        features['J_K'] = catalog['j_m'] - catalog['k_m']

    # WISE mid-IR colors (if available)
    if 'w1mpro' in catalog.colnames:
        features['W1_W2'] = catalog['w1mpro'] - catalog['w2mpro']
        features['W2_W3'] = catalog['w2mpro'] - catalog['w3mpro']

    # Optical-IR colors (if available)
    if 'r' in catalog.colnames and 'w1mpro' in catalog.colnames:
        features['r_W1'] = catalog['r'] - catalog['w1mpro']

    if 'i' in catalog.colnames and 'j_m' in catalog.colnames:
        features['i_J'] = catalog['i'] - catalog['j_m']

    return pd.DataFrame(features)


def calculate_proper_motions(catalog):
    """
    Calculate proper motion features from Gaia.

    Parameters
    ----------
    catalog : astropy.table.Table or pandas.DataFrame
        Catalog with Gaia astrometry

    Returns
    -------
    pandas.DataFrame
        Feature dataframe with proper motions
    """
    features = {}

    if 'pmra' in catalog.colnames and 'pmdec' in catalog.colnames:
        # Total proper motion
        features['pm_total'] = np.sqrt(
            catalog['pmra']**2 + catalog['pmdec']**2
        )

        # Proper motion direction
        features['pm_angle'] = np.arctan2(
            catalog['pmdec'], catalog['pmra']
        )

        # Parallax (if available)
        if 'parallax' in catalog.colnames:
            features['parallax'] = catalog['parallax']

            # Distance estimate (1/parallax in mas -> kpc)
            # Only for positive, significant parallaxes
            parallax_err = catalog.get('parallax_error', 0)
            good_plx = (catalog['parallax'] > 3*parallax_err) & (catalog['parallax'] > 0)
            features['distance_kpc'] = np.where(
                good_plx,
                1.0 / catalog['parallax'],  # mas -> kpc
                np.nan
            )

            # Reduced proper motion (proxy for absolute magnitude)
            if 'phot_g_mean_mag' in catalog.colnames:
                features['reduced_pm'] = (
                    catalog['phot_g_mean_mag'] +
                    5*np.log10(features['pm_total']) + 5
                )

    return pd.DataFrame(features)


def extract_morphology(catalog):
    """
    Extract morphological features from SDSS.

    Parameters
    ----------
    catalog : astropy.table.Table or pandas.DataFrame
        Catalog with SDSS morphology

    Returns
    -------
    pandas.DataFrame
        Feature dataframe with morphology
    """
    features = {}

    # SDSS type (3=galaxy, 6=star)
    if 'type' in catalog.colnames:
        features['sdss_type'] = catalog['type']

    # Petrosian radii (size indicators)
    if 'petroR50_r' in catalog.colnames:
        features['petroR50_r'] = catalog['petroR50_r']

    if 'petroR90_r' in catalog.colnames:
        features['petroR90_r'] = catalog['petroR90_r']

        # Concentration index
        if 'petroR50_r' in catalog.colnames:
            features['concentration'] = (
                catalog['petroR90_r'] / catalog['petroR50_r']
            )

    return pd.DataFrame(features)


def build_feature_matrix(catalog):
    """
    Build complete feature matrix for classification.

    Parameters
    ----------
    catalog : astropy.table.Table or pandas.DataFrame
        Multi-survey matched catalog

    Returns
    -------
    pandas.DataFrame
        Complete feature matrix for ML
    """
    print("Building feature matrix...")

    # Calculate all feature types
    colors = calculate_colors(catalog)
    proper_motions = calculate_proper_motions(catalog)
    morphology = extract_morphology(catalog)

    # Combine all features
    features = pd.concat([colors, proper_motions, morphology], axis=1)

    # Add basic photometry as features
    phot_cols = ['g', 'r', 'i', 'phot_g_mean_mag', 'j_m', 'w1mpro']
    for col in phot_cols:
        if col in catalog.colnames:
            features[col] = catalog[col]

    print(f"  Total features: {len(features.columns)}")
    print(f"  Total objects: {len(features)}")

    # Print feature completeness
    print("\nFeature completeness:")
    for col in features.columns:
        n_valid = (~features[col].isna()).sum()
        pct = 100 * n_valid / len(features)
        print(f"  {col:20s}: {n_valid:6d} / {len(features):6d} ({pct:5.1f}%)")

    return features


def normalize_features(features, method='standard'):
    """
    Normalize features for machine learning.

    Parameters
    ----------
    features : pandas.DataFrame
        Feature matrix
    method : str
        'standard' (z-score) or 'minmax' (0-1 scaling)

    Returns
    -------
    pandas.DataFrame
        Normalized features
    sklearn scaler
        Fitted scaler object
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Fit and transform
    normalized = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index
    )

    return normalized, scaler


def handle_missing_values(features, strategy='median'):
    """
    Handle missing values in feature matrix.

    Parameters
    ----------
    features : pandas.DataFrame
        Feature matrix with missing values
    strategy : str
        'median', 'mean', or 'zero'

    Returns
    -------
    pandas.DataFrame
        Features with imputed values
    sklearn imputer
        Fitted imputer object
    """
    from sklearn.impute import SimpleImputer

    if strategy in ['median', 'mean']:
        imputer = SimpleImputer(strategy=strategy)
    elif strategy == 'zero':
        imputer = SimpleImputer(strategy='constant', fill_value=0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Fit and transform
    imputed = pd.DataFrame(
        imputer.fit_transform(features),
        columns=features.columns,
        index=features.index
    )

    print(f"Imputed missing values using {strategy} strategy")

    return imputed, imputer
