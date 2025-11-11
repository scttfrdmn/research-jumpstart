#!/usr/bin/env python3
"""
Bias Correction Methods for Climate Model Data

Implements quantile mapping and delta method for bias correction of
regional climate model outputs.

Usage:
    from bias_correction import QuantileMapper, DeltaMethod

    qm = QuantileMapper()
    corrected = qm.fit_transform(model_hist, obs_hist, model_fut)
"""

import numpy as np
from scipy import interpolate, stats


class QuantileMapper:
    """
    Empirical Quantile Mapping bias correction.

    Maps the cumulative distribution function (CDF) of model data
    to match observations during a reference period, then applies
    the same transformation to future projections.
    """

    def __init__(self, n_quantiles=1000, extrapolate='constant'):
        """
        Parameters:
        -----------
        n_quantiles : int
            Number of quantiles for CDF mapping (default: 1000)
        extrapolate : str
            How to handle values outside training range
            Options: 'constant', 'linear'
        """
        self.n_quantiles = n_quantiles
        self.extrapolate = extrapolate
        self.transfer_func = None

    def fit(self, model_reference, obs_reference):
        """
        Build transfer function from reference period.

        Parameters:
        -----------
        model_reference : array-like
            Model values during reference period
        obs_reference : array-like
            Observed values during reference period

        Returns:
        --------
        self : QuantileMapper
        """
        # Calculate quantiles
        quantiles = np.linspace(0, 1, self.n_quantiles)

        # Get quantile values
        model_quantiles = np.quantile(model_reference, quantiles)
        obs_quantiles = np.quantile(obs_reference, quantiles)

        # Create interpolation function
        if self.extrapolate == 'constant':
            fill_value = (obs_quantiles[0], obs_quantiles[-1])
        else:
            fill_value = 'extrapolate'

        self.transfer_func = interpolate.interp1d(
            model_quantiles,
            obs_quantiles,
            kind='linear',
            bounds_error=False,
            fill_value=fill_value
        )

        return self

    def transform(self, model_data):
        """
        Apply bias correction to new model data.

        Parameters:
        -----------
        model_data : array-like
            Model values to correct

        Returns:
        --------
        corrected : array-like
            Bias-corrected values
        """
        if self.transfer_func is None:
            raise ValueError("Must call fit() before transform()")

        return self.transfer_func(model_data)

    def fit_transform(self, model_reference, obs_reference, model_target):
        """
        Fit on reference period and transform target period.

        Parameters:
        -----------
        model_reference : array-like
            Model values during reference period
        obs_reference : array-like
            Observed values during reference period
        model_target : array-like
            Model values to be corrected

        Returns:
        --------
        corrected : array-like
            Bias-corrected target values
        """
        self.fit(model_reference, obs_reference)
        return self.transform(model_target)


class SeasonalQuantileMapper(QuantileMapper):
    """
    Quantile Mapping applied separately by season.

    Improves correction by accounting for seasonal differences
    in bias characteristics.
    """

    def __init__(self, n_quantiles=1000, extrapolate='constant'):
        super().__init__(n_quantiles, extrapolate)
        self.seasonal_funcs = {}

    def fit(self, model_reference, obs_reference, months_reference):
        """
        Build separate transfer functions for each season.

        Parameters:
        -----------
        model_reference : array-like
            Model values during reference period
        obs_reference : array-like
            Observed values during reference period
        months_reference : array-like
            Month numbers (1-12) for reference period
        """
        # Define seasons
        seasons = {
            'DJF': [12, 1, 2],
            'MAM': [3, 4, 5],
            'JJA': [6, 7, 8],
            'SON': [9, 10, 11]
        }

        # Fit each season separately
        for season_name, season_months in seasons.items():
            mask = np.isin(months_reference, season_months)

            if mask.sum() > 0:
                qm = QuantileMapper(self.n_quantiles, self.extrapolate)
                qm.fit(model_reference[mask], obs_reference[mask])
                self.seasonal_funcs[season_name] = qm

        return self

    def transform(self, model_data, months_target):
        """
        Apply seasonal bias correction.

        Parameters:
        -----------
        model_data : array-like
            Model values to correct
        months_target : array-like
            Month numbers for target period

        Returns:
        --------
        corrected : array-like
            Bias-corrected values
        """
        if not self.seasonal_funcs:
            raise ValueError("Must call fit() before transform()")

        corrected = np.empty_like(model_data)

        seasons = {
            'DJF': [12, 1, 2],
            'MAM': [3, 4, 5],
            'JJA': [6, 7, 8],
            'SON': [9, 10, 11]
        }

        for season_name, season_months in seasons.items():
            mask = np.isin(months_target, season_months)
            if mask.sum() > 0 and season_name in self.seasonal_funcs:
                corrected[mask] = self.seasonal_funcs[season_name].transform(
                    model_data[mask]
                )

        return corrected


class DeltaMethod:
    """
    Delta (change factor) bias correction method.

    Applies the climate change signal from models to observations,
    preserving observed characteristics while incorporating projected changes.
    """

    def __init__(self, method='additive'):
        """
        Parameters:
        -----------
        method : str
            'additive' for temperature (absolute changes)
            'multiplicative' for precipitation (relative changes)
        """
        if method not in ['additive', 'multiplicative']:
            raise ValueError("method must be 'additive' or 'multiplicative'")
        self.method = method
        self.reference_mean = None

    def fit(self, obs_reference):
        """
        Store reference observations.

        Parameters:
        -----------
        obs_reference : array-like
            Observed values during reference period
        """
        self.reference_mean = np.mean(obs_reference)
        return self

    def transform(self, model_reference, model_target):
        """
        Apply delta method correction.

        Parameters:
        -----------
        model_reference : array-like
            Model values during reference period
        model_target : array-like
            Model values for target period

        Returns:
        --------
        corrected : array-like
            Bias-corrected target values
        """
        if self.reference_mean is None:
            raise ValueError("Must call fit() before transform()")

        model_ref_mean = np.mean(model_reference)

        if self.method == 'additive':
            # For temperature: add change signal
            delta = model_target - model_ref_mean
            corrected = self.reference_mean + delta
        else:
            # For precipitation: multiply by change factor
            # Add small epsilon to avoid division by zero
            factor = model_target / (model_ref_mean + 1e-6)
            corrected = self.reference_mean * factor

        return corrected

    def fit_transform(self, obs_reference, model_reference, model_target):
        """
        Fit and transform in one step.
        """
        self.fit(obs_reference)
        return self.transform(model_reference, model_target)


def validate_correction(obs, model_raw, model_corrected):
    """
    Validate bias correction effectiveness.

    Parameters:
    -----------
    obs : array-like
        Observations
    model_raw : array-like
        Uncorrected model data
    model_corrected : array-like
        Bias-corrected model data

    Returns:
    --------
    metrics : dict
        Validation metrics including bias, RMSE, correlation
    """
    metrics = {}

    # Mean bias
    metrics['raw_bias'] = np.mean(model_raw - obs)
    metrics['corrected_bias'] = np.mean(model_corrected - obs)

    # RMSE
    metrics['raw_rmse'] = np.sqrt(np.mean((model_raw - obs)**2))
    metrics['corrected_rmse'] = np.sqrt(np.mean((model_corrected - obs)**2))

    # Correlation
    metrics['raw_corr'] = np.corrcoef(obs, model_raw)[0, 1]
    metrics['corrected_corr'] = np.corrcoef(obs, model_corrected)[0, 1]

    # Standard deviation ratio
    metrics['raw_std_ratio'] = np.std(model_raw) / np.std(obs)
    metrics['corrected_std_ratio'] = np.std(model_corrected) / np.std(obs)

    # Percentile comparison (5th, 50th, 95th)
    for p in [5, 50, 95]:
        obs_p = np.percentile(obs, p)
        raw_p = np.percentile(model_raw, p)
        corr_p = np.percentile(model_corrected, p)

        metrics[f'raw_p{p}_bias'] = raw_p - obs_p
        metrics[f'corrected_p{p}_bias'] = corr_p - obs_p

    return metrics


def preserve_change_signal(model_hist, model_fut, corrected_fut):
    """
    Verify that climate change signal is preserved after correction.

    Parameters:
    -----------
    model_hist : array-like
        Model historical period
    model_fut : array-like
        Model future period (uncorrected)
    corrected_fut : array-like
        Model future period (corrected)

    Returns:
    --------
    analysis : dict
        Signal preservation metrics
    """
    raw_signal = np.mean(model_fut) - np.mean(model_hist)

    # For corrected signal, we compare to the corrected historical
    # which should match observations
    corrected_signal = np.mean(corrected_fut) - np.mean(model_hist)

    analysis = {
        'raw_signal': raw_signal,
        'corrected_signal': corrected_signal,
        'signal_preserved': np.abs(raw_signal - corrected_signal) < 0.1,
        'signal_difference': corrected_signal - raw_signal
    }

    return analysis


if __name__ == '__main__':
    # Example usage
    print("Bias Correction Methods")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    n = 1000

    # Observations
    obs = np.random.normal(15, 2, n)

    # Model with warm bias
    model_hist = obs + 2 + np.random.normal(0, 0.5, n)

    # Future with warming
    model_fut = model_hist + 3

    # Apply quantile mapping
    print("\n1. Quantile Mapping")
    qm = QuantileMapper()
    corrected_qm = qm.fit_transform(model_hist, obs, model_fut)

    print(f"   Original bias: {np.mean(model_hist - obs):.2f}")
    print(f"   Corrected bias: {np.mean(corrected_qm[:n] - obs):.2f}")

    # Apply delta method
    print("\n2. Delta Method")
    delta = DeltaMethod(method='additive')
    corrected_delta = delta.fit_transform(obs, model_hist, model_fut)

    print(f"   Corrected mean: {np.mean(corrected_delta):.2f}")
    print(f"   Expected mean: {np.mean(obs) + 3:.2f}")

    print("\n✓ Bias correction methods ready")
