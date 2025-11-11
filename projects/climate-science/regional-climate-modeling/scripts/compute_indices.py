#!/usr/bin/env python3
"""
Climate Extreme Indices Calculator

Implements ETCCDI (Expert Team on Climate Change Detection and Indices)
standard climate extreme indices for temperature and precipitation.

Usage:
    from compute_indices import TemperatureIndices, PrecipitationIndices

    temp_idx = TemperatureIndices()
    tx90p = temp_idx.tx90p(daily_max_temp, reference_period)
"""

import numpy as np
import pandas as pd
import xarray as xr


class TemperatureIndices:
    """
    Calculate temperature-based climate extreme indices.
    """

    @staticmethod
    def tx90p(tasmax, reference_period=None, threshold_percentile=90):
        """
        TX90p: Percentage of days with Tmax > 90th percentile.

        Warm days - indicates heat extremes.

        Parameters:
        -----------
        tasmax : array-like or xr.DataArray
            Daily maximum temperature
        reference_period : tuple or None
            (start_year, end_year) for baseline, e.g., (1971, 2000)
        threshold_percentile : float
            Percentile threshold (default: 90)

        Returns:
        --------
        percentage : float or xr.DataArray
            Percentage of days exceeding threshold
        """
        if isinstance(tasmax, xr.DataArray):
            if reference_period:
                ref_data = tasmax.sel(
                    time=slice(str(reference_period[0]), str(reference_period[1]))
                )
                threshold = ref_data.quantile(threshold_percentile / 100, dim='time')
            else:
                threshold = tasmax.quantile(threshold_percentile / 100, dim='time')

            return ((tasmax > threshold).sum(dim='time') / len(tasmax.time) * 100)
        else:
            threshold = np.percentile(tasmax, threshold_percentile)
            return (np.sum(tasmax > threshold) / len(tasmax)) * 100

    @staticmethod
    def tn10p(tasmin, reference_period=None, threshold_percentile=10):
        """
        TN10p: Percentage of days with Tmin < 10th percentile.

        Cold nights - indicates cold extremes.
        """
        if isinstance(tasmin, xr.DataArray):
            if reference_period:
                ref_data = tasmin.sel(
                    time=slice(str(reference_period[0]), str(reference_period[1]))
                )
                threshold = ref_data.quantile(threshold_percentile / 100, dim='time')
            else:
                threshold = tasmin.quantile(threshold_percentile / 100, dim='time')

            return ((tasmin < threshold).sum(dim='time') / len(tasmin.time) * 100)
        else:
            threshold = np.percentile(tasmin, threshold_percentile)
            return (np.sum(tasmin < threshold) / len(tasmin)) * 100

    @staticmethod
    def wsdi(tasmax, threshold_percentile=90, min_duration=6):
        """
        WSDI: Warm Spell Duration Index.

        Annual count of days in warm spells (>= 6 consecutive days
        with Tmax > 90th percentile).

        Parameters:
        -----------
        tasmax : array-like
            Daily maximum temperature
        threshold_percentile : float
            Percentile for warm threshold
        min_duration : int
            Minimum consecutive days for a warm spell

        Returns:
        --------
        count : int
            Number of days in warm spells
        """
        threshold = np.percentile(tasmax, threshold_percentile)
        above_threshold = tasmax > threshold

        # Find consecutive sequences
        spell_days = 0
        current_spell = 0

        for day in above_threshold:
            if day:
                current_spell += 1
            else:
                if current_spell >= min_duration:
                    spell_days += current_spell
                current_spell = 0

        # Check final spell
        if current_spell >= min_duration:
            spell_days += current_spell

        return spell_days

    @staticmethod
    def csdi(tasmin, threshold_percentile=10, min_duration=6):
        """
        CSDI: Cold Spell Duration Index.

        Annual count of days in cold spells (>= 6 consecutive days
        with Tmin < 10th percentile).
        """
        threshold = np.percentile(tasmin, threshold_percentile)
        below_threshold = tasmin < threshold

        spell_days = 0
        current_spell = 0

        for day in below_threshold:
            if day:
                current_spell += 1
            else:
                if current_spell >= min_duration:
                    spell_days += current_spell
                current_spell = 0

        if current_spell >= min_duration:
            spell_days += current_spell

        return spell_days

    @staticmethod
    def su(tasmax, threshold=25):
        """
        SU: Summer Days.

        Annual count of days with Tmax > threshold (default: 25°C).
        """
        if isinstance(tasmax, xr.DataArray):
            return (tasmax > threshold).sum(dim='time')
        else:
            return np.sum(tasmax > threshold)

    @staticmethod
    def fd(tasmin, threshold=0):
        """
        FD: Frost Days.

        Annual count of days with Tmin < threshold (default: 0°C).
        """
        if isinstance(tasmin, xr.DataArray):
            return (tasmin < threshold).sum(dim='time')
        else:
            return np.sum(tasmin < threshold)

    @staticmethod
    def tr(tasmin, threshold=20):
        """
        TR: Tropical Nights.

        Annual count of days with Tmin > threshold (default: 20°C).
        """
        if isinstance(tasmin, xr.DataArray):
            return (tasmin > threshold).sum(dim='time')
        else:
            return np.sum(tasmin > threshold)

    @staticmethod
    def gsl(tasmax, threshold=5, min_days=6):
        """
        GSL: Growing Season Length.

        Annual count of days between first span of at least min_days
        days with Tmean > threshold and first span after July 1 of
        min_days days with Tmean < threshold.

        Simplified version using Tmax as proxy for Tmean.
        """
        above_threshold = tasmax > threshold

        # Find first occurrence of min_days consecutive days above threshold
        start_day = None
        consecutive = 0

        for i, day in enumerate(above_threshold):
            if day:
                consecutive += 1
                if consecutive >= min_days and start_day is None:
                    start_day = i - min_days + 1
            else:
                consecutive = 0

        if start_day is None:
            return 0

        # Find end (after day 180 - approximately July 1)
        end_day = len(tasmax)
        consecutive = 0

        for i in range(180, len(above_threshold)):
            if not above_threshold[i]:
                consecutive += 1
                if consecutive >= min_days:
                    end_day = i - min_days + 1
                    break
            else:
                consecutive = 0

        return end_day - start_day


class PrecipitationIndices:
    """
    Calculate precipitation-based climate extreme indices.
    """

    @staticmethod
    def rx1day(pr):
        """
        Rx1day: Maximum 1-day precipitation.

        Annual maximum precipitation amount in a single day.

        Parameters:
        -----------
        pr : array-like or xr.DataArray
            Daily precipitation

        Returns:
        --------
        maximum : float or xr.DataArray
            Maximum daily precipitation
        """
        if isinstance(pr, xr.DataArray):
            return pr.max(dim='time')
        else:
            return np.max(pr)

    @staticmethod
    def rx5day(pr):
        """
        Rx5day: Maximum 5-day precipitation.

        Annual maximum consecutive 5-day precipitation amount.
        """
        if isinstance(pr, xr.DataArray):
            # Use rolling window
            rolling = pr.rolling(time=5).sum()
            return rolling.max(dim='time')
        else:
            if len(pr) < 5:
                return np.sum(pr)
            max_sum = 0
            for i in range(len(pr) - 4):
                window_sum = np.sum(pr[i:i+5])
                if window_sum > max_sum:
                    max_sum = window_sum
            return max_sum

    @staticmethod
    def sdii(pr, threshold=1.0):
        """
        SDII: Simple Daily Intensity Index.

        Annual total precipitation divided by number of wet days
        (days with precipitation >= threshold mm).

        Parameters:
        -----------
        pr : array-like
            Daily precipitation
        threshold : float
            Minimum precipitation for wet day (default: 1.0 mm)
        """
        wet_days = pr >= threshold
        if np.sum(wet_days) == 0:
            return 0
        return np.sum(pr[wet_days]) / np.sum(wet_days)

    @staticmethod
    def r10mm(pr, threshold=10):
        """
        R10mm: Heavy precipitation days.

        Annual count of days with precipitation >= threshold mm.

        Parameters:
        -----------
        pr : array-like or xr.DataArray
            Daily precipitation
        threshold : float
            Precipitation threshold (default: 10 mm)
        """
        if isinstance(pr, xr.DataArray):
            return (pr >= threshold).sum(dim='time')
        else:
            return np.sum(pr >= threshold)

    @staticmethod
    def r20mm(pr, threshold=20):
        """
        R20mm: Very heavy precipitation days.

        Annual count of days with precipitation >= threshold mm.
        """
        return PrecipitationIndices.r10mm(pr, threshold=threshold)

    @staticmethod
    def r95p(pr, reference_period=None, threshold_percentile=95):
        """
        R95p: Very wet days.

        Annual total precipitation from days with precipitation > 95th
        percentile of reference period wet days.
        """
        if isinstance(pr, xr.DataArray):
            if reference_period:
                ref_data = pr.sel(
                    time=slice(str(reference_period[0]), str(reference_period[1]))
                )
                wet_days = ref_data.where(ref_data >= 1.0, drop=True)
            else:
                wet_days = pr.where(pr >= 1.0, drop=True)

            threshold = wet_days.quantile(threshold_percentile / 100)
            return pr.where(pr > threshold, 0).sum(dim='time')
        else:
            wet_days = pr[pr >= 1.0]
            if len(wet_days) == 0:
                return 0
            threshold = np.percentile(wet_days, threshold_percentile)
            return np.sum(pr[pr > threshold])

    @staticmethod
    def cdd(pr, threshold=1.0):
        """
        CDD: Consecutive Dry Days.

        Maximum number of consecutive days with precipitation < threshold.

        Useful for drought analysis.
        """
        dry_days = pr < threshold

        max_consecutive = 0
        current_consecutive = 0

        for day in dry_days:
            if day:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    @staticmethod
    def cwd(pr, threshold=1.0):
        """
        CWD: Consecutive Wet Days.

        Maximum number of consecutive days with precipitation >= threshold.

        Useful for flooding analysis.
        """
        wet_days = pr >= threshold

        max_consecutive = 0
        current_consecutive = 0

        for day in wet_days:
            if day:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    @staticmethod
    def prcptot(pr, threshold=1.0):
        """
        PRCPTOT: Annual total precipitation in wet days.

        Sum of precipitation on days with precipitation >= threshold.
        """
        if isinstance(pr, xr.DataArray):
            return pr.where(pr >= threshold, 0).sum(dim='time')
        else:
            return np.sum(pr[pr >= threshold])


def compute_all_indices(tasmax=None, tasmin=None, pr=None, reference_period=None):
    """
    Compute all available climate indices.

    Parameters:
    -----------
    tasmax : array-like or xr.DataArray, optional
        Daily maximum temperature
    tasmin : array-like or xr.DataArray, optional
        Daily minimum temperature
    pr : array-like or xr.DataArray, optional
        Daily precipitation
    reference_period : tuple or None
        (start_year, end_year) for baseline

    Returns:
    --------
    indices : dict
        Dictionary of all computed indices
    """
    indices = {}

    temp_idx = TemperatureIndices()
    precip_idx = PrecipitationIndices()

    # Temperature indices
    if tasmax is not None:
        indices['TX90p'] = temp_idx.tx90p(tasmax, reference_period)
        indices['SU'] = temp_idx.su(tasmax)
        indices['WSDI'] = temp_idx.wsdi(tasmax)
        if len(tasmax) >= 365:
            indices['GSL'] = temp_idx.gsl(tasmax)

    if tasmin is not None:
        indices['TN10p'] = temp_idx.tn10p(tasmin, reference_period)
        indices['FD'] = temp_idx.fd(tasmin)
        indices['TR'] = temp_idx.tr(tasmin)
        indices['CSDI'] = temp_idx.csdi(tasmin)

    # Precipitation indices
    if pr is not None:
        indices['Rx1day'] = precip_idx.rx1day(pr)
        indices['Rx5day'] = precip_idx.rx5day(pr)
        indices['SDII'] = precip_idx.sdii(pr)
        indices['R10mm'] = precip_idx.r10mm(pr)
        indices['R20mm'] = precip_idx.r20mm(pr)
        indices['R95p'] = precip_idx.r95p(pr, reference_period)
        indices['CDD'] = precip_idx.cdd(pr)
        indices['CWD'] = precip_idx.cwd(pr)
        indices['PRCPTOT'] = precip_idx.prcptot(pr)

    return indices


if __name__ == '__main__':
    # Example usage
    print("Climate Extreme Indices Calculator")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    n_days = 365

    # Temperature data
    tasmax = 15 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 365) + \
             np.random.normal(0, 2, n_days)
    tasmin = tasmax - 5

    # Precipitation data (realistic: many zeros, occasional heavy rain)
    pr = np.random.exponential(2, n_days)
    pr[pr < 1] = 0  # Dry days

    print("\nTemperature Indices:")
    print(f"  TX90p (warm days): {TemperatureIndices.tx90p(tasmax):.1f}%")
    print(f"  Summer days (>25°C): {TemperatureIndices.su(tasmax):.0f} days")
    print(f"  Frost days (<0°C): {TemperatureIndices.fd(tasmin):.0f} days")
    print(f"  Growing season length: {TemperatureIndices.gsl(tasmax):.0f} days")

    print("\nPrecipitation Indices:")
    print(f"  Rx1day: {PrecipitationIndices.rx1day(pr):.1f} mm")
    print(f"  Rx5day: {PrecipitationIndices.rx5day(pr):.1f} mm")
    print(f"  R10mm (heavy days): {PrecipitationIndices.r10mm(pr):.0f} days")
    print(f"  CDD (max dry spell): {PrecipitationIndices.cdd(pr):.0f} days")
    print(f"  Total precipitation: {PrecipitationIndices.prcptot(pr):.1f} mm")

    print("\n✓ Climate indices ready")
