"""
Data access utilities for CMIP6 climate data on AWS S3.

This module provides functions to access and load CMIP6 model data from the
AWS Open Data registry without downloading files locally.
"""

import xarray as xr
import s3fs
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CMIP6DataAccess:
    """
    Client for accessing CMIP6 data from AWS S3.

    The CMIP6 archive on AWS (s3://cmip6-pds) is in Zarr format, optimized
    for cloud-native access. This class provides methods to discover and
    load model data without local downloads.
    """

    def __init__(self, use_anon: bool = True):
        """
        Initialize CMIP6 data access client.

        Parameters
        ----------
        use_anon : bool, default True
            Use anonymous access (public data). Set to False if using
            credentials for private data.
        """
        self.s3 = s3fs.S3FileSystem(anon=use_anon)
        self.bucket = 'cmip6-pds'

    def build_s3_path(
        self,
        model: str,
        experiment: str,
        variable: str,
        table: str = 'Amon',
        variant: str = 'r1i1p1f1',
        grid: str = 'gn'
    ) -> str:
        """
        Build S3 path for CMIP6 data.

        Parameters
        ----------
        model : str
            Model name (e.g., 'CESM2', 'GFDL-CM4', 'UKESM1-0-LL')
        experiment : str
            Experiment ID (e.g., 'historical', 'ssp245', 'ssp585')
        variable : str
            Variable name (e.g., 'tas', 'pr', 'tos')
        table : str, default 'Amon'
            MIP table (e.g., 'Amon' for monthly atmosphere, 'Omon' for ocean)
        variant : str, default 'r1i1p1f1'
            Variant label (realization, initialization, physics, forcing)
        grid : str, default 'gn'
            Grid label ('gn' = native grid, 'gr' = regridded)

        Returns
        -------
        str
            S3 path to Zarr store

        Examples
        --------
        >>> client = CMIP6DataAccess()
        >>> path = client.build_s3_path('CESM2', 'ssp245', 'tas')
        >>> print(path)
        s3://cmip6-pds/CMIP6/ScenarioMIP/NCAR/CESM2/ssp245/r1i1p1f1/Amon/tas/gn
        """
        # Determine activity (ScenarioMIP or CMIP for historical)
        activity = 'CMIP' if experiment == 'historical' else 'ScenarioMIP'

        # Model institution mapping (simplified - add more as needed)
        institutions = {
            'CESM2': 'NCAR',
            'GFDL-CM4': 'NOAA-GFDL',
            'UKESM1-0-LL': 'MOHC',
            'CNRM-CM6-1': 'CNRM-CERFACS',
            'MPI-ESM1-2-HR': 'MPI-M',
            'MIROC6': 'MIROC',
            'CanESM5': 'CCCma',
            'ACCESS-ESM1-5': 'CSIRO',
            'IPSL-CM6A-LR': 'IPSL',
            'NorESM2-LM': 'NCC'
        }

        institution = institutions.get(model, model)

        path = (
            f's3://{self.bucket}/CMIP6/{activity}/{institution}/{model}/'
            f'{experiment}/{variant}/{table}/{variable}/{grid}'
        )

        return path

    def load_model_data(
        self,
        model: str,
        experiment: str,
        variable: str,
        table: str = 'Amon',
        time_slice: Optional[Tuple[str, str]] = None,
        chunks: Optional[Dict] = None
    ) -> xr.Dataset:
        """
        Load CMIP6 model data from S3.

        Parameters
        ----------
        model : str
            Model name
        experiment : str
            Experiment ID
        variable : str
            Variable name
        table : str, default 'Amon'
            MIP table
        time_slice : tuple of str, optional
            (start, end) dates for temporal subsetting, e.g., ('2015', '2050')
        chunks : dict, optional
            Dask chunking specification, e.g., {'time': 12, 'lat': 90, 'lon': 180}
            Default: {'time': 12} for lazy loading

        Returns
        -------
        xr.Dataset
            Xarray dataset with requested variable

        Examples
        --------
        >>> client = CMIP6DataAccess()
        >>> data = client.load_model_data(
        ...     model='CESM2',
        ...     experiment='ssp245',
        ...     variable='tas',
        ...     time_slice=('2015', '2050')
        ... )
        >>> print(data)
        """
        if chunks is None:
            chunks = {'time': 12}  # Monthly chunks

        try:
            path = self.build_s3_path(model, experiment, variable, table)
            logger.info(f"Loading data from: {path}")

            # Open Zarr store
            store = s3fs.S3Map(root=path, s3=self.s3, check=False)
            ds = xr.open_zarr(store, chunks=chunks)

            # Temporal subset if requested
            if time_slice is not None:
                start, end = time_slice
                ds = ds.sel(time=slice(start, end))
                logger.info(f"Subsetted to time range: {start} to {end}")

            logger.info(f"Successfully loaded {model} {experiment} {variable}")
            return ds

        except Exception as e:
            logger.error(f"Error loading data for {model}: {e}")
            raise

    def load_ensemble(
        self,
        models: List[str],
        experiment: str,
        variable: str,
        table: str = 'Amon',
        time_slice: Optional[Tuple[str, str]] = None
    ) -> Dict[str, xr.Dataset]:
        """
        Load multiple models to create an ensemble.

        Parameters
        ----------
        models : list of str
            List of model names
        experiment : str
            Experiment ID
        variable : str
            Variable name
        table : str, default 'Amon'
            MIP table
        time_slice : tuple of str, optional
            (start, end) dates for temporal subsetting

        Returns
        -------
        dict
            Dictionary mapping model names to xarray Datasets

        Examples
        --------
        >>> client = CMIP6DataAccess()
        >>> ensemble = client.load_ensemble(
        ...     models=['CESM2', 'GFDL-CM4', 'UKESM1-0-LL'],
        ...     experiment='ssp245',
        ...     variable='tas',
        ...     time_slice=('2015', '2100')
        ... )
        >>> print(f"Loaded {len(ensemble)} models")
        """
        ensemble = {}
        failed_models = []

        for model in models:
            try:
                ds = self.load_model_data(
                    model=model,
                    experiment=experiment,
                    variable=variable,
                    table=table,
                    time_slice=time_slice
                )
                ensemble[model] = ds
                logger.info(f"✓ {model}")
            except Exception as e:
                logger.warning(f"✗ {model}: {e}")
                failed_models.append(model)

        if failed_models:
            logger.warning(
                f"Failed to load {len(failed_models)} models: {failed_models}"
            )

        logger.info(f"Successfully loaded {len(ensemble)}/{len(models)} models")
        return ensemble

    def get_available_models(self, experiment: str = 'ssp245') -> List[str]:
        """
        Get list of available models for a given experiment.

        Note: This is a simplified version with common models.
        For a complete list, query the S3 bucket directly.

        Parameters
        ----------
        experiment : str, default 'ssp245'
            Experiment ID

        Returns
        -------
        list of str
            List of model names
        """
        # Common CMIP6 models with good data availability
        common_models = [
            'ACCESS-CM2',
            'ACCESS-ESM1-5',
            'AWI-CM-1-1-MR',
            'BCC-CSM2-MR',
            'CAMS-CSM1-0',
            'CanESM5',
            'CESM2',
            'CESM2-WACCM',
            'CNRM-CM6-1',
            'CNRM-ESM2-1',
            'EC-Earth3',
            'EC-Earth3-Veg',
            'FGOALS-g3',
            'GFDL-CM4',
            'GFDL-ESM4',
            'GISS-E2-1-G',
            'HadGEM3-GC31-LL',
            'INM-CM4-8',
            'INM-CM5-0',
            'IPSL-CM6A-LR',
            'KACE-1-0-G',
            'MIROC6',
            'MIROC-ES2L',
            'MPI-ESM1-2-HR',
            'MPI-ESM1-2-LR',
            'MRI-ESM2-0',
            'NESM3',
            'NorESM2-LM',
            'NorESM2-MM',
            'TaiESM1',
            'UKESM1-0-LL',
        ]

        return common_models


def validate_region(region: Dict[str, float]) -> bool:
    """
    Validate regional bounding box specification.

    Parameters
    ----------
    region : dict
        Dictionary with keys 'lat_min', 'lat_max', 'lon_min', 'lon_max'

    Returns
    -------
    bool
        True if valid, raises ValueError otherwise

    Examples
    --------
    >>> region = {'lat_min': 31, 'lat_max': 37, 'lon_min': -114, 'lon_max': -109}
    >>> validate_region(region)
    True
    """
    required_keys = ['lat_min', 'lat_max', 'lon_min', 'lon_max']

    if not all(k in region for k in required_keys):
        raise ValueError(f"Region must contain keys: {required_keys}")

    if region['lat_min'] >= region['lat_max']:
        raise ValueError("lat_min must be less than lat_max")

    if region['lon_min'] >= region['lon_max']:
        raise ValueError("lon_min must be less than lon_max")

    if not (-90 <= region['lat_min'] <= 90):
        raise ValueError("lat_min must be in range [-90, 90]")

    if not (-90 <= region['lat_max'] <= 90):
        raise ValueError("lat_max must be in range [-90, 90]")

    return True


def check_s3_access() -> bool:
    """
    Verify that S3 access is working.

    Returns
    -------
    bool
        True if access works, False otherwise
    """
    try:
        s3 = s3fs.S3FileSystem(anon=True)
        # Try to list top-level CMIP6 directory
        s3.ls('cmip6-pds/CMIP6/', refresh=True)
        logger.info("✓ S3 access verified")
        return True
    except Exception as e:
        logger.error(f"✗ S3 access failed: {e}")
        return False
