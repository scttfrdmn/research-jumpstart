"""
Data access module for astronomical sky surveys.

Provides classes for accessing SDSS, Pan-STARRS, Legacy Survey, and other major surveys.
"""

import os
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.sdss import SDSS
from astroquery.mast import Catalogs
import boto3
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SDSSDataLoader:
    """Load data from Sloan Digital Sky Survey."""

    def __init__(self, bucket_name: Optional[str] = None, data_release: int = 18):
        """
        Initialize SDSS data loader.

        Args:
            bucket_name: S3 bucket for caching data
            data_release: SDSS data release (default: DR18)
        """
        self.bucket_name = bucket_name
        self.data_release = data_release
        self.s3_client = boto3.client('s3') if bucket_name else None

        # SDSS DR18 base URL
        self.base_url = f'https://data.sdss.org/sas/dr{data_release}/eboss/photoObj/'

    def query_region(
        self,
        ra: float,
        dec: float,
        radius: float = 0.1,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Query SDSS catalog in a sky region.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            radius: Search radius (degrees)
            filters: Additional filters (e.g., {'type': 'GALAXY'})

        Returns:
            DataFrame with SDSS photometry
        """
        logger.info(f"Querying SDSS at RA={ra:.2f}, Dec={dec:.2f}, radius={radius:.2f} deg")

        # Create coordinate
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

        # Query using astroquery
        try:
            result = SDSS.query_region(
                coord,
                radius=radius*u.deg,
                data_release=self.data_release,
                photoobj_fields=['ra', 'dec', 'u', 'g', 'r', 'i', 'z',
                               'petroMag_u', 'petroMag_g', 'petroMag_r',
                               'petroMag_i', 'petroMag_z',
                               'type', 'clean', 'objID']
            )
        except Exception as e:
            logger.error(f"SDSS query failed: {e}")
            return pd.DataFrame()

        if result is None:
            logger.warning("No objects found in region")
            return pd.DataFrame()

        # Convert to pandas
        df = result.to_pandas()

        # Apply filters
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    df = df[df[key].between(value[0], value[1])]
                else:
                    df = df[df[key] == value]

        logger.info(f"Found {len(df)} objects")
        return df

    def download_imaging(
        self,
        ra: float,
        dec: float,
        width: float = 0.5,
        bands: List[str] = ['g', 'r', 'i']
    ) -> Dict[str, Path]:
        """
        Download SDSS imaging cutouts.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            width: Image width (degrees)
            bands: List of bands to download

        Returns:
            Dictionary mapping bands to file paths
        """
        logger.info(f"Downloading SDSS images at RA={ra:.2f}, Dec={dec:.2f}")

        # SDSS imaging cutout service
        base_url = "https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"

        images = {}
        for band in bands:
            # Construct URL
            width_arcmin = width * 60
            url = f"{base_url}?ra={ra}&dec={dec}&width={width_arcmin}&height={width_arcmin}&scale=0.4"

            # Download
            local_path = Path(f"./data/sdss/cutout_ra{ra:.2f}_dec{dec:.2f}_{band}.jpg")
            local_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                response = requests.get(url)
                response.raise_for_status()

                with open(local_path, 'wb') as f:
                    f.write(response.content)

                logger.info(f"Downloaded {band}-band image to {local_path}")
                images[band] = local_path

                # Upload to S3 if bucket specified
                if self.bucket_name:
                    s3_key = f"sdss/images/ra{ra:.2f}_dec{dec:.2f}_{band}.jpg"
                    self.s3_client.upload_file(
                        str(local_path),
                        self.bucket_name,
                        s3_key
                    )

            except Exception as e:
                logger.error(f"Failed to download {band}-band: {e}")

        return images

    def load_spectrum(self, plate: int, mjd: int, fiber: int) -> Dict:
        """
        Load SDSS spectrum.

        Args:
            plate: SDSS plate number
            mjd: Modified Julian Date
            fiber: Fiber number

        Returns:
            Dictionary with wavelength, flux, error
        """
        # SDSS spectrum file path
        filename = f"spec-{plate:04d}-{mjd}-{fiber:04d}.fits"
        url = f"https://data.sdss.org/sas/dr{self.data_release}/eboss/spectro/redux/v5_13_2/spectra/lite/{plate}/{filename}"

        try:
            # Download and open FITS file
            with fits.open(url) as hdul:
                # Spectrum is in HDU 1
                data = hdul[1].data
                header = hdul[0].header

                # Extract arrays
                wavelength = 10**data['loglam']  # Convert log(wavelength) to wavelength
                flux = data['flux']
                ivar = data['ivar']  # Inverse variance
                error = np.sqrt(1.0 / ivar)

                spectrum = {
                    'wavelength': wavelength,
                    'flux': flux,
                    'error': error,
                    'redshift': header.get('Z', None),
                    'class': header.get('CLASS', None)
                }

                logger.info(f"Loaded spectrum: plate={plate}, mjd={mjd}, fiber={fiber}")
                return spectrum

        except Exception as e:
            logger.error(f"Failed to load spectrum: {e}")
            return {}

    def query_spectroscopic_sample(
        self,
        z_range: Tuple[float, float] = (0.0, 1.0),
        spec_class: str = 'GALAXY',
        max_rows: int = 10000
    ) -> pd.DataFrame:
        """
        Query SDSS spectroscopic sample.

        Args:
            z_range: Redshift range (min, max)
            spec_class: Spectroscopic class (GALAXY, STAR, QSO)
            max_rows: Maximum number of rows to return

        Returns:
            DataFrame with spectroscopic data
        """
        # CasJobs SQL query
        query = f"""
        SELECT TOP {max_rows}
            p.objID, p.ra, p.dec,
            p.u, p.g, p.r, p.i, p.z,
            s.plate, s.mjd, s.fiberID,
            s.z as redshift, s.zErr as redshift_err,
            s.class, s.subClass
        FROM PhotoObj AS p
        JOIN SpecObj AS s ON s.bestobjid = p.objid
        WHERE s.class = '{spec_class}'
          AND s.z BETWEEN {z_range[0]} AND {z_range[1]}
          AND s.zWarning = 0
        """

        logger.info(f"Querying {spec_class} spectra at z = {z_range[0]}-{z_range[1]}")

        try:
            result = SDSS.query_sql(query)
            df = result.to_pandas()
            logger.info(f"Found {len(df)} spectra")
            return df
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return pd.DataFrame()


class PanSTARRSLoader:
    """Load data from Pan-STARRS survey."""

    def __init__(self, bucket_name: Optional[str] = None):
        """
        Initialize Pan-STARRS data loader.

        Args:
            bucket_name: S3 bucket for caching data
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3') if bucket_name else None

    def query_region(
        self,
        ra: float,
        dec: float,
        radius: float = 0.1,
        catalog: str = 'mean'
    ) -> pd.DataFrame:
        """
        Query Pan-STARRS catalog.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            radius: Search radius (degrees)
            catalog: 'mean' for averaged photometry or 'stack' for coadd

        Returns:
            DataFrame with Pan-STARRS photometry
        """
        logger.info(f"Querying Pan-STARRS at RA={ra:.2f}, Dec={dec:.2f}")

        # Use astroquery MAST
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

        try:
            result = Catalogs.query_region(
                coord,
                radius=radius*u.deg,
                catalog="Panstarrs",
                table=catalog
            )

            if len(result) == 0:
                logger.warning("No objects found")
                return pd.DataFrame()

            df = result.to_pandas()
            logger.info(f"Found {len(df)} objects")
            return df

        except Exception as e:
            logger.error(f"Pan-STARRS query failed: {e}")
            return pd.DataFrame()

    def get_lightcurve(
        self,
        ra: float,
        dec: float,
        radius: float = 1.0  # arcseconds
    ) -> pd.DataFrame:
        """
        Get Pan-STARRS time-series photometry.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            radius: Match radius (arcseconds)

        Returns:
            DataFrame with multi-epoch photometry
        """
        logger.info(f"Getting lightcurve for RA={ra:.2f}, Dec={dec:.2f}")

        # Query detection table (individual epochs)
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

        try:
            result = Catalogs.query_region(
                coord,
                radius=radius*u.arcsec,
                catalog="Panstarrs",
                table="detection"
            )

            if len(result) == 0:
                return pd.DataFrame()

            df = result.to_pandas()

            # Sort by time
            if 'obsTime' in df.columns:
                df = df.sort_values('obsTime')

            logger.info(f"Found {len(df)} detections")
            return df

        except Exception as e:
            logger.error(f"Lightcurve query failed: {e}")
            return pd.DataFrame()


class LegacySurveyLoader:
    """Load data from Legacy Survey (DECaLS/BASS/MzLS)."""

    def __init__(self, bucket_name: Optional[str] = None):
        """
        Initialize Legacy Survey data loader.

        Args:
            bucket_name: S3 bucket for caching data
        """
        self.bucket_name = bucket_name
        self.base_url = "https://www.legacysurvey.org/viewer"

    def query_region(
        self,
        ra: float,
        dec: float,
        radius: float = 0.1
    ) -> pd.DataFrame:
        """
        Query Legacy Survey catalog.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            radius: Search radius (degrees)

        Returns:
            DataFrame with Legacy Survey photometry
        """
        logger.info(f"Querying Legacy Survey at RA={ra:.2f}, Dec={dec:.2f}")

        # Legacy Survey catalog service
        url = f"{self.base_url}/catalog-search/"
        params = {
            'ra': ra,
            'dec': dec,
            'radius': radius * 3600,  # Convert to arcseconds
            'format': 'json'
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data.get('rd'):
                logger.warning("No objects found")
                return pd.DataFrame()

            df = pd.DataFrame(data['rd'])
            logger.info(f"Found {len(df)} objects")
            return df

        except Exception as e:
            logger.error(f"Legacy Survey query failed: {e}")
            return pd.DataFrame()

    def download_cutout(
        self,
        ra: float,
        dec: float,
        size: int = 256,
        layer: str = 'ls-dr10',
        bands: str = 'grz'
    ) -> Path:
        """
        Download Legacy Survey image cutout.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            size: Image size (pixels)
            layer: Survey layer (e.g., 'ls-dr10')
            bands: Band combination (e.g., 'grz' for RGB)

        Returns:
            Path to downloaded image
        """
        # Cutout service URL
        url = f"{self.base_url}/jpeg-cutout/"
        params = {
            'ra': ra,
            'dec': dec,
            'size': size,
            'layer': layer,
            'bands': bands
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            # Save image
            local_path = Path(f"./data/legacy/cutout_ra{ra:.2f}_dec{dec:.2f}.jpg")
            local_path.parent.mkdir(parents=True, exist_ok=True)

            with open(local_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded cutout to {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download cutout: {e}")
            return None


def crossmatch_catalogs(
    catalog1: pd.DataFrame,
    catalog2: pd.DataFrame,
    ra_col1: str = 'ra',
    dec_col1: str = 'dec',
    ra_col2: str = 'ra',
    dec_col2: str = 'dec',
    max_separation: float = 1.0  # arcseconds
) -> pd.DataFrame:
    """
    Cross-match two astronomical catalogs by sky position.

    Args:
        catalog1: First catalog
        catalog2: Second catalog
        ra_col1: RA column name in catalog1
        dec_col1: Dec column name in catalog1
        ra_col2: RA column name in catalog2
        dec_col2: Dec column name in catalog2
        max_separation: Maximum match separation (arcseconds)

    Returns:
        Merged catalog with matches
    """
    from astropy.coordinates import match_coordinates_sky

    logger.info(f"Cross-matching {len(catalog1)} × {len(catalog2)} objects")

    # Create SkyCoord objects
    coord1 = SkyCoord(
        ra=catalog1[ra_col1].values*u.deg,
        dec=catalog1[dec_col1].values*u.deg,
        frame='icrs'
    )

    coord2 = SkyCoord(
        ra=catalog2[ra_col2].values*u.deg,
        dec=catalog2[dec_col2].values*u.deg,
        frame='icrs'
    )

    # Match
    idx, sep2d, _ = match_coordinates_sky(coord1, coord2)

    # Filter by separation
    mask = sep2d < max_separation * u.arcsec

    # Merge catalogs
    matched = catalog1[mask].copy()
    matched['match_idx'] = idx[mask]
    matched['match_sep'] = sep2d[mask].arcsec

    # Add columns from catalog2
    for col in catalog2.columns:
        if col not in [ra_col2, dec_col2]:
            matched[f'{col}_2'] = catalog2.iloc[idx[mask]][col].values

    logger.info(f"Found {len(matched)} matches within {max_separation}\"")

    return matched


def main():
    """
    Main function for data access demo.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Access astronomical survey data')
    parser.add_argument('--survey', type=str, default='sdss',
                       choices=['sdss', 'panstarrs', 'legacy'],
                       help='Survey to query')
    parser.add_argument('--ra', type=float, required=True,
                       help='Right ascension (degrees)')
    parser.add_argument('--dec', type=float, required=True,
                       help='Declination (degrees)')
    parser.add_argument('--radius', type=float, default=0.1,
                       help='Search radius (degrees)')
    args = parser.parse_args()

    # Query survey
    if args.survey == 'sdss':
        loader = SDSSDataLoader()
        catalog = loader.query_region(args.ra, args.dec, args.radius)
    elif args.survey == 'panstarrs':
        loader = PanSTARRSLoader()
        catalog = loader.query_region(args.ra, args.dec, args.radius)
    elif args.survey == 'legacy':
        loader = LegacySurveyLoader()
        catalog = loader.query_region(args.ra, args.dec, args.radius)

    print(f"\nFound {len(catalog)} objects:")
    print(catalog.head(10))

    # Save to CSV
    output_file = f"{args.survey}_catalog.csv"
    catalog.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")


if __name__ == '__main__':
    main()
