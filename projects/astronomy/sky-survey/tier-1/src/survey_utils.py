"""
Survey query utilities for SDSS, Gaia, 2MASS, and WISE.
"""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.sdss import SDSS
from astroquery.gaia import Gaia
from astroquery.irsa import Irsa
import warnings


def query_sdss(ra_center, dec_center, radius_deg=1.0, data_release=18):
    """
    Query SDSS photometric catalog.

    Parameters
    ----------
    ra_center : float
        Right ascension center (degrees)
    dec_center : float
        Declination center (degrees)
    radius_deg : float
        Search radius (degrees)
    data_release : int
        SDSS data release number

    Returns
    -------
    astropy.table.Table
        SDSS catalog with ugriz photometry
    """
    print(f"Querying SDSS DR{data_release}...")
    print(f"  Center: RA={ra_center}, Dec={dec_center}")
    print(f"  Radius: {radius_deg} deg")

    coords = SkyCoord(ra=ra_center*u.deg, dec=dec_center*u.deg)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = SDSS.query_region(
                coords,
                radius=radius_deg*u.deg,
                data_release=data_release,
                photoobj_fields=['ra', 'dec', 'u', 'g', 'r', 'i', 'z',
                                'err_u', 'err_g', 'err_r', 'err_i', 'err_z',
                                'type', 'petroR50_r', 'petroR90_r']
            )

        if result is not None:
            print(f"  Found {len(result)} sources")
            return result
        else:
            print("  No sources found")
            return None

    except Exception as e:
        print(f"  Error querying SDSS: {e}")
        return None


def query_gaia(ra_center, dec_center, radius_deg=1.0, max_sources=100000):
    """
    Query Gaia EDR3 catalog.

    Parameters
    ----------
    ra_center : float
        Right ascension center (degrees)
    dec_center : float
        Declination center (degrees)
    radius_deg : float
        Search radius (degrees)
    max_sources : int
        Maximum number of sources to return

    Returns
    -------
    astropy.table.Table
        Gaia catalog with astrometry and photometry
    """
    print("Querying Gaia EDR3...")
    print(f"  Center: RA={ra_center}, Dec={dec_center}")
    print(f"  Radius: {radius_deg} deg")

    query = f"""
    SELECT TOP {max_sources}
        source_id, ra, dec, pmra, pmdec, parallax,
        phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
        ra_error, dec_error, pmra_error, pmdec_error, parallax_error
    FROM gaiaedr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_deg})
    ) = 1
    """

    try:
        job = Gaia.launch_job_async(query)
        result = job.get_results()

        print(f"  Found {len(result)} sources")
        return result

    except Exception as e:
        print(f"  Error querying Gaia: {e}")
        return None


def query_2mass(ra_center, dec_center, radius_deg=1.0):
    """
    Query 2MASS point source catalog.

    Parameters
    ----------
    ra_center : float
        Right ascension center (degrees)
    dec_center : float
        Declination center (degrees)
    radius_deg : float
        Search radius (degrees)

    Returns
    -------
    astropy.table.Table
        2MASS catalog with JHK photometry
    """
    print("Querying 2MASS...")
    print(f"  Center: RA={ra_center}, Dec={dec_center}")
    print(f"  Radius: {radius_deg} deg")

    coords = SkyCoord(ra=ra_center*u.deg, dec=dec_center*u.deg)

    try:
        result = Irsa.query_region(
            coords,
            catalog='fp_psc',  # 2MASS Point Source Catalog
            spatial='Cone',
            radius=radius_deg*u.deg
        )

        if result is not None:
            print(f"  Found {len(result)} sources")
            return result
        else:
            print("  No sources found")
            return None

    except Exception as e:
        print(f"  Error querying 2MASS: {e}")
        return None


def query_wise(ra_center, dec_center, radius_deg=1.0):
    """
    Query WISE all-sky catalog.

    Parameters
    ----------
    ra_center : float
        Right ascension center (degrees)
    dec_center : float
        Declination center (degrees)
    radius_deg : float
        Search radius (degrees)

    Returns
    -------
    astropy.table.Table
        WISE catalog with W1-W4 photometry
    """
    print("Querying WISE...")
    print(f"  Center: RA={ra_center}, Dec={dec_center}")
    print(f"  Radius: {radius_deg} deg")

    coords = SkyCoord(ra=ra_center*u.deg, dec=dec_center*u.deg)

    try:
        result = Irsa.query_region(
            coords,
            catalog='allwise_p3as_psd',  # AllWISE Source Catalog
            spatial='Cone',
            radius=radius_deg*u.deg
        )

        if result is not None:
            print(f"  Found {len(result)} sources")
            return result
        else:
            print("  No sources found")
            return None

    except Exception as e:
        print(f"  Error querying WISE: {e}")
        return None


def save_catalog(catalog, filename):
    """
    Save catalog to FITS file.

    Parameters
    ----------
    catalog : astropy.table.Table
        Catalog to save
    filename : str
        Output filename (should end in .fits)
    """
    from pathlib import Path

    # Create directory if needed
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # Save to FITS
    catalog.write(filename, format='fits', overwrite=True)
    print(f"Saved {len(catalog)} sources to {filename}")


def load_catalog(filename):
    """
    Load catalog from FITS file.

    Parameters
    ----------
    filename : str
        Input filename

    Returns
    -------
    astropy.table.Table
        Loaded catalog
    """
    from astropy.table import Table

    catalog = Table.read(filename, format='fits')
    print(f"Loaded {len(catalog)} sources from {filename}")
    return catalog
