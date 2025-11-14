"""
Catalog cross-matching utilities using spatial indexing.
"""

import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u
import healpy as hp


def build_healpix_index(ra, dec, nside=1024):
    """
    Build HEALPix spatial index for catalog.

    Parameters
    ----------
    ra : array-like
        Right ascension (degrees)
    dec : array-like
        Declination (degrees)
    nside : int
        HEALPix resolution parameter (default: 1024 = 3.4 arcmin pixels)

    Returns
    -------
    array
        HEALPix pixel indices
    """
    pixels = hp.ang2pix(nside, ra, dec, lonlat=True)
    return pixels


def spatial_crossmatch(ra1, dec1, ra2, dec2, max_sep_arcsec=1.0):
    """
    Cross-match two catalogs using spherical distance.

    Parameters
    ----------
    ra1, dec1 : array-like
        Coordinates of catalog 1 (degrees)
    ra2, dec2 : array-like
        Coordinates of catalog 2 (degrees)
    max_sep_arcsec : float
        Maximum separation for match (arcseconds)

    Returns
    -------
    idx : array
        Indices of matches in catalog 2 (same length as catalog 1)
    sep : array
        Angular separations (arcseconds)
    mask : array
        Boolean mask: True where separation < max_sep_arcsec
    """
    # Create SkyCoord objects
    coords1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
    coords2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)

    # Match catalogs
    idx, sep2d, _ = match_coordinates_sky(coords1, coords2)

    # Convert separation to arcseconds
    sep_arcsec = sep2d.to(u.arcsec).value

    # Create mask for good matches
    mask = sep_arcsec < max_sep_arcsec

    print(f"Cross-match results:")
    print(f"  Total in catalog 1: {len(ra1)}")
    print(f"  Total in catalog 2: {len(ra2)}")
    print(f"  Matches within {max_sep_arcsec} arcsec: {mask.sum()}")
    print(f"  Match rate: {100*mask.sum()/len(ra1):.1f}%")

    return idx, sep_arcsec, mask


def match_catalogs(cat1, cat2, ra_col1='ra', dec_col1='dec',
                  ra_col2='ra', dec_col2='dec', max_sep_arcsec=1.0,
                  join_type='left'):
    """
    Match two astropy Tables by position.

    Parameters
    ----------
    cat1 : astropy.table.Table
        First catalog
    cat2 : astropy.table.Table
        Second catalog
    ra_col1, dec_col1 : str
        Column names for coordinates in catalog 1
    ra_col2, dec_col2 : str
        Column names for coordinates in catalog 2
    max_sep_arcsec : float
        Maximum separation for match (arcseconds)
    join_type : str
        'left' (keep all cat1) or 'inner' (only matches)

    Returns
    -------
    matched : astropy.table.Table
        Matched catalog with columns from both inputs
    """
    from astropy.table import Table, hstack

    # Extract coordinates
    ra1 = cat1[ra_col1]
    dec1 = cat1[dec_col1]
    ra2 = cat2[ra_col2]
    dec2 = cat2[dec_col2]

    # Perform cross-match
    idx, sep_arcsec, mask = spatial_crossmatch(ra1, dec1, ra2, dec2, max_sep_arcsec)

    # Build matched catalog
    if join_type == 'inner':
        # Only keep matches
        matched_cat1 = cat1[mask]
        matched_cat2 = cat2[idx[mask]]

        # Add separation column
        sep_col = Table()
        sep_col['separation_arcsec'] = sep_arcsec[mask]

        result = hstack([matched_cat1, matched_cat2, sep_col])

    elif join_type == 'left':
        # Keep all cat1, fill non-matches with masked values
        matched_cat2 = cat2[idx]

        # Mask non-matches
        for col in matched_cat2.colnames:
            matched_cat2[col].mask = ~mask

        # Add separation column
        sep_col = Table()
        sep_col['separation_arcsec'] = sep_arcsec
        sep_col['separation_arcsec'].mask = ~mask

        result = hstack([cat1, matched_cat2, sep_col])

    else:
        raise ValueError(f"Unknown join_type: {join_type}")

    return result


def build_multi_survey_catalog(sdss, gaia, tmass, wise,
                               max_sep_arcsec=1.0):
    """
    Cross-match catalogs from multiple surveys.

    Parameters
    ----------
    sdss : astropy.table.Table
        SDSS catalog
    gaia : astropy.table.Table
        Gaia catalog
    tmass : astropy.table.Table
        2MASS catalog
    wise : astropy.table.Table
        WISE catalog
    max_sep_arcsec : float
        Maximum separation for matches (arcseconds)

    Returns
    -------
    matched : astropy.table.Table
        Multi-survey matched catalog
    """
    print("Building multi-survey matched catalog...")

    # Start with SDSS as reference
    result = sdss.copy()
    result['catalog_source'] = 'SDSS'

    # Match with Gaia
    if gaia is not None:
        print("\n1. Matching SDSS + Gaia...")
        result = match_catalogs(
            result, gaia,
            ra_col1='ra', dec_col1='dec',
            ra_col2='ra', dec_col2='dec',
            max_sep_arcsec=max_sep_arcsec,
            join_type='left'
        )

    # Match with 2MASS
    if tmass is not None:
        print("\n2. Matching SDSS + Gaia + 2MASS...")
        result = match_catalogs(
            result, tmass,
            ra_col1='ra', dec_col1='dec',
            ra_col2='ra', dec_col2='dec',
            max_sep_arcsec=max_sep_arcsec,
            join_type='left'
        )

    # Match with WISE
    if wise is not None:
        print("\n3. Matching SDSS + Gaia + 2MASS + WISE...")
        result = match_catalogs(
            result, wise,
            ra_col1='ra', dec_col1='dec',
            ra_col2='ra', dec_col2='dec',
            max_sep_arcsec=max_sep_arcsec,
            join_type='left'
        )

    print(f"\nFinal matched catalog: {len(result)} sources")

    return result


def calculate_match_statistics(matched_catalog):
    """
    Calculate statistics on matched catalog.

    Parameters
    ----------
    matched_catalog : astropy.table.Table
        Multi-survey matched catalog

    Returns
    -------
    dict
        Statistics dictionary
    """
    stats = {}

    # Check which surveys have data for each source
    stats['total_sources'] = len(matched_catalog)

    # Count non-null entries from each survey
    # (This is simplified - adjust column names as needed)

    print("\nMatch Statistics:")
    print(f"  Total sources: {stats['total_sources']}")

    return stats
