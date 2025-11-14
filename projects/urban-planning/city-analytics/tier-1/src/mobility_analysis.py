"""
Mobility and traffic analysis functions.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple
from shapely.geometry import Point, LineString


def calculate_traffic_metrics(
    road_network: gpd.GeoDataFrame,
    traffic_counts: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate traffic flow metrics for road network.

    Parameters:
    -----------
    road_network : geopandas.GeoDataFrame
        Road network geometry and attributes
    traffic_counts : pandas.DataFrame
        Traffic count observations

    Returns:
    --------
    pandas.DataFrame
        Traffic metrics (AADT, V/C ratio, congestion)
    """
    metrics = pd.DataFrame()

    # Calculate Annual Average Daily Traffic (AADT)
    if 'count' in traffic_counts.columns:
        metrics['aadt'] = traffic_counts.groupby('road_id')['count'].mean()

    # Calculate Volume/Capacity ratio
    if 'capacity' in road_network.columns and 'volume' in traffic_counts.columns:
        capacity_dict = road_network.set_index('road_id')['capacity'].to_dict()
        metrics['v_c_ratio'] = traffic_counts.apply(
            lambda row: row['volume'] / capacity_dict.get(row['road_id'], 1),
            axis=1
        )

    # Identify congestion hotspots (V/C > 0.8)
    if 'v_c_ratio' in metrics.columns:
        metrics['congested'] = metrics['v_c_ratio'] > 0.8

    return metrics


def analyze_transit_accessibility(
    transit_stops: gpd.GeoDataFrame,
    population_grid: gpd.GeoDataFrame,
    buffer_distance: float = 400  # meters
) -> Dict[str, float]:
    """
    Analyze transit accessibility for population.

    Parameters:
    -----------
    transit_stops : geopandas.GeoDataFrame
        Transit stop locations
    population_grid : geopandas.GeoDataFrame
        Population distribution grid
    buffer_distance : float
        Walking distance buffer (meters)

    Returns:
    --------
    dict
        Accessibility metrics (coverage, avg_distance, service_gaps)
    """
    # Buffer transit stops
    transit_buffers = transit_stops.buffer(buffer_distance)
    service_area = transit_buffers.unary_union

    # Calculate population coverage
    population_grid['within_service'] = population_grid.geometry.within(service_area)
    total_pop = population_grid['population'].sum()
    served_pop = population_grid[population_grid['within_service']]['population'].sum()

    # Calculate metrics
    metrics = {
        'coverage_rate': served_pop / total_pop if total_pop > 0 else 0,
        'total_population': total_pop,
        'served_population': served_pop,
        'n_stops': len(transit_stops),
        'avg_stop_spacing': calculate_avg_stop_spacing(transit_stops)
    }

    return metrics


def compute_commute_patterns(
    origin_dest_matrix: pd.DataFrame,
    road_network: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Analyze commute patterns from origin-destination data.

    Parameters:
    -----------
    origin_dest_matrix : pandas.DataFrame
        Origin-destination flow matrix
    road_network : geopandas.GeoDataFrame
        Road network for routing

    Returns:
    --------
    pandas.DataFrame
        Commute statistics (avg_distance, avg_time, mode_share)
    """
    commute_stats = pd.DataFrame()

    # Calculate average commute distance
    if 'distance' in origin_dest_matrix.columns:
        commute_stats['avg_distance'] = origin_dest_matrix.groupby('origin')['distance'].mean()

    # Calculate average commute time
    if 'travel_time' in origin_dest_matrix.columns:
        commute_stats['avg_time'] = origin_dest_matrix.groupby('origin')['travel_time'].mean()

    # Calculate mode share
    if 'mode' in origin_dest_matrix.columns:
        mode_counts = origin_dest_matrix.groupby(['origin', 'mode']).size().unstack(fill_value=0)
        mode_share = mode_counts.div(mode_counts.sum(axis=1), axis=0)
        commute_stats = commute_stats.join(mode_share, rsuffix='_share')

    return commute_stats


def calculate_avg_stop_spacing(transit_stops: gpd.GeoDataFrame) -> float:
    """
    Calculate average spacing between transit stops.

    Parameters:
    -----------
    transit_stops : geopandas.GeoDataFrame
        Transit stop locations

    Returns:
    --------
    float
        Average stop spacing (meters)
    """
    if len(transit_stops) < 2:
        return 0.0

    # Group by route and calculate spacing
    spacings = []
    for route_id in transit_stops['route_id'].unique():
        route_stops = transit_stops[transit_stops['route_id'] == route_id].sort_values('stop_sequence')
        coords = [(p.x, p.y) for p in route_stops.geometry]

        for i in range(len(coords) - 1):
            dist = Point(coords[i]).distance(Point(coords[i + 1]))
            spacings.append(dist)

    return np.mean(spacings) if spacings else 0.0


def identify_service_gaps(
    transit_coverage: gpd.GeoDataFrame,
    demand_points: gpd.GeoDataFrame,
    threshold_distance: float = 800  # meters
) -> gpd.GeoDataFrame:
    """
    Identify areas with high demand but low transit service.

    Parameters:
    -----------
    transit_coverage : geopandas.GeoDataFrame
        Areas covered by transit
    demand_points : geopandas.GeoDataFrame
        Population/employment centers
    threshold_distance : float
        Maximum acceptable distance to transit (meters)

    Returns:
    --------
    geopandas.GeoDataFrame
        Service gap locations with demand intensity
    """
    # Find demand points outside service area
    demand_points['in_service'] = demand_points.geometry.within(
        transit_coverage.unary_union.buffer(threshold_distance)
    )

    service_gaps = demand_points[~demand_points['in_service']].copy()

    # Rank by demand intensity
    if 'demand' in service_gaps.columns:
        service_gaps['priority'] = service_gaps['demand'].rank(ascending=False)

    return service_gaps
