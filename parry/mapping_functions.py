import matplotlib.pyplot as plt
import numpy as np
import shapely.ops
import geopandas as gp
import pandas as pd
import geoplot.crs as gcrs
import geoplot as gplt
import multiprocessing
from shapely.geometry import LineString, Polygon
from pyroutelib3 import Router
from fiona.crs import from_epsg
import random
from datetime import datetime, timedelta


def lat_to_m(lat):
    '''
    This function that determines the conversion of latitude to
    meters.  
    Parameters
    ----------
    lat: float
        The latitude for conversion
    Returns
    -------
    The latitude to meters conversion rate
    '''
    return 111132.92-559.82*np.cos(2*lat)+1.175*np.cos(4*lat)

def long_to_m(lon):
    '''
    This function that determines the conversion of longitude to
    meters. 
    Parameters
    ----------
    lon: float
        The longitude for conversion
    Returns
    -------
    The longitude to meters conversion rate
    '''
    return 111412.84*np.cos(lon)-93.5*np.cos(3*lon)

def find_route(start_lat, start_long, end_lat, end_long, transport_type='car'):
    '''
    This function converts the lat/long distance into
    a meters distance.  
    Parameters
    ----------
    start_lat: float
        The start latitude
    start_long: float
        The end latitude
    end_lat: float
        The start longitude
    end_long: float
        The end longitude
    transport_type: string
        The type of transport (default is car)
    Returns
    -------
    route_latlongs: arraylike
        array of Points representing the vertices of the route. 
    '''
    router = Router(transport_type)
    start = router.findNode(start_lat, start_long)
    end = router.findNode(end_lat, end_long)
    try:
        status, route = router.doRoute(start, end)
    except:
        return None
    if status == 'success':
        route_latlongs = list(map(router.nodeLatLon, route))  # Get actual route coordinates
        return route_latlongs

def build_source_route(cords, t1, v, step_size, route=0):
    '''
    This function converts the lat/long distance into
    a meters distance.  
    Parameters
    ----------
    cords: array-like
        An array of the starting coordinates of the shape
        [lat1, long1, lat2, long2]
    t1: float
        The start time of the source route
    v: float
        Speed of the mobile source
    step_size: float
        The stepsize of the simulation in seconds.
    Returns
    -------
    gdf2: GeoDataFrame
        GeoDataFrame of containing the Points representing the
        movement of the source. 
    '''
    columns = ['route', 'geometry', 'time', 'date', 'type', 'count']
    source_coords = find_route(cords[0], cords[1], cords[2], cords[3])
    source_path = LineString(source_coords)
    if source_path is None:
        return None
    source_path = shapely.ops.transform(lambda x, y: (y, x), source_path)
    sources = []
    d = source_path.length*102385.08130853
    t2 = t1 + d/v
    for t in np.arange(t1,t2,step_size):
        d = (t-t1)*v
        sources.append([route, source_path.interpolate(d/102385.08130853), int(t), '2018-12-03', 'S', 0])
    source_df = pd.DataFrame(sources, columns=columns)
    gdf2 = gp.GeoDataFrame(source_df)
    return gdf2

    
def haversin_dist(lat1, lat2, lon1, lon2):
    '''
    This function converts the lat/long distance into
    a meters distance.  
    Parameters
    ----------
    lat1: float
        The start latitude
    lat2: float
        The end latitude
    lon1: float
        The start longitude
    lon2: float
        The end longitude
    Returns
    -------
    d: float
        Distance in meters
    '''
    radius = 6371 # km
    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) \
        * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = radius * c * 1000
    return d

def ray_dist(x1, x2, y1, y2):
    '''
    This function converts the lat/long distance into
    a meters distance.  
    Parameters
    ----------
    lat1: float
        The start latitude
    lat2: float
        The end latitude
    lon1: float
        The start longitude
    lon2: float
        The end longitude
    Returns
    -------
    d: float
        Distance in meters
    '''
    radius = 6371 # km
    dx = x2-x1
    dy = y2-y1
    d = np.sqrt(dx*2 + (dy**2))
    return d


def chop_geodataframe(frame, long_min, lat_min, long_max, lat_max):
    '''
    This function retrieves a subset of the dataframe passed, using
    the lat/longs.  
    Parameters
    ----------
    frame: GeoDataFrame
        DataFrame the subset is extracted from. 
    long_min: float
        The minimum longitude of the space extracted
    lat_min: float
        The minimum latitude of the space extracted
    long_max: float
        The maximum longitude of the space extracted
    lat_max: float
        The maximum latitude of the space extracted
    Returns
    -------
    frame: GeoDataFrame
        The chopped section of the original GeoDataFrame 
    '''
    gdf = gp.read_file(frame)
    gdf = gdf.cx[long_min:long_max,lat_min:lat_max]
    return gdf

def plot_point_plot(pointgeo, ax):
    gplt.pointplot(test1, ax=ax, scale='count', hue='count', limits=(1,20), cmap='Reds')

