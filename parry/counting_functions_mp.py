import geopandas as gp
import pandas as pd
import numpy as np
import multiprocessing

from shapely.strtree import STRtree
from shapely.geometry import LineString

import shapely.ops as ops
import parry.counting_functions as cf


def detector_counts_mp(gdf_d, gdf_s, buff, source, cores, mu=[0], polys=None):
    '''
    This function tallies the total counts each detector sees at each point
    in time. It does so by finding all detectors with the same timestep as
    the source, and then finding all detectors within range. 
    Parameters
    ----------
    gdf_d: GeoDataFrame
        The geodataframe containing the information for the mobile and 
        or stationary detectors. 
    gdf_s: GeoDataFrame
        The geodataframe containing the information for the mobile and 
        or stationary sources. 
    buff: float
        The maximum detection distance
    source: float
        The strength of the mobile source
    cores: int
        The number of cores to be used to calculate the interactions. 
    Returns
    -------
    gdf_d: GeoDataFrame
        The geodataframe containing the information for the mobile and 
        or stationary detectors updated with counts. 
    '''
    gdfs = []
    pool = multiprocessing.Pool(cores)
    ind = np.arange(0, len(gdf_d)+1, int(len(gdf_d)/cores))
    for i in range(1, len(ind)):
        gdfs.append(gdf_d.iloc[np.arange(ind[i-1], ind[i], 1)])
    iters = []
    for i in gdfs:
        iters.append((i, gdf_s, buff, source, mu, polys))
    p_array = pool.starmap(cf.detector_counts, iters)
    pd1 = pd.concat(p_array, ignore_index=True)
    return pd1

def detector_counts_time_mp(gdf_d, gdf_s, buff, source, cores, mu=[0], polys=None):
    '''
    This function tallies the total counts each detector sees at each point
    in time. It does so by finding all detectors with the same timestep as
    the source, and then finding all detectors within range. 
    Parameters
    ----------
    gdf_d: GeoDataFrame
        The geodataframe containing the information for the mobile and 
        or stationary detectors. 
    gdf_s: GeoDataFrame
        The geodataframe containing the information for the mobile and 
        or stationary sources. 
    buff: float
        The maximum detection distance
    source: float
        The strength of the mobile source
    Returns
    -------
    gdf_d: GeoDataFrame
        The geodataframe containing the information for the mobile and 
        or stationary detectors updated with counts. 
    '''
    gdfs = []
    pool = multiprocessing.Pool(cores)
    ind = np.arange(0, len(gdf_d)+1, int(len(gdf_d)/cores))
    for i in range(1, len(ind)):
        gdfs.append(gdf_d.iloc[np.arange(ind[i-1], ind[i], 1)])
    iters = []    
    for i in gdfs:
        iters.append((i, gdf_s, buff, source, mu, polys))
    p_array = pool.starmap(cf.detector_counts_time, iters)
    pd1 = pd.concat(p_array, ignore_index=True)
    return pd1

