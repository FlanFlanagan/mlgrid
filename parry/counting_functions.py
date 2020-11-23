import geopandas as gp
import pandas as pd
import numpy as np

from shapely.strtree import STRtree
from shapely.geometry import LineString
import shapely.ops as ops

def detector_counts(gdf_d, gdf_s, buff, source, mu=[0], polys=None):
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
    interaction_array = []
    for index_s in gdf_s.index:
        cols = ['detID', 'sourceID', 'counts', 'dist', 'dist2']
        spoint = gdf_s.at[index_s, 'geometry']
        geometries = np.append(gdf_d.geometry.values, gdf_s.geometry.values)
        tree=STRtree(geometries)
        buff1 = buff/111180
        d_list = tree.query(spoint.buffer(buff1))
        for d in d_list:
            if d in gdf_d['geometry']:
                index = gdf_d[gdf_d['geometry']==d].index
                for index_d in index:
                    time = gdf_s.at[index_s, 'time']
                    if polys is None:
                        dist = spoint.distance(d)*111180
                        #val = tally_counts(dist, mu[0], source)
                        #gdf_d.at[index_d, 'count'] += val
                        #gdf_s.at[index_s, 'count'] += val
                        interaction_array.append([index_d, index_s, 0, dist, 0.0])
                    else:
                        dist = attenu_lengths(gdf_d, gdf_s, index_d, index_s, polys)
                        #val = tally_counts_buildings(dist, mu, source)
                        #gdf_d.at[index_d, 'count'] += val
                        #gdf_s.at[index_s, 'count'] += val
                        interaction_array.append([index_d, index_s, 0, dist[0], dist[1]])
        pd1 = pd.DataFrame(interaction_array, columns=cols)
    return pd1

def detector_counts_time(gdf_d, gdf_s, buff, source, mu=[0], polys=None):
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
    for index_s in range(len(gdf_s)):
        cols = ['detID', 'sourceID', 'counts', 'dist', 'dist2']
        interaction_array = []
        spoint = gdf_s.at[index_s, 'geometry']
        time = gdf_s.at[index_s, 'time']
        gdf_t=gdf_d[gdf_d['time'] == time]
        geometries = np.append(gdf_t.geometry.values, gdf_s.geometry.values)
        tree=STRtree(geometries)
        buff1 = buff/111180
        d_list = tree.query(spoint.buffer(buff1))
        for d in d_list:
            if d in gdf_t['geometry']:
                index = gdf_t[gdf_t['geometry']==d].index
                for index_d in index:
                    if polys is None:
                        dist = spoint.distance(d)*111180
                        val = tally_counts(dist, mu[0], source)
                        gdf_d.at[index_d, 'count'] += val
                        gdf_s.at[index_s, 'count'] += val
                        interaction_array.append([index_d, index_s, val, dist, 0.0])
                    else:
                        dist = attenu_lengths(gdf_d, gdf_s, index_d, index_s, polys)
                        val = tally_counts_buildings(dist, mu, source)
                        gdf_d.at[index_d, 'count'] += val
                        gdf_s.at[index_s, 'count'] += val
                        interaction_array.append([index_d, index_s, val, dist[0], dist[1]])
        pd1 = pd.DataFrame(interaction_array, columns=cols)
    return pd1

def total_counts_front(gdf, dt):
    '''
    This function is used to calculate the total number of counts detected 
    from a mobile source. The sum of each detector used in the simulation
    is incorperated into the total.    
    Parameters
    ----------
    gdf: GeoDataFrame
        The geodataframe containing the information for the mobile and 
        or stationary detectors. 
    dt: float
        Delta time step. 
    Returns
    -------
    sum_counts: float
        The total number of counts detected of a source. 
    '''
    sum_counts = 0
    routes = gdf.route.unique()
    for r in routes:
        temp = gdf[gdf['route'] == r]
        temp = temp.sort_values(by='time')
        indexes = temp.index
        rang = len(indexes)-1
        i = 0
        while i < len(indexes)-1:
            t = temp.at[indexes[i+1], 'time']-temp.at[indexes[i], 'time']
            if t < dt:
                sum_counts += temp.at[indexes[i], 'count']*dt
            else:
                sum_counts += temp.at[indexes[i], 'count']*dt
            i += 1
    return sum_counts

def total_counts_center(gdf, dt):
    '''
    This function is used to calculate the total number of counts detected 
    from a mobile source. The sum of each detector used in the simulation
    is incorperated into the total.    
    Parameters
    ----------
    gdf: GeoDataFrame
        The geodataframe containing the information for the mobile and 
        or stationary detectors. 
    dt: float
        Delta time step. 
    Returns
    -------
    sum_counts: float
        The total number of counts detected of a source. 
    '''
    sum_counts = 0
    routes = gdf.route.unique()
    for r in routes:
        temp = gdf[gdf['route'] == r]
        temp = temp.sort_values(by='time')
        indexes = temp.index
        rang = len(indexes)-1
        i = 0
        while i < len(indexes)-1:
            t = temp.at[indexes[i+1], 'time']-temp.at[indexes[i], 'time']
            if t < dt:
                sum_counts += (temp.at[indexes[i], 'count']+0)/2*dt
            else:
                sum_counts += (temp.at[indexes[i], 'count']+temp.at[indexes[i+1], 'count'])/2*dt
            i += 1
    return sum_counts

def building_check(gdf_d, gdf_s, index_d, index_s, polys):
    '''
    This function determines if there are any buildings between two
    points.
    ----------
    gdf_d: GeoDataFrame
        The geodataframe containing the information for the mobile and 
        or stationary detectors. 
    gdf_d: GeoDataFrame
        The geodataframe containing the information for the mobile and 
        or stationary detectors. 
    index_d: int
        index of the detector in gdf_d 
    index_s: int
        index of the source in gdf_s
    polys: GeoDataFrame
        The geodataframe containing the polygons that represent any
        obstructions to the detector/source interaction.          
    Returns
    -------
    temp: Bool
        Boolean if there is a building or not. 
    '''
    d = gdf_d.at[index_d, 'geometry'].coords[:][0]
    s = gdf_s.at[index_s, 'geometry'].coords[:][0]
    line_t = LineString((d, s))
    lines_df = gp.GeoDataFrame(geometry = [line_t])
    result = gp.sjoin(polys, lines_df, op='intersects')
    temp = True
    if len(result) == 0:
        temp = False
    return temp

def attenu_lengths(gdf_d, gdf_s, index_d, index_s, polys):
    '''
    This function calculates the air attenuation and building
    attenuation distances for a source to detector ray.   
    Parameters
    ----------
    gdf_d: GeoDataFrame
        The geodataframe containing the information for the mobile and 
        or stationary detectors. 
    gdf_d: GeoDataFrame
        The geodataframe containing the information for the mobile and 
        or stationary detectors. 
    index_d: int
        index of the detector in gdf_d 
    index_s: int
        index of the source in gdf_s
    polys: GeoDataFrame
        The geodataframe containing the polygons that represent any
        obstructions to the detector/source interaction.          
    Returns
    -------
    array: array
        The attenuation distances (air and building) 
    '''
    d = gdf_d.at[index_d, 'geometry'].coords[:][0]
    s = gdf_s.at[index_s, 'geometry'].coords[:][0]
    line_t = LineString((d, s))
    lines_df = gp.GeoDataFrame(geometry = [line_t])
    result = gp.sjoin(polys, lines_df, op='intersects')
    d2 = 0
    for poly_t in result.geometry:
        temp = ops.split(line_t, poly_t)
        for line in temp:
            if line.within(poly_t):
                d2 += line.length
    d1 = line_t.length-d2
    return [d1*111180, d2*111180]

def tally_counts(d, mu, source):
    '''
    This functions attenuates the source strength based on
    distance from the detector.     
    Parameters
    ----------
    d: float
        distance from the source to the detector  
    mu: float
        linear attenuation coefficient of the medium 
        between the source and the detector. 
    source: float
        the strength of the source         
    Returns
    -------
    strength: float
        Determines the count rate calculated by the detector.  
    '''
    d = d*100
    if d <= 0.:
        return source
    else:
        return source/(4.*np.pi*d**2.)*np.exp(-mu*d)

def tally_counts_buildings(d, mu, source):
    '''
    This functions attenuates the source strength based on
    distance from the detector and accounts for buildings.     
    Parameters
    ----------
    d: array
        array of the air attenuation length and the building attenuation
        length  
    mu: array
        an array composed of the different attenuation coefficients for
        'air' and buildings 
    source: float
        the strength of the source         
    Returns
    -------
    strength: float
        Determines the count rate calculated by the detector.  
    '''
    d_tot = sum(d)
    return source/(4.*np.pi*d_tot**2.)*np.exp(-mu[0]*d[0]-mu[1]*d[1])

'''
def detector_counts(gdf_d, gdf_s, buff, source, mu, polys=None): 

    THIS FUNCTION IS CURRENTLY UNDER
    This function tallies the total counts each detector sees at each point
    in time. It does so by finding all detectors in range of the source point
    and then determining which are in the same timestep. 

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
      
    geometries = np.append(gdf_d.geometry.values, gdf_s.geometry.values)
    tree=STRtree(geometries)
    buff1 = buff/111180
    for index_s in range(len(gdf_s.geometry)):
        spoint = gdf_s.geometry[index_s]        
        d_list = tree.query(spoint.buffer(buff1))
        for d in d_list:
            if d in gdf_d['geometry']:
                index_d = -1
                index1 = gdf_d[gdf_d['geometry'] == d].index
                for i in index1:
                    if gdf_s.at[index_s, 'time'] == gdf_d.at[i, 'time']:
                        index_d = i
                if index_d != -1: 
                    if polys == None:             
                        dist = spoint.distance(d)
                        val = source/(4.*np.pi*dist**2.)*np.exp(-mu*dist)
                        gdf_d.at[index_d, 'count'] += val
                        gdf_s.at[index_s, 'count'] += val
                    else:
                        line = LineString(spoint, d)
                        lines = ops.split(line, poly)
                        if len(lines) > 1:
                            val = 0
                            gdf_d.at[index_d, 'count'] += val
                            gdf_s.at[index_s, 'count'] += val
    return (gdf_d)
'''

