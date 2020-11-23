import random
import matplotlib.pyplot as plt
import numpy as np
import shapely.ops
import geopandas as gp
import pandas as pd
import multiprocessing

import parry.mapping_functions as pmf

from fiona.crs import from_epsg
from shapely.geometry import LineString, Polygon, Point

def polygon_centroid_to_point(gdf):
    '''
    This function returns points representing the centroid
    of all polygons in GeoDataFrame gdf.  
    Parameters
    ----------
        A Geodataframe consisting of polygons.
    gdf: GeoDataFrame

    Returns
    -------
    points: arraylike
        An array of shapely Points 
    '''
    points = []
    for i in gdf.index:
        p = gdf['geometry'][i]
        points.append(Point(p.centroid.coords.xy[0][0], p.centroid.coords.xy[1][0]))
    return points
        
       
def build_lines_from_point(points, r, rays, lat_flag=True):
    '''
    Generates outward rays from the point passed to
    this function.     
    Parameters
    ----------
    point: Shapely Point
        A shapely point that will have rays cast out.
    r: Float
        The length of the rays cast out from the point.
    rays: Int
        The number of rays to be cast from the point.
    lines: Array Like
        An array used to contain the the lines generated from
        the function.
    '''
    temp_lines = []
    for point in points:
        lon, lat = point.coords[0][0], point.coords[0][1]
        if lat == True:
            lat_m = 111045 #meters per latitude
            long_m = np.cos(lat) * 111317
        else:
            lat_m, long_m = 1., 1.
        end_points = []
        for i in range(rays):
            angle = 2*np.pi*(i/rays)
            lat_t = lat + r*np.sin(angle)/lat_m
            long_t = lon + r*np.cos(angle)/long_m
            end_points.append((long_t, lat_t))
        for end in end_points:
            temp_lines.append(LineString([(lon, lat), end]))
    lines = gp.GeoDataFrame(geometry=temp_lines)
    return lines


def line_file(points):
    '''
    Builds an output shapefile for a GeoDataFrame consisting
    of LineStrings    
    Parameters
    ----------
    points: Array Like
        An array of Shapely Points that is used to generate
        the rays of the lines. 
    '''
    lines = []
    for point in points:
        build_lines_from_point(point, 10, 72, lines)

    schema = {
        'geometry': 'LineString',
        'properties': {'id': 'int'},
    }

    # Write a new Shapefile
    with fiona.open('my_lines.shp', 'w', 'ESRI Shapefile', schema) as c:
        ## If there are multiple geometries, put the "for" loop here
        for line in lines:
            i = 0
            c.write({
                'geometry': mapping(line),
                'properties': {'id': i},
            })
            i+=1

def build_bins(xbins, ybins, long_min, lat_min, long_max, lat_max):
    '''
    This function builds a shapefile of evenly spaced polygons
    fitted to the limits of lat/long. The number of polygons on
    each side is set by xbins/ybins.  
    Parameters
    ----------
    xbins: Int
        Number of bins to break up the x-dimension into.
    ybins: Int
        Number of bins to break up the y-dimension into.
    long_min: float
        The minimum longitude of the space being binned
    lat_min: float
        The minimum latitude of the space being binned
    long_max: float
        The maximum longitude of the space being binned
    lat_max: float
        The maximum latitude of the space being binned
    '''
    x = np.linspace(long_min, long_max, xbins+1)
    y = np.linspace(lat_min, lat_max, ybins+1)
    polygons = []
    for i in range(xbins):
        for j in range(ybins):
            poly = Polygon([(x[i], y[j]), (x[i], y[j+1]), (x[i+1], y[j+1]), (x[i+1], y[j])])
            polygons.append(poly)
    cols = ['index', 'geometry', 'count']
    grid = gp.GeoDataFrame(geometry=polygons, columns=cols)
    return grid


def write_bins(polys):
    '''
    This function writes a shapefile of evenly spaced polygons
    fitted to the limits of lat/long. The number of polygons on
    each side is set by xbins/ybins.  
    Parameters
    ----------
    polys: array
        Array of shapely polygons to be written to file. 
    '''
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int', 'count': 'float'},
    }
    # Write a new Shapefile
    with fiona.open('bins.shp', 'w', 'ESRI Shapefile', schema) as c:
        ## If there are multiple geometries, put the "for" loop here
        i = 0
        for poly in polygons:
            c.write({
                'geometry': mapping(poly),
                'properties': {'id': i, 'count': 0.0},
            })
            i+=1

def place_high_vis_detectors(gdf, r):
    '''
    Determines the location of the highest visibility detector 
    locations that are at least 'r' distance from each other.

    Parameters
    ----------
    gdf: GeoDataFrame
        A geodataframe containing a 'count' column. 
    r: float
        Minimum distance between stationary detector points. 

    Returns
    -------
    test: GeoDataFrame
        A GeoDataFrame containing the set of possible high visibility
        detector locations. 
    '''
    temp = gdf.sort_values(by='count', ascending=False)
    ind = temp.index
    for i in ind:
        if i in temp.index:
            for j in temp.index:
                if i == j:
                    continue
                if temp.at[i, 'geometry'].distance(temp.at[j, 'geometry']) < r:
                    if j in temp.index:
                        temp.drop(j, inplace=True)
    points = polygon_centroid_to_point(temp)
    temp['geometry'] = points
    return temp

def place_low_vis_detectors():
    '''
    Determines the location of the lowest visibility detector 
    locations that are at least 'r' distance from each other.
    Parameters
    ----------
    gdf: GeoDataFrame
        A geodataframe containing a 'count' column. 
    r: float
        Minimum distance between stationary detector points. 

    Returns
    -------
    test: GeoDataFrame
        A GeoDataFrame containing the set of possible low visibility
        detector locations. 
    '''
    temp = gdf.sort_values(by='count')
    ind = temp.index
    for i in ind:
        if i in temp.index:
            for j in temp.index:
                if i == j:
                    continue
                if temp.at[i, 'geometry'].distance(temp.at[j, 'geometry']) < r:
                    if j in temp.index:
                        temp.drop(j, inplace=True)
    return temp

def accumloop(g, i, ray_list, lat_flag):
    count = 0
    for ray in ray_list:
        if ray.geom_type is 'LineString':
            # if ray origin == bin centroid, don't accumulate.
            if g.centroid.coords.xy[0][0] == ray.coords.xy[0][0] and g.centroid.coords.xy[1][0] == ray.coords.xy[1][0]:
                I = 0.0
            else:
                if lat_flag is True:
                    d = pmf.haversin_dist(g.centroid.coords.xy[0][0], ray.coords.xy[0][0],
                                      g.centroid.coords.xy[1][0], ray.coords.xy[1][0])
                    I = 1/(np.pi*4*d**2)
                else:
                    d = pmf.ray_dist(g.centroid.coords.xy[0][0], ray.coords.xy[0][0],
                                      g.centroid.coords.xy[1][0], ray.coords.xy[1][0])
                    I=1/(np.pi*4*d**2)
            count += I
    c = (i, count)    
    return c


def accumulate_counts(strtree, grid, processes, lat_flag=True):
    iters = []   
    for i in grid.index:
        g = grid.at[i,'geometry']
        ray_list = strtree.query(g) 
        iters.append((g, i, ray_list, lat_flag))
    pool = multiprocessing.Pool(processes)
    p_array = pool.starmap(accumloop, iters)
    pool.close()
    for i in p_array:
        grid.at[i[0], 'count'] = i[1]


def polygeo_to_pointgeo(polygeo, background=1, cutoff=0):
    cols = ['index', 'xbkg', 'geometry']
    arrays = []
    for i in polygeo.index:
        counts = float(polygeo.at[i, 'count'])
        if counts/background < cutoff:
            continue
        poly = polygeo.at[i, 'geometry']
        point = Point(poly.centroid.coords.xy[0][0], poly.centroid.coords.xy[1][0])
        arrays.append([counts/background, point])
    pointgeo = gp.GeoDataFrame(arrays, columns = cols)
    return pointgeo


def polygeo_to_pointgeo_noct(polygeo):
    cols = ['xbkg', 'geometry']
    arrays = []
    for i in polygeo.index:
        poly = polygeo.at[i, 'geometry']
        point = Point(poly.centroid.coords.xy[0][0], poly.centroid.coords.xy[1][0])
        arrays.append([0.0, point])
    pointgeo = gp.GeoDataFrame(arrays, columns = cols)
    return pointgeo


def grid_shift(gdf, scale):
    cols = ['count', 'geometry']
    v = []
    for i in range(len(gdf)):
        g = gdf.at[i, 'geometry']
        x = g.centroid.coords.xy[0][0]
        y = g.centroid.coords.xy[0][1]
        x += (-0.5+random.uniform()) * scale
        y += (-0.5+random.uniform()) * scale
        p = Point(x,y)
        v.append([0.0, p])
    gdf_n = gp.GeoDataFrame(v, columns=cols)
    return gdf_n


def grid_to_closest_high_vis(gdf, hv):
    v = []
    for i in range(len(gdf)):
        p = gdf.at[i, 'geometry']
        min_d = [1e299, None]
        for j in range(len(hv)):
            d = p.distance(hv.at[j, 'geometry'])
            if d < min_d[0]:
                min_d[0] = d
                min_d[1] = hv.at[j, 'geometry']
        v.append([i, 0.0, min_d[1]])
    cols = ['index', 'count', 'geometry']
    gdf_n = gp.GeoDataFrame(v, columns=cols)
    return gdf_n
