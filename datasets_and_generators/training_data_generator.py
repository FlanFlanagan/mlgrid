from multiprocessing.dummy import Process, freeze_support

import pandas as pd
import geopandas as gp
import numpy as np
import geoplot as gplt
import random
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry.point import Point
import parry.binning_functions as pbf
from shapely.strtree import STRtree
import mapclassify as mc
import json
import copy
import os

#shape definitions

def gen_poly1(x1, x2, x3, y1, y2, y3):  #typical intersection with rectangles
    poly1 = Polygon(((0, 0), (0, y1), (x1, y1), (x1, 0)))
    poly2 = Polygon(((0, y1+y2), (0, y1+y2+y3), (x1, y1+y2+y3), (x1, y1+y2)))
    poly3 = Polygon(((x1+x2, 0), (x1+x2, y1), (x1+x2+x3, y1), (x1+x2+x3, 0)))
    poly4 = Polygon(((x1+x2, y1+y2), (x1+x2, y1+y2+y3), (x1+x2+x3, y1+y2+y3), (x1+x2+x3, y1+y2)))
    polys = [poly1, poly2, poly3, poly4]
    return gp.GeoDataFrame(geometry=polys)

def gen_poly2(x1, x2, x3, y1, y2, y3):   #triangles with wall
    poly1 = Polygon(((0, 0), (0, y3), (x1, 0)))
    poly2 = Polygon(((x1+x2, y3), (x1+x2+x3, y3), (x1+x2+x3, 0)))
    poly3 = Polygon(((0, y1+y2), (0, y1+y2+y3), (x1+x2+x3, y1+y2+y3), (x1+x2+x3, y1+y2)))
    polys = [poly1, poly2, poly3]
    return gp.GeoDataFrame(geometry=polys)

def gen_poly3(x1, x2, x3, y1, y2, y3):  #roundabout
    poly1 = Polygon(((0, 0), (0, (y1+y2+y3)/2-y2), ((x1+x2+x3)/2-y2*3, (y1+y2+y3)/2-y2), ((x1+x2+x3)/2-x2, (y1+y2+y3)/2-y2*3), ((x1+x2+x3)/2-x2, 0)))
    poly2 = Polygon(((0, (y1+y2+y3)/2+y2), (0, y1+y2+y3), ((x1+x2+x3)/2-x2, y1+y2+y3), ((x1+x2+x3)/2-x2, (y1+y2+y3)/2+y2*3), ((x1+x2+x3)/2-y2*3, (y1+y2+y3)/2+y2)))
    poly3 = Polygon((((x1+x2+x3)/2+x2, 0), ((x1+x2+x3)/2+x2, (y1+y2+y3)/2-y2*3), ((x1+x2+x3)/2+y2*3, (y1+y2+y3)/2-y2), (x1+x2+x3, (y1+y2+y3)/2-y2), (x1+x2+x3, 0)))
    poly4 = Polygon((((x1+x2+x3)/2+y2*3, (y1+y2+y3)/2+y2), ((x1+x2+x3)/2+x2, (y1+y2+y3)/2+y2*3), ((x1+x2+x3)/2+x2, y1+y2+y3), (x1+x2+x3, y1+y2+y3), (x1+x2+x3, (y1+y2+y3)/2+y2)))
    p = Point((x1+x2+x3)/2, (y1+y2+y3)/2)
    circle = p.buffer(y2)
    polyc = list(circle.exterior.coords)
    poly5 = Polygon(polyc)
    polys = [poly1, poly2, poly3, poly4, poly5]
    return gp.GeoDataFrame(geometry=polys)

def gen_poly4(x1, x2, x3, y1, y2, y3):  #many buildings
    n = 1
    poly1 = Polygon(((0, 0), (0, y1-y2*n), (x1-x2*n, y1-y2*n), (x1-x2*n, 0)))
    poly2 = Polygon(((x1+x2, 0), (x1+x3-x2*n, 0), (x1+x3-x2*n, y1-y2*n), (x1+x2, y1-y2*n)))
    poly3 = Polygon(((x1+x3, 0), (x1+x3, y1-y2*n), (x1+x2+x3, y1-y2*n), (x1+x2+x3, 0)))
    poly4 = Polygon(((0, y1+y2), (x1-x2*n, y1+y2), (x1-x2*n, y1+y3-y2*n), (0, y1+y3-y2*n)))
    poly5 = Polygon(((x1+x2, y1+y2), (x1+x3-x2*n, y1+y2), (x1+x3-x2*n, y1+y3-y2*n), (x1+x2, y1+y3-y2*n)))
    poly6 = Polygon(((x1+x3, y1+y2), (x1+x2+x3, y1+y2), (x1+x2+x3, y1+y3-y2*n), (x1+x3, y1+y3-y2*n)))
    poly7 = Polygon(((0, y1+y3), (0, y1+y2+y3), (x1-x2*n, y1+y2+y3), (x1-x2*n, y1+y3)))
    poly8 = Polygon(((x1+x2, y1+y3), (x1+x3-x2*n, y1+y3), (x1+x3-x2*n, y1+y2+y3), (x1+x2, y1+y2+y3)))
    poly9 = Polygon(((x1+x3, y1+y3), (x1+x3, y1+y2+y3), (x1+x2+x3, y1+y2+y3), (x1+x2+x3, y1+y3)))
    polys = [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8, poly9]
    return gp.GeoDataFrame(geometry=polys)

def gen_poly5(x1, x2, x3, y1, y2, y3): #roads with open space (possibly for parking lot)
    poly1 = Polygon(((0, 0), (x2*3, 0), (x2*3, y1+y2), (0, y1+y2)))
    poly2 = Polygon(((x2*3, 0), (x2*3, y1+y2), (x2*3+x2*2, y1+y2)))
    poly3 = Polygon(((x2*3+x2*1.5, 0), (x2*3+x2*3.5, 0), (x2*3+x2*3.5, y1+y2)))
    poly4 = Polygon(((x2*3+x2*3.5, y1+y2), (x1+x3-x2*3, y1+y2), (x1+x3-x2*3, 0), (x2*3+x2*3.5, 0)))
    poly5 = Polygon(((x1+x3, y1+y2), (x1+x3+x2, y1+y2), (x1+x3+x2, 0), (x1+x3, 0)))
    poly6 = Polygon(((0, y1+y3-y2*3), (x2*3+x2*2, y1+y3-y2*3), (x2*3+x2*2, y1+y3-y2), (0, y1+y3-y2)))
    poly7 = Polygon(((x1+x3+x2, y1+y3-y2*3), (x1+x3, y1+y3-y2*3), (x1+x3, y1+y3-y2), (x1+x3+x2, y1+y3-y2)))
    poly8 = Polygon(((0, y1+y3), (x1+x3-x2*3, y1+y3), (x1+x3-x2*3, y1+y2+y3), (0, y1+y2+y3)))
    poly9 = Polygon(((x1+x3, y1+y3), (x1+x3, y1+y2+y3), (x1+x2+x3, y1+y2+y3), (x1+x2+x3, y1+y3)))
    polys = [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8, poly9]
    return gp.GeoDataFrame(geometry=polys)

def gen_poly6(x1, x2, x3, y1, y2, y3):
    poly1 = Polygon(((0, 0), (0, y1-y2*15.5+y3), ((x1+x2+x3)/2-x2, y1-y2*15.5+y3), ((x1+x2+x3)/2-x2, 0)))
    poly2 = Polygon((((x1+x2+x3)/2+x2, 0), ((x1+x2+x3)/2+x2, y1-y2*15.5+y3), (x1+x2+x3, y1-y2*15.5+y3), (x1+x2+x3, 0)))
    poly3 = Polygon(((0, y1-y2*13+y3), (0, y1-y2*10+y3), (x1/2.5, y1-y2*10+y3), (x1/2.5, y1-y2*13+y3)))
    poly4 = Polygon(((x1+x2+x3*0.6, y1-y2*13+y3), (x1+x2+x3*0.6, y1-y2*10+y3), (x1+x2+x3, y1-y2*10+y3), (x1+x2+x3, y1-y2*13+y3)))
    poly5 = Polygon(((0, y1-y2*7.5+y3), (0, y1-y2*4.5+y3), (x1/2.5, y1-y2*4.5+y3), (x1/2.5, y1-y2*7.5+y3)))
    poly6 = Polygon(((x1+x2+x3*0.6, y1-y2*7.5+y3), (x1+x2+x3*0.6, y1-y2*4.5+y3), (x1+x3+x2, y1-y2*4.5+y3), (x1+x2+x3, y1-y2*7.5+y3)))
    poly7 = Polygon(((0, y1-y2*2+y3), (0, y1+y2+y3), ((x1+x2+x3)/2-x2, y1+y2+y3), ((x1+x2+x3)/2-x2, y1-y2*2+y3)))
    poly8 = Polygon((((x1+x2+x3)/2+x2, y1-y2*2+y3), ((x1+x2+x3)/2+x2, y1+y2+y3), (x1+x2+x3, y1+y3+y2), (x1+x2+x3, y1-y2*2+y3)))
    polys = [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8]
    return gp.GeoDataFrame(geometry=polys)

def gen_poly7(x1, x2, x3, y1, y2, y3):  #1-3 parallel buildings
    poly1 = Polygon(((x2*2, (y1+y3+y2)/2 - y2*4), (x2*2, (y1+y3+y2)/2 - y2*2), (x1+x3-x2, (y1+y3+y2)/2 - y2*2), (x1+x3-x2, (y1+y3+y2)/2 - y2*4)))
    poly2 = Polygon(((x2*2, (y1+y3+y2)/2 - y2*1), (x2*2, (y1+y3+y2)/2 + y2*1), (x1+x3-x2, (y1+y3+y2)/2 + y2*1), (x1+x3-x2, (y1+y3+y2)/2 - y2*1)))
    poly3 = Polygon(((x2*2, (y1+y3+y2)/2 + y2*2), (x2*2, (y1+y3+y2)/2 + y2*4), (x1+x3-x2, (y1+y3+y2)/2 + y2*4), (x1+x3-x2, (y1+y3+y2)/2 + y2*2)))
    polys = [poly1, poly2, poly3]
    return gp.GeoDataFrame(geometry=polys)

def gen_poly8(x1, x2, x3, y1, y2, y3):  #multiple diagonal parallel roads
    poly1 = Polygon(((0, y2+y1+y3), (0, y2+y1+y3-y3/1.5), (y3/3, y2+y1+y3)))
    poly2 = Polygon(((x2+x3/2, y1+y2+y3), (x2+x3, y1+y2+y3), (x2+x3-y3/3, y2+y1+y3-y3/1.5), (x2+x3/2-y3/3, y2+y1+y3-y3/1.5)))
    poly3 = Polygon((((x2+x3/2)-(y2+y1+y3)/2, 0), ((x2+x3)-(y2+y1+y3)/2, 0), ((x2+x3)-(y2+y1+y3)/2 +y3/3, y3/1.5), ((x2+x3/2)-(y2+y1+y3)/2+y3/3, y3/1.5)))
    poly4 = Polygon(((x2+x3+x1-y3/3, 0), (x2+x3+x1, 0), (x1+x2+x3, y3/1.5)))
    poly5 = Polygon(((x2+x3+x1/2, y1+y2+y3), (x1+x2+x3, y1+y2+y3), (x2+x3+x1-y3/3, y2+y1+y3-y3/1.5), (x2+x3+x1/2-y3/3, y2+y1+y3-y3/1.5)))
    poly6 = Polygon((((x2+x3+x1/2)-(y2+y1+y3)/2, 0), ((x2+x3+x1)-(y2+y1+y3)/2, 0), ((x2+x3+x1)-(y2+y1+y3)/2 +y3/3, y3/1.5), ((x2+x3+x1/2)-(y2+y1+y3)/2+y3/3, y3/1.5)))
    polys = [poly1, poly2, poly3, poly4, poly5, poly6]
    return gp.GeoDataFrame(geometry=polys)

def gen_poly9(x1, x2, x3, y1, y2, y3):  #one diagonal parallel road
    poly1 = Polygon(((0, y2+y1+y3), (0, y2+y1+y3-y3/1.5), (y3/3, y2+y1+y3)))
    poly2 = Polygon(((x2+x3, y1+y2+y3), (x1+x2+x3, y1+y2+y3), (x2+x3+x1-y3/3, y2+y1+y3-y3/1.5), (x2+x3-y3/3, y2+y1+y3-y3/1.5)))
    poly3 = Polygon((((x2+x3)-(y2+y1+y3)/2, 0), ((x2+x3+x1)-(y2+y1+y3)/2, 0), ((x2+x3+x1)-(y2+y1+y3)/2 +y3/3, y3/1.5), ((x2+x3)-(y2+y1+y3)/2+y3/3, y3/1.5)))
    polys = [poly1, poly2, poly3]
    return gp.GeoDataFrame(geometry=polys)

def gen_poly10(x1, x2, x3, y1, y2, y3): #complicated intersection
    poly1 = Polygon(((0, 0), (0, y1+y2+y3 - y2*10), (x2*2, y1+y2+y3 - y2*10), (x2*2, 0)))
    poly2 = Polygon(((0, y1+y2+y3 - y2*5), (0, y1+y2+y3), (x2*2, y1+y2+y3), (x2*2, y1+y2+y3 - y2*5)))
    poly3 = Polygon(((x1+x2+x3 - y2*15, 0), (x1+x2+x3 - y2*15, y1+y2+y3 - y2*15), (x1+x2+x3 - y2*10, y1+y2+y3 - y2*10), (x1+x2+x3, y1+y2+y3 - y2*10), (x1+x2+x3, 0)))
    poly4 = Polygon(((x1+x2+x3 - y2*5, y1+y2+y3 - y2*5), (x1+x2+x3, y1+y2+y3), (x1+x2+x3, y1+y2+y3 - y2*5)))
    poly5 = Polygon(((x1+x2+x3 - y2*15, y1+y2+y3 - y2*5), (x1+x2+x3 - y2*15, y1+y2+y3), (x1+x2+x3 - y2*10, y1+y2+y3)))
    polys = [poly1, poly2, poly3, poly4, poly5]
    return gp.GeoDataFrame(geometry=polys)

def gen_poly11(x1, x2, x3, y1, y2, y3):  #roundabout w/out circle
    poly1 = Polygon(((0, 0), (0, (y1+y2+y3)/2-y2), ((x1+x2+x3)/2-y2*3, (y1+y2+y3)/2-y2), ((x1+x2+x3)/2-x2, (y1+y2+y3)/2-y2*3), ((x1+x2+x3)/2-x2, 0)))
    poly2 = Polygon(((0, (y1+y2+y3)/2+y2), (0, y1+y2+y3), ((x1+x2+x3)/2-x2, y1+y2+y3), ((x1+x2+x3)/2-x2, (y1+y2+y3)/2+y2*3), ((x1+x2+x3)/2-y2*3, (y1+y2+y3)/2+y2)))
    poly3 = Polygon((((x1+x2+x3)/2+x2, 0), ((x1+x2+x3)/2+x2, (y1+y2+y3)/2-y2*3), ((x1+x2+x3)/2+y2*3, (y1+y2+y3)/2-y2), (x1+x2+x3, (y1+y2+y3)/2-y2), (x1+x2+x3, 0)))
    poly4 = Polygon((((x1+x2+x3)/2+y2*3, (y1+y2+y3)/2+y2), ((x1+x2+x3)/2+x2, (y1+y2+y3)/2+y2*3), ((x1+x2+x3)/2+x2, y1+y2+y3), (x1+x2+x3, y1+y2+y3), (x1+x2+x3, (y1+y2+y3)/2+y2)))
    polys = [poly1, poly2, poly3, poly4]
    return gp.GeoDataFrame(geometry=polys)

def gen_poly12(x1, x2, x3, y1, y2, y3):
    poly1 = Polygon(((0, 0), ((x1+x2+x3)/2-x2*2, 0), ((x1+x2+x3)/2-x2*2, (y1+y3+y2)/2 - y2*3), (0, (y1+y3+y2)/2 - y2*3)))
    poly2 = Polygon((((x1+x2+x3)/2+x2*2, 0), ((x1+x2+x3)/2+x2*2, (y1+y3+y2)/2 - y2*3), (x1+x2+x3, (y1+y3+y2)/2 - y2*3), (x1+x2+x3, 0)))
    poly3 = Polygon(((0, (y1+y3+y2)/2 - y2*1), ((x1+x2+x3)/2-x2*2, (y1+y3+y2)/2 - y2*1), ((x1+x2+x3)/2-x2*2-y2, (y1+y3+y2)/2 + y2*1), (0, (y1+y3+y2)/2 + y2*1)))
    poly4 = Polygon((((x1+x2+x3)/2+x2*2, (y1+y3+y2)/2 - y2*1), (x2+x3+x1, (y1+y3+y2)/2 - y2*1), (x1+x2+x3, (y1+y3+y2)/2 + y2*1), ((x1+x2+x3)/2+x2*2+y2, (y1+y3+y2)/2 + y2*1)))
    poly5 = Polygon(((0, (y1+y3+y2)/2 + y2*3), ((x1+x2+x3)/2-x2*2-y2*2, (y1+y3+y2)/2 + y2*3), ((x1+x2+x3)/2-x2*2-y2*2-((y1+y3+y2)/2 - y2*3)/2, y2+y1+y3), (0, y2+y1+y3)))
    poly6 = Polygon((((x1+x2+x3)/2-x2*1.75, (y1+y3+y2)/2 + y2*3), ((x1+x2+x3)/2+x2*1.75, (y1+y3+y2)/2 + y2*3), ((x1+x2+x3)/2+x2*1.75+ ((y1+y3+y2)/2 - y2*3)/2, y1+y3+y2), ((x1+x2+x3)/2-x2*1.75- ((y1+y3+y2)/2 - y2*3)/2, y1+y3+y2)))
    poly7 = Polygon((((x1+x2+x3)/2+x2*2+y2*2, (y1+y3+y2)/2 + y2*3), (x1+x2+x3, (y1+y3+y2)/2 + y2*3), (x1+x2+x3, y1+y3+y2), ((x1+x2+x3)/2+x2*2+y2*2 + ((y1+y3+y2)/2 - y2*3)/2, y1+y3+y2)))
    polys = [poly1, poly2, poly3, poly4, poly5, poly6, poly7]
    return gp.GeoDataFrame(geometry=polys)

def master_poly(x1, x2, x3, y1, y2, y3, s):  #determines which poly generator to run based on s
    if s == 1: poly = gen_poly1(x1, x2, x3, y1, y2, y3)
    if s == 2: poly = gen_poly2(x1, x2, x3, y1, y2, y3)
    if s == 3: poly = gen_poly3(x1, x2, x3, y1, y2, y3)
    if s == 4: poly = gen_poly4(x1, x2, x3, y1, y2, y3)
    if s == 5: poly = gen_poly5(x1, x2, x3, y1, y2, y3)
    if s == 6: poly = gen_poly6(x1, x2, x3, y1, y2, y3)
    if s == 7: poly = gen_poly7(x1, x2, x3, y1, y2, y3)
    if s == 8: poly = gen_poly8(x1, x2, x3, y1, y2, y3)
    if s == 9: poly = gen_poly9(x1, x2, x3, y1, y2, y3)
    if s == 10: poly = gen_poly10(x1, x2, x3, y1, y2, y3)
    if s == 11: poly = gen_poly11(x1, x2, x3, y1, y2, y3)
    if s == 12: poly = gen_poly12(x1, x2, x3, y1, y2, y3)
    return poly

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def grabCNNdata(grids, nxpixels):
    masterlist = []
    input_list = copy.deepcopy(grids)
    for i in range(len(grids)):
        tempa = []
        for j in input_list[i].index:
            if grids[i].at[j, 'count'] != 0:
                input_list[i].at[j, 'count'] = 0  # street- white
            else:
                input_list[i].at[j, 'count'] = 255  # wall- black
            tempa = list(input_list[i]['count'])
        tempa = chunks(tempa, nxpixels)
        masterlist.append(list(tempa))
    return masterlist


def main():
    polys = []
    maxes = []
    range_size = 2000  #global range variable
    slist = []
    for i in range(range_size):
        x1 = np.random.uniform(10., 20)
        x2 = np.random.uniform(0., 5)
        y1 = np.random.uniform(10., 20)
        y2 = np.random.uniform(0., 5)
        s = int(np.random.uniform(1, 13))  # random # between 1-12 (number of poly functions)
        slist.append(s)
        polys.append(master_poly(x1, x2, x1, y1, y2, y1, s))
        maxes.append((x1 + x2 + x1, y1 + y2 + y1))

    grids = []
    for i in range(range_size):
        grid = []
        x_max = maxes[i][0]
        y_max = maxes[i][1]
        xpix = 29 #this is equal to number of lines - 1 (xpix = ypix for now)
        ypix = 29
        xs = np.linspace(0, x_max, xpix + 1)
        ys = np.linspace(0, y_max, ypix + 1)
        for x in range(len(xs) - 1):
            for y in range(len(ys) - 1):
                poly = Polygon(((xs[x], ys[y]), (xs[x], ys[y + 1]), (xs[x + 1], ys[y + 1]), (xs[x + 1], ys[y])))
                grid.append(poly)
        grids.append(gp.GeoDataFrame(geometry=grid))

    streets = []
    building_grid = []
    for i in range(len(grids)):
        print(i)
        x = gp.sjoin(grids[i], polys[i], op='intersects')
        grids[i]['count'] = 0.0
        street = grids[i].drop(x.index)
        building_grid.append(x)
        streets.append(street)
        street['count'] = 0.0
        samplePoints = pbf.polygon_centroid_to_point(street)
        rays = pbf.build_lines_from_point(samplePoints, maxes[i][0]/3, 30)
        try:
            raysWithBuildings = gp.sjoin(rays, polys[i], op="intersects")
        except:
            print("An exception has occured: there were not enough bins which contained buildings")
            continue
        rays = rays.drop(raysWithBuildings.index.values.tolist())
        tree_list = list(rays['geometry']) + list(street['geometry'])
        strtree = STRtree(tree_list)
        pbf.accumulate_counts(strtree, street, 5)
        for j in street.index:
            grids[i].at[j, 'count'] = street.at[j, 'count']
        with open('datasets_and_generators/ANN_rawtraindata.txt', 'a') as outfile:
            json.dump(list(grids[i]['count']), outfile)
        ax = x.plot()
        scheme = mc.Quantiles(street['count'], k=15)
        gplt.choropleth(street, ax=ax, hue='count', legend=True, scheme=scheme,
                        legend_kwargs={'bbox_to_anchor': (1, 0.9)})
        plt.savefig('ANN_trainimages/x_'+str(i)+'.png')
        plt.close()

#saves dataset to json file
    # os.remove("ANN_trainingdata.json")
    masterlist = []
    input_list = copy.deepcopy(grids)
    output_list = copy.deepcopy(grids)
    for i in range(len(grids)):
        temp = ()
        for j in input_list[i].index:
            if grids[i].at[j, 'count'] != 0:
                input_list[i].at[j, 'count'] = 0  # street
            else:
                output_list[i].at[j, 'count'] = -1  # wall
                input_list[i].at[j, 'count'] = -1  # wall
        temp = (list(input_list[i]['count']), list(output_list[i]['count']))
        masterlist.append(temp)
    with open('datasets_and_generators/ANN_trainingdata.json', 'a') as outfile:
        json.dump(masterlist, outfile)

#save black and white street images and their labels for CNN
    # os.remove("CNNdata_images.json")
    with open('datasets_and_generators/CNNdata_images.json', 'a') as outfile:
        json.dump(grabCNNdata(grids, xpix), outfile)
    # os.remove("CNNdata_labels.json")
    with open('datasets_and_generators/CNNdata_labels.json', 'a') as outfile:
        json.dump(slist, outfile)

if __name__ == '__main__':
    freeze_support()
    p = Process(target=main)
    p.start()

