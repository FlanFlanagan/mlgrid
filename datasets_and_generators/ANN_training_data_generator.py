import pandas as pd
import geopandas as gp
import numpy as np
import geoplot as gplt
import random
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry.point import Point
import parry.binning_functions as pbf
from shapely.strtree import STRtree
import mapclassify as mc
import json
import copy
import os

"""
All gen_poly functions below get randomly called by 
master_poly and return the geodataframe of the polys 
created by each poly definition.
Parameters
----------
x1, x2, x3, y1, y2, y3: Floats
    define side lengths of randomly generated poly.

Returns
-------
gp.GeoDataFrame(geometry=polys): GeoDataFrame
    A GeoDataFrame containing the polygon shapes defined
    in each poly function
"""


def gen_poly1(x1, x2, x3, y1, y2, y3):  # typical intersection with rectangles
    poly1 = Polygon(((0, 0), (0, y1), (x1, y1), (x1, 0)))
    poly2 = Polygon(((0, y1 + y2), (0, y1 + y2 + y3), (x1, y1 + y2 + y3), (x1, y1 + y2)))
    poly3 = Polygon(((x1 + x2, 0), (x1 + x2, y1), (x1 + x2 + x3, y1), (x1 + x2 + x3, 0)))
    poly4 = Polygon(
        ((x1 + x2, y1 + y2), (x1 + x2, y1 + y2 + y3), (x1 + x2 + x3, y1 + y2 + y3), (x1 + x2 + x3, y1 + y2)))
    polys = [poly1, poly2, poly3, poly4]
    return gp.GeoDataFrame(geometry=polys)


def gen_poly2(x1, x2, x3, y1, y2, y3):  # triangles with wall
    poly1 = Polygon(((0, 0), (0, y3), (x1, 0)))
    poly2 = Polygon(((x1 + x2, y3), (x1 + x2 + x3, y3), (x1 + x2 + x3, 0)))
    poly3 = Polygon(((0, y1 + y2), (0, y1 + y2 + y3), (x1 + x2 + x3, y1 + y2 + y3), (x1 + x2 + x3, y1 + y2)))
    polys = [poly1, poly2, poly3]
    return gp.GeoDataFrame(geometry=polys)


def gen_poly3(x1, x2, x3, y1, y2, y3):  # roundabout
    poly1 = Polygon(((0, 0), (0, (y1 + y2 + y3) / 2 - y2), ((x1 + x2 + x3) / 2 - y2 * 3, (y1 + y2 + y3) / 2 - y2),
                     ((x1 + x2 + x3) / 2 - x2, (y1 + y2 + y3) / 2 - y2 * 3), ((x1 + x2 + x3) / 2 - x2, 0)))
    poly2 = Polygon(((0, (y1 + y2 + y3) / 2 + y2), (0, y1 + y2 + y3), ((x1 + x2 + x3) / 2 - x2, y1 + y2 + y3),
                     ((x1 + x2 + x3) / 2 - x2, (y1 + y2 + y3) / 2 + y2 * 3),
                     ((x1 + x2 + x3) / 2 - y2 * 3, (y1 + y2 + y3) / 2 + y2)))
    poly3 = Polygon((((x1 + x2 + x3) / 2 + x2, 0), ((x1 + x2 + x3) / 2 + x2, (y1 + y2 + y3) / 2 - y2 * 3),
                     ((x1 + x2 + x3) / 2 + y2 * 3, (y1 + y2 + y3) / 2 - y2), (x1 + x2 + x3, (y1 + y2 + y3) / 2 - y2),
                     (x1 + x2 + x3, 0)))
    poly4 = Polygon((((x1 + x2 + x3) / 2 + y2 * 3, (y1 + y2 + y3) / 2 + y2),
                     ((x1 + x2 + x3) / 2 + x2, (y1 + y2 + y3) / 2 + y2 * 3), ((x1 + x2 + x3) / 2 + x2, y1 + y2 + y3),
                     (x1 + x2 + x3, y1 + y2 + y3), (x1 + x2 + x3, (y1 + y2 + y3) / 2 + y2)))
    p = Point((x1 + x2 + x3) / 2, (y1 + y2 + y3) / 2)
    circle = p.buffer(y2)
    polyc = list(circle.exterior.coords)
    poly5 = Polygon(polyc)
    polys = [poly1, poly2, poly3, poly4, poly5]
    return gp.GeoDataFrame(geometry=polys)


def gen_poly4(x1, x2, x3, y1, y2, y3):  # many buildings
    n = 1
    poly1 = Polygon(((0, 0), (0, y1 - y2 * n), (x1 - x2 * n, y1 - y2 * n), (x1 - x2 * n, 0)))
    poly2 = Polygon(((x1 + x2, 0), (x1 + x3 - x2 * n, 0), (x1 + x3 - x2 * n, y1 - y2 * n), (x1 + x2, y1 - y2 * n)))
    poly3 = Polygon(((x1 + x3, 0), (x1 + x3, y1 - y2 * n), (x1 + x2 + x3, y1 - y2 * n), (x1 + x2 + x3, 0)))
    poly4 = Polygon(((0, y1 + y2), (x1 - x2 * n, y1 + y2), (x1 - x2 * n, y1 + y3 - y2 * n), (0, y1 + y3 - y2 * n)))
    poly5 = Polygon(((x1 + x2, y1 + y2), (x1 + x3 - x2 * n, y1 + y2), (x1 + x3 - x2 * n, y1 + y3 - y2 * n),
                     (x1 + x2, y1 + y3 - y2 * n)))
    poly6 = Polygon(
        ((x1 + x3, y1 + y2), (x1 + x2 + x3, y1 + y2), (x1 + x2 + x3, y1 + y3 - y2 * n), (x1 + x3, y1 + y3 - y2 * n)))
    poly7 = Polygon(((0, y1 + y3), (0, y1 + y2 + y3), (x1 - x2 * n, y1 + y2 + y3), (x1 - x2 * n, y1 + y3)))
    poly8 = Polygon(
        ((x1 + x2, y1 + y3), (x1 + x3 - x2 * n, y1 + y3), (x1 + x3 - x2 * n, y1 + y2 + y3), (x1 + x2, y1 + y2 + y3)))
    poly9 = Polygon(
        ((x1 + x3, y1 + y3), (x1 + x3, y1 + y2 + y3), (x1 + x2 + x3, y1 + y2 + y3), (x1 + x2 + x3, y1 + y3)))
    polys = [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8, poly9]
    return gp.GeoDataFrame(geometry=polys)


def gen_poly5(x1, x2, x3, y1, y2, y3):  # roads with open space (possibly for parking lot)
    poly1 = Polygon(((0, 0), (x2 * 3, 0), (x2 * 3, y1 + y2), (0, y1 + y2)))
    poly2 = Polygon(((x2 * 3, 0), (x2 * 3, y1 + y2), (x2 * 3 + x2 * 2, y1 + y2)))
    poly3 = Polygon(((x2 * 3 + x2 * 1.5, 0), (x2 * 3 + x2 * 3.5, 0), (x2 * 3 + x2 * 3.5, y1 + y2)))
    poly4 = Polygon(
        ((x2 * 3 + x2 * 3.5, y1 + y2), (x1 + x3 - x2 * 3, y1 + y2), (x1 + x3 - x2 * 3, 0), (x2 * 3 + x2 * 3.5, 0)))
    poly5 = Polygon(((x1 + x3, y1 + y2), (x1 + x3 + x2, y1 + y2), (x1 + x3 + x2, 0), (x1 + x3, 0)))
    poly6 = Polygon(((0, y1 + y3 - y2 * 3), (x2 * 3 + x2 * 2, y1 + y3 - y2 * 3), (x2 * 3 + x2 * 2, y1 + y3 - y2),
                     (0, y1 + y3 - y2)))
    poly7 = Polygon(((x1 + x3 + x2, y1 + y3 - y2 * 3), (x1 + x3, y1 + y3 - y2 * 3), (x1 + x3, y1 + y3 - y2),
                     (x1 + x3 + x2, y1 + y3 - y2)))
    poly8 = Polygon(((0, y1 + y3), (x1 + x3 - x2 * 3, y1 + y3), (x1 + x3 - x2 * 3, y1 + y2 + y3), (0, y1 + y2 + y3)))
    poly9 = Polygon(
        ((x1 + x3, y1 + y3), (x1 + x3, y1 + y2 + y3), (x1 + x2 + x3, y1 + y2 + y3), (x1 + x2 + x3, y1 + y3)))
    polys = [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8, poly9]
    return gp.GeoDataFrame(geometry=polys)


def gen_poly6(x1, x2, x3, y1, y2, y3):
    poly1 = Polygon(((0, 0), (0, y1 - y2 * 15.5 + y3), ((x1 + x2 + x3) / 2 - x2, y1 - y2 * 15.5 + y3),
                     ((x1 + x2 + x3) / 2 - x2, 0)))
    poly2 = Polygon((((x1 + x2 + x3) / 2 + x2, 0), ((x1 + x2 + x3) / 2 + x2, y1 - y2 * 15.5 + y3),
                     (x1 + x2 + x3, y1 - y2 * 15.5 + y3), (x1 + x2 + x3, 0)))
    poly3 = Polygon(
        ((0, y1 - y2 * 13 + y3), (0, y1 - y2 * 10 + y3), (x1 / 2.5, y1 - y2 * 10 + y3), (x1 / 2.5, y1 - y2 * 13 + y3)))
    poly4 = Polygon(((x1 + x2 + x3 * 0.6, y1 - y2 * 13 + y3), (x1 + x2 + x3 * 0.6, y1 - y2 * 10 + y3),
                     (x1 + x2 + x3, y1 - y2 * 10 + y3), (x1 + x2 + x3, y1 - y2 * 13 + y3)))
    poly5 = Polygon(((0, y1 - y2 * 7.5 + y3), (0, y1 - y2 * 4.5 + y3), (x1 / 2.5, y1 - y2 * 4.5 + y3),
                     (x1 / 2.5, y1 - y2 * 7.5 + y3)))
    poly6 = Polygon(((x1 + x2 + x3 * 0.6, y1 - y2 * 7.5 + y3), (x1 + x2 + x3 * 0.6, y1 - y2 * 4.5 + y3),
                     (x1 + x3 + x2, y1 - y2 * 4.5 + y3), (x1 + x2 + x3, y1 - y2 * 7.5 + y3)))
    poly7 = Polygon(((0, y1 - y2 * 2 + y3), (0, y1 + y2 + y3), ((x1 + x2 + x3) / 2 - x2, y1 + y2 + y3),
                     ((x1 + x2 + x3) / 2 - x2, y1 - y2 * 2 + y3)))
    poly8 = Polygon((((x1 + x2 + x3) / 2 + x2, y1 - y2 * 2 + y3), ((x1 + x2 + x3) / 2 + x2, y1 + y2 + y3),
                     (x1 + x2 + x3, y1 + y3 + y2), (x1 + x2 + x3, y1 - y2 * 2 + y3)))
    polys = [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8]
    return gp.GeoDataFrame(geometry=polys)


def gen_poly7(x1, x2, x3, y1, y2, y3):  # 1-3 parallel buildings
    poly1 = Polygon(((x2 * 2, (y1 + y3 + y2) / 2 - y2 * 4), (x2 * 2, (y1 + y3 + y2) / 2 - y2 * 2),
                     (x1 + x3 - x2, (y1 + y3 + y2) / 2 - y2 * 2), (x1 + x3 - x2, (y1 + y3 + y2) / 2 - y2 * 4)))
    poly2 = Polygon(((x2 * 2, (y1 + y3 + y2) / 2 - y2 * 1), (x2 * 2, (y1 + y3 + y2) / 2 + y2 * 1),
                     (x1 + x3 - x2, (y1 + y3 + y2) / 2 + y2 * 1), (x1 + x3 - x2, (y1 + y3 + y2) / 2 - y2 * 1)))
    poly3 = Polygon(((x2 * 2, (y1 + y3 + y2) / 2 + y2 * 2), (x2 * 2, (y1 + y3 + y2) / 2 + y2 * 4),
                     (x1 + x3 - x2, (y1 + y3 + y2) / 2 + y2 * 4), (x1 + x3 - x2, (y1 + y3 + y2) / 2 + y2 * 2)))
    polys = [poly1, poly2, poly3]
    return gp.GeoDataFrame(geometry=polys)


def gen_poly8(x1, x2, x3, y1, y2, y3):  # multiple diagonal parallel roads
    poly1 = Polygon(((0, y2 + y1 + y3), (0, y2 + y1 + y3 - y3 / 1.5), (y3 / 3, y2 + y1 + y3)))
    poly2 = Polygon(((x2 + x3 / 2, y1 + y2 + y3), (x2 + x3, y1 + y2 + y3), (x2 + x3 - y3 / 3, y2 + y1 + y3 - y3 / 1.5),
                     (x2 + x3 / 2 - y3 / 3, y2 + y1 + y3 - y3 / 1.5)))
    poly3 = Polygon((((x2 + x3 / 2) - (y2 + y1 + y3) / 2, 0), ((x2 + x3) - (y2 + y1 + y3) / 2, 0),
                     ((x2 + x3) - (y2 + y1 + y3) / 2 + y3 / 3, y3 / 1.5),
                     ((x2 + x3 / 2) - (y2 + y1 + y3) / 2 + y3 / 3, y3 / 1.5)))
    poly4 = Polygon(((x2 + x3 + x1 - y3 / 3, 0), (x2 + x3 + x1, 0), (x1 + x2 + x3, y3 / 1.5)))
    poly5 = Polygon(((x2 + x3 + x1 / 2, y1 + y2 + y3), (x1 + x2 + x3, y1 + y2 + y3),
                     (x2 + x3 + x1 - y3 / 3, y2 + y1 + y3 - y3 / 1.5),
                     (x2 + x3 + x1 / 2 - y3 / 3, y2 + y1 + y3 - y3 / 1.5)))
    poly6 = Polygon((((x2 + x3 + x1 / 2) - (y2 + y1 + y3) / 2, 0), ((x2 + x3 + x1) - (y2 + y1 + y3) / 2, 0),
                     ((x2 + x3 + x1) - (y2 + y1 + y3) / 2 + y3 / 3, y3 / 1.5),
                     ((x2 + x3 + x1 / 2) - (y2 + y1 + y3) / 2 + y3 / 3, y3 / 1.5)))
    polys = [poly1, poly2, poly3, poly4, poly5, poly6]
    return gp.GeoDataFrame(geometry=polys)


def gen_poly9(x1, x2, x3, y1, y2, y3):  # one diagonal parallel road
    poly1 = Polygon(((0, y2 + y1 + y3), (0, y2 + y1 + y3 - y3 / 1.5), (y3 / 3, y2 + y1 + y3)))
    poly2 = Polygon(((x2 + x3, y1 + y2 + y3), (x1 + x2 + x3, y1 + y2 + y3),
                     (x2 + x3 + x1 - y3 / 3, y2 + y1 + y3 - y3 / 1.5), (x2 + x3 - y3 / 3, y2 + y1 + y3 - y3 / 1.5)))
    poly3 = Polygon((((x2 + x3) - (y2 + y1 + y3) / 2, 0), ((x2 + x3 + x1) - (y2 + y1 + y3) / 2, 0),
                     ((x2 + x3 + x1) - (y2 + y1 + y3) / 2 + y3 / 3, y3 / 1.5),
                     ((x2 + x3) - (y2 + y1 + y3) / 2 + y3 / 3, y3 / 1.5)))
    polys = [poly1, poly2, poly3]
    return gp.GeoDataFrame(geometry=polys)


def gen_poly10(x1, x2, x3, y1, y2, y3):  # complicated intersection
    poly1 = Polygon(((0, 0), (0, y1 + y2 + y3 - y2 * 10), (x2 * 2, y1 + y2 + y3 - y2 * 10), (x2 * 2, 0)))
    poly2 = Polygon(
        ((0, y1 + y2 + y3 - y2 * 5), (0, y1 + y2 + y3), (x2 * 2, y1 + y2 + y3), (x2 * 2, y1 + y2 + y3 - y2 * 5)))
    poly3 = Polygon(((x1 + x2 + x3 - y2 * 15, 0), (x1 + x2 + x3 - y2 * 15, y1 + y2 + y3 - y2 * 15),
                     (x1 + x2 + x3 - y2 * 10, y1 + y2 + y3 - y2 * 10), (x1 + x2 + x3, y1 + y2 + y3 - y2 * 10),
                     (x1 + x2 + x3, 0)))
    poly4 = Polygon(((x1 + x2 + x3 - y2 * 5, y1 + y2 + y3 - y2 * 5), (x1 + x2 + x3, y1 + y2 + y3),
                     (x1 + x2 + x3, y1 + y2 + y3 - y2 * 5)))
    poly5 = Polygon(((x1 + x2 + x3 - y2 * 15, y1 + y2 + y3 - y2 * 5), (x1 + x2 + x3 - y2 * 15, y1 + y2 + y3),
                     (x1 + x2 + x3 - y2 * 10, y1 + y2 + y3)))
    polys = [poly1, poly2, poly3, poly4, poly5]
    return gp.GeoDataFrame(geometry=polys)


def gen_poly11(x1, x2, x3, y1, y2, y3):  # roundabout w/out circle
    poly1 = Polygon(((0, 0), (0, (y1 + y2 + y3) / 2 - y2), ((x1 + x2 + x3) / 2 - y2 * 3, (y1 + y2 + y3) / 2 - y2),
                     ((x1 + x2 + x3) / 2 - x2, (y1 + y2 + y3) / 2 - y2 * 3), ((x1 + x2 + x3) / 2 - x2, 0)))
    poly2 = Polygon(((0, (y1 + y2 + y3) / 2 + y2), (0, y1 + y2 + y3), ((x1 + x2 + x3) / 2 - x2, y1 + y2 + y3),
                     ((x1 + x2 + x3) / 2 - x2, (y1 + y2 + y3) / 2 + y2 * 3),
                     ((x1 + x2 + x3) / 2 - y2 * 3, (y1 + y2 + y3) / 2 + y2)))
    poly3 = Polygon((((x1 + x2 + x3) / 2 + x2, 0), ((x1 + x2 + x3) / 2 + x2, (y1 + y2 + y3) / 2 - y2 * 3),
                     ((x1 + x2 + x3) / 2 + y2 * 3, (y1 + y2 + y3) / 2 - y2), (x1 + x2 + x3, (y1 + y2 + y3) / 2 - y2),
                     (x1 + x2 + x3, 0)))
    poly4 = Polygon((((x1 + x2 + x3) / 2 + y2 * 3, (y1 + y2 + y3) / 2 + y2),
                     ((x1 + x2 + x3) / 2 + x2, (y1 + y2 + y3) / 2 + y2 * 3), ((x1 + x2 + x3) / 2 + x2, y1 + y2 + y3),
                     (x1 + x2 + x3, y1 + y2 + y3), (x1 + x2 + x3, (y1 + y2 + y3) / 2 + y2)))
    polys = [poly1, poly2, poly3, poly4]
    return gp.GeoDataFrame(geometry=polys)


def gen_poly12(x1, x2, x3, y1, y2, y3):
    poly1 = Polygon(((0, 0), ((x1 + x2 + x3) / 2 - x2 * 2, 0),
                     ((x1 + x2 + x3) / 2 - x2 * 2, (y1 + y3 + y2) / 2 - y2 * 3), (0, (y1 + y3 + y2) / 2 - y2 * 3)))
    poly2 = Polygon((((x1 + x2 + x3) / 2 + x2 * 2, 0), ((x1 + x2 + x3) / 2 + x2 * 2, (y1 + y3 + y2) / 2 - y2 * 3),
                     (x1 + x2 + x3, (y1 + y3 + y2) / 2 - y2 * 3), (x1 + x2 + x3, 0)))
    poly3 = Polygon(((0, (y1 + y3 + y2) / 2 - y2 * 1), ((x1 + x2 + x3) / 2 - x2 * 2, (y1 + y3 + y2) / 2 - y2 * 1),
                     ((x1 + x2 + x3) / 2 - x2 * 2 - y2, (y1 + y3 + y2) / 2 + y2 * 1), (0, (y1 + y3 + y2) / 2 + y2 * 1)))
    poly4 = Polygon((((x1 + x2 + x3) / 2 + x2 * 2, (y1 + y3 + y2) / 2 - y2 * 1),
                     (x2 + x3 + x1, (y1 + y3 + y2) / 2 - y2 * 1), (x1 + x2 + x3, (y1 + y3 + y2) / 2 + y2 * 1),
                     ((x1 + x2 + x3) / 2 + x2 * 2 + y2, (y1 + y3 + y2) / 2 + y2 * 1)))
    poly5 = Polygon(((0, (y1 + y3 + y2) / 2 + y2 * 3),
                     ((x1 + x2 + x3) / 2 - x2 * 2 - y2 * 2, (y1 + y3 + y2) / 2 + y2 * 3),
                     ((x1 + x2 + x3) / 2 - x2 * 2 - y2 * 2 - ((y1 + y3 + y2) / 2 - y2 * 3) / 2, y2 + y1 + y3),
                     (0, y2 + y1 + y3)))
    poly6 = Polygon((((x1 + x2 + x3) / 2 - x2 * 1.75, (y1 + y3 + y2) / 2 + y2 * 3),
                     ((x1 + x2 + x3) / 2 + x2 * 1.75, (y1 + y3 + y2) / 2 + y2 * 3),
                     ((x1 + x2 + x3) / 2 + x2 * 1.75 + ((y1 + y3 + y2) / 2 - y2 * 3) / 2, y1 + y3 + y2),
                     ((x1 + x2 + x3) / 2 - x2 * 1.75 - ((y1 + y3 + y2) / 2 - y2 * 3) / 2, y1 + y3 + y2)))
    poly7 = Polygon((((x1 + x2 + x3) / 2 + x2 * 2 + y2 * 2, (y1 + y3 + y2) / 2 + y2 * 3),
                     (x1 + x2 + x3, (y1 + y3 + y2) / 2 + y2 * 3), (x1 + x2 + x3, y1 + y3 + y2),
                     ((x1 + x2 + x3) / 2 + x2 * 2 + y2 * 2 + ((y1 + y3 + y2) / 2 - y2 * 3) / 2, y1 + y3 + y2)))
    polys = [poly1, poly2, poly3, poly4, poly5, poly6, poly7]
    return gp.GeoDataFrame(geometry=polys)


def master_poly(x1, x2, x3, y1, y2, y3, s):
    """
    Calls randomly generated poly function based on s and
    returns the geodataframe of that poly.
    Parameters
    ----------
    x1, x2, x3, y1, y2, y3: Float
        defines side lengths of randomly generated polys.
    s: Int
        randomly generated to indicate which poly is to
        be created.

    Returns
    -------
    poly: GeoDataFrame
        A GeoDataFrame containing the polygon shapes defined
        in whichever poly function was called
    """
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
    """
    Yields successive n-sized chunks from lst.
    Obtained at: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    Parameters
    ----------
    lst: List
        lst to be broken into n sized chunks.
    n: Int
        size of chunks to break lst into.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def grabANNdata(grid):
    """
    Puts data in correct format for ANN and places
    it into a list to be stored in a json file later.
    Parameters
    ----------
    grid: GeoDataFrame
        geodataframe containing data on the polys generated
        in part 3 of main().

    Returns
    -------
    master: List of Tuples
        A list of tuples with all ANN training data to be stored
        in a json file.
    """
    input_list = copy.deepcopy(grid)
    output_list = copy.deepcopy(grid)
    for j in input_list.index:
        if grid.at[j, 'count'] != 0:
            input_list.at[j, 'count'] = 0  # street
        else:
            output_list.at[j, 'count'] = -1  # wall
            input_list.at[j, 'count'] = -1  # wall
    master = (list(input_list['count']), list(output_list['count']))
    return master


def append_to_json(_dict, path):
    """
    Manually appends data to MASTER data files, without pulling
    out data in file into a list to get appended to first, thus avoiding
    any memory leaks and major slow downs.
    Obtained at: https://stackoverflow.com/questions/12994442/how-to-append-data-to-a-json-file
    Parameters
    ----------
    _dict: List/Array-Like Object
        contains data to be stored in file
    path: String
        file path/file name
    """
    with open(path, 'ab+') as f:
        f.seek(0, 2)  # Go to the end of file
        if f.tell() == 0:  # Check if file is empty
            f.write(json.dumps([_dict]).encode())  # If empty, write an array
        else:
            f.seek(-1, 2)
            f.truncate()  # Remove the last character, open the array
            f.write(' , '.encode())  # Write the separator
            f.write(json.dumps(_dict).encode())  # Dump the dictionary
            f.write(']'.encode())


# main---------------------------------------------
range_size = 5  # USER DEFINED: number of data points to be added to CNN data set

for i in range(range_size):
    '''
    note about this entire section: python runs more efficiently 
    when having everything stored in, appended to, and extracted from 
    lists which is why, despite it making the code look less cleanly,
    everything is done using lists
    '''

    print(i)  # tells us which iteration we are on

    '''
    list variables declared below along with where 
    they are initially appended to
    '''
    polys = []  # part 1
    maxes = []  # part 1
    grids = []  # part 2
    grid = []  # part 2
    building_grid = []  # part 3
    streets = []  # part 3
    raysWithBuildings = None  # part 3

    '''
    part 1: the section below randomly generates the 
    lengths that will be used to define each polygon 
    and randomly decides which polygon will be generated. 
    Then it calls master_poly() which will return the 
    defined polygon and gets appended to polys[]
    and the max lengths get appended to maxes[].
    '''
    x1 = np.random.uniform(.02, .04)
    x2 = np.random.uniform(0., .01)
    y1 = np.random.uniform(.02, .04)
    y2 = np.random.uniform(0., .01)
    s = int(np.random.uniform(1, 13))  # random # between 1-12 (number of poly functions)
    polys.append(master_poly(x1, x2, x1, y1, y2, y1, s))
    maxes.append((x1 + x2 + x1, y1 + y2 + y1))

    '''
    part 2: this section of the code holds the purpose
    of building a grid over the geodataframe just created 
    previously and xpix and ypix are where the user 
    defines how many pixels or matrix columns should be 
    used to describe each polygon. It then takes these 
    grid polygons in a list grid[] which then gets
    appended to grids[] as a geopandas dataframe. 

    note: ensure that the pixel #s are equivalent to 
    those defined in ANN_CNN_test_data_generator.py
    as well as CNN_training_data_generator.py
    '''

    x_max = maxes[0][0]
    y_max = maxes[0][1]
    xpix = 39  # this is equal to number of lines - 1 (xpix = ypix for now)
    ypix = 39
    xs = np.linspace(0, x_max, xpix + 1)
    ys = np.linspace(0, y_max, ypix + 1)
    for x in range(len(xs) - 1):
        for y in range(len(ys) - 1):
            poly = Polygon(((xs[x], ys[y]), (xs[x], ys[y + 1]), (xs[x + 1], ys[y + 1]), (xs[x + 1], ys[y])))
            grid.append(poly)
    grids.append(gp.GeoDataFrame(geometry=grid))

    '''
    part 3: this portion of the code is what determines the data
    going into the ANN data set. It calculates the visibility 
    for each point in the street defined by the grid and number 
    of pixels in it. Each pixel is like a point. It then places 
    the calculated street visibility values into grids[] in the 
    correct format to be reformatted by grabANNdata() for storage 
    in a json file in part 4. This part also saves pngs of the 
    plots of data for the first 10 iterations to visualize the outcomes.
    
    NOTE: for the lines marked with '**************' the values for 
    ray_length and ray_number need to be tweaked whenever the number of 
    pixels gets changed in order to obtain a reasonable run time along
    with accurate results based on the plotted outcomes. This will 
    require a bit of trial and error. 
    (it was found that for 39x39 pixels: ray_length = x_max * 1.2 & ray_number = 16 yielded good results)
    (it was found that for 29x29 pixels: ray_length = x_max/3 & ray_number = 30 yielded good results)
    '''
    for j in range(len(grids)):
        x = gp.sjoin(grids[j], polys[j], op='intersects')
        grids[j]['count'] = 0.0
        street = grids[j].drop(x.index)
        building_grid.append(x)
        streets.append(street)
        street['count'] = 0.0
        samplePoints = pbf.polygon_centroid_to_point(street)
        ray_length = x_max * 1.2  # **************
        ray_number = int(16)  # **************
        rays = pbf.build_lines_from_point(samplePoints, ray_length, ray_number)  # **************
        try:
            raysWithBuildings = gp.sjoin(rays, polys[j], op="intersects")
        except:
            print("An exception has occured: there were not enough bins which contained buildings")
            continue
        rays = rays.drop(raysWithBuildings.index.values.tolist())
        tree_list = list(rays['geometry']) + list(street['geometry'])
        strtree = STRtree(tree_list)
        pbf.accumulate_counts(strtree, street, 7)
        for k in street.index:
            grids[j].at[k, 'count'] = street.at[k, 'count']
        if i < 10:  # saves images of the plots of data for first 10 iterations
            ax = x.plot()
            scheme = mc.Quantiles(street['count'], k=20)
            gplt.choropleth(street, ax=ax, hue='count', legend=True, scheme=scheme, cmap="jet",
                            legend_kwargs={'bbox_to_anchor': (1, 0.9)})
            plt.savefig('../datasets_and_generators/ANN_trainimages/x_' + str(i) + '.png')
            plt.close()

    '''
    part 4: concatenates data to MASTER training data files 

    ATTENTION: when changing this file DO NOT immediately concatenate
    generated data to MASTER files, use a temporary json file by changing 
    the file name below until it is certain that the data is compatible 
    with the rest of the data set
    -compatibility involves: the data being the same shape (xpix & ypix),
    and that the format it is being saved in is consistent; refer to functions 
    grabANNdata() and append_to_json().
    '''
    append_to_json(grabANNdata(grids[0]), '../datasets_and_generators/ANN_trainingdata_MASTER.json')
