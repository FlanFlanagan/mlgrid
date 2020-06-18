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
import numpy as np


def poly6_1():
    x_min = -73.98708012650212
    y_min = 40.736886547453686
    x_max = -73.98479591049974
    y_max = 40.738999835727995
    return [x_min, y_min, x_max, y_max]

#TODO: add more poly functions here

def clipmaker(x_min, y_min, x_max, y_max):
    return Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min)])


def clippoly(master, clipper):
    # clip NewYork to the different polygons
    NY_clipped = gp.clip(master, clipper)
    poly = []
    for index, row in NY_clipped.iterrows():
        poly.append(row['geometry'])
    return gp.GeoDataFrame(geometry=poly)


def master_poly():
    # place Manhattan shape file intp gp dataframe
    NewYork = gp.read_file('ManhattanBuildings.shp')

    # points is the list that with store all the poly clip points
    points = []
    points.append(poly6_1())
    # TODO: add polys to points list here......................
    # print(points)
    polys = []
    for i in points:
        polys.append(clippoly(NewYork, clipmaker(i[0], i[1], i[2], i[3])))
    return polys, points


def main():
    polys, ranges = master_poly()
    range_size = len(polys)

    grids = []
    for i in range(range_size):
        grid = []
        xpix = 29  # this is equal to number of lines - 1 (xpix = ypix for now)
        ypix = 29
        xs = np.linspace(ranges[i][0], ranges[i][2], xpix + 1)  #[0]-x-min #[1]-y_min #[2]-x_max #[3]-y_max
        ys = np.linspace(ranges[i][1], ranges[i][3], ypix + 1)
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
        rays = pbf.build_lines_from_point(samplePoints, .0008, 30)
        raysWithBuildings = gp.sjoin(rays, polys[i], op="intersects")
        rays = rays.drop(raysWithBuildings.index.values.tolist())
        tree_list = list(rays['geometry']) + list(street['geometry'])
        strtree = STRtree(tree_list)
        pbf.accumulate_counts(strtree, street, 5)
        for j in street.index:
            grids[i].at[j, 'count'] = street.at[j, 'count']
        with open('ANN_rawtestdata.txt', 'a') as outfile:
            json.dump(list(grids[i]['count']), outfile)
        ax = x.plot()
        scheme = mc.Quantiles(street['count'], k=15)
        gplt.choropleth(street, ax=ax, hue='count', legend=True, scheme=scheme,
                        legend_kwargs={'bbox_to_anchor': (1, 0.9)})
        plt.savefig('testpictures/x_' + str(i) + '.png')
        plt.close()

    # saves dataset to json file
    os.remove("ANN_testdata.json")
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
    with open('ANN_testdata.json', 'a') as outfile:
        json.dump(masterlist, outfile)


if __name__ == '__main__':
    freeze_support()
    p = Process(target=main)
    p.start()
