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


def poly1_1():
    x_min = -73.98527333395992
    y_min = 40.7479771463825
    x_max = -73.98398531682531
    y_max = 40.74901713319052
    return [x_min, y_min, x_max, y_max]


def poly2_1():
    x_min = -73.9889235881017
    y_min = 40.72925502438974
    x_max = -73.9878500933199
    y_max = 40.73006272599238
    return [x_min, y_min, x_max, y_max]


def poly4_1():
    x_min = -74.0008045464351
    y_min = 40.74089665338382
    x_max = -73.99462691943303
    y_max = 40.74505876812792
    return [x_min, y_min, x_max, y_max]


def poly5_1():
    x_min = -73.98841905652145
    y_min = 40.729550854778644
    x_max = -73.98548539656309
    y_max = 40.7309091095344
    return [x_min, y_min, x_max, y_max]


def poly6_1():
    x_min = -73.98708012650212
    y_min = 40.736886547453686
    x_max = -73.98479591049974
    y_max = 40.738999835727995
    return [x_min, y_min, x_max, y_max]

def poly6_2():
    x_min = -73.98577258157727
    y_min = 40.732374335722696
    x_max = -73.98209379753659
    y_max = 40.73526967501397
    return [x_min, y_min, x_max, y_max]

def poly7_1():
    x_min = -74.01030028529303
    y_min = 40.71870248810824
    x_max = -74.00864686326192
    y_max = 40.72066867985905
    return [x_min, y_min, x_max, y_max]

def poly8or9_1():
    x_min = -73.99187739737646
    y_min = 40.7325038982852
    x_max = -73.9903432804347
    y_max = 40.734636695009115
    return [x_min, y_min, x_max, y_max]

def poly9or8_1():
    x_min = -73.99187739737646
    y_min = 40.73307639346226
    x_max = -73.9903681093166
    y_max = 40.7339940591359
    return [x_min, y_min, x_max, y_max]

def poly10_1():
    x_min = -73.9900583314379
    y_min = 40.72940971602
    x_max = -73.98807894640817
    y_max = 40.73054036923789
    return [x_min, y_min, x_max, y_max]

def poly12_1():
    x_min = -73.98657426755938
    y_min = 40.75751315956297
    x_max = -73.98398914368272
    y_max = 40.759315510560945
    return [x_min, y_min, x_max, y_max]

def poly12_2():
    x_min = -74.00997699640905
    y_min = 40.71517261147056
    x_max = -74.00802781186117
    y_max = 40.71666092335309
    return [x_min, y_min, x_max, y_max]

#add new poly definitions here

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
    points.append(poly1_1())
    # points.append(poly2_1())
    # points.append(poly4_1())
    # points.append(poly5_1())
    # points.append(poly6_1())
    # points.append(poly6_2())
    # points.append(poly7_1())
    # points.append(poly8or9_1())
    # points.append(poly9or8_1())
    # points.append(poly10_1())
    # points.append(poly12_1())
    # points.append(poly12_2())
    #add polys to points list here......................
    polys = []
    for i in points:
        polys.append(clippoly(NewYork, clipmaker(i[0], i[1], i[2], i[3])))
    for j in range(len(polys)):
        polys[j].plot()
        plt.savefig('CNN_testpictures/x_' + str(j) + '.jpg')  # creating test images for CNN
        plt.close()
    return polys, points


def main():
    polys, ranges = master_poly()
    range_size = len(polys)

    grids = []
    for i in range(range_size):
        grid = []
        xpix = 29  # this is equal to number of lines - 1 (xpix = ypix for now)
        ypix = 29
        xs = np.linspace(ranges[i][0], ranges[i][2], xpix + 1)  # [0]-x-min #[1]-y_min #[2]-x_max #[3]-y_max
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
