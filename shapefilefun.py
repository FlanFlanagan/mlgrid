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
import pprint
import warnings
import numpy as np
import tensorflow as tf
import time
import cv2


def main():
    # place Manhattan shape file intp gp dataframe
    NewYork = gp.read_file('ManhattanBuildings.shp')
    # print(NewYork.head())

    range_size = 1
    # create polygons on New York City #TODO: create list of polygons so that multiple can be saved or created at once
    polys = []
    x_min = -73.98708012650212
    y_min = 40.736886547453686
    x_max = -73.98479591049974
    y_max = 40.738999835727995
    poly6_1 = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min)])
    # print(poly6_1)
    poly_size = int(gp.clip(NewYork, poly6_1).count().count())
    poly_gdf = gp.GeoDataFrame([1], geometry=[poly6_1], crs=NewYork.crs)

    # clip NewYork to the different polygons
    NY_clipped = gp.clip(NewYork, poly_gdf)
    # NY_clipped = gp.reindex(NY_clipped, labels=np.arange(NY_clipped.count()))
    # print(NY_clipped.head())
    polys.append(NY_clipped)
    print(polys[0])
    print(polys[0].count())

    grids = []
    for i in range(range_size):
        grid = []
        xpix = 39 #this is equal to number of lines - 1 (xpix = ypix for now)
        ypix = 39
        xs = np.linspace(x_min, x_max, xpix + 1)
        ys = np.linspace(x_max, y_max, ypix + 1)
        for x in range(len(xs) - 1):
            for y in range(len(ys) - 1):
                poly = Polygon(((xs[x], ys[y]), (xs[x], ys[y + 1]), (xs[x + 1], ys[y + 1]), (xs[x + 1], ys[y])))
                grid.append(poly)
        grids.append(gp.GeoDataFrame(geometry=grid))
    print(grids[0])
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
        rays = pbf.build_lines_from_point(samplePoints, 20, 20)
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
        scheme = mc.Quantiles(street['count'], k=10)
        gplt.choropleth(street, ax=ax, hue='count', legend=True, scheme=scheme,
                        legend_kwargs={'bbox_to_anchor': (1, 0.9)})
        plt.savefig('testpictures/x_'+str(i)+'.png')
        plt.close()

    # rays.plot()
#saves dataset to json file
    # os.remove("ANN_testdata.json")
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
'''
#save black and white street images and their labels for CNN
    os.remove("CNNdata_images.json")
    with open('CNNdata_images.json', 'a') as outfile:
        json.dump(grabCNNdata(grids, xpix), outfile)
    os.remove("CNNdata_labels.json")
    with open('CNNdata_labels.json', 'a') as outfile:
        json.dump(slist, outfile)'''


if __name__ == '__main__':
    # freeze_support()
    p = Process(target=main)
    p.start()

'''fig, ax = plt.subplots(figsize=(29, 29), frameon=False)
# h = w = 1
# fig.set_size_inches(h, w)
# ax = plt.Axes(fig, [0., 0., 1., 1.])
# ax.set_axis_off()
# fig.add_axes(ax)
NY_clipped.plot(ax=ax, color="black")
NY_clipped.boundary.plot(ax=ax, color="black")
ax.set_axis_off()
# plt.show()
os.remove("testplot.jpg")
plt.savefig('testplot.jpg', bbox_inches='tight', transparent=True, pad_inches=0)
# Image.open('testplot.png').save('testplot.jpg','JPEG')
# import warnings; warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
#     return self.notna()
# print(prepare('testplot.jpg', 29))
'''
