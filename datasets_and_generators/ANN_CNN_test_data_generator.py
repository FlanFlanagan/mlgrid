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

"""
All poly functions below get sequentially called by 
master_poly and return a list of points used to clip the Manhattan 
geodataframe to create test set based on real images.
It also returns the function name to name the files where 
the images get saved particularly for the CNN.
Parameters
----------
none

Returns
-------
[x_min, y_min, x_max, y_max, name]: list of 4 floats and a str
    the four points mark where on the manhattan shp file the shapes
    get clipped from and the name will be the name of the file where 
    the images get saved for the CNN test images
"""


def poly1_1():
    x_min = -73.98527333395992
    y_min = 40.7479771463825
    x_max = -73.98398531682531
    y_max = 40.74901713319052
    name = 'poly1_1.jpg'
    return [x_min, y_min, x_max, y_max, name]


def poly2_1():
    x_min = -73.9889235881017
    y_min = 40.72925502438974
    x_max = -73.9878500933199
    y_max = 40.73006272599238
    name = 'poly2_1.jpg'
    return [x_min, y_min, x_max, y_max, name]


def poly4_1():
    x_min = -74.0008045464351
    y_min = 40.74089665338382
    x_max = -73.99462691943303
    y_max = 40.74505876812792
    name = 'poly4_1.jpg'
    return [x_min, y_min, x_max, y_max, name]


def poly5_1():
    x_min = -73.98841905652145
    y_min = 40.729550854778644
    x_max = -73.98548539656309
    y_max = 40.7309091095344
    name = 'poly5_1.jpg'
    return [x_min, y_min, x_max, y_max, name]


def poly6_1():
    x_min = -73.98708012650212
    y_min = 40.736886547453686
    x_max = -73.98479591049974
    y_max = 40.738999835727995
    name = 'poly6_1.jpg'
    return [x_min, y_min, x_max, y_max, name]


def poly6_2():
    x_min = -73.98577258157727
    y_min = 40.732374335722696
    x_max = -73.98209379753659
    y_max = 40.73526967501397
    name = 'poly6_2.jpg'
    return [x_min, y_min, x_max, y_max, name]


def poly7_1():
    x_min = -74.01030028529303
    y_min = 40.71870248810824
    x_max = -74.00864686326192
    y_max = 40.72066867985905
    name = 'poly7_1.jpg'
    return [x_min, y_min, x_max, y_max, name]


def poly8or9_1():
    x_min = -73.99187739737646
    y_min = 40.7325038982852
    x_max = -73.9903432804347
    y_max = 40.734636695009115
    name = 'poly8or9_1.jpg'
    return [x_min, y_min, x_max, y_max, name]


def poly9or8_1():
    x_min = -73.99187739737646
    y_min = 40.73307639346226
    x_max = -73.9903681093166
    y_max = 40.7339940591359
    name = 'poly9or8_1.jpg'
    return [x_min, y_min, x_max, y_max, name]


def poly10_1():
    x_min = -73.9900583314379
    y_min = 40.72940971602
    x_max = -73.98807894640817
    y_max = 40.73054036923789
    name = 'poly10_1.jpg'
    return [x_min, y_min, x_max, y_max, name]


def poly12_1():
    x_min = -73.98657426755938
    y_min = 40.75751315956297
    x_max = -73.98398914368272
    y_max = 40.759315510560945
    name = 'poly12_1.jpg'
    return [x_min, y_min, x_max, y_max, name]


def poly12_2():
    x_min = -74.00997699640905
    y_min = 40.71517261147056
    x_max = -74.00802781186117
    y_max = 40.71666092335309
    name = 'poly12_2.jpg'
    return [x_min, y_min, x_max, y_max, name]


# add new poly definitions here

def clipmaker(x_min, y_min, x_max, y_max):
    """
    takes in the x_min, y_min, x_max, y_max of a desired
    dataframe clipping and returns a polygon defined by these
    points
    Parameters
    ----------
    x_min, y_min, x_max, y_max: floats
        the points in the Manhattan shp file

    Returns
    -------
    Polygon(points): Polygon
        a polygon defined by the specified points on the Manhattan
        shp file which will be used to clip out target shapes
    """
    return Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min)])


def clippoly(master, clipper):
    """
    returns the clipped geodataframe by clipping the master geodataframe
    containing all of the polygons in Manhattan with the clipper polygon
    defined by the points where we want manhattan to get clipped.
    Parameters
    ----------
    master: geodataframe
        the points in the Manhattan shp file
    clipper: polygon
        polygon defined by the points used to clip the
        master geodataframe

    Returns
    -------
    gp.GeoDataFrame(geometry=poly): geodataframe
        the clipped geodataframe
    """
    # clip NewYork to the different polygons
    NY_clipped = gp.clip(master, clipper)
    poly = []
    for index, row in NY_clipped.iterrows():
        poly.append(row['geometry'])
    return gp.GeoDataFrame(geometry=poly)


def save_image(data, fn):
    """
    Saves CNN training images in the proper format. This
    involves primarily removing the white border from the
    figure containing the data before saving it as fn
    Some code obtained at: https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image-in-matplotlib
    Parameters
    ----------
    data: polygon/geodataframe (clipped geodataframe)
        clipped polys of Manhattan to be saved as CNN test pngs.
    fn: str
        file path/name
    """
    data.plot(color="black", edgecolor='black')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(fn, bbox_inches='tight', pad_inches=-0.0)
    plt.close()


def master_poly():
    """
    the master_poly function does all the data preparation to turn the
    raw Manhattan shp file into preselected geodataframes defined by
    each gen_poly function. It loads in the shp file using geopandas and
    then after creating a list of all the desired clips using the information
    in the gen_poly funcs and making them into polygons with clipmaker()
    it then clips the NewYork dataframe by calling clippoly(). It saves
    the new geodataframes as pngs by calling save_image() for the CNN testing
    image dataand returns the polys list of geodataframes as well as their
    defining points to main
    Parameters
    ----------
    none

    Returns
    -------
    polys: list of geodataframes
        list of the clipped geodataframes
    points: list of lists of floats
        list of lists of the clipping points-> (x_min, y_min, x_max, y_max)
    """
    # place Manhattan shape file intp gp dataframe
    NewYork = gp.read_file('../Manhattan files/ManhattanBuildings.shp')

    # points is the list that will store all the poly clip points
    points = []
    points.append(poly1_1())
    points.append(poly2_1())
    points.append(poly4_1())
    points.append(poly5_1())
    points.append(poly6_1())
    points.append(poly6_2())
    points.append(poly7_1())
    points.append(poly8or9_1())
    points.append(poly9or8_1())
    points.append(poly10_1())
    points.append(poly12_1())
    points.append(poly12_2())
    # add polys to points list here......................
    polys = []
    # placing clipped polys into a list with a for loop
    for i in points:
        polys.append(clippoly(NewYork, clipmaker(i[0], i[1], i[2], i[3])))
    # saving images of the polys using save_image() and storing with name of poly generated
    for j in range(len(polys)):
        save_image(polys[j], '../datasets_and_generators/CNN_testimages/' + str(points[j][4]) + '')
    return polys, points


def main():

    polys, ranges = master_poly()
    range_size = len(polys)

    '''
    part 1: this section of the code holds the purpose
    of building the grid over the polygons just clipped 
    previously using master_poly() which come from the 
    original Manhattan shp file. It then takes the 
    polygon in a list [grid] which then gets
    appended to grids as a geopandas dataframe. 

    note: ensure that the pixel #s are equivalent to 
    those defined in ANN_training_data_generator.py
    as well as CNN_training_data_generator.py
    '''
    grids = []
    for i in range(range_size):
        grid = []
        xpix = 39  # this is equal to number of lines - 1 (xpix = ypix for now)
        ypix = 39
        xs = np.linspace(ranges[i][0], ranges[i][2], xpix + 1)  # [0]-x-min #[1]-y_min #[2]-x_max #[3]-y_max
        ys = np.linspace(ranges[i][1], ranges[i][3], ypix + 1)
        for x in range(len(xs) - 1):
            for y in range(len(ys) - 1):
                poly = Polygon(((xs[x], ys[y]), (xs[x], ys[y + 1]), (xs[x + 1], ys[y + 1]), (xs[x + 1], ys[y])))
                grid.append(poly)
        grids.append(gp.GeoDataFrame(geometry=grid))

    '''
    part 2: this portion of the code is what determines the data
    going into the ANN data set. It calculates the visibility 
    for each point in the street defined by the grids and number 
    of pixels in them. Each pixel is like a point. It then places 
    the calculated street visibility values into grids in the 
    correct format to be reformatted later and stored 
    in a json file in part 3. This part also saves pngs of the 
    plots of data to visualize the outcomes.

    NOTE: for the lines marked with '**************' the values for 
    ray_length and ray_number need to be tweaked whenever the number of 
    pixels gets changed in order to obtain a reasonable run time along
    with accurate results based on the plotted outcomes. This will 
    require a bit of trial and error. 
    (it was found that for 39x39 pixels: ray_length = ranges[i][2] * 1.2 & ray_number = 16 yielded good results)
    (it was found that for 29x29 pixels: ray_length = ranges[i][2]/3 & ray_number = 30 yielded good results)
    '''
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
        ray_length = (ranges[i][2] - ranges[i][0]) * 1.2
        ray_length = ranges[i][2] * 1.2  # **************
        ray_number = int(16)  # **************
        rays = pbf.build_lines_from_point(samplePoints, ray_length, ray_number)  # **************
        raysWithBuildings = gp.sjoin(rays, polys[i], op="intersects")
        rays = rays.drop(raysWithBuildings.index.values.tolist())
        tree_list = list(rays['geometry']) + list(street['geometry'])
        strtree = STRtree(tree_list)
        pbf.accumulate_counts(strtree, street, 5)
        for j in street.index:
            grids[i].at[j, 'count'] = street.at[j, 'count']
        ax = x.plot()
        scheme = mc.Quantiles(street['count'], k=15)
        gplt.choropleth(street, ax=ax, hue='count', legend=True, scheme=scheme,
                        legend_kwargs={'bbox_to_anchor': (1, 0.9)})
        plt.savefig('../datasets_and_generators/ANN_testimages/x_' + str(i) + '.png')

        plt.close()

    '''
    part 3: Puts data in correct format for ANN and places
    it into a list to then be stored in file.
    '''
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
