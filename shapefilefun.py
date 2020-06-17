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

'''
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
def prepare(filepath, num_pix):
    img = cv2.imread(filepath)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize image
    resized = cv2.resize(grayImage, (num_pix, num_pix), interpolation=cv2.INTER_AREA)
    # turn b&w
    (thresh, bawimg) = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    print(bawimg)# show image
    cv2.imshow('b&w', bawimg)
    print("press enter\n")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # turn data into proper format
    piclist = bawimg.tolist()
    flat_list = [item for sublist in piclist for item in sublist]
    masterlist = []
    mainlist = []
    for j in flat_list:
        if j == 0:
            mainlist.append(0.)
        else:
            mainlist.append(1.)
    tempa = chunks(mainlist, num_pix)
    masterlist.append(list(tempa))
    matrix = np.asarray(list(masterlist))
    matrix = matrix / 255
    matrix = np.expand_dims(matrix, axis=3)  # TensorFlow expects a channel dimension
    matrix = tf.cast(matrix, tf.float32)
    return matrix'''

NewYork = gp.read_file('ManhattanBuildings.shp')
# gplt.polyplot(NewYork, figsize=(8,4))
print(NewYork.head())

#create polygon
polygon = Polygon([(-73.98708012650212, 40.736886547453686), (-73.98708012650212, 40.738999835727995), (-73.98479591049974, 40.738999835727995), (-73.98479591049974, 40.736886547453686), (-73.98708012650212, 40.736886547453686)])
poly_gdf = gp.GeoDataFrame([1], geometry=[polygon], crs=NewYork.crs)


NY_clipped = gp.clip(NewYork, polygon)
fig, ax = plt.subplots(figsize=(29, 29), frameon=False)
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
