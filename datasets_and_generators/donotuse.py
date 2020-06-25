import pandas as pd
import geopandas as gp
import numpy as np
import geoplot as gplt
import random
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry.point import Point
# import parry.binning_functions as pbf
# from shapely.strtree import STRtree
import mapclassify as mc
import json
import copy
import os


masterlist = []
with open('../datasets_and_generators/ANN_trainingdata_MASTER.json', 'r') as openfile:
    mylist = list(json.load(openfile))
    masterlist = mylist[0:len(mylist)-3]
# print(masterlist)
os.remove("../datasets_and_generators/ANN_trainingdata_MASTER.json")
with open('../datasets_and_generators/ANN_trainingdata_MASTER.json', 'a') as outfile:
    json.dump(masterlist, outfile)