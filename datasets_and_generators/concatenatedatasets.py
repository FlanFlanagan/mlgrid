import json
import numpy as np
import os

# CNN files to concatenate
imagefilelist = ['../datasets_and_generators/CNN_MASTER2trainingimages.json', '../datasets_and_generators/CNN_trainingimages.json']
labelfilelist = ['../datasets_and_generators/CNN_MASTER2traininglabels.json', '../datasets_and_generators/CNN_traininglabels.json']

masterimagelist = []
for i in imagefilelist:
    with open(i, 'r') as openfile:
        mylist = json.load(openfile)
        masterimagelist.extend(mylist)
os.remove("../datasets_and_generators/CNN_MASTER2trainingimages.json")
with open('../datasets_and_generators/CNN_MASTER2trainingimages.json', 'a') as outfile:
    json.dump(masterimagelist, outfile)

masterlabellist = []
for i in labelfilelist:
    with open(i, 'r') as openfile:
        mylist = json.load(openfile)
        masterlabellist.extend(mylist)
os.remove("../datasets_and_generators/CNN_MASTER2traininglabels.json")
with open('../datasets_and_generators/CNN_MASTER2traininglabels.json', 'a') as outfile:
    json.dump(masterlabellist, outfile)
