import json
import numpy as np
import os

# list files to concatenate
filelist = ['file1', 'file2']

masterdatalist = []
for i in filelist:
    with open(i, 'r') as openfile:
        mylist = json.load(openfile)
        masterdatalist.extend(mylist)
os.remove("masterlist filename")
with open('masterlist filename', 'a') as outfile:
    json.dump(masterdatalist, outfile)