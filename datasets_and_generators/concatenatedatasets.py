import json
import numpy as np
import os

'''
this script was originally made for the purpose of 
combining large data files when more than one of the
same kind of training data set was made and wanted to 
be combined in the same file but overall can be used 
to concatenate any files by placing the contents into 
one list and saving it into a master file.
'''


# list files to concatenate
filelist = ['path/file1', 'path/file2']

masterdatalist = []
for i in filelist:
    with open(i, 'r') as openfile:
        mylist = json.load(openfile)
        masterdatalist.extend(mylist)
os.remove("masterlist path/filename")
with open('masterlist path/filename', 'a') as outfile:
    json.dump(masterdatalist, outfile)