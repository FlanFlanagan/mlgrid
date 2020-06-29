import json
import os

'''
this code holds the purpose of fixing datasets if something ever
gets appended wrong and it wouldn't make sense to remake an
entire dataset. This should ONLY be done if you know exactly what is 
wrong and what specifically needs to be removed from the file 
to fix it.

It places the files contents into a list, removes items based on
chosen indices, and replaces file contents with new list.
'''


masterlist = []
with open('../datasets_and_generators/ANN_trainingdata_MASTER.json', 'r') as openfile:
    mylist = list(json.load(openfile))
    masterlist = mylist[0:len(mylist)-3]
# print(masterlist)
os.remove("../datasets_and_generators/ANN_trainingdata_MASTER.json")
with open('../datasets_and_generators/ANN_trainingdata_MASTER.json', 'a') as outfile:
    json.dump(masterlist, outfile)