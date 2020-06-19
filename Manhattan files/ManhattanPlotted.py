import geopandas as gp
import geoplot as gplt
import matplotlib.pyplot as plt


NewYork = gp.read_file('ManhattanBuildings.shp')
gplt.polyplot(NewYork, figsize=(12, 12))
plt.show()