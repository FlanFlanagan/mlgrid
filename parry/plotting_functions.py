import matplotlib.pyplot as plt
import geopandas as gp
import pandas as pd
import geoplot.crs as gcrs
import geoplot as gplt

def plot_gdf_kde(gdf, plotname, gridsize=300):
    '''
    This function converts the lat/long distance into
    a meters distance.  
    Parameters
    ----------
    gdf: GeoDataFrame
        The dataframe used to generate the heatmap
    plotname: string
        Name of the save file for the plot
    gridsize: int
        The x, y size of the grid for the plot
    '''
    gdf = gdf[gdf['count'] > 0]
    fig, ax = plt.subplots(figsize=(50,50))
    try:
        ax=gplt.kdeplot(gdf, shade=True, color="blue", figsize=(50,50), gridsize=gridsize)
    except:
        print('Unable to plot kde for the following GeoDataFrame')
        print(gdf)
    #TODO Need to modify this to determine max lat/long
    ax.set_xlim(-74.0,-73.96)
    ax.set_ylim(40.7, 40.76)
    manhattan_buildings.plot(ax=ax, color="Black", figsize=(50,50))
    plt.savefig(plotname)

'''
def plot_multi_gdf_kde(gdf_array, plotname, gridsize=300):
    This function converts the lat/long distance into
    a meters distance.  
    Parameters
    ----------
    gdf_array: array-like
        array of dataframes to generate heatmaps from
    plotname: string
        Name of the save file for the plot
    gridsize: int
        The x, y size of the grid for the plot
    fig, ax = plt.subplots(figsize=(50,50))
    for a in gdf_array:
        a = a[a['count'] > 0]         
        try:
            gplt.kdeplot(a, ax=ax, shade=True, color="blue", figsize=(50,50), gridsize=gridsize)
        except:
            print('Unable to plot kde for the following GeoDataFrame')
            print(gdf)
    #TODO Need to modify this to determine max lat/long
    ax.set_xlim(-74.0,-73.96)
    ax.set_ylim(40.7, 40.76)
    manhattan_buildings.plot(ax=ax, color="Black", figsize=(50,50))
    plt.savefig(plotname)
'''
