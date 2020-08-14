#! /usr/bin/env python3
# coding: utf-8

""" This module gathers functions to visualize data.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def corr_transformed(data, pvalue=False):
    """ Function which transforms the datafram returned by corr() pandas method
    to be used by corr_heatmap() function
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding the data
    pvalue: bool, default False
        to get the pvalue instead of the correlation factor
    -----------
    Return : a (x, y, value) tuple
    """
    if pvalue:
        corr = data.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(len(data.columns)-1)
        corr_unpivot = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe
        corr_unpivot.columns = ['x', 'y', 'value']
    else:
        corr = data.corr()
        corr_unpivot = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe
        corr_unpivot.columns = ['x', 'y', 'value']
    return corr_unpivot['x'], corr_unpivot['y'], corr_unpivot['value']


def corr_heatmap(x, y, corr_val, pvalue=False, size_scale=500, ax=False):
    """ Display a heatmap with color and size corresponding to the magnitude of the correlations
    Parameters:
    -----------
    x: Series
        the pandas object holding the x labels (first parameter returned by corr_transformed)
    y: Series
        the pandas object holding the y labels (second parameter returned by corr_transformed)
    corr_val: Series
        the pandas object holding the correlation values (third parameter returned by corr_transformed)
    pvalue: bool, default False
        if pvalue and not correlation factor
    size scale: int, default 500
        enable to change square marker size
    ax: matplotlib.axes.Axes, default False
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.
    -----------
    Return :
    matplotlib.axes.Axes
    """
    if not ax:
            ax = plt.gca()
    if pvalue:
        ax.set_title("p-values of the correlation matrix")
    else:
        ax.set_title("Heatmap of the correlation matrix")
    # Mapping from column names to integer coordinates
    x_labels = [x_label for x_label in sorted(x.unique())]
    y_labels = [y_label for y_label in sorted(y.unique(), reverse=True)]
    x_to_num = {x_label: i for i, x_label in enumerate(x_labels)}
    y_to_num = {y_label: j for j, y_label in enumerate(y_labels)}
    # Define a color map
    if pvalue:
        color_map = plt.cm.get_cmap('Reds').reversed()
    else:
        color_map = plt.cm.get_cmap('seismic').reversed()
    # Creation of a scatter plot
    if pvalue:
        heatmap = ax.scatter(x=x.map(x_to_num), # Use mapping for x
                             y=y.map(y_to_num), # Use mapping for y
                             marker='s', # Use square as scatterplot marker
                             s=size_scale,
                             c=corr_val,
                             cmap=color_map
                            )
    else:
        heatmap = ax.scatter(x=x.map(x_to_num), # Use mapping for x
                             y=y.map(y_to_num), # Use mapping for y
                             s=corr_val.abs() * size_scale,
                             marker='s', # Use square as scatterplot marker
                             c=corr_val,
                             cmap=color_map
                            )
    # Show column labels on the axes
    ax.set_xticks([x_to_num[x_label] for x_label in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[y_label] for y_label in y_labels])
    ax.set_yticklabels(y_labels)
    # change grid parameters to get squares in the center of cells
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    # change axis parameters
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    heatmap.set_clim(-1, 1)
    return heatmap


def corr_plot(data, size_scale=500, save=True):
    """Get a nice plot of correlation matrix including
    - a heatmap subplot for pearson correlations
    - a heatmap with relative p-values
    Parameters:
    -----------
    data: DataFrame
        the pandas object holding the data
    size scale: int, default 500
        enable to change square marker size
    save: bool, default True
        to save the plot in a png format.
    -----------
    Return :
    matplotlib.axes.Axes
    """
    # get correlation matrix in a proper format
    corr_pearson = corr_transformed(data)
    corr_pvalue = corr_transformed(data, pvalue=True)
    # Create the figure
    fig = plt.figure(figsize=(12, 4.8))
    # Create the pearson heatmap
    ax_1 = fig.add_subplot(121)
    heatmap_1 = corr_heatmap(corr_pearson[0], corr_pearson[1], corr_pearson[2],
                             size_scale=size_scale, ax=ax_1)
    heatmap_1.set_clim(-1, 1) # manually setup the range of the colorscale
    plt.colorbar(heatmap_1)
    # Create the p-value heatmap
    ax_2 = fig.add_subplot(122)
    heatmap_2 = corr_heatmap(corr_pvalue[0], corr_pvalue[1], corr_pvalue[2],
                             pvalue=True, size_scale=size_scale, ax=ax_2)
    heatmap_2.set_clim(0, 1) # manually setup the range of the colorscale
    plt.colorbar(heatmap_2)
    # Save the plot
    if save:
        folder_path=os.path.join("charts")
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        plt.savefig("charts/coor_matrix.png")
    plt.show()
