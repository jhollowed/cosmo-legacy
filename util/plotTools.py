'''
Joe Hollowed
COSMO-HEP 2017

Providing a set of tools for quick and convenient plotting tasks
'''

import numpy as np
import matplotlib.pyplot as plt


# ==================================================================================================


def hist2d_to_contour(hist2d, ax=None, log=False, **kwargs):
    '''
    Function to convert a 2d histogram to a contour density plot

    :param hist2d: a numpy.2dhistogram() return tuple
    :param ax: if supplied, plot the contour plot on the axis ax, else create a new figure
    :param log: whether or not to measure the density (bin heights) on a log scale
    :param kwargs: optional keyword arguments follow those of the matplotlib contour() docs
    :return: handle to the contour plot
    '''
    
    x_edges = hist2d[1]
    y_edges = hist2d[2]
    if(log): height = np.log(hist2d[0])
    else: height = hist2d[0]

    centerWidth_x = np.diff(x_edges) / 2.0
    centerWidth_y = np.diff(y_edges) / 2.0
    centerPos_x = x_edges[:-1] + centerWidth_x
    centerPos_y = y_edges[:-1] + centerWidth_y
    
    if(ax == None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        contour = ax.contour(centerPos_x, centerPos_y, height.T, **kwargs)
    else:
        contour = ax.contour(centerPos_x, centerPos_y, height.T, **kwargs)
    return contour
