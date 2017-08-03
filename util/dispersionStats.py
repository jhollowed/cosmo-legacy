'''
Joe Hollowed
Last edited 6/27/16

Providing set of functions for measuring statistics relevant to cluster velocity dispersions. Functions
with a 'b' suffix are biweight functions, and those with a 'g' suffix are gapper functions
'''

import pdb
import numpy as np
from astropy import constants as const
c = const.c.value


def MAD(x, M=None):
    '''
    Function to calculate the Median Absolute Deviation, as given by Eq.7 in Beers et al. 1990

    :param x: data vector
    :param M: estimate of location for the sample. If None, use the sample median
    :return: the Median Abolsute Deviation
    '''

    if(M is None): M = np.median(x)
    MAD = np.mean( abs(x - M) )
    return MAD


def percentDiff(val1, val2):
    ans = abs( (val1 - val2)/((val1 + val2) / 2) )
    return ans


def bAverage(z, tol = 0, maxIters = 6, C=6.0):
    '''
    Function for finding the biweight average (C_BI), or biweight location estimator, of
    an array of values (Beers et al. 1990) iteratively, based on user defined tolerance
    or maximum iterations.

    :param z: array of values
    :param tol: desired tolerance (percent difference bteween last and new value). If
                tol = 0, then C_BI will be recalculated until maxIters
    :param maxIters: maximum iterations to force return if desired tolerance not achieved
    :param C: the"tuning constant" (default value is 6.0
    :returns: the biweight average of the dataset z
    '''
    z = np.array(z)
    M = np.median(z)
    i = 0
    Cbi = M
    oldCbi = 0

    while(percentDiff(Cbi, oldCbi) > tol and i < maxIters):

        oldCbi = Cbi
        if(i > 0): M = oldCbi
        i+= 1

        try:
            #find weighting factor of each entry in array, using the median
            u = (z - M) / (MAD(z, M)*C)
        except TypeError:
            print('Expected array or array-like object; got {}'.format(type(z)))
            return

        # find weights u_i that meet summation requirement
        mask = abs(u) > 1

        # apply mask to terms of C_BI
        term1 = (z - M)[~mask]
        term2 = ((1-(u**2))**2)[~mask]

        # calculate C_BI
        num = sum(term1*term2)
        den = sum(term2)
        Cbi = M + (num / den)

    return Cbi


def bAverage_err(sigBI,z,N):
    '''
    returns the error in the biweight average (cluster redshift z) (Ruel et al. 2014).

    :param sigBI: the calculated velocity dispersion of the data (cluster)
    :param z: the calculated biweight average (cluster redshift)
    :param N: size of dataset that calculated z (number of cluster members)
    :returns: error in biweight average
    '''
    d_z = (1/c) * (sigBI*(1+z) / np.sqrt(N))
    return d_z


def bootstrap_bAverage_err(vals, size = None, draws = 1000, replace = True):

    if(size == None):
        size = len(vals)
    results = [bAverage(np.random.choice(vals, size=size, replace=replace)) for i in range(draws)]
    err = np.std(results)
    return err


def bVariance(v, iter=True, tol=0, maxIters=6, C=9.0):
    '''
    Returns the biweight sample variance of a cluster (Sigma_BI), as presented in Ruel et al. 2014,
    from Mosteller&Tukey 1977

    :param v: velocity array
    :param iter: whether or not to use the iteratively computed biweight mean 
		 rather than the sample median
    :param C: tuning constant (default is C=9.0)
    :return: the biweight variance in v
    '''
    try:
        N = len(v)
        v = np.array(v)
    except TypeError:
        print('Expected array or array-like object; got {}'.format(type(v)))
        return

    if(iter): M = bAverage(v, tol, maxIters)
    else: M = np.median(v)

    u = (v - M) / (MAD(v, M) * C)  # usual biweight weighting
    mask = abs(u) > 1  # find weights whose absolute value is greater than one
    D = np.sum( ((1-(u**2)) * (1-(5*(u**2))))[~mask] )
    try:
        #apply mask to both terms of Sigma_BI, calculate dispersion
        term1v = ((1-u**2)**4)[~mask]
        term2v = ((v - M)**2)[~mask]
        num = np.sum(term1v*term2v)
        den = D * (D - 1)
        sampleVar = N * (num / den)
        return sampleVar

    except ZeroDivisionError:
        #dispersion cannot be calculated on given cluster
        return None


def bDispersion(v, iter=True, tol=0, maxIters=6, C=9.0):
    '''
    Returns the biweight (velocity) dispersion of a (cluster) dataset (Sigma_BI), which is the square
    root of the biweight sample variance.

    :param v: an array of (velocity) values
    :param iter: whether or not to use the iteratively computed biweight 
		 mean rather than the sample median
    :param tol: desired tolerance (percent difference bteween last and new value). If
                tol = 0, then C_BI will be recalculated until maxIters
    :param maxIters: maximum iterations to force return if desired tolerance not achieves
    :param C: the "tuning constant" (default is C=9.0)
    :returns: biweight dispersion and uncertainty in a 2-element list
    '''
    try:
        N = len(v)
        v = sorted(v)
        sigBI = np.sqrt(bVariance(v, iter, tol, maxIters, C))
        return sigBI
    except AttributeError:
        #in case the sample variance returned None
        return None


def bDispersion_err(sigBI, N):
    # Find uncertainty in velocity dispersion
    
    if(N<15): c = 0.91
    if(N>=15): c = 0.92 
    d_sigBI = (c * sigBI) / np.sqrt(N - 1)
    return d_sigBI


def bootstrap_bDispersion(vals, size = None, draws = 1000, repl = True):

    if(size == None):
        size = len(vals)
    results = [bDispersion(np.random.choice(vals, size=size, replace=repl)) for i in range(draws)]
    avg = np.mean(results)
    err = np.sqrt(np.std(results))
    return [avg, err]


def bDispersion_beers(v, iter = True, maxIters=6, tol = 0, C=9.0):
    '''
    Returns the biweight scale estimator (S_BI), as presented in Beers et al. 1990

    :param v: velocity array
    :param iter: whether or not to use the iteratively computed biweight mean rather than the sample median
    :param tol: desired tolerance (percent difference bteween last and new value). If
                tol = 0, then C_BI will be recalculated until maxIters
    :param maxIters: maximum iterations to force return if desired tolerance not achieves
    :param C: tuning constant (default is C=9.0)
    :return: the biweight scale estimator (dispersion) in v
    '''
    try:
        N = len(v)
        v = np.array(v)
    except TypeError:
        print('Expected array or array-like object; got {}'.format(type(v)))
        return

    if (iter): M = bAverage(v, tol, maxIters)
    else: M = np.median(v)

    u = (v - M) / (MAD(v, M) * C)  # usual biweight weighting
    mask = abs(u) > 1  # find weights whose absolute value is greater than one
    D = np.sum( ((1-(u**2)) * (1-(5*(u**2))))[~mask] )
    try:
        #apply mask to both terms of Sigma_BI, calculate dispersion
        term1v = ((1-u**2)**4)[~mask]
        term2v = ((v - M)**2)[~mask]
        num = np.sqrt(np.sum(term1v*term2v))
        den = abs(D)
        s_BI = np.sqrt(N) * (num / den)
        return s_BI

    except ZeroDivisionError:
        #dispersion cannot be calculated on given cluster
        return None


def gDispersion(v):
    '''
    Returns the gapper (velocity) dispersion of a (cluster) dataset (Sigma_G)
    :param v: an array of (velocity) values (assumes unsorted).
    :returns: gapper estimation of dispersion
    '''
    try:
        n = len(v)
    except TypeError:
        print('Expected array or array-like object; got {}'.format(type(v)))
        return

    # find gaussian weights and gaps
    v = sorted(v)
    w = np.arange(1,n) * np.arange(n-1,0,-1)
    g = np.diff(v)

    sigG = (np.sqrt(np.pi))/(n*(n-1)) * np.dot(w,g)
    return sigG


def haloDispersion():
	'''
	Returns the full 3D velocity dispersion, as defined 	
	'''


def sigmaClip(v, center = None, sigma=3, cutoff=15):
    '''
    Preforms sigma clipping on a (velocity) array of data. In this context, the variance is taken to be
    the biweight sample variance, and the center about which the clip is preformed is the biweight 
    average

    :param v: array of (velocity) data
    :param center: bulk velocity of cluster (default is center=None, meaning velocities are assumed
                   to be normalized with respect to the cluster dispersion)
    :param sigma: where to preform clipping (default is sigma=3)
    :param cutoff: the datasize value at which to use the gapper rather than the biweight dispersion
                   (default is cutoff=15)
    :returns: a numpy masked array, with only the un-clipped elements unmasked
    '''

    # Plots for debugging!
    # y = np.random.random(len(v))
    # plt.plot(v, y, '.', color = 'g')
    # plt.title(len(v))

    v = np.array(v)
    N = len(v)

    #find the deviation by either the gapper or biweight dispersion
    if(N>cutoff):
        deviation = sigma * bDispersion(v)
    else:
        deviation = sigma * gDispersion(v)

    #find the center (biweight average), find clip bounds
    if center == None: center = bAverage(v)
    clipMin = center - deviation
    clipMax = center + deviation

    # Plots for debugging!
    # plt.hold(True)
    # plt.plot([center, center], [0,max(y)], '--', color = 'black', linewidth = 2)
    # plt.plot([clipMin, clipMin], [0, max(y)], '--', color='red', linewidth=2)
    # plt.plot([clipMax, clipMax], [0, max(y)], '--', color='blue', linewidth=2)
    # plt.show()

    # mask out all elements outside clip bounds
    mask = (v < clipMin) | (v > clipMax)
    members = np.ma.masked_array(v,mask)
    clipped = len(v) - len(v[~mask])

    #return original array with clipped values masked as a numpy masked array
    return members
