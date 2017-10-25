'''
Joe Hollowed
Last edited 6/27/16

Providing set of functions for measuring statistics relevant to cluster velocity dispersions. Functions
with a 'b' suffix are biweight functions, and those with a 'g' suffix are gapper functions
'''

import pdb
import warnings
import numpy as np
from astropy import constants as const
from numpy.core import umath_tests as npm
c = const.c.value


def MAD(x, M=None):
    '''
    Function to calculate the Median Absolute Deviation, as given by Eq.7 in Beers et al. 1990

    :param x: data vector or matrix (in which case, the MAD is computed and returned per row)
    :param M: estimate of location for the sample. If None, use the sample median
    :return: the Median Abolsute Deviation
    '''

    if(M is None): 
        if(len(x.shape) == 1): x = np.reshape(x, (1, len(x)))
        M = np.reshape( np.median(x, axis=1), (len(x), 1))
    MAD = np.mean( abs(x - M), axis=1 )
    return MAD

# ---------------------------------------------------------------------------------------------------

def percentDiff(val1, val2):
    ans = abs( (val1 - val2)/((val1 + val2) / 2) )
    return ans

# ---------------------------------------------------------------------------------------------------

def random_choice_noreplace(x, realizations, size, axis=-1):
    '''
    This function addresses a missing functionality in np.random.choice()
    In that numpy function, you cannot set `replace` = False, and also pass a 2d
    tuple for the argument `size` with reliable results. For instance, if you would 
    like to sample an array x of length 20 five times, each time choosing 10 elements, 
    then you would like a return 20x10 array, with each row being the new realization 
    without replacement. Numpy will instead try to ensure that the entire 200-element 
    matrix is done without replacement, which of course cannot be done on a 20-element 
    input list

    :param x: the array from which to sample
    :param realizations: number of realizations to take from x
    :param size: the size of each realization (must be <= len(x), though that 
                 would be pretty dumb if size==len(x))
    :param axis: the axis on which to force the no replacement condition. axis=-1 ensures
                 unique elements across rows, and axis=0 does the same across columns
    :return: a matrix of size realizations x size, each row being a realization with no 
             replacement (assuming axis=-1)
    '''
    resample_indices = np.random.rand(realizations, size).argsort(axis=axis)
    resamples = x[resample_indices]
    return resamples

# ==============================================================================================================
# ==============================================================================================================

def bAverage(z, tol = 0, iterate = True, maxIters = 6, C = 6.0):
    '''
    Function for finding the biweight average (C_BI), or biweight location estimator, of
    an array of values (Beers et al. 1990) iteratively, based on user defined tolerance
    or maximum iterations.

    :param z: data array or matrix (if matrix, the biweight average is computed and returned on each row)
    :param tol: desired tolerance (percent difference beteween last and new value). If
                tol = 0, then C_BI will be recalculated until maxIters (NO LONGER IMPLEMENTED)
    :param iterate: whether or not to calculate the statistic iteratively
    :param maxIters: maximum iterations after which to force return
    :param C: the"tuning constant" (default value is 6.0
    :returns: the biweight average of the dataset z
    '''
    
    if(len(z.shape) == 1): z = np.reshape(z, (1, len(z)))
    M = np.reshape( np.median(z, axis=1), (len(z), 1))
    if not iterate: maxIters = 1 
    Cbi = M
    i = 0
    while(i < maxIters):
        i += 1

        #find weighting factor of each entry in array, using the median
        u = (z - Cbi) / np.reshape( (MAD(z, M)*C), (len(z), 1))

        # find weights u_i that meet summation requirement
        mask = abs(u) < 1

        # apply mask to terms of C_BI
        term1 = (z - Cbi)
        term2 = ((1-(u**2))**2)
        term2[~mask] = 0

        # calculate C_BI
        num = np.sum(term1*term2, axis=-1)
        den = np.sum(term2, axis=-1)
        Cbi = M.flatten() + (num/den)

    return Cbi

# ---------------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------------

def bootstrap_bAverage_err(vals, draws = 1000, conf=68.3, avgErr=True, iterate=True):
    '''
    Simulates the error in the biweight location estimator via a bootstrap resampling approach. 
    1000 resamples are drawn from the input data vector, with replacement. For
    each realization, the location estimator is calculated. The error is then estimated from the spread of
    these results.
    
    :param vals: input data vector (must be 1d)
    :param draws: number of realizations to use in the bootstrap (less than 1000 is not suggested)
    :param conf: the confidence within which to return the error - default is 1sigma or 68.3%
    :param avgErr: whether or not to average the high and low confidence bounds to give a symmetric error
    :param iterate: whether or not to calculate the statistic iteratively
    :return: if avgErr == True, return a 2-element list of the dispersion, and the symmetric error
             if avgErr == False, return a 3-element list of the uncertainty lower bound, the dispersion, 
             and the uncertainty upper bound
    '''
    
    size = len(vals)
    avg = bAverage(vals, iterate=iterate)
    results = bAverage( np.random.choice(vals, size=(draws, size), replace=True), iterate=iterate)
    scatter = results - avg

    critPoints = [0+(100-conf)/2, 100-(100-conf)/2]
    critVals = [np.percentile(scatter, critPoints[i], interpolation='nearest') for i in range(2)]
    confidence = [disp - crit for crit in critVals]
    
    if(avgErr):
        err = np.mean(abs(confidence - avg))
        return [avg, err]
    else:
        return [confidence[1], disp, confidence[0]]


# ==============================================================================================================
# ==============================================================================================================


def bVariance(v, iterate=True, tol=0, maxIters=6, C=9.0):
    '''
    Returns the biweight sample variance of a cluster (Sigma_BI), as presented in Ruel et al. 2014,
    from Mosteller&Tukey 1977

    :param v: velocity array or matrix (in which case, the variance is computed and returnred per row)
    :param iter: whether or not to use the iteratively computed biweight mean 
		 rather than the sample median
    :param C: tuning constant (default is C=9.0)
    :return: the biweight variance in v
    '''
    
    if(len(v.shape) == 1): v = np.reshape(v, (1, len(v)))
    median = np.reshape( np.median(v, axis=1), (len(v), 1))
    if(iterate): M = np.reshape( bAverage(v, tol, maxIters), (len(v), 1))
    else: M = median
    
    N = np.shape(v)[1]
    u = (v - M) / np.reshape((MAD(v, median) * C), (len(v), 1))  # usual biweight weighting
    mask = abs(u) < 1  # find weights whose absolute value is less than one
    summand = ((1-(u**2)) * (1-(5*u**2)))
    summand[~mask] = 0
    D = np.sum( summand, axis=-1)

    #apply mask to both terms of Sigma_BI, calculate dispersion
    term1v = ((1-u**2)**4)
    term1v[~mask] = 0
    term2v = ((v - M)**2)
    num = np.sum(term1v*term2v, axis=-1)
    den = D * (D - 1)
    sampleVar = N * (num / den)
    return sampleVar

# ---------------------------------------------------------------------------------------------------

def bDispersion(v, iterate=True, tol=0, maxIters=6, C=9.0):
    '''
    Returns the biweight scale estimator of a dataset (Sigma_BI), which is the square
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
    
    var = bVariance(v, iterate, tol, maxIters, C)
    sigBI = np.sqrt(var)
    return sigBI

# ---------------------------------------------------------------------------------------------------

def bootstrap_bDispersion(vals, size = None, draws = 1000, conf=68.3, avgErr = True,
                          iterate = True, ignoreWarn=False):
    '''
    Simulates the error in the biweight dispersion statistic via a bootstrap resampling approach. 
    1000 resamples are drawn from the input data vector, without replacement (drawing with displacement
    is no advised when computing a second-order statistic, as this can bia the results low). For
    each realization, the dispersion is calculated. The error is then estimated from the spread of
    these results.
    
    :param vals: input data vector (must be 1d)
    :param size: size of each realization - if not specified, defaults to len(vals)/2
    :param draws: number of realizations to use in the bootstrap (less than 1000 is not suggested)
    :param conf: the confidence within which to return the error - default is 1sigma or 68.3%
    :param avgErr: whether or not to average the high and low confidence bounds to give a symmetric error
    :param iterate: whether or not to calculate the biweight dispersion iteratively
    :param ignoreWarn: whether or not to ignore the warning encouraging you to use the gapper 
                       dispersion estimator for samples of size < 15
    :return: if avgErr == True, return a 2-element list of the dispersion, and the symmetric error
             if avgErr == False, return a 3-element list of the uncertainty lower bound, the dispersion, 
             and the uncertainty upper bound
    '''

    if(size == None):
        size = int(len(vals)/2)
        if not ignoreWarn and size < 15: 
            warnings.warn('resample size is less than 15; consider using gapper scale estimator')
    else:
        size = int(size)

    disp = bDispersion(vals, iterate=iterate)
    results = bDispersion( random_choice_noreplace(vals, draws, size) )
    scatter = results - disp

    critPoints = [0+(100-conf)/2, 100-(100-conf)/2]
    critVals = [np.percentile(scatter, critPoints[i], interpolation='nearest') for i in range(2)]
    confidence = [disp - crit for crit in critVals]
    
    if(avgErr):
        err = np.mean(abs(confidence - disp))
        return [disp, err]
    else:
        return [confidence[1], disp, confidence[0]]


# ==============================================================================================================
# ==============================================================================================================


def gDispersion(v):
    '''
    Returns the gapper (velocity) dispersion of a (cluster) dataset (Sigma_G)
    :param v: an array of (velocity) values (assumes unsorted).
    :returns: gapper estimation of dispersion
    '''
    if(len(v.shape) == 1): v = np.reshape(v, (1, len(v)))
    n = np.shape(v)[1]
    # find gaussian weights and gaps
    v = np.sort(v, axis=-1)
    w = np.arange(1,n) * np.arange(n-1,0,-1)
    g = np.diff(v)

    sigG = (np.sqrt(np.pi))/(n*(n-1)) * npm.inner1d(w,g)
    return sigG

# ---------------------------------------------------------------------------------------------------

def bootstrap_gDispersion(vals, size = None, draws = 1000, conf=68.3, avgErr = True):
    '''
    Simulates the error in the gapper dispersion statistic via a bootstrap resampling approach. 
    1000 resamples are drawn from the input data vector, without replacement (drawing with displacement
    is no advised when computing a second-order statistic, as this can bia the results low). For
    each realization, the dispersion is calculated. The error is then estimated from the spread of
    these results.
    
    :param vals: input data vector (must be 1d)
    :param size: size of each realization - if not specified, defaults to len(vals)/2
    :param draws: number of realizations to use in the bootstrap (less than 1000 is not suggested)
    :param conf: the confidence within which to return the error - default is 1sigma or 68.3%
    :param avgErr: whether or not to average the high and low confidence bounds to give a symmetric error
    :return: if avgErr == True, return a 2-element list of the dispersion, and the symmetric error
             if avgErr == False, return a 3-element list of the uncertainty lower bound, the dispersion, 
             and the uncertainty upper bound
    '''

    if(size == None):
        if(repl): size = len(vals)
        else: size = int(len(vals)/2)
    else:
        size = int(size)

    disp = gDispersion(vals)
    results = gDispersion( random_choice_noreplace(vals, draws, size) )
    scatter = results - disp

    critPoints = [0+(100-conf)/2, 100-(100-conf)/2]
    critVals = [np.percentile(scatter, critPoints[i], interpolation='nearest') for i in range(2)]
    confidence = [disp - crit for crit in critVals]
    
    if(avgErr):
        err = np.mean(abs(confidence - disp))
        return [disp, err]
    else:
        return [confidence[1], disp, confidence[0]]


# ==============================================================================================================
# ==============================================================================================================


def dmDispersion(vx, vy, vz):
    '''
    Returns the 1D velocity dispersion estiamted from the full 3d component velocity vectors of each
    system member (as applied to halo simulation particles, or simulated galaxies), as given in 
    Evrard+ 2003

    :param vx: an array of x-component velocity values
    :param vy: an array of x-component velocity values
    :param vz: an array of x-component velocity values
    :return: the 1d dispersion of the input velocities
    '''
    if( sum(abs(np.diff([len(vx), len(vy), len(vz)]))) != 0 ):
        raise ValueError('velocity component arrays must all be of equal length')
    
    Np = len(vx)
    vx_diff = vx - np.median(vx)
    vy_diff = vy - np.median(vy)
    vz_diff = vz - np.median(vz)
    comps = [vx_diff, vy_diff, vz_diff]

    var = 1/(3*Np) * np.sum([comp**2 for comp in comps])
    disp = np.sqrt(var)
    return disp


# ==============================================================================================================
# ==============================================================================================================


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
