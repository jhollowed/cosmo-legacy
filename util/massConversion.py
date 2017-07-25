'''
Joe Hollowed
Last edited 2016

Providing set of functions to preform conversions between cluster mass defenitions,
as provided by Sebastian Boquet
'''

import numpy as np
import scipy.optimize as op


# 200crit from Duffy et al 2008, input [M200c/h]
def calC200(m, z):
    # Preform Duffy et al. concentration calculation
    return 5.71 * (m / 2.e12) ** (-0.084) * (1. + z) ** (-0.47)  # full sample


##### Actual input functions
# Input in [Msun/h]
def MDelta_to_M200(m_delta, input_overdensity, z):
    ratio = 200. / input_overdensity
    Mmin = m_delta / ratio / 10.
    Mmax = m_delta / ratio * 10.
    return op.brentq(mdiff_findM200, Mmin, Mmax, args=(m_delta, input_overdensity, z), xtol=1.e-6)


# Input in [Msun/h]
def M200_to_MDelta(m200, output_overdensity, z):
    ratio = output_overdensity / 200.
    Mmin = m200 / ratio / 10.
    Mmax = m200 / ratio * 10.
    return op.brentq(mdiff_findMDelta, Mmin, Mmax, args=(m200, output_overdensity, z), xtol=1.e-6)


##### Functions used for conversion
# calculate the coefficient for NFW aperture mass given c
def calcoef(c):
    return np.log(1 + c) - c / (1 + c)


# root function for concentration
def diffc(c_delta, c200, ratio):
    return calcoef(c200) / calcoef(c_delta) - ratio * (c200 / c_delta) ** 3


def findc(c200, overdensity):
    ratio = 200. / overdensity
    return op.brentq(diffc, .1, 100, args=(c200, ratio), xtol=1.e-6)


# Root function for mass
def mdiff_findM200(m200, m_delta, input_overdensity, z):
    c200 = calC200(m200, z)
    c_delta = findc(c200, input_overdensity)
    return m200 / m_delta - calcoef(c200) / calcoef(c_delta)


def mdiff_findMDelta(mguess, m200, output_overdensity, z):
    conin = calC200(m200, z)
    conguess = findc(conin, output_overdensity)
    return m200 / mguess - calcoef(conin) / calcoef(conguess)
