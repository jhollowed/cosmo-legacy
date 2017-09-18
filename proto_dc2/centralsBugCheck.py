import numpy as np
import matplotlib.pyplot as plt
import h5py
import pdb

f = h5py.File('data/protoDC2_catalog.hdf5', 'r')
halos = list(f.keys())
numCen = np.zeros(len(halos))
numMatch = np.zeros(len(halos))
rat = np.zeros(len(halos))
steps = np.zeros(len(halos))
zs = np.zeros(len(halos))

for i in range(len(halos)):
    halo = f[halos[i]]
    centrals = halo['nodeIsIsolated'][:]
    totCentrals = sum(centrals)
    numCen[i] = totCentrals

    galIdx = halo['nodeIndex'][:]
    hostIdx = halo.attrs['hostIndex']
    steps[i] = halo.attrs['step']
    zs[i] = halo.attrs['z']
    totIdxMatch = sum(galIdx == hostIdx)
    numMatch[i] = totIdxMatch
    
    uniqueIdx = np.unique(galIdx)
    rat[i] = len(uniqueIdx) / len(galIdx)   
    

badMask = numCen == 1 
pdb.set_trace()
