
'''
Joe Hollowed
COSMO-HEP 2017
'''

import numpy as np
import gio2hdf_cores as sort
import stack_cores as stack
import stack_analysis as analyze
import time
import sys, traceback

class noprint(object):
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self
    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            raise
    def write(self, x): pass


def runPipeline(catalog = 0, runUnprocessed = False):

    print('\nStarting...')
    startAll = time.time()
    startProcessed = time.time()
    
    print('\nsorting core data & processing')
    start = time.time()
    #with noprint():
    #    sort.group_halos(catalog=catalog, process = True)
    end = time.time()
    print('done; took {} s'.format(end-start))

    print('stacking processed core data')
    start = time.time()
    #with noprint():
    #    stack.stackCores(catalog=catalog, processed = True)
    end = time.time()
    print('done; took {} s'.format(end-start))
    
    print('segregating core data into mixed bins')
    start = time.time()
    #with noprint():
    #    analyze.stack_segregation(catalog=catalog, processed = True)
    end = time.time()
    print('done; took {} s'.format(end-start))
    
    print('segregating core data into discrete bins')
    start = time.time()
    #with noprint():
    analyze.segregation_binning(catalog=catalog, processed = True)
    end = time.time()
    print('done; took {} s'.format(end-start))

    endProcessed = time.time()
    print('processed cores have gone through entire pipeline; took {} s'
          .format(endProcessed - startProcessed))

    if(runUnprocessed):
        startUnprocessed = time.time()

        print('\nsorting core data')
        start = time.time()
        with noprint():
            sort.group_halos(catalog=catalog, process = False)
        end = time.time()
        print('done; took {} s'.format(end-start))

        print('stacking unprocessed core data')
        start = time.time()
        with noprint():
            stack.stackCores(catalog=catalog, processed = False)
        end = time.time()
        print('done; took {} s'.format(end-start))
        
        print('segregating core data into mixed bins')
        start = time.time()
        with noprint():
            analyze.stack_segregation(catalog=catalog, processed = False)
        end = time.time()
        print('done; took {} s'.format(end-start))
        
        print('segregating core data into discrete bins')
        start = time.time()
        with noprint():
            analyze.segregation_binning(catalog=catalog, processed = False)
        end = time.time()
        print('done; took {} s'.format(end-start))

        endUnprocessed = time.time()
        print('processed cores have gone through entire pipeline; took {} s'
              .format(endUnprocessed - starUnprocessed))

    endAll = time.time()

    print('\nDONE. everything took {} s'.format(endAll - startAll))
    return 0
