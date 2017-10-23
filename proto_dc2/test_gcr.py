import pdb

import sys
sys.path.insert(0, '/home/jphollowed/code/repos/gcr-catalogs')
sys.path.insert(0, '/home/jphollowed/code/repos/gcr-catalog-reader')

import GCRCatalogs
cat = GCRCatalogs.load_catalog('proto-dc2-clusters-v1.0')
q = cat.get_quantities([''])
print(q)

