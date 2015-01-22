#! /usr/bin/env python
import numpy as np
from numpy.random import randn
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as pp
tsize=256
PATH='/home/shai/plankton/fields/'
sigma=15
for i in range(4000):
    field=randn(tsize,tsize)
    field=np.pi*sigma*gaussian_filter(field,sigma=sigma)
    name=PATH+'field'+str(i)+'.fld'
    fid=open(name,"w")
    field.tofile(fid)
    fid.close()
#pp.imshow(field)
#pp.show()

