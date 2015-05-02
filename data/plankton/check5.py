#!/usr/bin/env python
import random
import h5py
import numpy as np

root="/home/shai/plankton/train/"
list_h5='/home/shai/plankton/list.h5'
train_h5='/home/shai/plankton/train.h5'
val_h5='/home/shai/plankton/val.h5'



with h5py.File(train_h5, 'r') as h5dat:
    X=h5dat['data'].value
    Y=h5dat['label'].value


for i in range(len(X)):
    print X[i],Y[i]

print np.mean(X),np.median(X)
