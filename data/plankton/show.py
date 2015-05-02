#!/usr/bin/env python


from os import listdir
import os.path as op
from glob import glob
import cv2
from matplotlib import pyplot as pp   
import math
import h5py
import numpy as np


def main():
    ROOT_DIR="/home/shai/plankton/train"
    TEST_DIR="/home/shai/plankton/test"
    plist=[]
    tlist=[]
    elist=[]
    X_train=[]
    Y_train=[]    
    X_test=[]
    test_images=glob(op.join(TEST_DIR,"*.jpg"))
    for test_image  in test_images:
        print test_image
        img=255-cv2.imread(test_image,-1)
        print np.min(img),np.max(img)

if __name__ == "__main__":
    main()
