#!/usr/bin/env python


from os import listdir
import os.path as op
from glob import glob
import cv2
from matplotlib import pyplot as pp   
import math
import h5py
import numpy as np

def augment(image):
    tsize=400
    img=cv2.imread(image,-1)

    img=cv2.bitwise_not(img)
    rows,cols = img.shape
    img1=img.copy()

    rows1,cols1=img1.shape   
    actual_size=max(cols1,rows1)
    f=float(tsize)/max(cols1,rows1)
    #inter=cv2.cv.CV_INTER_AREA
    inter=cv2.cv.CV_INTER_LINEAR
    if f>1.0:
        f=1.0
        #inter=cv2.cv.CV_INTER_CUBIC
    img1=cv2.resize(img1,(int(f*cols1),int(f*rows1)),interpolation=inter) 
    rows2,cols2=img1.shape        
    top=(tsize-rows2)//2
    bottom=tsize-top-rows2
    left=(tsize-cols2)//2
    right=tsize-left-cols2
    img1 = cv2.copyMakeBorder(img1,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)        
    if (img1.shape[0]!=tsize or img1.shape[1] != tsize):
        print "wrong final size ",img.shape
    split1=op.splitext(image)
    name=split1[0]+".png"

    cv2.imwrite(name,img1)
    return name,f,actual_size

def main():
    ROOT_DIR="/home/shai/plankton/train"
    TEST_DIR="/home/shai/plankton/test"
    plist=[]
    tlist=[]
    elist=[]
    X_train=[]
    Y_train=[]    
    X_test=[]
    list_h5_names="list.h5.txt"
    test_h5_names="test.h5.txt"
    list_h5="/home/shai/plankton/list.h5"
    test_h5="/home/shai/plankton/test.h5"
    list_file="list.txt"
    test_file="test.txt"
    encode_file='encode.txt'
    test_images=glob(op.join(TEST_DIR,"*.jpg"))
    for test_image  in test_images:
        print test_image
        name,scaling,actual_size=augment(test_image)

        ti=name.split('/')[-1]
        tlist.append(ti+" 121 "+str(actual_size))
        X_test.append(scaling)
                
    categories=listdir(op.join(ROOT_DIR))
    categories.sort()

    for encode,category in enumerate(categories):
        elist.append(",".join([str(encode),category.split('/')[-1]]))

        images=glob(op.join(ROOT_DIR,category,'*.jpg'))
        
        for image in images:
            print image
            aug,scaling,actual_size=augment(image)
            aug='/'.join(aug.split('/')[-2:])
            plist.append(" ".join([aug,str(encode),str(actual_size)]))
            X_train.append(scaling)
            Y_train.append(encode)


    f1=open(list_file,'w') 
    for line in plist:
        f1.write(line+'\n')
    f1.close()

    f2=open(test_file,'w') 
    for line in tlist:
        f2.write(line+'\n')
    f2.close()


    f3=open(encode_file,'w') 
    for line in elist:
        f3.write(line+'\n')
    f3.close()

    X_train=np.asarray(X_train)
    X_train=np.expand_dims(X_train, axis=1)
    X_test=np.asarray(X_test)
    X_test=np.expand_dims(X_test, axis=1)
    Y_train=np.asarray(Y_train,dtype=np.float32)
    with h5py.File(list_h5, 'w') as f:
        f['data'] = X_train
        f['label'] = Y_train
    with open(list_h5_names, 'w') as f:
        f.write(list_h5 + '\n')
    with h5py.File(test_h5, 'w') as f:
        f['data'] = X_test
        f['label'] = np.zeros(X_test.shape[0],dtype=np.float32)
    with open(test_h5_names, 'w') as f:
        f.write(test_h5 + '\n')

    

if __name__ == "__main__":
    main()
