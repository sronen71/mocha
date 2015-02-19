#!/usr/bin/env python


from os import listdir
import os.path as op
from glob import glob
import cv2
from matplotlib import pyplot as pp   
import math
#import h5py
import numpy as np


### config section
TRAIN_DIR="/home/jamie/data/Plankton/train"
TEST_DIR="/home/jamie/data/Plankton/test/"

IMG_SIZE=256
SUFFIX="_"+str(IMG_SIZE)

WRITE_ROTATED_IMAGES=False

# number of levels
USE_LEVEL_NAME = False
LEVEL_NAME = "levela"
LEVEL_SUFFIX = "_"+LEVEL_NAME
# names of level mappings files, from lowest level to highest level
# size should be NUM_LEVELS-1
MAPPING_FILENAMES = ["leveld_levelc_mapping.csv",
                      "levelc_levelb_mapping.csv",
                      "levelb_levela_mapping.csv"]

### end config section


def augment(image,data_suffix,write_files=True):
    tsize=IMG_SIZE
    img=cv2.imread(image,-1)
    img=cv2.bitwise_not(img)
    rows,cols = img.shape
    img_copy=img.copy()

    rows_copy,cols_copy=img_copy.shape   
    actual_size=max(cols_copy,rows_copy)
    f=float(tsize)/max(cols_copy,rows_copy)
    #inter=cv2.cv.CV_INTER_AREA
    inter=cv2.cv.CV_INTER_LINEAR
    if f>1.0:
        f=1.0
        #inter=cv2.cv.CV_INTER_CUBIC
    img_copy=cv2.resize(img_copy,(int(f*cols_copy),int(f*rows_copy)),interpolation=inter) 
    rows_resized,cols_resized=img_copy.shape        
    top=(tsize-rows_resized)//2
    bottom=tsize-top-rows_resized
    left=(tsize-cols_resized)//2
    right=tsize-left-cols_resized
    img_copy = cv2.copyMakeBorder(img_copy,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)        
    if (img_copy.shape[0]!=tsize or img_copy.shape[1] != tsize):
        print "wrong final size ",img.shape
    if write_files:
        split=op.splitext(image)
        name=split[0]+data_suffix+".png"
        cv2.imwrite(name,img_copy)
    return name,f,actual_size

def get_files(root_dir,test_dir,data_suffix,level_suffix,level_map=None,write_rotated_images=True):
    plist=[]
    tlist=[]
    elist=[]
    # X_train=[]
    # Y_train=[]    
    # X_test=[]
    full_suffix = data_suffix+level_suffix
    #list_h5_names="list"+full_suffix+".h5.txt"
    #test_h5_names="test"+full_suffix+".h5.txt"
    #list_h5 = "list" + full_suffix + ".h5"
    #test_h5 = "test" + full_suffix + ".h5"
    list_file="list"+full_suffix+".txt"
    test_file="test"+full_suffix+".txt"
    encode_file="encode"+full_suffix+".txt"

    test_images=glob(op.join(test_dir,"*.jpg"))
    print "Processing test images"

    for test_image  in test_images:
        # print test_image
        name,scaling,actual_size=augment(test_image,data_suffix)

        ti=name.split('/')[-1]
        tlist.append(ti+" 0 "+str(actual_size))
        X_test.append(scaling)
    print "Test images done"

    categories=listdir(op.join(root_dir))
    categories.sort()
    for category in categories:
        category = category.split('/')[-1]


    for encode,category in enumerate(categories):
        print "Processing training images for category %s (%d)" % (category,encode)
        if level_map:
            print "Mapping %s (%d) to %s (%d)" % \
                (category,encode,level_map[category][1],level_map[category][0])
            encode = level_map[category][0]
            elist.append(",".join([str(encode),level_map[category][1]]))
        else:
            elist.append(",".join([str(encode),category]))            

        images=glob(op.join(root_dir,category,'*.jpg'))
        
        for image in images:
            # print image
            aug,scaling,actual_size=augment(image,data_suffix)
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

    # X_train=np.asarray(X_train)
    # X_train=np.expand_dims(X_train, axis=1)
    # X_test=np.asarray(X_test)
    # X_test=np.expand_dims(X_test, axis=1)
    # Y_train=np.asarray(Y_train,dtype=np.float32)
    # with h5py.File(list_h5, 'w') as f:
    #     f['data'] = X_train
    #     f['label'] = Y_train
    # with open(list_h5_names, 'w') as f:
    #     f.write(list_h5 + '\n')
    # with h5py.File(test_h5, 'w') as f:
    #     f['data'] = X_test
    #     f['label'] = np.zeros(X_test.shape[0],dtype=np.float32)
    # with open(test_h5_names, 'w') as f:
    #     f.write(test_h5 + '\n')

def main():
    if not USE_LEVEL_NAME:
        level_suffix = ""
        get_files(TRAIN_DIR,TEST_DIR,SUFFIX,level_suffix,write_rotated_images=WRITE_ROTATED_IMAGES)
    elif USE_LEVEL_NAME and len(MAPPING_FILENAMES) == 0:
        get_files(TRAIN_DIR,TEST_DIR,SUFFIX,LEVEL_SUFFIX,write_rotated_images=WRITE_ROTATED_IMAGES)
    else:
        get_files(TRAIN_DIR,TEST_DIR,SUFFIX,LEVEL_SUFFIX,\
            level_map=level_map.get_level_map(MAPPING_FILENAMES),write_rotated_images=WRITE_ROTATED_IMAGES)

if __name__ == "__main__":
    main()
