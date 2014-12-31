#!/usr/bin/env python


from os import listdir
import os.path as op
from glob import glob
import cv2
from matplotlib import pyplot as pp   
import math

def augment(image,rotate=False):
    tsize=48
    auglist=[]
    img=cv2.imread(image,-1)

    img=cv2.bitwise_not(img)
    rows,cols = img.shape
    angles=[0]
    if rotate:
        angles=range(0,360,360/8)
    for angle in angles:

        if angle!=0:
            M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            rangle=angle/180.*math.pi
            cols1=math.ceil(cols*math.fabs(math.cos(rangle))+rows*math.fabs(math.sin(rangle)))
            rows1=math.ceil(cols*math.fabs(math.sin(rangle))+rows*math.fabs(math.cos(rangle)))
            cols1=int(cols1)
            rows1=int(rows1)
            M[0][2] += cols1/2-cols/2
            M[1][2] += rows1/2-rows/2
            img1 = cv2.warpAffine(img,M,(cols1,rows1))
        else:
            img1=img.copy()

        rows1,cols1=img1.shape    
        f=float(tsize)/max(cols1,rows1)
        if f>1.0:
            f=1.0
        img1=cv2.resize(img1,(int(f*cols1),int(f*rows1))) 
        rows2,cols2=img1.shape        
        top=(tsize-rows2)//2
        bottom=tsize-top-rows2
        left=(tsize-cols2)//2
        right=tsize-left-cols2
        img1 = cv2.copyMakeBorder(img1,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)        
        if (img1.shape[0]!=tsize or img1.shape[1] != tsize):
            print "wrong final size ",img.shape
        split=op.splitext(image)
        name=split[0]+"R"+str(angle)+".png"
 
        auglist.append(name)
        cv2.imwrite(name,img1)
    return auglist

def main():
    ROOT_DIR="/home/sronen/plankton/train"
    TEST_DIR="/home/sronen/plankton/test/"
    plist=[]
    tlist=[]
    elist=[]
    list_file="list.txt"
    test_file="test.txt"
    encode_file='encode.txt'
    test_images=glob(op.join(TEST_DIR,"*.jpg"))
    for test_image  in test_images:
        print test_image
        auglist=augment(test_image,rotate=False)
        name=auglist[0]

        ti=name.replace(TEST_DIR,'')
        tlist.append(ti+" 0")
    categories=listdir(op.join(ROOT_DIR))
    for encode,category in enumerate(categories):
        elist.append(",".join([str(encode),category.split('/')[-1]]))

        images=glob(op.join(ROOT_DIR,category,'*.jpg'))

        for image in images:
            print image
            auglist=augment(image,rotate=True)
            for aug in auglist:
                plist.append(" ".join([aug,str(encode)]))

            


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



if __name__ == "__main__":
    main()
