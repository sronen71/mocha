#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import lmdb
import caffe
import matplotlib.cm as cm
import sklearn.metrics as metrics
import sys
import csv
import cv2

caffe_root = '../'
sys.path.insert(0,caffe_root+'python')
#SUBMIT=True
SUBMIT=False

oversample=True
supersample=True

#oversample=False
#supersample=False

tsize=64 # 64


if SUBMIT:
    TEST_DB="plankton/plankton_test_lmdb"
    SUBMIT_FILE="plankton/submit.csv"
else:
    TEST_DB='plankton/plankton_val_lmdb'

ENCODE_FILE="/home/shai/mocha/data/plankton/encode.txt"
MODEL_FILE='plankton/inet_deploy11.prototxt'
PRETRAINED='plankton/inet_model_val_11.caffemodel'
#pretrained='plankton/inet10-iter35000.caffemodel'
#PRETRAINED='plankton/inet11-full-iter56000.caffemodel'




print "Try to create net..."
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
#	mean=np.load('plankton/mean.npy'),
        image_dims=(tsize, tsize))
print "Created net"
net.set_phase_test()
net.set_mode_gpu()

env=lmdb.open(TEST_DB)
datum = caffe.proto.caffe_pb2.Datum()

def write_submit(filename,keys,predictions,labels=[]):
    import csv
    with open(ENCODE_FILE, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        encode=[]
        for row in reader:
            encode.append(row[1])
    with open(filename,'w') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(["image"]+encode)
        for i,predict in enumerate(predictions):
            name=(keys[i].split("."))[0]+".jpg"
            name=''.join(name.split('_')[1:])
            row=[name]+predict.tolist()
            if labels:
                label=[str(labels[i])]
                row=row+label
            writer.writerow(row) 


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]



def getimages(datum,angle=0,rescale=1.0,lumin=1.0):
    with env.begin() as txn:
        cursor = txn.cursor()           # Cursor on main database.
        cursor.first()
        images=[]
        labels=[]
        keys=[]
        
        for key,value in cursor:
            datum.ParseFromString(value)
            arr=caffe.io.datum_to_array(datum)  # [c,h,w]
            arr=arr.reshape(arr.shape[1],arr.shape[2],arr.shape[0]) # [h,w,c]
            h=arr.shape[0]
            actual_size=datum.actual_size
            if actual_size>arr.shape[0]:
                actual_size=arr.shape[0]    
            #inter=cv2.cv.CV_INTER_LINEAR
            inter=cv2.cv.CV_INTER_CUBIC + cv2.cv.CV_WARP_FILL_OUTLIERS

            f=float(tsize)/actual_size*rescale
            img=np.squeeze(arr)
            M=cv2.getRotationMatrix2D((h/2,h/2),angle,f)
            M[0][2]-=(h-tsize)/2
            M[1][2]-=(h-tsize)/2
            img=img.astype(np.float32)
            img1 = cv2.warpAffine(img,M,(tsize,tsize),flags=inter,borderMode=cv2.BORDER_CONSTANT)
            #img1=cv2.resize(img,(0,0),fx=f,fy=f,interpolation=inter)
            #l=(img1.shape[0]-tsize)/2
            #img1=img1[l:l+tsize,l:l+tsize]

            img1=lumin*img1;
            img1=np.clip(img1,-1000,255)
            img1=img1.astype(np.int)

            #print np.min(img),np.max(img),np.min(img1),np.max(img1)
            arr=img1[:,:,None]


            images.append(arr.astype(np.float32)) # python wrapper needs float32, e.g for resize
            labels.append(datum.label)
            keys.append(key)
            #plt.imshow(img,cmap=cm.Greys_r)
            #plt.show()
            #plt.imshow(img1,cmap=cm.Greys_r)
            #plt.show()
        print "#images: ", len(images)    
    return images,labels,keys


if supersample:
    angles=range(0,360,45)
    rescales=[0.9,1.0,1.1]
    lumins=[1.0] # didn't help much to diversify this
#  rescales=[1.0]
else:
    angles=[0]
    rescales=[1.0]
    lumins=[1.0]

pre=[]
for angle in angles:
    for rescale in rescales:
        for lumin in lumins:
            pre.append((angle,rescale,lumin))

predictions=np.zeros(1)
for k,config in enumerate(pre):
    angle=config[0]
    rescale=config[1]
    lumin=config[2]
    print "angle,rescale,lumin :", angle,rescale,lumin
    images,labels,keys=getimages(datum,angle,rescale,lumin)
    if k==0:
        predcitions=np.zeros(len(images))
    predictions1=[]
    count=0
    for chunk in chunks(images,len(images)/10): 
        count+=1
        print count
        predictions_chunk = net.predict(chunk,oversample=oversample)  
        predictions1.extend(predictions_chunk)
    predictions=predictions+predictions1
    
predictions=predictions/len(pre)


if SUBMIT:
    write_submit(SUBMIT_FILE,keys,predictions)
else:
    predictions=np.array(predictions)
    nlabels=np.zeros(predictions.shape)
    probs=np.zeros(predictions.shape[0])
    for i,label in enumerate(labels):
        nlabels[i,label]=1
        probs[i]=predictions[i,label]
    score=-np.mean(np.log(probs))
    loss=metrics.log_loss(nlabels,predictions,eps=1e-15)
    print "log loss score:", score

    print "regularized score:",loss
    write_submit('val.csv',keys,predictions,labels)
