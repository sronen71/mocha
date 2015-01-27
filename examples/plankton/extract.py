#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import lmdb
import caffe
import matplotlib.cm as cm
import sklearn.metrics as metrics
from sklearn.manifold import TSNE
import sys
import csv
import cv2

caffe_root = '../'
sys.path.insert(0,caffe_root+'python')
#SUBMIT=True
SUBMIT=False

tsize=90 # 64


if SUBMIT:
    TEST_DB="plankton_test_lmdb"
    FEATURES_FILE="test_features.npz"
else:
    TEST_DB='plankton_val_lmdb'
    FEATURES_FILE="val_features.npz"

ENCODE_FILE="/home/shai/mocha/data/plankton/encode.txt"
MODEL_FILE='inet_deploy8.prototxt'
PRETRAINED='inet8.caffemodel'



print "Try to create net..."
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
#	mean=np.load('mean.npy'),
        image_dims=(tsize, tsize))
print "Created net"
net.set_phase_test()
net.set_mode_gpu()

env=lmdb.open(TEST_DB)
datum = caffe.proto.caffe_pb2.Datum()
def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]



def getimages(datum,angle=0,rescale=1.0):
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
            inter=cv2.cv.CV_INTER_CUBIC | cv2.cv.CV_INTER_CUBIC

            f=float(tsize)/actual_size*rescale
            img=np.squeeze(arr)
            M=cv2.getRotationMatrix2D((h/2,h/2),angle,f)
            M[0][2]-=(h-tsize)/2
            M[1][2]-=(h-tsize)/2
            img1 = cv2.warpAffine(img,M,(tsize,tsize),flags=inter)

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


angles=[0]
rescales=[1.0]

pre=[]
for angle in angles:
    for rescale in rescales:
        pre.append((angle,rescale))

predictions=np.zeros(1)
for config in pre:
    angle=config[0]
    rescale=config[1]
    print "angle,rescale:", angle,rescale
    images,labels,keys=getimages(datum,angle,rescale)
    if angle is 0:
        predcitions=np.zeros(len(images))
    predictions1=[]
    features=[]
    count=0
#    for chunk in chunks(images,len(images)/10): 
    for chunk in chunks(images,256): 
        count+=1
        print count
        predictions_chunk = net.predict(chunk,oversample=False)  
        predictions1.extend(predictions_chunk)
        features_chunk=np.squeeze(net.blobs['fc1'].data)
        features.extend(features_chunk)
        
    predictions=predictions+predictions1
    features=np.array(features)
    features=features[:len(images),:]
    features = np.asfarray( features, dtype='float' )
    '''    
    model=TSNE(n_components=2,random_state=0)       
    X=model.fit_transform(features)
    print X.shape
    fig, ax =plt.subplots()
    ax.scatter(X[:,0],X[:,1],c=labels)
    for i,txt in enumerate(labels):
        ax.annotate(txt, (X[i,0], X[i,1]))

    plt.show()
    '''
    np.savez(FEATURES_FILE,features,labels)

    
predictions=predictions/len(pre)


if not SUBMIT:
    predictions=np.array(predictions)
    nlabels=np.zeros(predictions.shape)
    probs=np.zeros(predictions.shape[0])
    for i,label in enumerate(labels):
        nlabels[i,label]=1
        probs[i]=predictions[i,label]
    loss=metrics.log_loss(nlabels,predictions,eps=1e-15)
    print "log loss:", loss


