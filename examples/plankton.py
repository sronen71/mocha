import numpy as np
import matplotlib.pyplot as plt
import lmdb
import caffe
import matplotlib.cm as cm
import sklearn.metrics as metrics
import sys
import csv

caffe_root = '../'
sys.path.insert(0,caffe_root+'python')


SUBMIT=True

if SUBMIT:
    TEST_DB="plankton/plankton_test_lmdb"
    ENCODE_FILE="plankton/encode.txt"
    SUBMIT_FILE="plankton/submit.csv"
else:
    TEST_DB='plankton/plankton_val_lmdb'
MODEL_FILE='plankton/inet_deploy1.prototxt'
PRETRAINED='plankton/inet_iter_15000.caffemodel'


print "Try to create net..."
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
	mean=np.load('plankton/mean.npy'),
        image_dims=(48, 48))
print "Created net"
net.set_phase_test()
net.set_mode_gpu()

env=lmdb.open(TEST_DB)
datum = caffe.proto.caffe_pb2.Datum()
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
        images.append(arr.astype(np.float32)) # python wrapper needs float32, e.g for resize
        labels.append(datum.label)
        keys.append(key)
        #plt.imshow(img,cmap=cm.Greys_r)
        #plt.show()
    print "prediction, #images: ", len(images)    
    predictions = net.predict(images,oversample=False)  # predict takes any number of images, and formats them for the Caffe net automatically
    if SUBMIT:
        import csv
        with open(ENCODE_FILE, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            encode=[]
            for row in reader:
                encode.append(row[1])
        with open(SUBMIT_FILE,'w') as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow(["image"]+encode)
            for i,predict in enumerate(predictions):
                name=keys[i].split("R")[0]+".jpg"
                print name
                name=name.split("_")[1]
                writer.writerow([name]+predict.tolist()) 
    
    else:
        loss=metrics.log_loss(labels,predictions)
        print "loss",loss
