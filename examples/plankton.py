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

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


SUBMIT=True
if SUBMIT:
    TEST_DB="plankton/plankton_test_lmdb"
    ENCODE_FILE="/home/ubuntu/mocha/caffe/data/plankton/encode.txt"
    SUBMIT_FILE="plankton/submit.csv"
else:
    TEST_DB='plankton/plankton_val_lmdb'
MODEL_FILE='plankton/inet_deploy6.prototxt'
PRETRAINED='plankton/inet7.caffemodel'


print "Try to create net..."
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
#	mean=np.load('plankton/mean.npy'),
        image_dims=(66, 66))
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
    #predictions = net.predict(images,oversample=False)  # predict takes any number of images, and formats them for the Caffe net automatically
    predictions=[]   
    count=0
    for chunk in chunks(images,len(images)/1): 
        count+=1
        print count
        predictions1 = net.predict(chunk,oversample=False)  
        predictions.extend(predictions1)

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
                name=(keys[i].split("."))[0]+".jpg"
                name=name.split('_')[1]
                print name
                writer.writerow([name]+predict.tolist()) 
    
    else:
        loss=metrics.log_loss(labels,predictions)
        print "loss",loss
