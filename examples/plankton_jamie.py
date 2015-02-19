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
import time

caffe_root = '../'
sys.path.insert(0,caffe_root+'python')

## Three configurations:
## SUBMIT=True and DATA_SUFFIX="_train": bagging
## SUBMIT=True and DATA_SUFFIX="_full": full model submission
## SUBMIT=False and DATA_SUFFIX="_train": out-of-sample loss estimate
## SUBMIT=False and DATA_SUFFIX="_full": partial in-sample loss (not preferred)
SUBMIT=False
#DATA_SUFFIX="_full"
DATA_SUFFIX="_train"


ORIG_IMAGE_SIZE=256
RESIZED_IMAGE_SIZE=64
LEVEL_SUFFIX=""
SUFFIX="_"+str(ORIG_IMAGE_SIZE)+LEVEL_SUFFIX
MODEL="256_inet10"
MODEL_SUFFIX="_"+MODEL
NUM_ITERS=43500
SPLIT_SUFFIX=""
ITERS_SUFFIX=DATA_SUFFIX+SPLIT_SUFFIX+"_iter_"+str(NUM_ITERS)
EXTRA_OUT_SUFFIX=""
OVERSAMPLE=True
SUPERSAMPLE=True
if OVERSAMPLE:
    EXTRA_OUT_SUFFIX += "_oversampled" # only used when submitting    
if SUPERSAMPLE:
    EXTRA_OUT_SUFFIX += "_supersampled" # only used when submitting    
    


if SUBMIT:
    TEST_DB="plankton/plankton_test_lmdb"+SUFFIX
    SUBMIT_FILE="plankton/submissions/submit"+DATA_SUFFIX+MODEL_SUFFIX+SPLIT_SUFFIX+EXTRA_OUT_SUFFIX+".csv"
else:
    TEST_DB="plankton/plankton_val_lmdb"+SUFFIX+SPLIT_SUFFIX
    VAL_FILE="plankton/submissions/val"+MODEL_SUFFIX+SPLIT_SUFFIX+EXTRA_OUT_SUFFIX+".csv"
    VAL_WLABELS_FILE="plankton/submissions/val_w_labels"+MODEL_SUFFIX+SPLIT_SUFFIX+EXTRA_OUT_SUFFIX+".csv"

MAPPING_PROBABILITIES = ""
MAPPING_FILENAMES = []
#MAPPING_PROBABILITIES = "plankton/submissions/submit_inet7.csv"
#MAPPING_PROBABILITIES = "plankton/submissions/submit_48_3x3-6-5-3_iter20000_oversampled.csv"
# MAPPING_FILENAMES = ["../data/plankton/leveld_levelc_mapping.csv",
#                      "../data/plankton/levelc_levelb_mapping.csv",
#                      "../data/plankton/levelb_levela_mapping.csv"]
# MAPPING_FILENAMES = ["../data/plankton/leveld_levelc_mapping.csv",
#                  "../data/plankton/levelc_levelb_mapping.csv"]
#MAPPING_FILENAMES = ["../data/plankton/leveld_levelc_mapping.csv"]


ENCODE_FILE="plankton/encode"+SUFFIX+".txt"
MODEL_FILE="plankton/models/"+MODEL+"/inet_deploy.prototxt"
PRETRAINED="plankton/snapshots/"+MODEL+"/inet"+ITERS_SUFFIX+".caffemodel"

print "Using test DB:", TEST_DB
print "Using deploy model:", MODEL_FILE
print "Using pretrained model:", PRETRAINED


print "Try to create net..."
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
    # mean=np.load("plankton/mean"+DATA_SUFFIX+SUFFIX+".npy"),
        image_dims=(RESIZED_IMAGE_SIZE, RESIZED_IMAGE_SIZE))
print "Created net"
net.set_phase_test()
net.set_mode_gpu()

images=[]
labels=[]
keys=[]

def image_name_from_db_key(db_key):
    # remove everything before first "_"
    key = "_".join(db_key.split("_")[1:])
    if "/" in key:
        # remove everything before the slash
        key = key.split("/")[1]
    return key.split("R")[0]+".jpg"

def write_submit(filename,keys,predictions,labels=[]):
    import csv
    with open(ENCODE_FILE, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        encode=[]
        for row in reader:
            encode.append(row[1])

    # map out the labels to higher level (if necessary)
    if len(MAPPING_FILENAMES) > 0 and len(MAPPING_PROBABILITIES) > 0:
        print "Mapping higher-level labels to expected labels"
        mapped_predictions = []
        # prob_map = extract.get_derived_probabilities(MAPPING_PROBABILITIES,MAPPING_FILENAMES)
        prob_map = extract.get_derived_probabilities_ind(MAPPING_PROBABILITIES,MAPPING_FILENAMES)
        mapped_encode = []
        mapped_encode_to_index = {}

        example_prob_map = prob_map[prob_map.keys()[0]]
        for high_level_key in example_prob_map:
            for low_level_key in example_prob_map[high_level_key]:
                # low_level_value = prob_map[high_level_key][low_level_key]
                mapped_encode_to_index[low_level_key] = len(mapped_encode)
                mapped_encode.append(low_level_key)
        # print "encode",len(encode),encode
        # print "mapped_encode",len(mapped_encode),mapped_encode
        # print "mapped_encode_to_index",len(mapped_encode_to_index),mapped_encode_to_index
        # print "prob_map",len(prob_map),prob_map
        printed_warning = False;
        total_predictions = predictions.shape[0]
        predictions_processed = 0
        for i,predict in enumerate(predictions):
            mapped_prediction_row = np.zeros((1,len(mapped_encode)))
            # if i < 2:
            #     print "predict",len(predict),predict
            if len(predict) > len(encode):
                # we accidentally trained with too many outputs.  cut off all the extras
                if not printed_warning:
                    print "Prediction has too many classes (%d).  Cutting down to expected (%d)" % \
                        (len(predict),len(encode))
                    printed_warning = True;
                predict = predict[0:len(encode)]
                predict /= np.sum(predict)
            # if i < 2:
            #     print "re-normalized predict",len(predict),predict
            name = image_name_from_db_key(keys[i])
            ind_prob_map = prob_map[name]    
            for j in range(len(predict)):
                # print j
                high_level_key = encode[j]
                # print high_level_key
                hl_prob_map = ind_prob_map[high_level_key]
                # print "hl_prob_map",len(hl_prob_map),hl_prob_map
                # print mapped_prediction_row.shape
                for low_level_key in hl_prob_map:
                    # print low_level_key
                    mapped_prediction_row[0,mapped_encode_to_index[low_level_key]] = \
                        predict[j] * hl_prob_map[low_level_key]
            mapped_predictions.append(mapped_prediction_row)

            predictions_processed +=1 
            if predictions_processed * 10 % total_predictions == 0:
                print "Mapping progress: %d / %d" % (predictions_processed,total_predictions)

        # print "original predictions",predictions.shape,predictions[0:2,0:len(encode)],\
        #     sum(predictions[0,:]),sum(predictions[1,:])
        
        predictions = np.vstack(mapped_predictions)
        # print "mapped predictions",predictions.shape,predictions[0:2,:],\
        #     sum(predictions[0,:]),sum(predictions[1,:])
        encode = mapped_encode
        print "Mapping complete"

    print "Writing submit file", filename

    with open(filename,'w') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(["image"]+encode)
        for i,predict in enumerate(predictions):
            # the strings in the keys array is of form <idx>_<key>_<data_size>.jpg
            # extract the key part, and add .jpg
            name = keys[i].split("_")[1]+".jpg"
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



def getimages(test_db,image_size,angle=0,rescale=1.0):
    env=lmdb.open(test_db)
    datum = caffe.proto.caffe_pb2.Datum()
    images=[]
    labels=[]
    keys=[]
    with env.begin() as txn:
        cursor = txn.cursor()           # Cursor on main database.
        cursor.first()
        
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

            f=float(image_size)/actual_size*rescale
            img=np.squeeze(arr)
            M=cv2.getRotationMatrix2D((h/2,h/2),angle,f)
            M[0][2]-=(h-image_size)/2
            M[1][2]-=(h-image_size)/2
            img=img.astype(np.float32)
            img1 = cv2.warpAffine(img,M,(image_size,image_size),flags=inter,borderMode=cv2.BORDER_CONSTANT)
            #img1=cv2.resize(img,(0,0),fx=f,fy=f,interpolation=inter)
            #l=(img1.shape[0]-image_size)/2
            #img1=img1[l:l+image_size,l:l+image_size]
            img1=np.clip(img1,-1000,255)
            #img1=np.floor(img1)
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


images,labels,keys=getimages(TEST_DB,RESIZED_IMAGE_SIZE,0,1.0)

if SUPERSAMPLE:
    angles=range(0,360,45)
    rescales=[0.95,1.0,1.05]
    # angles=[0]
    # rescales=[1.0]
else:
    angles=[0]
    rescales=[1.0]

pre=[]
for angle in angles:
    for rescale in rescales:
        pre.append((angle,rescale))

predictions=np.zeros(1)
remaining_configs = len(pre)
for config in pre:
    angle=config[0]
    rescale=config[1]
    config_start = time.clock()
    print "Fetching images for angle:",angle,"rescale:",rescale
    img_start = time.clock()
    images,labels,keys=getimages(TEST_DB,RESIZED_IMAGE_SIZE,angle,rescale)
    img_end = time.clock()
    print "Image read time:",img_end-img_start,"seconds"
    if SUBMIT:
        print "prediction, #images:",len(images)
    else:
        print "prediction, #images:",len(images),"#classes(true):",len(set(labels))
    # if angle is 0:
        # predictions=np.zeros(len(images))
    predictions1=[]
    count=0
    NUM_CHUNKS=10
    print "Computing predictions for angle:",angle,"rescale:",rescale
    for chunk in chunks(images,len(images)/NUM_CHUNKS): 
        count+=1
        chunk_start = time.clock()
        predictions_chunk = net.predict(chunk,oversample=OVERSAMPLE)  
        chunk_end = time.clock()
        print "Chunk",count,"of",NUM_CHUNKS,"took",chunk_end-chunk_start,"seconds"
        predictions1.extend(predictions_chunk)

    predictions=predictions+predictions1
    config_end = time.clock()
    print "Processing took",(config_end-config_start)/60,"minutes for angle:",angle,"rescale:",rescale
    remaining_configs -= 1
    if remaining_configs != 0:
        print "Expected time remaining:",remaining_configs*(config_end-config_start)/60,"minutes"

    
predictions=predictions/len(pre)


if SUBMIT:
    write_submit(SUBMIT_FILE,keys,predictions)
    print "Wrote submission to",SUBMIT_FILE
else:
    # num_prediction_classes = predictions.shape[1]
    # num_true_classes = len(set(labels))
    # if num_prediction_classes != num_true_classes:
    #     # different number of labels
    #     print "Shape mismatch, removing extra columns from predictions"
    #     label_exists = [0] * num_prediction_classes
    #     for label in labels:
    #         label_exists[label] = 1
    #     pred_col_shape = (predictions.shape[0],1)
    #     prediction_columns = []
    #     for class_idx in range(num_prediction_classes):
    #         if label_exists[class_idx]:
    #             prediction_columns.append(predictions[:,class_idx].reshape(pred_col_shape))
    #     predictions = np.hstack(prediction_columns)
    #     print "Removed %d column(s)" % (len(label_exists)-sum(label_exists))
    # assert predictions.shape[1] == num_true_classes

    predictions=np.array(predictions)
    nlabels=np.zeros(predictions.shape)
    probs=np.zeros(predictions.shape[0])
    for i,label in enumerate(labels):
        # print i, label
        nlabels[i,label]=1
        probs[i]=predictions[i,label]
    score=-np.mean(np.log(probs))
    loss=metrics.log_loss(nlabels,predictions,eps=1e-15)
    print "log loss score:", score
    print "regularized score:",loss

    write_submit(VAL_FILE,keys,predictions)
    print "Wrote validation file",VAL_FILE
    write_submit(VAL_WLABELS_FILE,keys,predictions,labels)
    print "Wrote validation (with labels) file",VAL_WLABELS_FILE
