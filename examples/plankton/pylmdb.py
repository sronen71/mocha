import lmdb
import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
env=lmdb.open('starvision_test_lmdb')
datum = caffe.proto.caffe_pb2.Datum()
with env.begin() as txn:
    cursor = txn.cursor()           # Cursor on main database.
    cursor.first()
    for key,value in cursor:
        datum.ParseFromString(value)
        arr=np.array( caffe.io.datum_to_array(datum) )
        arr=arr[0]
        print key,datum.label
        plt.imshow(arr,cmap=cm.Greys_r)
        plt.show()

