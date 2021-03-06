#!/usr/bin/env python
import random
import h5py
import numpy as np

root="/home/shai/plankton/train/"
fr=open('list.txt','r')
ftrain=open('train.txt','w')
fval=open('val.txt','w')
ffulltrain=open('fulltrain.txt','w')
random.seed(2)
plist=[]
perm=[]
for i,line in enumerate(fr):
    line=line.replace(root,'')
    plist.append(line)
    perm.append(i)
random.shuffle(perm)
train_h5='/home/shai/plankton/train.h5'
val_h5="/home/shai/plankton/val.h5"
train_h5_names="train.h5.txt"
val_h5_names="val.h5.txt"
list_h5="/home/shai/plankton/list.h5"

with h5py.File(list_h5, 'r') as h5dat:
    X=h5dat['data'].value
    Y=h5dat['label'].value
    X1 = X[perm,:]
    Y1 = Y[perm]

plist = [ plist[i] for i in perm]
frac=0.8
n=int(len(plist)*frac)
n1=(n//10)*10
n2=(len(plist)/10)*10


#n=int(len(plist))
#n1=n
#n2=n

train=plist[0:n1]
val=plist[n1:n2]

Xtrain=X1[0:n1,:]
Xval=X1[n1:n2,:]
Ytrain=Y1[0:n1]
Yval=Y1[n1:n2]

for line in train:
    info=' '.join(line.split()[0:3])+'\n'
    ftrain.write(info)
for line in val:
    info=' '.join(line.split()[0:3])+'\n'
    fval.write(info)

for line in plist:
    info=' '.join(line.split()[0:3])+'\n'
    ffulltrain.write(info)

fr.close()
ftrain.close()
ffulltrain.close()
fval.close()


Xtrain=[float(x) for x in Xtrain]
Xval=[float(x) for x in Xval]

Xtrain=np.array(Xtrain)
Xval=np.array(Xval)
Xtrain=np.reshape(Xtrain,(Xtrain.shape[0],1))
Xval=np.reshape(Xval,(Xval.shape[0],1))

Ytrain=np.array(Ytrain).astype(np.float32)
Yval=np.array(Yval).astype(np.float32)
Ytrain=np.reshape(Ytrain,(Ytrain.shape[0],1))
Yval=np.reshape(Yval,(Yval.shape[0],1))


with h5py.File(train_h5, 'w') as f:
    f['data'] = Xtrain
    f['label'] = Ytrain

with h5py.File(val_h5, 'w') as f:
    f['data'] = Xval
    f['label'] = Yval

with open(train_h5_names, 'w') as f:
    f.write(train_h5 + '\n')
with open(val_h5_names, 'w') as f:
    f.write(val_h5 + '\n')




