#!/usr/bin/env python
import random
import h5py
import numpy as np

root_train="train/"
root_test="test/"
fr=open('list.txt','r')
ftest=open('test.txt','r')
ftrain=open('train.txt','w')
fval=open('val.txt','w')
ffulltrain=open('fulltrain.txt','w')
fsemi=open('semi.txt','w')
random.seed(2)
plist1=[]
plist2=[]
for i,line in enumerate(fr):
    plist1.append(line)

for j,line in enumerate(ftest):
    plist2.append(line)

random.shuffle(plist1)
random.shuffle(plist2)

frac=0.8
n=int(len(plist1)*frac)
n1=(n//10)*10
n2=(len(plist1)/10)*10

train=plist1[0:n1]
val=plist1[n1:n2]

addsemi=30000


trainR = [root_train+x for x in train]
plist2R = [root_test+x for x in plist2]
semilist = trainR+plist2R[:addsemi]
random.shuffle(semilist)
print len(semilist)

for line in train:
    info=' '.join(line.split()[0:3])+'\n'
    ftrain.write(info)
for line in val:
    info=' '.join(line.split()[0:3])+'\n'
    fval.write(info)

for line in plist1:
    info=' '.join(line.split()[0:3])+'\n'
    ffulltrain.write(info)

for line in semilist:
    info=' '.join(line.split()[0:3])+'\n'
    fsemi.write(info)


fr.close()
ftest.close()
ftrain.close()
ffulltrain.close()
fval.close()
fsemi.close()



