#!/usr/bin/env python
import random
root="/home/sronen/plankton/train/"
fr=open('list.txt','r')
ftrain=open('train.txt','w')
fval=open('val.txt','w')
random.seed(1)
plist=[]
for line in fr:
    line=line.replace(root,'')
    plist.append(line)
random.shuffle(plist)
frac=0.8
n=int(len(plist)*frac)
train=plist[:n]
test=plist[n:]
train=train[:(len(train)//10)*10]
test=test[:(len(train)//10)*10]

for line in train:
    info=' '.join(line.split()[0:2])+'\n'
    ftrain.write(info)
for line in test:
    info=' '.join(line.split()[0:2])+'\n'
    fval.write(info)
fr.close()
ftrain.close()
fval.close()



