#!/usr/bin/env python
import random
import h5py
import numpy as np


root_train="train/"
root_test="test/"
fr=open('list.txt','r')
ftest=open('test.txt','r')
fextend=open('extend.txt','w')
fsubmit=open('submit-inet10.csv')

random.seed(2)
plist1=[]
plist2=[]
psub=[]
for i,line in enumerate(fr):
    plist1.append(line)

for j,line in enumerate(ftest):
    plist2.append(line)

for k,line in enumerate(fsubmit):
    if (k>0):
        psub.append(line)

random.shuffle(plist1)

frac=0.8
n=int(len(plist1)*frac)
n1=(n//10)*10
n2=(len(plist1)/10)*10

train=plist1[0:n1]
val=plist1[n1:n2]

NEXT=6000


trainR = [root_train+x for x in train]

fractions=np.zeros(121)
for row in plist1:
    row=row.split(" ")
    category=int(row[1])
    fractions[category]+=1.0

fractions = fractions/sum(fractions)


sorting=[]
bprobs=[]
for row in psub:
    row=row.split(',')
    imgname=row[0]
    
    probs=map(float,row[1:])
    bestarg=np.argmax(probs)
    bestprob=probs[bestarg]
    bprobs.append(bestprob)
    sorting.append([imgname,bestarg,bestprob,probs])

bprobs=np.array(bprobs)
sort=np.argsort(-bprobs)
sorting=[sorting[i] for i in sort]

extension=[]
counter=np.zeros(121)
cuts=np.zeros(121)
ind=[x.split(' ')[0] for x in plist2]
tcount=0
fulfill=np.zeros(121)
for row in sorting:
    categ=row[1]
    if counter[categ] < np.ceil(NEXT*fractions[categ]):
        l= ind.index(row[0][:-3]+'png')
        extension.append(plist2[l])

        cuts[categ]=row[2]
        counter[categ]+=1
        tcount+=1
    else:
        fulfill[categ]=1
    if sum(fulfill)==121:
        break

#print "fractions: ", fractions
print "Cut probs: "
print cuts

lows=[]
for i,c in enumerate(cuts):
    if c<0.5:
        lows.append(i)

print "categories with low confidence: ",lows 

print "Added stats: "
print counter

plist2R = [root_test+x for x in plist2]
extendlist = trainR+extension

print len(extendlist)
"""
for line in train:
    info=' '.join(line.split()[0:3])+'\n'
    ftrain.write(info)
for line in val:
    info=' '.join(line.split()[0:3])+'\n'
    fval.write(info)

for line in plist1:
    info=' '.join(line.split()[0:3])+'\n'
    ffulltrain.write(info)

for line in extendlist:
    info=' '.join(line.split()[0:3])+'\n'
    fextend.write(info)
"""

fr.close()
ftest.close()
fextend.close()



