#! /usr/bin/env python
import csv
import math
import numpy as np

with open('val.csv','rb') as f:
    reader=csv.reader(f)
    names=reader.next()[1:]
    classloss=np.zeros(121)
    classcount=np.zeros(121)
    classent=np.zeros(121)
    prob=np.zeros(121)
    ent=np.zeros(121)
    k=0
    results=[]
    labels=[]
    for row in reader:
        res=np.array(map(float,row[1:-1]))
        results.append(res)
        label=int(row[-1])
        labels.append(label)
        loss=-math.log(res[label]+1e-15)
        prob+=res
        classloss[label]+=loss
        classcount[label]+=1
        ent=-sum(res*np.log(res+1e-15))
        classent[label]+=ent
        k+=1
    totloss=np.sum(classloss)/np.sum(classcount)
    totcounts=sum(classcount)
    classloss=classloss/classcount
    classent=classent/classcount
    s=np.argsort(classloss*classcount)
    classloss=classloss[s]
    classcount=classcount[s]
    prob=prob[s]
    classent=classent[s]
    names=[names[i] for i in s]
    for i,name in enumerate(names):
        print s[i],name, int(classcount[i]),classloss[i],prob[i]/totcounts, \
                classent[i]
    print "Total loss:", totloss
    print sum(prob)




