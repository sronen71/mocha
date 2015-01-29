#! /usr/bin/env python
import csv
import math
import numpy as np

with open('val.csv','rb') as f:
    reader=csv.reader(f)
    names=reader.next()[1:]
    classloss=np.zeros(121)
    classcount=np.zeros(121)
    classmax=np.zeros(121)
    classent=np.zeros(121)
    prob=np.zeros(121)
    ent=np.zeros(121)
    k=0
    results=[]
    labels=[]
    for row in reader:
        res=np.array(map(float,row[1:-1]))
        label=int(row[-1])
        if max(res)<0.4 or np.argmax(res)!=120:
            continue
        results.append(res)
        if np.argmax(res)!=label:
            print "confusing:",names[label],names[np.argmax(res)]
        labels.append(label)
        loss=-math.log(res[label]+1e-15)
        prob+=res
        classloss[label]+=loss
        classcount[label]+=1
        ent=-sum(res*np.log(res+1e-15))
        pmax=max(res)
        classent[label]+=ent
        classmax[label]+=pmax
        k+=1
    totloss=np.sum(classloss)/np.sum(classcount)
    totcounts=sum(classcount)
    classloss=classloss/classcount
    classent=classent/classcount
    classmax=classmax/classcount
    s=np.argsort(classloss*classcount)
    classloss=classloss[s]
    classcount=classcount[s]
    prob=prob[s]
    classent=classent[s]
    classmax=classmax[s]
    names=[names[i] for i in s]
    for i,name in enumerate(names):
        print s[i],name, "count:",int(classcount[i]),"loss:",classloss[i], \
                "entropy:",classent[i],"confidence:",classmax[i]
    print "Total loss:", totloss
    print sum(prob)




