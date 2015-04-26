import csv
import numpy as np

FILES=["ensemble_10H_10AR_11H_11AR_DH_DAR.csv","submit-deeper-B.csv"]
weights=[6.0,1.0]
SUBMIT="ensemble-10H-10AR-11H-11AR-DH_DAR_DB.csv"
PATH="plankton/"
average={}
count=0
for ifile,f in enumerate(FILES):
    with open(PATH+f,'rb') as csvfile:
        count+=1
        print "count: ",count
        rows=csv.reader(csvfile)
        header=next(rows, None)
        encode=header[1:]
        for k,row in enumerate(rows): 
            predictions=np.array([float(x) for x in row[1:]])
            #print "sum: ",np.sum(predictions)
            name=row[0]
            if count==1:
                #print "ifile",ifile
                average[name]=predictions*weights[ifile]
            else:
                average[name]=[predictions[i]+average[name][i]*weights[ifile] 
                        for i in range(0,len(predictions))]
print len(average)
with open(PATH+SUBMIT,'w') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow(header)
    for key  in average:
        average[key]=[average[key][k]/sum(weights) for k in range(0,len(average[key]))]
        writer.writerow([key]+average[key])
    
