import csv
import math

FILES=["val9.csv","val10.csv"]
SUBMIT="avgval.csv"
PATH="./"
average={}
label={}
loss=0
count=0
for f in FILES:
    with open(PATH+f,'rb') as csvfile:
        count+=1
        print count
        rows=csv.reader(csvfile)
        header=next(rows, None)
        encode=header[1:]
        for k,row in enumerate(rows): 
            predictions=[float(x) for x in row[1:-1]]
            name=row[0]
            label[name]=row[-1]
            if count==1:
                average[name]=predictions  
            else:
                average[name]=[predictions[i]+average[name][i] 
                        for i in range(0,len(predictions))]
print len(average)
with open(PATH+SUBMIT,'w') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow(header)
    for key  in average:
        average[key]=[average[key][k]/len(FILES) for k in range(0,len(average[key]))]
        writer.writerow([key]+average[key]+[label[key]])
        loss=loss-math.log(average[key][int(label[key])])
    
print "loss of avg: ",loss/len(average)
